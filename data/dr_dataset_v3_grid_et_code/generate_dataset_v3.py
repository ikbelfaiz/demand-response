"""
Dataset Demand Response v3 - quartier de N foyers, Tunis, 2025, pas de 1 minute.

Logique DR :
  consommation globale des foyers (somme des N compteurs)  vs  production disponible (STEG + PV)
  -> is_dr_peak = 1 quand la consommation dépasse la production
  -> un événement DR est lancé la veille à partir d'une prévision (imparfaite) de cet écart
  -> les foyers qui acceptent baissent leur consommation (clim +2,2 °C, report chauffe-eau / machine)

Simulation en deux passes :
  passe 1 : comportement naturel, sans DR  (sert à la prévision et donne le vrai contrefactuel)
  passe 2 : même comportement + réponse aux événements DR  (données "observées")
Le hasard lié au DR utilise un générateur séparé : la seule différence entre les deux passes
est donc l'effet réel du DR, ce qui permet de valider la baseline.

Fichiers produits :
  grid_1min.csv              courbe globale du quartier : consommation, production STEG, PV, pics, événements
  households_1min.parquet    compteur intelligent de chaque foyer (+ sous-compteurs pour les foyers "panel")
  households_info.csv        caractéristiques des foyers
  dr_events.csv              événements DR lancés
  dr_participation.csv       réponse de chaque foyer à chaque événement
  ground_truth_savings.csv   vraie économie par foyer et par événement (pour évaluer la baseline)
"""
import argparse
import os
import numpy as np
import pandas as pd
import dr_core as core

NON_RES_SHARE = 0.35
APPLIANCE_COLS = ["ac_power_w", "water_heater_power_w", "washing_machine_power_w"]


def sample_profiles(n, n_panel, rng):
    profs = []
    for i in range(n):
        has_ac = rng.random() < 0.8
        p = dict(
            client_id=f"C{i + 1:03d}",
            n_people=int(rng.choice([1, 2, 3, 4, 5, 6], p=[.08, .17, .22, .28, .17, .08])),
            has_ac=bool(has_ac),
            has_ac2=bool(has_ac and rng.random() < 0.45),
            has_ewh=bool(rng.random() < 0.55),        # sinon chauffe-eau au gaz / solaire
            wh_kw=float(rng.choice([1.2, 1.5, 2.0])),
            has_wm=bool(rng.random() < 0.92),
            home_all_day=bool(rng.random() < 0.18),   # retraités, parent au foyer
            back=float(rng.normal(17.4, 0.5)),
            shift=float(rng.normal(0, 0.4)),
            base_w=float(np.clip(rng.lognormal(np.log(48), 0.35), 20, 120)),
            thr_on=float(rng.normal(27.3, 1.0)),
            sp=float(rng.normal(24.3, 0.8)),
            guests=float(rng.uniform(0.03, 0.12)),
            wfh=float(rng.uniform(0, 0.2)),
            trip=int(rng.integers(182, 228)) if rng.random() < 0.7 else None,
            trip_len=int(rng.integers(7, 16)),
            n_wkend_away=int(rng.integers(0, 5)),
            accept_base=float(rng.uniform(0.45, 0.9)),
            comply=float(rng.uniform(0.7, 0.95)),
        )
        profs.append(p)
    # foyers panel (sous-compteurs) : choisis parmi ceux qui ont les 3 appareils
    full = [i for i, p in enumerate(profs) if p["has_ac"] and p["has_ewh"] and p["has_wm"]]
    panel = set(rng.choice(full, min(n_panel, len(full)), replace=False).tolist())
    for i, p in enumerate(profs):
        p["has_submeter"] = i in panel
    return profs


def rolling_mean(x, w=15):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


def calibrate_supply(cons, nat_supply, ctx, target_days):
    """Facteur d'échelle de la production pour obtenir ~target_days jours de pic."""
    cs = rolling_mean(cons)
    lo, hi = 0.0, cs.max() / nat_supply.min() * 2
    for _ in range(50):
        k = (lo + hi) / 2
        days = np.unique(ctx["d"][cs > k * nat_supply]).size
        lo, hi = (k, hi) if days > target_days else (lo, k)
    return (lo + hi) / 2


def build_events(margin, supply_mean, ctx, rng):
    """Décision J-1 sur une prévision bruitée de la marge (production - consommation) du soir."""
    n, ND = ctx["n"], ctx["ND"]
    is_event = np.zeros(n, bool)
    windows = {}
    for day in range(ND):
        i0 = day * 1440
        seg = margin[i0 + 11 * 60:i0 + 1440].reshape(-1, 60).mean(1)       # moyennes horaires 11h-24h
        fc = seg + rng.normal(0, 0.035 * supply_mean) + rng.normal(0, 0.012 * supply_mean, len(seg))
        k = int(np.argmin(fc))
        if fc[k] < 0:
            launch = rng.random() < 0.9
        elif fc[k] < 0.08 * supply_mean:
            launch = rng.random() < 0.6
        else:
            launch = False
        if not launch:
            continue
        peak_h = 11 + k + rng.choice([-1, 0, 0, 0, 1])
        start_h = int(np.clip(peak_h - 1 if rng.random() < 0.5 else peak_h, 12, 22))
        s = i0 + start_h * 60
        e = min(s + 120, n)
        is_event[s:e] = True
        windows[day] = (s, e)
    return is_event, windows


def gaps(v, n_gaps, med_min, single_rate, rng):
    v = v.astype(float).copy()
    n = len(v)
    v[rng.random(n) < single_rate] = np.nan
    for _ in range(n_gaps):
        s = int(rng.integers(0, n))
        v[s:s + int(np.clip(rng.lognormal(np.log(med_min), 1.0), 5, 3 * 1440))] = np.nan
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--households", type=int, default=50)
    ap.add_argument("--panel", type=int, default=10, help="foyers équipés de sous-compteurs (NILM)")
    ap.add_argument("--peak-days", type=int, default=45, help="nombre visé de jours de pic sans DR")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="dr_dataset_v3")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    ctx = core.build_context(rng)
    n = ctx["n"]
    profs = sample_profiles(args.households, args.panel, np.random.default_rng(args.seed + 1))
    seeds = [args.seed * 1000 + i for i in range(args.households)]

    # ---------------- passe 1 : sans DR (contrefactuel)
    natural = np.zeros((args.households, n), dtype=np.float32)
    for i, p in enumerate(profs):
        tot, _, _ = core.build_household(ctx, {}, np.random.default_rng(seeds[i]), p,
                                         np.random.default_rng(seeds[i] + 500_000))
        natural[i] = tot
    cons0_kw = natural.sum(0) / 1000

    # ---------------- production disponible du quartier
    # part du quartier dans la capacité STEG + PV, calibrée pour ~peak_days jours de pic
    # capacité restante pour le résidentiel = capacité STEG + PV - demande des autres secteurs
    # (industrie, services : hypothèse 35 % de la demande nationale)
    steg_res = ctx["steg"] - NON_RES_SHARE * ctx["zone"]
    nat_supply = steg_res + ctx["pv"]
    k = calibrate_supply(cons0_kw, nat_supply, ctx, args.peak_days)
    steg_kw, pv_kw = k * steg_res, k * ctx["pv"]
    supply_kw = steg_kw + pv_kw

    # ---------------- événements DR (décidés sur prévision bruitée)
    margin0 = supply_kw - rolling_mean(cons0_kw)
    is_event, windows = build_events(margin0, supply_kw.mean(), ctx, np.random.default_rng(args.seed + 2))

    # ---------------- passe 2 : avec DR
    rows, part, truth = [], [], []
    actual_kw = np.zeros(n)
    rng_q = np.random.default_rng(args.seed + 3)
    ts = ctx["ts"]
    for i, p in enumerate(profs):
        tot, L, accept = core.build_household(ctx, windows, np.random.default_rng(seeds[i]), p,
                                              np.random.default_rng(seeds[i] + 500_000))
        actual_kw += tot / 1000
        for ev_id, (day, (s, e)) in enumerate(sorted(windows.items()), start=1):
            a, comply, resp = accept[day]
            part.append((ev_id, p["client_id"], resp))
            truth.append((ev_id, p["client_id"], resp, bool(comply),
                          round(natural[i, s:e].sum() / 60 / 1000, 3), round(tot[s:e].sum() / 60 / 1000, 3)))

        # compteur intelligent : trous, pics aberrants, valeurs figées
        agg = gaps(np.round(tot), int(rng_q.integers(2, 8)), 180, 0.0015, rng_q)
        spk = rng_q.integers(0, n, int(rng_q.integers(3, 12)))
        agg[spk] = agg[spk] * rng_q.uniform(5, 15, len(spk)) + 500
        for _ in range(int(rng_q.integers(0, 4))):
            s0 = int(rng_q.integers(0, n - 60))
            agg[s0:s0 + int(rng_q.uniform(20, 50))] = agg[s0]
        d = {"timestamp": ts, "client_id": p["client_id"], "aggregate_power_w": agg}
        for col, key in zip(APPLIANCE_COLS, ["ac", "wh", "wm"]):
            if p["has_submeter"]:
                x = np.clip(L[key] * rng_q.normal(1, 0.015) + rng_q.normal(0, 2, n), 0, None)
                d[col] = gaps(np.round(x), 2, 300, 0.001, rng_q)
            else:
                d[col] = np.full(n, np.nan)
        hh = pd.DataFrame(d)
        for c in ["aggregate_power_w"] + APPLIANCE_COLS:
            hh[c] = hh[c].round().astype("Int32")
        rows.append(hh)

    # ---------------- courbe globale du quartier (compteur du poste de distribution)
    feeder_kw = actual_kw * 1.03 + rng_q.normal(0, 0.15, n)          # + 3 % de pertes réseau
    is_peak = rolling_mean(feeder_kw) > supply_kw
    grid = pd.DataFrame({
        "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
        "region_id": "TUN",
        "temperature_c": ctx["temp_obs"],
        "is_holiday": ctx["hol"][ctx["d"]].astype(int),
        "is_ramadan": ctx["ram"][ctx["d"]].astype(int),
        "households_consumption_kw": gaps(np.round(feeder_kw, 2), 3, 30, 0.0002, rng_q),
        "steg_production_kw": np.round(steg_kw, 2),
        "pv_production_kw": gaps(np.round(pv_kw, 2), 2, 60, 0.0002, rng_q),
        "is_dr_event": is_event.astype(int),
        "is_dr_peak": is_peak.astype(int),
    })
    grid.loc[grid.households_consumption_kw.isna() | grid.pv_production_kw.isna(), "is_dr_peak"] = pd.NA
    grid["is_dr_peak"] = grid["is_dr_peak"].astype("Int8")
    grid.to_csv(f"{args.outdir}/grid_1min.csv", index=False)

    import pyarrow as pa, pyarrow.parquet as pq
    hh_all = pd.concat(rows, ignore_index=True)
    hh_all["timestamp"] = hh_all["timestamp"].astype("datetime64[s]")
    tbl = pa.Table.from_pandas(hh_all, preserve_index=False)
    pq.write_table(tbl, f"{args.outdir}/households_1min.parquet", compression="zstd", compression_level=9,
                   use_dictionary=["client_id"], column_encoding={"timestamp": "DELTA_BINARY_PACKED"})
    info = pd.DataFrame([{
        "client_id": p["client_id"], "region_id": "TUN", "n_occupants": p["n_people"],
        "has_ac": p["has_ac"], "has_second_ac": p["has_ac2"], "has_electric_water_heater": p["has_ewh"],
        "water_heater_kw": p["wh_kw"] if p["has_ewh"] else None, "has_washing_machine": p["has_wm"],
        "occupied_daytime": p["home_all_day"], "has_submeter": p["has_submeter"],
    } for p in profs])
    info.to_csv(f"{args.outdir}/households_info.csv", index=False)

    ev = pd.DataFrame([{
        "event_id": i, "region_id": "TUN",
        "start_ts": ts[s].strftime("%Y-%m-%d %H:%M"), "end_ts": ts[min(e, n - 1)].strftime("%Y-%m-%d %H:%M"),
        "surcharge_level": 2, "reduction_target_pct": 20,
    } for i, (day, (s, e)) in enumerate(sorted(windows.items()), start=1)])
    ev.to_csv(f"{args.outdir}/dr_events.csv", index=False)
    pd.DataFrame(part, columns=["event_id", "client_id", "response"]).to_csv(
        f"{args.outdir}/dr_participation.csv", index=False)
    pd.DataFrame(truth, columns=["event_id", "client_id", "response", "complied",
                                 "true_baseline_kwh", "actual_kwh"]).to_csv(
        f"{args.outdir}/ground_truth_savings.csv", index=False)

    np.save(f"{args.outdir}/_natural_kw.npy", cons0_kw.astype(np.float32))   # contrôle interne
    print(f"foyers: {args.households} | facteur production: {k:.5f} | événements: {len(windows)}")


if __name__ == "__main__":
    main()
