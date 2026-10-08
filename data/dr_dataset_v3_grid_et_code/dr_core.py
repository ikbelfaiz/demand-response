"""
Dataset demand response réaliste - Tunis, année 2025, pas de 1 minute.

Réalisme ajouté par rapport à la v1 :
- Météo : vagues de chaleur, régimes nuageux persistants, station horaire interpolée, trous.
- PV : position réelle du soleil (lat 36.8), passages nuageux, encrassement, déclassement thermique.
- Zone : inertie thermique, horaire d'été (séance unique), Ramadan, bruit autocorrélé.
- STEG : maintenances planifiées, pannes aléatoires (plus fréquentes par forte chaleur), déclassement.
- Événements DR décidés sur une prévision imparfaite (faux positifs / pics manqués).
- Foyer : horaires variables, télétravail, vacances, invités, coupures de courant,
  climatiseur inverter, 2e climatiseur NON sous-compté, appareils résistifs parasites
  (four, fer, sèche-cheveux, radiateur) qui ressemblent au chauffe-eau, réponse DR partielle.
- Qualité des données : trous de communication, pics aberrants, valeurs figées,
  erreurs de calibration des sous-compteurs.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.signal import lfilter

YEAR = 2025
LAT = np.deg2rad(36.8)
HOLIDAYS = pd.to_datetime([
    "2025-01-01", "2025-03-20", "2025-03-30", "2025-03-31", "2025-04-01", "2025-04-09",
    "2025-05-01", "2025-06-06", "2025-06-07", "2025-06-26", "2025-07-25", "2025-08-13",
    "2025-09-04", "2025-10-15", "2025-12-17"])
RAMADAN = ("2025-03-01", "2025-03-29")
SUMMER_SCHEDULE = ("2025-07-01", "2025-08-31")
SCHOOL_VAC = [("2025-01-01", "2025-01-05"), ("2025-03-16", "2025-03-30"),
              ("2025-06-15", "2025-09-14"), ("2025-10-27", "2025-11-02"), ("2025-12-21", "2025-12-31")]
FAMILY_TRIP = ("2025-08-05", "2025-08-17")


def ar1(rng, n, phi, sigma):
    return lfilter([1.0], [1.0, -phi], rng.normal(0, sigma, n))


def bump(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)


def in_range(dates, a, b):
    return (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))


# --------------------------------------------------------------------------- contexte
def build_context(rng):
    ts = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31 23:59", freq="1min")
    n = len(ts)
    m = np.arange(n)
    d = m // 1440
    h = (m % 1440) / 60.0
    days = pd.date_range(f"{YEAR}-01-01", periods=365)
    ND = len(days)
    doy_d = np.arange(1, ND + 1)
    doy = d + 1

    hol = days.isin(HOLIDAYS)
    ram = in_range(days, *RAMADAN)
    wkend = days.dayofweek.values >= 5
    summer_sched = in_range(days, *SUMMER_SCHEDULE)
    school_vac = np.zeros(ND, bool)
    for a, b in SCHOOL_VAC:
        school_vac |= in_range(days, a, b)

    # --- soleil
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    ha = np.deg2rad(15 * (h - 12.35))
    sin_el = np.sin(LAT) * np.sin(decl) + np.cos(LAT) * np.cos(decl) * np.cos(ha)
    elev = np.rad2deg(np.arcsin(np.clip(sin_el, -1, 1)))
    s = np.sin(2 * np.pi * (doy - 80) / 365)
    sunset = 18.6 + 1.35 * s

    # --- régimes nuageux (0 clair, 1 partiel, 2 couvert) avec persistance
    seas = np.sin(2 * np.pi * (doy_d - 110) / 365)
    p_clear = 0.675 + 0.225 * seas
    regime = np.zeros(ND, int)
    for i in range(ND):
        if i > 0 and rng.random() < 0.55:
            regime[i] = regime[i - 1]
        else:
            u = rng.random()
            regime[i] = 0 if u < p_clear[i] else (1 if u < p_clear[i] + (1 - p_clear[i]) * 0.6 else 2)

    # --- température journalière
    t_mean = 19.5 + 8.0 * np.sin(2 * np.pi * (doy_d - 112) / 365)
    anom = ar1(rng, ND, 0.75, 1.5)
    for _ in range(4):  # vagues de chaleur (sirocco)
        st = rng.integers(155, 235); ln = rng.integers(3, 7)
        anom[st:st + ln] += 6.5 * np.sin(np.linspace(0.3, np.pi - 0.3, len(anom[st:st + ln])))
    for _ in range(2):  # vagues de froid
        st = rng.choice(np.r_[0:50, 330:360]); ln = rng.integers(3, 6)
        anom[st:st + ln] -= 4.0
    anom = np.where(regime == 2, anom - 1.8, anom)
    amp = (4.5 + 1.5 * seas) * np.where(regime == 2, 0.45, np.where(regime == 1, 0.8, 1.0))
    hours = np.arange(ND * 24)
    hd = hours // 24; hh = hours % 24
    t_hour = t_mean[hd] + anom[hd] + amp[hd] * np.cos(2 * np.pi * (hh - 15) / 24) + rng.normal(0, 0.35, len(hours))
    temp_true = np.interp(m / 60.0, hours, t_hour)
    temp_obs = np.round(temp_true, 1)
    miss_h = rng.random(len(hours)) < 0.005
    temp_obs[np.repeat(miss_h, 60)] = np.nan
    t_eff = lfilter([1 / 180], [1, -(1 - 1 / 180)], temp_true, zi=[temp_true[0] * (1 - 1 / 180)])[0]

    # --- PV de zone
    cloud_pass = (lfilter([1 / 8], [1, -(1 - 1 / 8)], (rng.random(n) < 0.04).astype(float) * 8) > 0.35)
    cloud_pass = lfilter([1 / 10], [1, -0.9], cloud_pass.astype(float))
    reg_m = regime[d]
    clear = np.where(reg_m == 0, 0.93 + ar1(rng, n, 0.99, 0.002),
             np.where(reg_m == 1, 0.80 - 0.40 * cloud_pass + ar1(rng, n, 0.995, 0.004),
                      0.32 + ar1(rng, n, 0.998, 0.006)))
    soil = np.ones(ND)
    for i in range(1, ND):
        soil[i] = 1.0 if (regime[i] == 2 and rng.random() < 0.5) else max(0.9, soil[i - 1] - 0.0012)
    cap = np.where(days >= pd.Timestamp("2025-06-15"), 340.0, 300.0)  # mise en service d'une centrale
    derate = 1 - 0.004 * np.clip(temp_true + 22 - 25, 0, None)
    pv = cap[d] * 0.9 * np.clip(sin_el, 0, None) ** 1.2 * np.clip(clear, 0.05, 1) * soil[d] * derate
    pv = np.clip(pv, 0, None)

    # --- consommation de zone
    ram_m, hol_m, wk_m, ss_m = ram[d], hol[d], wkend[d], summer_sched[d]
    shape_std = (0.60 + 0.17 * bump(h, 9.8, 2.2) + 0.20 * bump(h, 13.0, 2.8)
                 + 0.36 * bump(h, sunset + 1.2, 1.5) + 0.09 * bump(h, 22.6, 1.1))
    shape_ss = (0.60 + 0.20 * bump(h, 10.5, 2.3) + 0.12 * bump(h, 14.5, 2.5)
                + 0.36 * bump(h, sunset + 1.3, 1.6) + 0.12 * bump(h, 23.0, 1.2))
    shape_ram = (0.58 + 0.15 * bump(h, 11.0, 2.8) + 0.10 * bump(h, sunset - 0.8, 0.7)
                 + 0.40 * bump(h, sunset + 2.0, 1.7) + 0.18 * bump(h, 0.8, 1.1)
                 + 0.18 * bump(h, 24.8, 1.1) + 0.08 * bump(h, 3.9, 0.5))
    shape = np.where(ram_m, shape_ram, np.where(ss_m, shape_ss, shape_std))
    dayf = np.where(hol_m, 0.86, np.where(wk_m, 0.93, 1.0)) * (1 + ar1(rng, ND, 0.7, 0.012))[d]
    cool = (22.0 * np.clip(t_eff - 23, 0, None) + 0.9 * np.clip(t_eff - 30, 0, None) ** 2) \
        * (0.75 + 0.35 * bump(h, 16.5, 5))
    heat = 14.0 * np.clip(14 - t_eff, 0, None) * (0.4 + 0.6 * np.clip(bump(h, sunset + 1.5, 2.5) + bump(h, 7.5, 1.5), 0, 1))
    growth = 1 + 0.025 * (doy / 365)
    zone = (830 * shape * dayf + cool + heat) * growth + ar1(rng, n, 0.995, 0.4) + rng.normal(0, 1.5, n)

    # --- capacité STEG disponible
    steg = np.full(n, 1015.0)
    steg -= np.where(in_range(days, "2025-03-03", "2025-04-12"), 110, 0)[d]
    steg -= np.where(in_range(days, "2025-10-13", "2025-11-16"), 130, 0)[d]
    t = 0
    while t < n:
        hot = t_eff[t] > 32
        t += int(rng.exponential(1440 * (9 if hot else 20)))
        if t >= n:
            break
        dur = int(np.clip(rng.lognormal(np.log(1440), 0.9), 240, 1440 * 6))
        steg[t:t + dur] -= rng.uniform(60, 170)
        t += dur
    steg -= 3.5 * np.clip(t_eff - 28, 0, None)
    steg += ar1(rng, n, 0.998, 0.25)

    return dict(ts=ts, n=n, m=m, d=d, h=h, ND=ND, days=days, doy=doy, elev=elev, sunset=sunset,
                temp_true=temp_true, temp_obs=temp_obs, t_eff=t_eff, pv=pv, zone=zone, steg=steg,
                hol=hol, ram=ram, wkend=wkend, summer_sched=summer_sched, school_vac=school_vac)


# --------------------------------------------------------------------------- événements DR
def build_events(ctx, rng):
    n, h, d, ND = ctx["n"], ctx["h"], ctx["d"], ctx["ND"]
    margin = ctx["steg"] + ctx["pv"] - ctx["zone"]
    is_event = np.zeros(n, bool)
    windows = {}
    for day in range(ND):
        i0 = day * 1440
        seg = margin[i0 + 11 * 60:i0 + 1440].reshape(-1, 60).mean(1)  # moyennes horaires 11h-24h
        fc = seg + rng.normal(0, 28) + rng.normal(0, 10, len(seg))  # prévision J-1 imparfaite
        k = int(np.argmin(fc)); fmin = fc[k]
        if fmin < 0:
            launch = rng.random() < 0.9
        elif fmin < 25:
            launch = rng.random() < 0.35
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


# --------------------------------------------------------------------------- foyer
def build_household(ctx, windows, rng, prof, rng_dr):
    n, h, d, ND = ctx["n"], ctx["h"], ctx["d"], ctx["ND"]
    days, temp, t_eff, elev, sunset = ctx["days"], ctx["temp_true"], ctx["t_eff"], ctx["elev"], ctx["sunset"]
    ram, hol, wkend, ss, svac = ctx["ram"], ctx["hol"], ctx["wkend"], ctx["summer_sched"], ctx["school_vac"]
    free_day = hol | wkend

    L = {k: np.zeros(n) for k in ["base", "fridge", "light", "elec", "small", "ac", "ac2", "wh", "wm"]}
    occupied = np.zeros(n, bool)
    awake = np.zeros(n, bool)
    volt = 1 + 0.012 * ar1(rng, n, 0.999, 0.045)  # tension relative -> charges résistives

    away = np.zeros(ND, bool)
    if prof['trip'] is not None:
        away[prof['trip']:prof['trip'] + prof['trip_len']] = True
    for _ in range(prof['n_wkend_away']):  # week-ends hors de la maison
        sat = rng.choice(np.where(days.dayofweek == 5)[0])
        away[sat:sat + 2] = True
    guests = rng.random(ND) < prof['guests']
    wfh = (rng.random(ND) < prof['wfh']) & ~free_day

    # acceptation DR : dépend de la chaleur et de la fatigue (événements consécutifs)
    accept, done_prev = {}, 0
    for day in sorted(windows):
        p = prof['accept_base'] - 0.04 * max(0, t_eff[windows[day][0]] - 32) - 0.08 * done_prev
        a = rng_dr.random() < np.clip(p, 0.1, 0.95)
        comply = a and rng_dr.random() < prof['comply']  # accepte mais n'agit pas toujours
        resp = 'accept' if a else ('no_response' if rng_dr.random() < 0.35 else 'decline')
        accept[day] = (a, comply, resp)
        done_prev = done_prev + 1 if (day - 1) in windows else 0
    in_evt = np.zeros(n, bool)
    for day, (s, e) in windows.items():
        if accept[day][1]:
            in_evt[s:e] = True

    # -------- routines journalières
    plan = []
    for day in range(ND):
        i0 = day * 1440
        if away[day]:
            plan.append(None)
            continue
        if ram[day]:
            wake = rng.normal(7.3, 0.3) if not free_day[day] else rng.normal(9.5, 0.8)
            sleep = rng.normal(24.9, 0.4)
            leave, back = rng.normal(8.1, 0.2), rng.normal(15.2, 0.3)
            awake[i0 + int(rng.normal(3.8, 0.1) * 60): i0 + int(4.4 * 60)] = True
        else:
            wake = rng.normal(6.3 + prof['shift'], 0.25) if not free_day[day] else rng.normal(8.4 + prof['shift'], 0.7)
            sleep = rng.normal(23.3, 0.45) + (0.6 if svac[day] else 0) + (0.7 if free_day[day] else 0)
            leave = rng.normal(7.55, 0.15)
            back = rng.normal(14.6, 0.4) if ss[day] else rng.normal(prof['back'], 0.6)
        awake[i0 + int(wake * 60): i0 + int(min(sleep, 26) * 60)] = True
        occupied[i0:i0 + 1440] = True
        daytime_home = prof['home_all_day'] or free_day[day] or wfh[day] or (svac[day] and rng.random() < 0.7)
        if not daytime_home:
            occupied[i0 + int(leave * 60): i0 + int(back * 60)] = False
        elif rng.random() < 0.25:  # sortie en famille
            o = rng.uniform(10, 17)
            occupied[i0 + int(o * 60): i0 + int((o + rng.uniform(1.5, 4)) * 60)] = False
        if rng.random() < 0.10 and not ram[day]:  # sortie le soir
            o = rng.uniform(19, 21)
            occupied[i0 + int(o * 60): i0 + int((o + rng.uniform(1.5, 3)) * 60)] = False
        plan.append(dict(wake=wake, sleep=sleep, back=back if not daytime_home else wake, home=daytime_home))
    home_awake = occupied & awake

    # -------- veille + électronique + éclairage
    L["base"] = prof['base_w'] + np.repeat(rng.normal(0, 5, ND), 1440) + ar1(rng, n, 0.995, 0.3)
    L["base"] = np.where(occupied, L["base"], L["base"] * 0.8)
    tv_on = home_awake & ((h >= 18.5) | (h < 1.5) | (np.repeat(free_day | ctx["school_vac"], 1440) & (h > 13)))
    L["elec"] = np.where(tv_on, 90 + 60 * (1 + np.tanh(ar1(rng, n, 0.99, 0.08))), 0) \
        + np.where(home_awake, 15 + 10 * rng.random(n), 0)
    dark = elev < 5
    u = 1 / (1 + np.exp(-ar1(rng, n, 0.99, 0.12)))
    L["light"] = np.where(home_awake & dark, 40 + 180 * u, 0) + np.where(home_awake & dark & np.repeat(guests, 1440), 110, 0)

    # -------- réfrigérateur (cycles irréguliers + dégivrage)
    t_in = np.clip(21 + 0.45 * (t_eff - 21), 16, 31)
    t = int(rng.integers(0, 30))
    next_defrost = int(rng.uniform(900, 1500))
    while t < n:
        k = min(t, n - 1)
        on = int(max(7, rng.normal(13, 2.5) * (1 + 0.04 * (t_in[k] - 22)) * (1.25 if home_awake[k] and rng.random() < 0.3 else 1)))
        off = int(max(8, rng.normal(28, 5) * (1 - 0.035 * (t_in[k] - 22))))
        L["fridge"][t:t + on] = rng.normal(102, 5)
        L["fridge"][t] = 165
        t += on + off
        if t > next_defrost:
            L["fridge"][t:t + 22] = rng.normal(235, 8)
            t += 22 + 5
            L["fridge"][t:t + 35] = 110
            t += 45
            next_defrost = t + int(rng.uniform(1300, 1700))

    # -------- utilitaires
    def place(center_h, sd, day, lo=0.0, hi=23.9):
        hh_ = float(np.clip(rng.normal(center_h, sd), lo, hi))
        idx = day * 1440 + int(hh_ * 60)
        return idx if idx < n and home_awake[idx] else None

    def cycling(length, p_on, duty, period, lvl_off=0.0):
        x = np.full(length, lvl_off)
        t_ = 0
        while t_ < length:
            a = max(1, int(round(period * duty * rng.uniform(0.7, 1.3))))
            b = max(1, int(round(period * (1 - duty) * rng.uniform(0.7, 1.3))))
            x[t_:t_ + a] = p_on
            t_ += a + b
        return x

    def add(arr, s, prof):
        e = min(s + len(prof), n)
        arr[s:e] += prof[:e - s]

    # -------- petits appareils non sous-comptés (dont charges résistives "parasites")
    for day in range(ND):
        p = plan[day]
        if p is None:
            continue
        mult = (1.6 if guests[day] else 1.0) * prof['n_people'] / 4
        for _ in range(rng.integers(1, 3)):
            s = place(p["wake"] + 0.3, 0.2, day)
            if s: add(L["small"], s, np.full(rng.integers(2, 5), rng.uniform(1800, 2200)))
        for c in ([13.0, 20.3] if not ram[day] else [sunset[day * 1440] - 0.6, 3.9]):
            for _ in range(rng.poisson(1.0 * mult)):
                s = place(c, 0.8, day)
                if s: add(L["small"], s, np.full(rng.integers(2, 7), rng.uniform(850, 1250)))
        if not free_day[day] and rng.random() < 0.5:
            s = place(p["wake"] + 0.4, 0.2, day)
            if s: add(L["small"], s, np.full(rng.integers(4, 11), rng.uniform(1300, 1800)))  # sèche-cheveux
        p_oven = (0.5 if ram[day] else 0.32 if free_day[day] else 0.16) * mult
        if rng.random() < p_oven:
            c = sunset[day * 1440] - 1.5 if ram[day] else (12.2 if free_day[day] else 19.3)
            s = place(c, 0.6, day)
            if s:
                ln = int(rng.uniform(40, 95))
                oven = np.r_[np.full(12, 2150.0), cycling(ln - 12, 2150, rng.uniform(0.35, 0.55), 5)]
                add(L["small"], s, oven)
        if rng.random() < 0.28:  # fer à repasser
            s = place(18.0 if not free_day[day] else 11.0, 1.5, day)
            if s: add(L["small"], s, cycling(int(rng.uniform(20, 45)), rng.uniform(1000, 1300), 0.4, 3))
        if rng.random() < (0.5 if free_day[day] else 0.08):  # aspirateur
            s = place(10.5, 1.0, day)
            if s: add(L["small"], s, np.full(int(rng.uniform(10, 25)), rng.uniform(1200, 1500)))
        if t_eff[day * 1440 + 8 * 60] < 13 and rng.random() < 0.5:  # radiateur soufflant salle de bain
            s = place(p["wake"] + 0.2, 0.2, day)
            if s: add(L["small"], s, np.full(int(rng.uniform(15, 45)), rng.uniform(1400, 1650)))
        if rng.random() < 0.18:  # plaque de cuisson électrique
            s = place(12.8 if free_day[day] else 19.8, 0.7, day)
            if s: add(L["small"], s, cycling(int(rng.uniform(20, 55)), rng.uniform(1300, 1700), 0.75, 6))
        if t_eff[day * 1440 + 20 * 60] < 11 and rng.random() < 0.45:  # radiateur bain d'huile
            s = place(20.5, 0.5, day)
            if s:
                ln = int(max(30, (p["sleep"] - 20.5) * 60))
                add(L["small"], s, cycling(ln, rng.choice([1000, 2000]), 0.6, 12))

    # -------- climatiseur séjour (inverter, sous-compté) + chambre (NON sous-compté)
    for day in range(ND):
        p = plan[day]
        i0 = day * 1440
        seg = slice(i0, i0 + 1440)
        L["ac"][seg] = 3.0  # veille de l'unité
        if p is None:
            continue
        T = temp[seg]; Te = t_eff[seg]
        ha_ = home_awake[seg]
        thr_on = rng.normal(prof['thr_on'], 0.6)
        use_day = rng.random() < 0.93
        cooling = ha_ & (Te > thr_on) & use_day & prof['has_ac']
        heating = ha_ & (Te < rng.normal(12.5, 0.8)) & (rng.random() < 0.65) & prof['has_ac'] & ((h[seg] > 18) | (h[seg] < 9))
        for mode, mask in (("c", cooling), ("h", heating)):
            if not mask.any():
                continue
            # lisser les sessions (supprime les micro-arrêts < 25 min et sessions < 20 min)
            idx = np.where(mask)[0]
            runs = np.split(idx, np.where(np.diff(idx) > 25)[0] + 1)
            for r in runs:
                if len(r) < 20:
                    continue
                a, b = r[0] + int(rng.uniform(0, 20)), r[-1]
                if b <= a:
                    continue
                sp = rng.normal(prof['sp'], 0.7) if mode == "c" else rng.normal(21.5, 0.6)
                k = np.arange(a, b)
                sp_arr = np.full(len(k), sp) + np.where(in_evt[i0 + k], 2.2, 0)
                delta = (T[k] - sp_arr) if mode == "c" else (sp_arr - T[k])
                pw = 230 + 105 * delta + ar1(rng, len(k), 0.95, 25)
                pw = np.clip(pw, 0, 1550)
                low = delta < 3
                cyc = cycling(len(k), 1.0, 0.55, 18)
                pw = np.where(low & (cyc < 0.5), 28, np.maximum(pw, np.where(low, 320, 0)))
                pw[:int(rng.uniform(8, 15))] = rng.uniform(1400, 1600)
                L["ac"][i0 + k] = pw
        # chambre, la nuit
        night_T = t_eff[min(i0 + 23 * 60, n - 1)]
        if night_T > 25.5 and rng.random() < 0.8 and prof['has_ac2']:
            a = i0 + int((p["sleep"] - 0.4) * 60)
            ln = int(rng.uniform(150, 420))
            k = np.arange(a, min(a + ln, n))
            dl = temp[k] - 25.5
            pw = np.clip(180 + 80 * dl + ar1(rng, len(k), 0.95, 15), 0, 900)
            pw = np.where(cycling(len(k), 1.0, 0.6, 20) > 0.5, np.maximum(pw, 220), 20)
            pw[:10] = 900
            L["ac2"][k] = pw

    # -------- chauffe-eau (sous-compté) : réchauffes après puisages + pertes
    t_inlet = 20 + 6 * np.sin(2 * np.pi * (np.arange(ND) - 115) / 365)
    need = np.zeros(n, bool)
    for day in range(ND):
        p = plan[day]
        i0 = day * 1440
        if p is None:
            if rng.random() < 0.5:  # chauffe-eau laissé allumé : pertes seulement
                need[i0 + int(rng.uniform(0, 1400)): i0 + int(rng.uniform(0, 1400)) + 10] = True
            continue
        n_show = rng.binomial(prof['n_people'], 0.85 if t_eff[i0 + 900] > 24 else 0.6)
        draws = []
        for j in range(n_show):
            if ram[day]:
                c = sunset[i0] + rng.uniform(1.5, 4.0)
            elif j < 2 and not free_day[day] and rng.random() < 0.55:
                c = p["wake"] + rng.uniform(0.1, 0.8)
            else:
                c = rng.normal(20.8, 0.6) if not free_day[day] else rng.normal(19.5, 2.0)
            draws.append((c, rng.uniform(22, 38)))  # litres d'eau chaude
        for c in (13.3, 21.0):
            if rng.random() < 0.7:
                draws.append((rng.normal(c, 0.4), rng.uniform(4, 10)))  # vaisselle
        for c, vol in draws:
            s = i0 + int(np.clip(c, 0, 26.5) * 60) + int(rng.uniform(2, 8))
            kwh = vol * 4.186 * (58 - t_inlet[day]) / 3600
            dur = int(kwh / 1.5 * 60 * rng.uniform(0.9, 1.1))
            need[s:s + dur] = True
    # pertes thermiques : petite réchauffe après ~6-8 h sans chauffe
    last = 0
    for t_ in range(0, n, 30):
        if need[t_:t_ + 30].any():
            last = t_
        elif t_ - last > rng.uniform(360, 480):
            need[t_:t_ + int(rng.uniform(7, 13))] = True
            last = t_
    # report pendant les événements DR acceptés (effet rebond)
    for day, (s, e) in windows.items():
        if accept[day][1] and need[s:e].any():
            k = int(need[s:e].sum())
            need[s:e] = False
            need[e:e + k] = True
    L["wh"] = np.where(need, prof['wh_kw'] * 1000 * volt ** 2, 0) * prof['has_ewh']

    # -------- machine à laver (sous-comptée)
    def wm_program(kind, day):
        tin = t_inlet[day]
        target = {"c60": 60, "e40": 40, "q30": 30, "s40": 40}[kind]
        heat = max(3, int(12 * 4.186 * (target - tin) / 3600 / 2.0 * 60 * rng.uniform(0.9, 1.15)))
        wash = {"c60": 55, "e40": 85, "q30": 12, "s40": 38}[kind]
        rinses = 1 if kind == "q30" else rng.integers(2, 4)
        prof = [np.full(rng.integers(2, 5), 20.0)]
        prof.append(2000 * volt[0] ** 2 + rng.normal(0, 30, heat) + np.where(rng.random(heat) < 0.3, 80, 0))
        tumble = 10 + 300 * np.clip(rng.normal(0.45, 0.15, wash), 0.1, 0.9)
        if kind == "c60":
            tumble[rng.integers(10, max(11, wash - 5), 2)] = 2000
        prof.append(tumble)
        for _ in range(rinses):
            prof += [np.full(1, 45.0), np.full(2, 20.0), 10 + 250 * np.clip(rng.normal(0.4, 0.1, 5), 0.1, 0.9),
                     np.full(2, rng.uniform(180, 300))]
        spin = int(rng.uniform(6, 12))
        prof.append(np.linspace(150, rng.uniform(380, 620), spin) + rng.normal(0, 25, spin))
        prof.append(np.full(1, 45.0))
        prof.append(np.full(int(rng.uniform(0, 60)), 2.0))
        return np.concatenate(prof)

    after_trip = False
    for day in range(ND):
        p = plan[day]
        if p is None:
            after_trip = True
            continue
        pl = min(0.95, 0.45 * prof['n_people'] / 4)
        n_loads = 2 if after_trip else (rng.binomial(2, pl) if free_day[day] else int(rng.random() < pl))
        n_loads *= int(prof['has_wm'])
        after_trip = False
        for j in range(n_loads):
            if free_day[day] or p["home"]:
                c = rng.uniform(9, 13) + j * 2.2
            else:
                c = rng.uniform(17.8, 21.0) if p["back"] < 18.5 else p["back"] + 0.5
            s = day * 1440 + int(c * 60)
            if s >= n or not occupied[s]:
                continue
            kind = rng.choice(["c60", "e40", "q30", "s40"], p=[0.2, 0.45, 0.25, 0.1])
            wprog = wm_program(kind, day)
            if day in windows and accept[day][1]:
                ws, we = windows[day]
                if s < we and s + len(wprog) > ws:
                    s = we + int(rng_dr.uniform(0, 40))
            add(L["wm"], s, wprog)

    # -------- coupures de courant (réseau local)
    for _ in range(6):
        day = int(rng.choice(np.r_[150:260, 0:365], p=None))
        s = day * 1440 + int(rng.uniform(12, 23) * 60)
        dur = int(rng.uniform(15, 150))
        for k_ in L:
            L[k_][s:s + dur] = 0
        L["fridge"][s + dur:s + dur + 40] = 110

    true_total = sum(L.values()) + 12 * (1 + np.tanh(ar1(rng, n, 0.98, 0.15))) + rng.normal(0, 6, n)
    true_total = np.clip(true_total, 0, None)
    return true_total, L, accept


# --------------------------------------------------------------------------- qualité de données
def degrade(df, rng):
    n = len(df)

    def gaps(col, n_gaps, med_min, single_rate):
        v = df[col].to_numpy(dtype=float, copy=True)
        v[rng.random(n) < single_rate] = np.nan
        for _ in range(n_gaps):
            s = int(rng.integers(0, n))
            v[s:s + int(np.clip(rng.lognormal(np.log(med_min), 1.0), 5, 3 * 1440))] = np.nan
        df[col] = v

    gaps("aggregate_power_w", 6, 180, 0.0015)
    gaps("ac_power_w", 2, 360, 0.001)
    gaps("water_heater_power_w", 2, 240, 0.001)
    gaps("washing_machine_power_w", 1, 2 * 1440, 0.001)
    gaps("zone_consumption_mw", 4, 40, 0.0002)
    gaps("pv_production_mw", 2, 60, 0.0002)

    agg = df["aggregate_power_w"].to_numpy(dtype=float, copy=True)
    spk = rng.integers(0, n, 10)
    agg[spk] = agg[spk] * rng.uniform(5, 15, 10) + 500  # pics aberrants
    for _ in range(3):  # valeurs figées
        s = int(rng.integers(0, n - 60))
        agg[s:s + int(rng.uniform(20, 50))] = agg[s]
    df["aggregate_power_w"] = agg
    return df


