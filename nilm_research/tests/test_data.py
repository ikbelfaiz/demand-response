import numpy as np

from nilm_research.data import prepare_partition, valid_endpoints


def test_dataset_and_panel_verification(prepared):
    repo, panel, *_ = prepared
    report = repo.verify()
    assert report["rows"] == 26_280_000
    assert report["households"] == 50
    assert len(panel) == 10
    assert report["bad_intervals"] == 0
    assert report["duplicate_keys"] == 0


def test_household_and_split_isolation(prepared, cfg):
    repo, _, fit_homes, transforms, _, _ = prepared
    train = prepare_partition(repo, cfg, transforms, fit_homes[0], "train")
    val = prepare_partition(repo, cfg, transforms, fit_homes[0], "validation")
    assert train.client_id == val.client_id == fit_homes[0]
    assert train.timestamps.max() < val.timestamps.min()
    assert train.inputs[1, 0] == 0 and val.inputs[1, 0] == 0
    assert train.timestamps.min() >= np.datetime64(cfg["splits"]["train"][0])
    assert train.timestamps.max() < np.datetime64(cfg["splits"]["train"][1])


def test_transforms_are_training_only_and_exclude_heldout(prepared, cfg):
    _, _, fit_homes, transforms, _, _ = prepared
    assert transforms["fitted_split"] == "train"
    assert transforms["fitted_households"] == fit_homes
    assert not set(cfg["evaluation"]["held_out_households"]) & set(transforms["fitted_households"])


def test_missing_imputed_target_mask_and_contaminated_context(prepared):
    repo, panel, _, transforms, _, _ = prepared
    found_target = found_context = False
    for client in panel:
        part = prepare_partition(repo, repo.cfg, transforms, client, "train")
        if part.target_imputed.any():
            assert np.all(~part.target_valid[part.target_imputed])
            found_target = True
        endpoints = valid_endpoints(part, 256)
        bad_cs = np.r_[0, np.cumsum(part.bad_context_point.astype(int))]
        assert np.all((bad_cs[endpoints + 1] - bad_cs[endpoints - 255]) == 0)
        found_context |= bool(part.bad_context_point.any())
    assert found_target and found_context

