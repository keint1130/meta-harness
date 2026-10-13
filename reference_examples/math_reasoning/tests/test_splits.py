from splits import build_splits, load_config, validate_splits


def test_balanced_sizes_and_no_overlap():
    config = load_config()
    splits = build_splits(config)
    validate_splits(splits, config)
    for split_name, size in config["splits"].items():
        assert all(len(splits[split_name][domain]) == size for domain in config["domains"])


def test_prior_pilot_is_search_only():
    splits = build_splits()
    for domain in load_config()["domains"]:
        pilot = {f"imo-bench-{domain}-{number:03d}" for number in range(1, 4)}
        assert pilot.issubset(splits["search"][domain])
        assert pilot.isdisjoint(splits["dev"][domain])
        assert pilot.isdisjoint(splits["test"][domain])


def test_split_is_deterministic():
    assert build_splits() == build_splits()

