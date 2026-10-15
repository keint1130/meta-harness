from splits import build_splits, load_config, load_problem_rows, validate_splits


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


def test_answerbench_multiline_row_is_not_shifted_or_leaked():
    row = next(
        item
        for item in load_problem_rows()
        if item["problem_id"] == "imo-bench-algebra-036"
    )
    assert row["groundtruth"] == r"$Y(x)=A+\frac{B}{x}-x$"
    assert row["domain"] == "algebra"
    assert row["groundtruth"] not in row["problem"]

