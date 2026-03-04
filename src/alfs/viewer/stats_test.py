from alfs.viewer.stats import compute_year_kde


def test_empty_input():
    assert compute_year_kde({}, {}) == {}


def test_single_sense_single_year():
    result = compute_year_kde({"S1": {2000: 10}}, {2000: 10})
    assert "S1" in result
    pts = result["S1"]
    assert len(pts) > 0
    # All proportions should be ~1.0 (only one sense covers all occurrences)
    assert all(abs(v - 1.0) < 0.01 for _, v in pts)


def test_two_senses_same_year_sum_to_one():
    sense_year_counts = {"S1": {2000: 3}, "S2": {2000: 7}}
    year_totals = {2000: 10}
    result = compute_year_kde(sense_year_counts, year_totals)
    assert "S1" in result and "S2" in result
    # Find a grid point near year 2000 in both series
    s1_pts = {round(t, 4): v for t, v in result["S1"]}
    s2_pts = {round(t, 4): v for t, v in result["S2"]}
    common_ts = set(s1_pts) & set(s2_pts)
    assert common_ts
    for t in common_ts:
        total = s1_pts[t] + s2_pts[t]
        assert abs(total - 1.0) < 0.01, f"Sum at {t} = {total}"


def test_smoothing_between_years():
    # Sense present only in 2000 and 2010; should have non-zero value at 2005
    sense_year_counts = {"S1": {2000: 5, 2010: 5}}
    year_totals = {2000: 5, 2010: 5}
    result = compute_year_kde(sense_year_counts, year_totals)
    pts_dict = dict(result["S1"])
    # Find closest grid point to 2005
    closest_t = min(pts_dict, key=lambda t: abs(t - 2005))
    assert pts_dict[closest_t] > 0.01, "Expected non-zero smoothed value between years"
