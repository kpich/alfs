"""KDE-based year statistics for viewer."""

import math


def compute_year_kde(
    sense_year_counts: dict[str, dict[int, int]],
    year_totals: dict[int, int],
    bandwidth: float = 2.5,
) -> dict[str, list[tuple[float, float]]]:
    """Nadaraya-Watson kernel smooth of per-year sense proportions.

    Returns sense_key -> [(year_float, proportion), ...] over a dense grid.
    """
    if not sense_year_counts or not year_totals:
        return {}

    all_years = list(year_totals.keys())
    year_min = min(all_years)
    year_max = max(all_years)

    n_grid = 300
    step = (year_max + bandwidth - (year_min - bandwidth)) / (n_grid - 1)
    grid = [year_min - bandwidth + i * step for i in range(n_grid)]

    def kernel(u: float) -> float:
        return math.exp(-0.5 * u * u)

    result: dict[str, list[tuple[float, float]]] = {}
    for sk, year_counts in sense_year_counts.items():
        pts: list[tuple[float, float]] = []
        for t in grid:
            num = sum(
                count * kernel((t - y) / bandwidth) for y, count in year_counts.items()
            )
            den = sum(
                total * kernel((t - y) / bandwidth) for y, total in year_totals.items()
            )
            if den > 1e-10:
                pts.append((t, num / den))
        result[sk] = pts

    return result
