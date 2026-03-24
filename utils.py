def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def half_saturation_score(value: float, half_sat: float) -> float:
    if value <= 0:
        return 0.0
    return clamp(100.0 * (value / (value + half_sat)))


def center_fit_score(value: float, target: float, minimum: float, maximum: float) -> float:
    if value < minimum or value > maximum:
        return 0.0

    left_span = target - minimum
    right_span = maximum - target
    max_distance = max(left_span, right_span)

    if max_distance <= 0:
        return 100.0 if value == target else 0.0

    distance = abs(value - target)
    return clamp(100.0 * (1.0 - distance / max_distance))


def min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []

    vmin = min(values)
    vmax = max(values)

    if vmax == vmin:
        return [50.0 for _ in values]

    return [100.0 * (v - vmin) / (vmax - vmin) for v in values]
