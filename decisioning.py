from typing import Iterable


HIGH_SEVERITY_WARNING_REASONS = {
    "earnings_before_expiry",
    "earnings_date_unknown",
    "spread_moderately_wide",
    "oi_below_min",
    "volume_below_min",
}

MEDIUM_SEVERITY_WARNING_REASONS = {
    "quality_data_incomplete",
    "oi_unknown",
    "volume_unknown",
}

HIGH_SEVERITY_WARNING_FLAGS = {
    "delta_near_upper_band",
    "liquidity_borderline",
}

MEDIUM_SEVERITY_WARNING_FLAGS = {
    "thin_yield",
}


def _normalize_iterable(values) -> list[str]:
    if values is None:
        return []

    if isinstance(values, str):
        text = values.strip()
        if not text:
            return []
        return [item.strip() for item in text.split(";") if item.strip()]

    if isinstance(values, Iterable):
        normalized = []
        for item in values:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized

    text = str(values).strip()
    return [text] if text else []


def confidence_rank(value: str | None) -> int:
    mapping = {"High": 3, "Medium": 2, "Low": 1}
    return mapping.get(value or "", 0)


def technical_rank(value: str | None) -> int:
    mapping = {
        "Strongly supportive": 4,
        "Supportive": 3,
        "Mildly supportive": 3,
        "Mixed": 2,
        "Neutral": 2,
        "Cautious": 1,
        "Weak": 1,
    }
    return mapping.get(value or "", 0)


def score_warning_severity(
    stock_warning_reasons=None,
    contract_warning_reasons=None,
    warning_flags=None,
) -> dict:
    stock_reasons = _normalize_iterable(stock_warning_reasons)
    contract_reasons = _normalize_iterable(contract_warning_reasons)
    flags = _normalize_iterable(warning_flags)

    all_reasons = stock_reasons + contract_reasons

    high = 0
    medium = 0
    low = 0

    for reason in all_reasons:
        if reason in HIGH_SEVERITY_WARNING_REASONS:
            high += 1
        elif reason in MEDIUM_SEVERITY_WARNING_REASONS:
            medium += 1
        else:
            low += 1

    for flag in flags:
        if flag in HIGH_SEVERITY_WARNING_FLAGS:
            high += 1
        elif flag in MEDIUM_SEVERITY_WARNING_FLAGS:
            medium += 1
        else:
            low += 1

    total_items = len(stock_reasons) + len(contract_reasons) + len(flags)
    severity_points = high * 3 + medium * 2 + low * 1

    if high >= 2 or severity_points >= 8:
        severity_label = "High"
    elif high >= 1 or medium >= 1 or severity_points >= 2:
        severity_label = "Medium"
    elif total_items > 0:
        severity_label = "Low"
    else:
        severity_label = "None"

    return {
        "stock_warning_reasons": stock_reasons,
        "contract_warning_reasons": contract_reasons,
        "warning_flags": flags,
        "warning_count_total": total_items,
        "high_severity_count": high,
        "medium_severity_count": medium,
        "low_severity_count": low,
        "severity_points": severity_points,
        "severity_label": severity_label,
    }


def derive_decision_status(
    *,
    final_score: float | None,
    confidence: str | None,
    technical_label: str | None,
    contract_eligibility_status: str | None,
    stock_eligibility_status: str | None = None,
    stock_warning_reasons=None,
    contract_warning_reasons=None,
    warning_flags=None,
) -> tuple[str, dict]:
    severity = score_warning_severity(
        stock_warning_reasons=stock_warning_reasons,
        contract_warning_reasons=contract_warning_reasons,
        warning_flags=warning_flags,
    )

    final_score_value = float(final_score) if final_score is not None else None
    confidence_value = confidence or "Low"
    technical_value = technical_label or "Unknown"
    contract_status = contract_eligibility_status or "eligible_with_warnings"
    stock_status = stock_eligibility_status or "eligible"

    blockers = []
    caution_notes = []

    if contract_status == "ineligible":
        blockers.append("contract_ineligible")
    if stock_status == "ineligible":
        blockers.append("stock_ineligible")

    if confidence_rank(confidence_value) == 0:
        caution_notes.append("missing_confidence")
    if technical_rank(technical_value) == 0:
        caution_notes.append("missing_technical_label")
    if final_score_value is None:
        caution_notes.append("missing_final_score")

    if blockers:
        return (
            "Pass",
            {
                **severity,
                "decision_rationale": "Hard ineligibility present.",
                "decision_blockers": blockers,
                "decision_cautions": caution_notes,
            },
        )

    if contract_status == "eligible_with_warnings":
        caution_notes.append("contract_has_warnings")
    if stock_status == "eligible_with_warnings":
        caution_notes.append("stock_has_warnings")

    if final_score_value is not None and final_score_value < 50:
        caution_notes.append("score_below_50")
    elif final_score_value is not None and final_score_value < 65:
        caution_notes.append("score_below_65")
    elif final_score_value is not None and final_score_value < 75:
        caution_notes.append("score_below_75")

    if confidence_rank(confidence_value) < confidence_rank("Medium"):
        caution_notes.append("confidence_low")
    elif confidence_rank(confidence_value) < confidence_rank("High"):
        caution_notes.append("confidence_not_high")

    if technical_rank(technical_value) < technical_rank("Mixed"):
        caution_notes.append("technical_weak")
    elif technical_rank(technical_value) < technical_rank("Supportive"):
        caution_notes.append("technical_not_supportive")

    if severity["severity_label"] == "High":
        caution_notes.append("warning_severity_high")
    elif severity["severity_label"] == "Medium":
        caution_notes.append("warning_severity_medium")
    elif severity["severity_label"] == "Low":
        caution_notes.append("warning_severity_low")

    ready_conditions = [
        final_score_value is not None and final_score_value >= 75,
        confidence_rank(confidence_value) >= confidence_rank("High"),
        technical_rank(technical_value) >= technical_rank("Supportive"),
        contract_status == "eligible",
        stock_status == "eligible",
        severity["warning_count_total"] == 0,
    ]

    if all(ready_conditions):
        return (
            "Ready",
            {
                **severity,
                "decision_rationale": "High score, strong confidence, supportive technicals, and no warning burden.",
                "decision_blockers": blockers,
                "decision_cautions": caution_notes,
            },
        )

    review_conditions = [
        final_score_value is not None and final_score_value >= 58,
        confidence_rank(confidence_value) >= confidence_rank("Medium"),
        technical_rank(technical_value) >= technical_rank("Mixed"),
        severity["severity_label"] != "High",
    ]

    if all(review_conditions):
        return (
            "Review",
            {
                **severity,
                "decision_rationale": "Tradable candidate, but warning burden or quality threshold keeps it out of Ready.",
                "decision_blockers": blockers,
                "decision_cautions": caution_notes,
            },
        )

    return (
        "Pass",
        {
            **severity,
            "decision_rationale": "Overall trade quality is too weak after confidence, technicals, and warning burden are considered.",
            "decision_blockers": blockers,
            "decision_cautions": caution_notes,
        },
    )
