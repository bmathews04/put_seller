from decisioning import derive_decision_status, score_warning_severity


def test_ready_when_clean_and_strong():
    status, meta = derive_decision_status(
        final_score=82.0,
        confidence="High",
        technical_label="Supportive",
        contract_eligibility_status="eligible",
        stock_eligibility_status="eligible",
        stock_warning_reasons=[],
        contract_warning_reasons=[],
        warning_flags=[],
    )

    assert status == "Ready"
    assert meta["severity_label"] == "None"
    assert meta["warning_count_total"] == 0
    assert meta["decision_blockers"] == []


def test_review_when_warning_burden_is_low_but_not_clean():
    status, meta = derive_decision_status(
        final_score=72.0,
        confidence="High",
        technical_label="Supportive",
        contract_eligibility_status="eligible_with_warnings",
        stock_eligibility_status="eligible",
        stock_warning_reasons=[],
        contract_warning_reasons=["volume_unknown"],
        warning_flags=[],
    )

    assert status == "Review"
    assert meta["severity_label"] in {"Low", "Medium"}
    assert "contract_has_warnings" in meta["decision_cautions"]


def test_pass_when_high_severity_warning_burden_present():
    status, meta = derive_decision_status(
        final_score=78.0,
        confidence="High",
        technical_label="Supportive",
        contract_eligibility_status="eligible_with_warnings",
        stock_eligibility_status="eligible",
        stock_warning_reasons=[],
        contract_warning_reasons=["earnings_before_expiry", "oi_below_min"],
        warning_flags=[],
    )

    assert status == "Pass"
    assert meta["severity_label"] == "High"
    assert "warning_severity_high" in meta["decision_cautions"]


def test_pass_when_stock_ineligible():
    status, meta = derive_decision_status(
        final_score=90.0,
        confidence="High",
        technical_label="Strongly supportive",
        contract_eligibility_status="eligible",
        stock_eligibility_status="ineligible",
        stock_warning_reasons=[],
        contract_warning_reasons=[],
        warning_flags=[],
    )

    assert status == "Pass"
    assert "stock_ineligible" in meta["decision_blockers"]
    assert meta["decision_rationale"] == "Hard ineligibility present."


def test_pass_when_contract_ineligible():
    status, meta = derive_decision_status(
        final_score=90.0,
        confidence="High",
        technical_label="Strongly supportive",
        contract_eligibility_status="ineligible",
        stock_eligibility_status="eligible",
        stock_warning_reasons=[],
        contract_warning_reasons=[],
        warning_flags=[],
    )

    assert status == "Pass"
    assert "contract_ineligible" in meta["decision_blockers"]
    assert meta["decision_rationale"] == "Hard ineligibility present."


def test_review_when_score_is_good_but_confidence_only_medium():
    status, meta = derive_decision_status(
        final_score=80.0,
        confidence="Medium",
        technical_label="Supportive",
        contract_eligibility_status="eligible",
        stock_eligibility_status="eligible",
        stock_warning_reasons=[],
        contract_warning_reasons=[],
        warning_flags=[],
    )

    assert status == "Review"
    assert "confidence_not_high" in meta["decision_cautions"]


def test_review_when_technicals_only_mixed():
    status, meta = derive_decision_status(
        final_score=76.0,
        confidence="High",
        technical_label="Mixed",
        contract_eligibility_status="eligible",
        stock_eligibility_status="eligible",
        stock_warning_reasons=[],
        contract_warning_reasons=[],
        warning_flags=[],
    )

    assert status == "Review"
    assert "technical_not_supportive" in meta["decision_cautions"]


def test_pass_when_score_too_low_even_without_warnings():
    status, meta = derive_decision_status(
        final_score=47.0,
        confidence="High",
        technical_label="Supportive",
        contract_eligibility_status="eligible",
        stock_eligibility_status="eligible",
        stock_warning_reasons=[],
        contract_warning_reasons=[],
        warning_flags=[],
    )

    assert status == "Pass"
    assert "score_below_50" in meta["decision_cautions"]


def test_score_warning_severity_low_bucket():
    meta = score_warning_severity(
        stock_warning_reasons=[],
        contract_warning_reasons=["some_unclassified_warning"],
        warning_flags=[],
    )

    assert meta["severity_label"] == "Low"
    assert meta["warning_count_total"] == 1
    assert meta["severity_points"] == 1
    assert meta["low_severity_count"] == 1


def test_score_warning_severity_medium_bucket():
    meta = score_warning_severity(
        stock_warning_reasons=["quality_data_incomplete"],
        contract_warning_reasons=[],
        warning_flags=[],
    )

    assert meta["severity_label"] == "Medium"
    assert meta["warning_count_total"] == 1
    assert meta["severity_points"] == 2
    assert meta["medium_severity_count"] == 1


def test_score_warning_severity_high_bucket_from_flags_and_reasons():
    meta = score_warning_severity(
        stock_warning_reasons=["quality_data_incomplete"],
        contract_warning_reasons=["earnings_before_expiry"],
        warning_flags=["liquidity_borderline"],
    )

    assert meta["severity_label"] == "High"
    assert meta["warning_count_total"] == 3
    assert meta["high_severity_count"] == 2
    assert meta["medium_severity_count"] == 1
    assert meta["severity_points"] == 8


def test_warning_inputs_accept_semicolon_strings():
    meta = score_warning_severity(
        stock_warning_reasons="quality_data_incomplete; oi_unknown",
        contract_warning_reasons="volume_below_min",
        warning_flags="thin_yield",
    )

    assert meta["warning_count_total"] == 4
    assert meta["medium_severity_count"] == 3
    assert meta["high_severity_count"] == 1
    assert meta["severity_label"] == "High"
