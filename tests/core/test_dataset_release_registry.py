"""Policy and availability behavior for release recommendations."""

from src.utils.dataset_release_registry import (
    get_release_alias,
    get_release_recommendation,
    recommended_release_for,
)


def test_mcns_alias_and_recommendation_are_explicit():
    alias = get_release_alias("male-cns:v0.9", "male-cns:v1.0")
    assert alias["mapping_strategy"] == "same_name_release_alias"
    assert alias["recommend_if_selected"] is True
    assert recommended_release_for("male-cns:v0.9") == "male-cns:v1.0"

    record = get_release_recommendation(
        "male-cns:v0.9",
        ["male-cns:v0.9", "male-cns:v1.0"],
    )
    assert record["actionable"] is True
    assert record["target"] == "male-cns:v1.0"


def test_release_recommendation_never_promotes_uncertified_manc():
    assert get_release_recommendation(
        "manc:v1.0", ["manc:v1.0", "manc:v1.2.1"]
    ) is None
    assert get_release_recommendation(
        "manc:v1.2.1", ["manc:v1.2.1", "manc:v1.2.3"]
    ) is None


def test_missing_new_release_is_not_actionable():
    record = get_release_recommendation("male-cns:v0.9", ["male-cns:v0.9"])
    assert record["actionable"] is False
    assert "not currently available" in record["message"]
