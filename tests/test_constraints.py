from __future__ import annotations

import pytest

from gemma_labeling.constraints import ConstraintRuntime
from gemma_labeling.taxonomy import Taxonomy


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        self.mapping = {
            "alpha": [1],
            "beta": [2, 3],
            "gamma": [4],
            ", alpha": [5, 1],
            ", beta": [5, 2, 3],
            ", gamma": [5, 4],
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self.mapping[text])


@pytest.fixture()
def taxonomy() -> Taxonomy:
    return Taxonomy.from_records(
        [
            {"canonical": "alpha", "category": "subject"},
            {"canonical": "beta", "category": "style"},
            {"canonical": "gamma", "category": "detail", "prerequisites": ["alpha"]},
        ]
    )


@pytest.fixture()
def runtime(taxonomy: Taxonomy) -> ConstraintRuntime:
    return ConstraintRuntime(FakeTokenizer(), taxonomy)


def test_duplicate_tags_are_rejected(taxonomy: Taxonomy) -> None:
    state = taxonomy.initial_state()
    state = taxonomy.commit(state, "alpha")
    with pytest.raises(ValueError):
        taxonomy.commit(state, "alpha")


def test_tags_stay_illegal_until_prerequisite_is_met(taxonomy: Taxonomy) -> None:
    state = taxonomy.initial_state()
    assert "gamma" not in taxonomy.legal_tags(state)

    next_state = taxonomy.commit(state, "alpha")
    assert "gamma" in taxonomy.legal_tags(next_state)


def test_newly_unlocked_tags_show_up_immediately(taxonomy: Taxonomy) -> None:
    trace, final_state = taxonomy.trace_for_tags(["alpha"])
    assert trace[0].unlocked_tags == ("gamma",)
    assert "gamma" in final_state.unlocked_tags


def test_separator_is_illegal_before_first_tag(runtime: ConstraintRuntime) -> None:
    allowed, snapshot = runtime.allowed_token_ids([])
    assert snapshot.at_boundary is True
    assert 5 not in allowed
    assert allowed == {1, 2}


def test_partial_subword_path_allows_only_valid_continuation(runtime: ConstraintRuntime) -> None:
    allowed, snapshot = runtime.allowed_token_ids([2])
    assert snapshot.at_boundary is False
    assert snapshot.invalid_prefix is False
    assert allowed == {3}


def test_duplicate_is_blocked_after_separator(runtime: ConstraintRuntime) -> None:
    allowed, _ = runtime.allowed_token_ids([1, 5])
    assert 1 not in allowed
    assert allowed == {2, 4}


def test_eos_is_allowed_only_at_valid_boundary(runtime: ConstraintRuntime) -> None:
    allowed_mid, _ = runtime.allowed_token_ids([2])
    assert 0 not in allowed_mid

    allowed_boundary, snapshot = runtime.allowed_token_ids([1])
    assert snapshot.at_boundary is True
    assert 0 in allowed_boundary
    assert 5 in allowed_boundary


def test_only_eos_remains_when_no_followup_tags_exist() -> None:
    taxonomy = Taxonomy.from_records(
        [{"canonical": "alpha", "category": "subject"}]
    )
    runtime = ConstraintRuntime(FakeTokenizer(), taxonomy)

    allowed, snapshot = runtime.allowed_token_ids([1])
    assert snapshot.at_boundary is True
    assert allowed == {0}


def test_implications_are_auto_applied_and_block_duplicates() -> None:
    taxonomy = Taxonomy.from_records(
        [
            {"canonical": "wolf", "category": "species", "implications": ["canine", "canid"]},
            {"canonical": "canine", "category": "species", "implications": ["mammal"]},
            {"canonical": "canid", "category": "species"},
            {"canonical": "mammal", "category": "species"},
        ]
    )

    parsed = taxonomy.parse_generated_tags("wolf, canine, mammal")

    assert parsed.explicit_tags == ("wolf",)
    assert parsed.implied_tags == ("canine", "canid", "mammal")
    assert parsed.normalized_tags == ("wolf", "canine", "canid", "mammal")
    assert parsed.duplicate_tags == ("canine", "mammal")


def test_aliases_normalize_to_canonical_tags() -> None:
    taxonomy = Taxonomy.from_records(
        [
            {"canonical": "1girl", "category": "general", "aliases": ["female", "one_girl"]},
            {"canonical": "solo", "category": "general"},
        ]
    )

    parsed = taxonomy.parse_generated_tags("female, solo")
    assert parsed.explicit_tags == ("1girl", "solo")
    assert parsed.normalized_tags == ("1girl", "solo")
