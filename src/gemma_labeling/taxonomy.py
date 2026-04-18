from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import yaml


_SEPARATOR_PATTERN = re.compile(r"[,;\n]+")
_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9_]+")


@dataclass(frozen=True, slots=True)
class TagRule:
    canonical: str
    category: str
    aliases: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    implications: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TagState:
    used_tags: frozenset[str]
    unlocked_tags: frozenset[str]


@dataclass(frozen=True, slots=True)
class TraceStep:
    tag: str
    unlocked_tags: tuple[str, ...]
    implied_tags: tuple[str, ...]
    active_tags_after: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParseResult:
    raw_text: str
    normalized_tags: tuple[str, ...]
    explicit_tags: tuple[str, ...]
    implied_tags: tuple[str, ...]
    unknown_tags: tuple[str, ...]
    duplicate_tags: tuple[str, ...]
    illegal_tags: tuple[str, ...]


def normalize_tag_key(value: str) -> str:
    lowered = value.strip().lower().replace(" ", "_").replace("-", "_")
    lowered = _NORMALIZE_PATTERN.sub("", lowered)
    return re.sub(r"_+", "_", lowered).strip("_")


class Taxonomy:
    def __init__(self, rules: list[TagRule]) -> None:
        self.rules = {rule.canonical: rule for rule in rules}
        self.order = [rule.canonical for rule in rules]
        self.alias_map = self._build_alias_map(rules)
        self.reverse_unlocks = self._build_reverse_unlocks(rules)
        self.starter_tags = frozenset(
            rule.canonical for rule in rules if not rule.prerequisites
        )
        self._validate()

    @classmethod
    def from_yaml(cls, path: Path) -> "Taxonomy":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        rules = [
            TagRule(
                canonical=entry["canonical"],
                category=entry["category"],
                aliases=tuple(entry.get("aliases", [])),
                prerequisites=tuple(entry.get("prerequisites", [])),
                exclusions=tuple(entry.get("exclusions", [])),
                implications=tuple(entry.get("implications", [])),
            )
            for entry in payload["tags"]
        ]
        return cls(rules)

    @classmethod
    def from_records(cls, records: list[dict[str, object]]) -> "Taxonomy":
        rules = [
            TagRule(
                canonical=str(entry["canonical"]),
                category=str(entry["category"]),
                aliases=tuple(entry.get("aliases", []) or []),
                prerequisites=tuple(entry.get("prerequisites", []) or []),
                exclusions=tuple(entry.get("exclusions", []) or []),
                implications=tuple(entry.get("implications", []) or []),
            )
            for entry in records
        ]
        return cls(rules)

    def initial_state(self) -> TagState:
        return TagState(used_tags=frozenset(), unlocked_tags=self._compute_unlocked(frozenset()))

    def legal_tags(self, state: TagState) -> list[str]:
        legal: list[str] = []
        for canonical in self.order:
            if canonical in state.used_tags:
                continue
            rule = self.rules[canonical]
            if not set(rule.prerequisites).issubset(state.used_tags):
                continue
            if set(rule.exclusions).intersection(state.used_tags):
                continue
            if any(
                canonical in self.rules[used_tag].exclusions for used_tag in state.used_tags
            ):
                continue
            legal.append(canonical)
        return legal

    def commit(self, state: TagState, tag: str) -> TagState:
        next_state, _, _ = self.apply_tag(state, tag)
        return next_state

    def apply_tag(self, state: TagState, tag: str) -> tuple[TagState, tuple[str, ...], tuple[str, ...]]:
        if tag not in self.legal_tags(state):
            raise ValueError(f"Illegal tag transition: {tag}")

        used_before = set(state.used_tags)
        expanded = self._implication_closure(tag)
        used = frozenset(used_before | expanded)
        next_state = TagState(used_tags=used, unlocked_tags=self._compute_unlocked(used))
        implied_tags = self._ordered_tags(expanded - {tag})
        unlocked_tags = self._ordered_tags(set(next_state.unlocked_tags) - set(state.unlocked_tags))
        return next_state, implied_tags, unlocked_tags

    def trace_for_tags(self, tags: list[str]) -> tuple[list[TraceStep], TagState]:
        state = self.initial_state()
        trace: list[TraceStep] = []
        for tag in tags:
            state, implied_tags, unlocked_tags = self.apply_tag(state, tag)
            trace.append(
                TraceStep(
                    tag=tag,
                    unlocked_tags=tuple(unlocked_tags),
                    implied_tags=tuple(implied_tags),
                    active_tags_after=tuple(self.legal_tags(state)),
                )
            )
        return trace, state

    def parse_generated_tags(self, raw_text: str, limit: int | None = None) -> ParseResult:
        normalized_tags: list[str] = []
        explicit_tags: list[str] = []
        implied_tags: list[str] = []
        unknown_tags: list[str] = []
        duplicate_tags: list[str] = []
        illegal_tags: list[str] = []
        state = self.initial_state()

        for raw_chunk in _SEPARATOR_PATTERN.split(raw_text):
            cleaned = raw_chunk.strip(" \t\r\n.:")
            if not cleaned:
                continue
            key = normalize_tag_key(cleaned)
            canonical = self.alias_map.get(key)
            if canonical is None:
                unknown_tags.append(cleaned)
                continue
            if canonical in state.used_tags:
                duplicate_tags.append(canonical)
                continue

            try:
                state, auto_implied, _ = self.apply_tag(state, canonical)
            except ValueError:
                illegal_tags.append(canonical)
                continue

            explicit_tags.append(canonical)
            normalized_tags.append(canonical)
            for implied_tag in auto_implied:
                implied_tags.append(implied_tag)
                normalized_tags.append(implied_tag)

            if limit is not None and len(explicit_tags) >= limit:
                break

        return ParseResult(
            raw_text=raw_text.strip(),
            normalized_tags=tuple(normalized_tags),
            explicit_tags=tuple(explicit_tags),
            implied_tags=tuple(implied_tags),
            unknown_tags=tuple(unknown_tags),
            duplicate_tags=tuple(duplicate_tags),
            illegal_tags=tuple(illegal_tags),
        )

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for canonical in self.order:
            category = self.rules[canonical].category
            counts[category] = counts.get(category, 0) + 1
        return counts

    def _validate(self) -> None:
        missing_refs: set[str] = set()
        for rule in self.rules.values():
            for item in (*rule.prerequisites, *rule.exclusions, *rule.implications):
                if item not in self.rules:
                    missing_refs.add(item)
        if missing_refs:
            joined = ", ".join(sorted(missing_refs))
            raise ValueError(f"Unknown prerequisite, exclusion, or implication tags: {joined}")

    def _build_alias_map(self, rules: list[TagRule]) -> dict[str, str]:
        alias_map: dict[str, str] = {}
        for rule in rules:
            for alias in (rule.canonical, *rule.aliases):
                alias_map[normalize_tag_key(alias)] = rule.canonical
        return alias_map

    def _build_reverse_unlocks(self, rules: list[TagRule]) -> dict[str, tuple[str, ...]]:
        graph: dict[str, list[str]] = {rule.canonical: [] for rule in rules}
        for rule in rules:
            for parent in rule.prerequisites:
                graph.setdefault(parent, []).append(rule.canonical)
        return {key: tuple(sorted(value)) for key, value in graph.items()}

    def _compute_unlocked(self, used_tags: frozenset[str]) -> frozenset[str]:
        return frozenset(
            canonical
            for canonical in self.order
            if canonical not in used_tags
            and set(self.rules[canonical].prerequisites).issubset(used_tags)
        )

    def _implication_closure(self, tag: str) -> set[str]:
        closure: set[str] = set()
        stack = [tag]

        while stack:
            current = stack.pop()
            if current in closure:
                continue
            closure.add(current)
            for implied in self.rules[current].implications:
                stack.append(implied)

        return closure

    def _ordered_tags(self, names: set[str]) -> tuple[str, ...]:
        return tuple(canonical for canonical in self.order if canonical in names)
