from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from transformers import LogitsProcessor

from gemma_labeling.taxonomy import TagState, Taxonomy


class SupportsTokenEncoding(Protocol):
    eos_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ...


class TokenTrieNode:
    __slots__ = ("children", "terminals", "descendant_tags")

    def __init__(self) -> None:
        self.children: dict[int, TokenTrieNode] = {}
        self.terminals: set[str] = set()
        self.descendant_tags: set[str] = set()


@dataclass(frozen=True, slots=True)
class PrefixSnapshot:
    state: TagState
    current_prefix: str
    at_boundary: bool
    invalid_prefix: bool
    pending_tag: str | None = None
    active_tags: tuple[str, ...] = ()
    node: TokenTrieNode | None = None
    first_segment: bool = True


class SegmentTrie:
    def __init__(self) -> None:
        self.root = TokenTrieNode()

    def insert(self, token_ids: list[int], tag: str) -> None:
        node = self.root
        node.descendant_tags.add(tag)
        for token_id in token_ids:
            node = node.children.setdefault(token_id, TokenTrieNode())
            node.descendant_tags.add(tag)
        node.terminals.add(tag)


@dataclass(frozen=True, slots=True)
class StepStats:
    allowed_count: int
    masked_count: int
    eos_allowed: bool


class ConstraintRuntime:
    def __init__(self, tokenizer: SupportsTokenEncoding, taxonomy: Taxonomy) -> None:
        self.tokenizer = tokenizer
        self.taxonomy = taxonomy
        self.first_segment = SegmentTrie()
        self.next_segment = SegmentTrie()
        for canonical in taxonomy.order:
            self.first_segment.insert(
                tokenizer.encode(canonical, add_special_tokens=False),
                canonical,
            )
            self.next_segment.insert(
                tokenizer.encode(f", {canonical}", add_special_tokens=False),
                canonical,
            )

    def allowed_token_ids(
        self,
        generated_ids: list[int],
    ) -> tuple[set[int], PrefixSnapshot]:
        snapshot = self._parse_prefix(generated_ids)
        allowed: set[int] = set()
        eos_token_id = self.tokenizer.eos_token_id

        if snapshot.invalid_prefix:
            if eos_token_id is not None:
                allowed.add(eos_token_id)
            return allowed, snapshot

        if snapshot.at_boundary and snapshot.pending_tag is not None:
            next_state = self.taxonomy.commit(snapshot.state, snapshot.pending_tag)
            next_tags = self.taxonomy.legal_tags(next_state)
            if next_tags:
                allowed.update(self._children_for_tags(self.next_segment.root, set(next_tags)))
            if eos_token_id is not None:
                allowed.add(eos_token_id)
            return allowed, PrefixSnapshot(
                state=next_state,
                current_prefix="",
                at_boundary=True,
                invalid_prefix=False,
                pending_tag=None,
                active_tags=tuple(next_tags),
                node=None,
                first_segment=False,
            )

        if snapshot.at_boundary:
            trie = self.first_segment.root if snapshot.first_segment else self.next_segment.root
            allowed.update(self._children_for_tags(trie, set(snapshot.active_tags)))
            return allowed, snapshot

        if snapshot.node is not None:
            allowed.update(self._children_for_tags(snapshot.node, set(snapshot.active_tags)))

        return allowed, snapshot

    def summarize_steps(self, steps: list[StepStats]) -> dict[str, float | int | list[int]]:
        if not steps:
            return {
                "total_steps": 0,
                "average_allowed_tokens": 0.0,
                "min_allowed_tokens": 0,
                "max_allowed_tokens": 0,
                "eos_allowed_steps": 0,
                "masked_fraction": 0.0,
                "per_step_allowed": [],
            }

        per_step_allowed = [step.allowed_count for step in steps]
        total_allowed = sum(per_step_allowed)
        total_masked = sum(step.masked_count for step in steps)
        total_candidates = total_allowed + total_masked
        return {
            "total_steps": len(steps),
            "average_allowed_tokens": total_allowed / len(steps),
            "min_allowed_tokens": min(per_step_allowed),
            "max_allowed_tokens": max(per_step_allowed),
            "eos_allowed_steps": sum(1 for step in steps if step.eos_allowed),
            "masked_fraction": (total_masked / total_candidates) if total_candidates else 0.0,
            "per_step_allowed": per_step_allowed,
        }

    def _parse_prefix(self, token_ids: list[int]) -> PrefixSnapshot:
        state = self.taxonomy.initial_state()
        if not token_ids:
            return PrefixSnapshot(
                state=state,
                current_prefix="",
                at_boundary=True,
                invalid_prefix=False,
                active_tags=tuple(self.taxonomy.legal_tags(state)),
                first_segment=True,
            )

        remaining = token_ids
        first_segment = True

        while remaining:
            legal_tags = set(self.taxonomy.legal_tags(state))
            if not legal_tags:
                return PrefixSnapshot(
                    state=state,
                    current_prefix="",
                    at_boundary=False,
                    invalid_prefix=True,
                    active_tags=(),
                    first_segment=first_segment,
                )

            trie = self.first_segment.root if first_segment else self.next_segment.root
            node = trie
            last_terminal: tuple[str, int] | None = None
            index = 0

            while index < len(remaining):
                token_id = remaining[index]
                child = node.children.get(token_id)
                if child is None:
                    break
                node = child
                index += 1
                terminals = node.terminals.intersection(legal_tags)
                if terminals:
                    last_terminal = (sorted(terminals)[0], index)

            if index == len(remaining):
                if last_terminal and last_terminal[1] == len(remaining):
                    pending_tag = last_terminal[0]
                    prefix_text = pending_tag if first_segment else f", {pending_tag}"
                    return PrefixSnapshot(
                        state=state,
                        current_prefix=prefix_text,
                        at_boundary=True,
                        invalid_prefix=False,
                        pending_tag=pending_tag,
                        active_tags=tuple(sorted(legal_tags)),
                        node=node,
                        first_segment=first_segment,
                    )
                active_tags = tuple(sorted(node.descendant_tags.intersection(legal_tags)))
                if not active_tags:
                    return PrefixSnapshot(
                        state=state,
                        current_prefix="",
                        at_boundary=False,
                        invalid_prefix=True,
                        active_tags=(),
                        node=node,
                        first_segment=first_segment,
                    )
                return PrefixSnapshot(
                    state=state,
                    current_prefix="",
                    at_boundary=False,
                    invalid_prefix=False,
                    active_tags=active_tags,
                    node=node,
                    first_segment=first_segment,
                )

            if last_terminal is None:
                return PrefixSnapshot(
                    state=state,
                    current_prefix="",
                    at_boundary=False,
                    invalid_prefix=True,
                    active_tags=(),
                    node=node,
                    first_segment=first_segment,
                )

            committed_tag, consumed = last_terminal
            state = self.taxonomy.commit(state, committed_tag)
            remaining = remaining[consumed:]
            first_segment = False

        return PrefixSnapshot(
            state=state,
            current_prefix="",
            at_boundary=True,
            invalid_prefix=False,
            active_tags=tuple(self.taxonomy.legal_tags(state)),
            first_segment=first_segment,
        )

    @staticmethod
    def _children_for_tags(node: TokenTrieNode, legal_tags: set[str]) -> set[int]:
        allowed: set[int] = set()
        for token_id, child in node.children.items():
            if child.descendant_tags.intersection(legal_tags):
                allowed.add(token_id)
        return allowed


class ConstrainedTagLogitsProcessor(LogitsProcessor):
    def __init__(self, runtime: ConstraintRuntime, prompt_length: int) -> None:
        self.runtime = runtime
        self.prompt_length = prompt_length
        self.step_stats: list[StepStats] = []
        initial_state = runtime.taxonomy.initial_state()
        self.last_snapshot = PrefixSnapshot(
            state=initial_state,
            current_prefix="",
            at_boundary=True,
            invalid_prefix=False,
            active_tags=tuple(runtime.taxonomy.legal_tags(initial_state)),
            first_segment=True,
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[0] != 1:
            raise ValueError("The PoC only supports batch size 1.")

        generated_ids = input_ids[0, self.prompt_length :].tolist()
        allowed_token_ids, snapshot = self.runtime.allowed_token_ids(generated_ids)
        self.last_snapshot = snapshot

        if not allowed_token_ids:
            eos_token_id = self.runtime.tokenizer.eos_token_id
            if eos_token_id is not None:
                allowed_token_ids = {eos_token_id}

        allowed_list = sorted(allowed_token_ids)
        masked_count = int(scores.shape[-1] - len(allowed_list))
        self.step_stats.append(
            StepStats(
                allowed_count=len(allowed_list),
                masked_count=masked_count,
                eos_allowed=self.runtime.tokenizer.eos_token_id in allowed_token_ids,
            )
        )

        mask = torch.full_like(scores, float("-inf"))
        mask[:, allowed_list] = scores[:, allowed_list]
        return mask
