from __future__ import annotations

from io import BytesIO

from PIL import Image

from gemma_labeling.config import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MAX_TAGS, DEFAULT_TEMPERATURE
from gemma_labeling.model import GemmaModelRuntime
from gemma_labeling.schemas import (
    ConstrainedLabelResponse,
    FinalConstraintState,
    InvalidTokenStats,
    ParsedTagResult,
    PromptSnapshot,
    TagRuleModel,
    TaxonomyResponse,
    TraceEntry,
)
from gemma_labeling.taxonomy import ParseResult, Taxonomy


DEFAULT_SYSTEM_PROMPT = """You are a booru-style image tagger.
Return only canonical comma-separated tags with no prose, numbering, or explanations.
Prefer concrete visual attributes that are visible in the image."""

DEFAULT_USER_PROMPT = """Tag the image using the provided canonical vocabulary.
Prioritize subject count, composition, setting, visible appearance details, attire, and pose."""


class LabelingService:
    def __init__(self, taxonomy: Taxonomy, model_runtime: GemmaModelRuntime) -> None:
        self.taxonomy = taxonomy
        self.model_runtime = model_runtime

    def taxonomy_response(self) -> TaxonomyResponse:
        rules = [
            TagRuleModel(
                canonical=rule.canonical,
                category=rule.category,
                aliases=list(rule.aliases),
                prerequisites=list(rule.prerequisites),
                exclusions=list(rule.exclusions),
                implications=list(rule.implications),
            )
            for rule in (self.taxonomy.rules[name] for name in self.taxonomy.order)
        ]
        return TaxonomyResponse(
            tag_count=len(self.taxonomy.order),
            categories=self.taxonomy.categories(),
            starter_tags=sorted(self.taxonomy.starter_tags),
            rules=rules,
        )

    def label_image(
        self,
        image_bytes: bytes,
        *,
        hint: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_tags: int = DEFAULT_MAX_TAGS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> ConstrainedLabelResponse:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        system_prompt = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
        user_prompt = self._build_user_prompt(
            base_prompt=(user_prompt or DEFAULT_USER_PROMPT).strip(),
            hint=hint,
        )

        baseline = self.model_runtime.generate(
            image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            constrained=False,
        )
        constrained = self.model_runtime.generate(
            image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            constrained=True,
        )

        baseline_parsed = self.taxonomy.parse_generated_tags(baseline.text, limit=max_tags)
        constrained_parsed = self.taxonomy.parse_generated_tags(constrained.text, limit=max_tags)
        trace, final_state = self.taxonomy.trace_for_tags(list(constrained_parsed.explicit_tags))

        return ConstrainedLabelResponse(
            model_id=self.model_runtime.model_id,
            prompt=PromptSnapshot(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                hint=hint or None,
            ),
            baseline=self._to_parsed_result(baseline_parsed),
            constrained=self._to_parsed_result(constrained_parsed),
            trace=[
                TraceEntry(
                    tag=step.tag,
                    unlocked_tags=list(step.unlocked_tags),
                    implied_tags=list(step.implied_tags),
                    active_tags_after=list(step.active_tags_after),
                )
                for step in trace
            ],
            final_state=FinalConstraintState(
                used_tags=sorted(final_state.used_tags),
                unlocked_tags=sorted(final_state.unlocked_tags),
                blocked_duplicate_tags=sorted(final_state.used_tags),
                current_prefix=str(constrained.snapshot.get("current_prefix", "")),
                at_boundary=bool(constrained.snapshot.get("at_boundary", True)),
                invalid_prefix=bool(constrained.snapshot.get("invalid_prefix", False)),
            ),
            invalid_token_stats=InvalidTokenStats(**constrained.stats),
        )

    def _build_user_prompt(self, *, base_prompt: str, hint: str | None) -> str:
        if len(self.taxonomy.order) <= 120:
            rules_text = "\n".join(
                f"- {tag}: {self.taxonomy.rules[tag].category}"
                for tag in self.taxonomy.order
            )
            taxonomy_block = f"Allowed canonical tags:\n{rules_text}\n"
        else:
            category_summary = ", ".join(
                f"{category}={count}"
                for category, count in sorted(self.taxonomy.categories().items())
            )
            taxonomy_block = (
                "Use only canonical tags from the active e621 taxonomy snapshot.\n"
                f"Category counts: {category_summary}\n"
            )
        hint_block = f"\nUser hint: {hint.strip()}" if hint and hint.strip() else ""
        return (
            f"{base_prompt}\n\n"
            f"{taxonomy_block}"
            "Output format: comma-separated canonical tags only."
            f"{hint_block}"
        )

    @staticmethod
    def _to_parsed_result(parsed: ParseResult) -> ParsedTagResult:
        return ParsedTagResult(
            raw_text=parsed.raw_text,
            normalized_tags=list(parsed.normalized_tags),
            explicit_tags=list(parsed.explicit_tags),
            implied_tags=list(parsed.implied_tags),
            unknown_tags=list(parsed.unknown_tags),
            duplicate_tags=list(parsed.duplicate_tags),
            illegal_tags=list(parsed.illegal_tags),
        )
