from __future__ import annotations

from pydantic import BaseModel, Field


class TagRuleModel(BaseModel):
    canonical: str
    category: str
    aliases: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    implications: list[str] = Field(default_factory=list)


class TaxonomyResponse(BaseModel):
    tag_count: int
    categories: dict[str, int]
    starter_tags: list[str]
    rules: list[TagRuleModel]


class PromptSnapshot(BaseModel):
    system_prompt: str
    user_prompt: str
    hint: str | None = None


class ParsedTagResult(BaseModel):
    raw_text: str
    normalized_tags: list[str]
    explicit_tags: list[str] = Field(default_factory=list)
    implied_tags: list[str] = Field(default_factory=list)
    unknown_tags: list[str] = Field(default_factory=list)
    duplicate_tags: list[str] = Field(default_factory=list)
    illegal_tags: list[str] = Field(default_factory=list)


class TraceEntry(BaseModel):
    tag: str
    unlocked_tags: list[str] = Field(default_factory=list)
    implied_tags: list[str] = Field(default_factory=list)
    active_tags_after: list[str] = Field(default_factory=list)


class FinalConstraintState(BaseModel):
    used_tags: list[str]
    unlocked_tags: list[str]
    blocked_duplicate_tags: list[str]
    current_prefix: str = ""
    at_boundary: bool = True
    invalid_prefix: bool = False


class InvalidTokenStats(BaseModel):
    total_steps: int = 0
    average_allowed_tokens: float = 0.0
    min_allowed_tokens: int = 0
    max_allowed_tokens: int = 0
    eos_allowed_steps: int = 0
    masked_fraction: float = 0.0
    per_step_allowed: list[int] = Field(default_factory=list)


class ConstrainedLabelResponse(BaseModel):
    model_id: str
    prompt: PromptSnapshot
    baseline: ParsedTagResult
    constrained: ParsedTagResult
    trace: list[TraceEntry]
    final_state: FinalConstraintState
    invalid_token_stats: InvalidTokenStats


class HealthResponse(BaseModel):
    status: str
    model_id: str
    cuda_available: bool
    model_loaded: bool
    detail: str | None = None
