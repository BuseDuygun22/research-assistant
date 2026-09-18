"""Process-wide settings (JOINT).

Everything configurable is read here and nowhere else, so a CI run and a laptop
run differ only by environment, never by an edited literal. Secrets come from the
environment; corpus/model choices come from `configs/*.yaml` so they can be
version-controlled and diffed in a PR.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
EVAL_DIR = REPO_ROOT / "eval"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RA_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- corpus / index identity -------------------------------------------
    corpus_version: str = Field(
        "dev", description="Bump on any re-ingest that changes chunk boundaries."
    )
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- services -----------------------------------------------------------
    retrieval_api_url: str = Field(
        "http://127.0.0.1:8000",
        description="FastAPI service the MCP tools call. Overridden to the compose "
        "service name inside Docker.",
    )
    tool_deadline_seconds: float = Field(
        10.0,
        ge=0.0,
        description="Per-tool wall-clock budget. A hung backend is the one failure "
        "an agent cannot observe - no error, no budget spent, no escalation - so it "
        "is converted into an ordinary retryable ToolError. 0 disables the check.",
    )
    mcp_transport: Literal["stdio", "sse"] = "stdio"
    mcp_port: int = 8080

    # --- judge / generation -------------------------------------------------
    judge_backend: Literal["stub", "anthropic", "gemini"] = Field(
        "stub",
        description="'stub' is deterministic and free — the default so CI never "
        "depends on a paid API. 'anthropic' and 'gemini' are both real backends; "
        "which one a run uses is whoever's key is set, not a preference this repo "
        "takes — Gemini's free tier is why it exists as an option at all.",
    )
    # No single default is right for both real backends (a Claude model id is
    # meaningless to Gemini and vice versa). This default assumes 'anthropic';
    # set RA_JUDGE_MODEL in .env to a Gemini model id (e.g. gemini-flash-latest)
    # when running on 'gemini', or get_llm() will hand that id to the wrong API.
    judge_model: str = "claude-sonnet-5"
    judge_prompt_version: str = "v1"
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None

    # --- agent graph --------------------------------------------------------
    # Budgets are separate because the two revision types cost differently: a
    # rewrite is one writer call, a re-retrieval is a retrieval round plus a
    # rerank plus a writer call. One shared counter would let cheap rewrites
    # starve the expensive repair that actually fixes an evidence gap.
    max_rewrites: int = Field(3, ge=0, le=10, description="Writer revision cap.")
    max_re_retrievals: int = Field(2, ge=0, le=5, description="Retrieval round cap.")
    max_steps: int = Field(
        12, ge=1, le=50, description="Global backstop against a cycle no specific cap catches."
    )
    writer_top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Spans delivered to the writer. The context-rot budget: retrieve "
        "wide, deliver narrow. Raising this is the easiest way to quietly make "
        "drafts worse.",
    )
    # Kept as a deprecated alias so anything still reading it gets a sane value
    # rather than an AttributeError; nothing in the graph consults it.
    max_revisions: int = Field(3, ge=0, le=10, description="Deprecated: use max_rewrites.")
    cost_aware_escalation: bool = Field(
        False,
        description="Escalate once finishing the remaining budget would cost more "
        "than a human review. Off by default: it trades a little autonomy for "
        "spend, which is an operator decision rather than one to inherit.",
    )
    escalation_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Judge confidence below which a draft goes to flag_for_human "
        "regardless of whether it passed. Ships at 0.0 — escalate on nothing — "
        "because an uncalibrated confidence is not a threshold anyone can defend. "
        "Raise it only once the judge is calibrated against Buse's qrels and we "
        "know what a given number means.",
    )

    # --- observability ------------------------------------------------------
    tracing_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    def yaml_config(self, name: str) -> dict[str, Any]:
        """Load `configs/<name>.yaml`; missing file is an empty dict so a track
        that has not landed its config yet does not break the other track."""
        path = CONFIG_DIR / f"{name}.yaml"
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
