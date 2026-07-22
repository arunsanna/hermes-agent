"""ZAI / GLM provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class ZaiProfile(ProviderProfile):
    """Z.AI / GLM — top-level reasoning_effort, no extra_body reasoning toggle.

    Z.AI accepts max|xhigh|high|medium|low|minimal|none for
    ``reasoning_effort``, but only "high", "max", and "none" are distinct
    on the wire: low/medium map to high server-side, xhigh maps to max,
    ultra is our own alias for max, and none/minimal skip thinking.
    Clamp to the meaningful values here; default to max when no effort was
    explicitly requested. IMPORTANT: disabled/none must SEND
    ``reasoning_effort: "none"`` — omitting the field makes Z.AI apply its
    own default (max), the opposite of "off".
    """

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        top_level: dict[str, Any] = {}

        thinking_off = bool(
            reasoning_config
            and isinstance(reasoning_config, dict)
            and reasoning_config.get("enabled") is False
        )
        if thinking_off:
            top_level["reasoning_effort"] = "none"
            return {}, top_level

        raw_effort = ""
        if reasoning_config and isinstance(reasoning_config, dict):
            raw_effort = (reasoning_config.get("effort") or "").strip().lower()

        if raw_effort in {"none", "minimal"}:
            # Explicitly requested no reasoning — Z.AI skips thinking for
            # "none"; must be sent explicitly (omission = server default max).
            effort = "none"
        elif raw_effort == "":
            # No explicit effort requested (config absent, or present
            # without an "effort" key) — default to max.
            effort = "max"
        elif raw_effort in {"ultra", "xhigh"}:
            effort = "max"
        elif raw_effort in {"low", "medium"}:
            effort = "high"
        elif raw_effort in {"high", "max"}:
            effort = raw_effort
        else:
            effort = "max"

        if effort is not None:
            top_level["reasoning_effort"] = effort

        return {}, top_level


zai = ZaiProfile(
    name="zai",
    aliases=("glm", "z-ai", "z.ai", "zhipu"),
    env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
    display_name="Z.AI (GLM)",
    description="Z.AI / GLM — Zhipu AI models",
    signup_url="https://z.ai/",
    fallback_models=(
        "glm-5.2",
        "glm-5",
        "glm-4-9b",
    ),
    base_url="https://api.z.ai/api/paas/v4",
    default_aux_model="glm-4.5-flash",
)

register_provider(zai)
