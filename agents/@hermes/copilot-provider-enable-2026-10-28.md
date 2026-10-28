# Hermes Copilot provider enablement — 2026-10-28

Task: `tsk-6688d7eb-e71c-4f47-84c1-d582c2587eaa`; related verification `tsk-95550780-7b9e-4b0b-acff-ac9f2c9ff7dc`.

## Outcome

The Switchboard Hermes default profile now has `model.provider: copilot` and `model.default: gpt-6-luna`. Previously they were `synapse` and `qwen3.8-twin-turbo`. The deployed Hermes CLI's `config set` command changed only these two parsed YAML values; all other sections and model keys compared equal against the pre-change copy. The default had to change because the ACP picker filters ambient GitHub CLI credentials unless Copilot is explicitly configured; a normal ACP inventory before the change contained no Copilot row. The change affects defaults read by new sessions; existing running sessions have their own model/provider state and were not switched or restarted.

## Evidence

- Live PID 30339, session `sess-A92AC9D0-81DB-4493-ABB2-F5EB212FA4CA`, has `HERMES_HOME=/Users/jarvis_arunlab/code/research-lab/.hermes/home` with no profile override; active launcher runtime is `/Users/jarvis_arunlab/code/hermes-agent-rt-2e496e05b9`. This is an observed Switchboard process, not a claim it is the chair process.
- Normal Hermes `load_picker_context` read back provider `copilot` and model `gpt-6-luna`. `resolve_api_key_provider_credentials` reported source `gh auth token`; no token was displayed or credentials intentionally modified.
- Live `fetch_github_model_catalog` returned 20 picker-enabled IDs. The normal `build_models_payload` returned the same 20 in its Copilot row, and deployed ACP `build_model_state` returned 20 `copilot:<id>` choices and `copilot:gpt-6-luna` as current. These are fresh equivalent processes using the deployed runtime and actual profile, not calls inside PID 30339 or a screenshot of the running app.
- Prior separate isolated real inference returned `COPILOT_OK` for `gpt-6-luna`; see `agents/@hermes/copilot-verification-2026-10-28.md`. No post-change inference was performed. A provider-plugin circular-import warning appeared in the standalone ACP state probe; it still returned the 20 choices, including Synapse. It is not proof of a live UI defect.
- Token exchange degraded to the raw GitHub CLI OAuth token. GitHub account/integrator entitlements apply; individual listed models may still reject inference. Picker exposure is established, not inference success for each model. The earlier raw account response contained 18 additional IDs with picker disabled, including embeddings; these are not claimed selectable.

## Live picker-enabled model IDs

```text
claude-haiku-5.5
claude-opus-4.8
claude-opus-5.5
claude-opus-5
claude-sonnet-5.5
claude-sonnet-5
gemini-3.8-flash
gemini-4-argon
gpt-5.3-codex
gpt-5.6-luna
gpt-5.6-sol
gpt-5.6-terra
gpt-6.1-sol
gpt-6.1-terra
gpt-6-astra
gpt-6-luna
gpt-6-sol
grok-4.7
mai-code-1.1-flash
claude-haiku-4.5
```

## Rollback

Exact pre-change config retained at `/Users/jarvis_arunlab/code/research-lab/.hermes/home/config.yaml.pre-copilot-20261028`. To restore just the previous selection while retaining subsequent unrelated settings, run these supported product commands against the same profile:

```sh
HERMES_HOME=/Users/jarvis_arunlab/code/research-lab/.hermes/home /Users/jarvis_arunlab/code/hermes-agent-rt-2e496e05b9/.hermes/bin/hermes config set model.default qwen3.8-twin-turbo
HERMES_HOME=/Users/jarvis_arunlab/code/research-lab/.hermes/home /Users/jarvis_arunlab/code/hermes-agent-rt-2e496e05b9/.hermes/bin/hermes config set model.provider synapse
```

Check current config before rollback; changing defaults may affect new sessions. The backup is a point-in-time copy and should not overwrite subsequent settings. No gateway restart, install, deployment, Directory route change, code edit, or commit was performed. Pre-existing dirty repo files remain untouched.
