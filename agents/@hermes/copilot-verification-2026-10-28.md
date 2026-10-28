# Hermes Copilot verification — 2026-10-28

Tracking task: `tsk-95550780-7b9e-4b0b-acff-ac9f2c9ff7dc`. Prior task unchanged.

## Verdict

The Hermes Copilot catalog is accessible using the GitHub CLI OAuth credential. An authorized corrective inference through Hermes model-specific resolution succeeded: `gpt-6-luna`, `codex_responses`, one HTTP inference request, response `COPILOT_OK`. The original test failed because the probe forced the wrong endpoint. This proves the deployed runtime path in a fresh isolated process, not an inference inside an existing Hermes PID.

## Runtime and credential evidence

- At 2026-10-28 00:56–00:57 ET, `~/.local/bin/hermes-acp` points to `/Users/jarvis_arunlab/code/hermes-agent-rt-2e496e05b9/.hermes/bin/hermes-acp`. Live process 30339 was running this runtime with `HERMES_HOME=/Users/jarvis_arunlab/code/research-lab/.hermes/home`, default profile (session `sess-A92AC9D0-81DB-4493-ABB2-F5EB212FA4CA`). Other Hermes processes and the standalone `hermes` launcher use a different runtime; no claim that PID 30339 is the chair session.
- `ps` startup environment for the inspected Hermes processes exposed none of `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, or `GITHUB_TOKEN`; the checking shell also had none. `ps` is startup-environment evidence, not proof of values added or removed in a running process after launch.
- A separate probe using the running ACP runtime's Python, source tree, same home and default profile called `resolve_copilot_token()`: source `gh auth token`, supported OAuth token family. This shows that an equivalent process can retrieve it via GitHub CLI; it does not inspect the existing process's dynamic memory or prove that process already resolved it.
- The default profile's Copilot credential pool reported one entry with no `access_token`; no token values were printed. The resolver's Copilot-token exchange fell back to the raw OAuth token; exchange failure reason was not exposed. The raw token successfully retrieved the account catalog. Exchange cache writes were disabled in the probe.
- The catalog used Hermes `resolve_api_key_provider_credentials('copilot')`, `fetch_github_model_catalog`, and the same provider request helpers. This was a separate process using the deployed runtime code, not an invocation inside PID 30339. A plugin import warning for Synapse occurred during the catalog probe; Copilot data was still returned.
- Inference used deployed `AIAgent.run_conversation` in a disposable home, passed the resolved credential, used no tools, `max_iterations=1`, `max_tokens=32`, and disabled client retries. That first inference attempt was made once. A request debug dump in the disposable home was removed with that home. This direct harness explicitly forced `chat_completions` instead of resolving the model's endpoint as the normal runtime does; HTTP 400 is therefore inconclusive for normal routing. No credential, settings, routing, gateway, or installation changes were made.
- Corrective retry: after two local setup attempts stopped before inference because dependencies were not loaded, the deployed runtime was bootstrapped with the live home, then an empty disposable `HERMES_HOME` was used for execution. Hermes `resolve_runtime_provider(requested="copilot", target_model="gpt-6-luna")` resolved provider `copilot`, source `gh_cli`, and mode `codex_responses`; token exchange degraded to the raw token. `AIAgent.run_conversation` returned exactly `COPILOT_OK`, with error false. A request guard counted one inference HTTP attempt and prevented any second attempt; SDK retries were disabled. Disposable state was removed. This isolated home does not reproduce all persisted configuration of the running profile, and does not demonstrate the existing process has loaded the credential.

## Hermes picker-enabled chat IDs (20)

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

## Other IDs returned in the same live account catalog (18)

These had `model_picker_enabled=false`; embedding IDs are labeled. Their presence in the catalog does not prove they are enabled for interactive inference.

```text
gpt-5.4-nano
gpt-4o-mini-2024-07-18
gpt-4o-2024-11-20
gpt-4o-2024-08-06
text-embedding-3-small [embedding]
text-embedding-3-small-inference [embedding]
gpt-4.1-2025-04-14
gpt-3.5-turbo-0613
gpt-4
gpt-4-0613
gpt-4-0125-preview
gpt-4o-2024-05-13
gpt-4-o-preview
gpt-4.1
gpt-3.5-turbo
gpt-4o-mini
gpt-4o
text-embedding-ada-002 [embedding]
```

The raw catalog contained 38 IDs total. The successful corrective inference establishes this one model and runtime route, not availability of every listed model.
