# Integration Test Pipeline (CI + Full Models)

This document defines a unified integration test pipeline that runs the full
backend flow without a front-end dependency, with two profiles:

- CI profile (no models): fast, deterministic, runs locally and in GitHub Actions.
- Full profile (real models): production-like run using embedded Python.

The goal is that adding or modifying features only requires updating test cases
or adding a small phase assertion plugin. When a failure occurs, the pipeline
must produce enough information to locate the issue.

## Principles

- Single harness, multiple profiles.
- Phase-driven validation for stable, extensible assertions.
- Centralized logging control to avoid noisy debug output.
- Reproducible failures via artifacts and event traces.

## Profiles

### integration_ci (no models)

Purpose:
- Verify pipeline wiring, state transitions, SSE, outputs, and checkpoints.
- Run fast and deterministically for CI and local dev.

Characteristics:
- Uses DummyEngine or stub ASR engines.
- GPU and model loading disabled.
- Uses small synthetic or fixture audio.

### integration_full (real models)

Purpose:
- End-to-end verification with production-like behavior.
- Matches front-end operations and real model loading.

Characteristics:
- Uses embedded Python runtime and packaged models.
- Uses real media fixtures (audio/video).
- Validates SSE, output files, and quality gates.

## Environment Selection

Do not use the global Python environment.

- CI profile: `.venv/Scripts/python.exe`
- Full profile: `build/.../tools/python/python.exe` (embedded runtime)

The harness should support specifying an explicit Python executable.

## Phase Model

All tests share the same phase checkpoints:

1. Preprocess
   - Chunk metadata exists and is readable.
   - Preprocess cache files (if enabled) are present.
2. Pipeline
   - Draft/patch outputs exist and are well-formed.
   - Progress events are monotonic.
3. Output
   - SRT/VTT/ASS files exist and parse correctly.
4. State
   - job.status, checkpoint, and job.srt_path are consistent.
5. SSE
   - Expected event types are emitted in valid order.

Each phase can define assertions with a consistent interface.

## Harness Layout (proposed)

```
tests/
  integration_harness/
    runner.py
    case_loader.py
    phases/
      preprocess.py
      pipeline.py
      output.py
      state.py
      sse.py
    artifacts/
      collector.py
```

## Test Case Format (example)

Use YAML or JSON. Example YAML:

```
id: ci_smoke_audio
profile: ci
entry_mode: service   # service | api
input:
  audio: tests/fixtures/integration/sample.wav
expected:
  outputs: [srt]
timeouts:
  preprocess: 30
  pipeline: 120
  output: 30
assertions:
  preprocess: true
  pipeline: true
  output: true
  state: true
  sse: true
```

Notes:
- `entry_mode=service` for CI speed.
- `entry_mode=api` for full parity with front-end flow.

## Logging and Noise Control

To avoid debug-level noise masking failures:

- Default test log level: WARNING.
- Allow override with `TEST_LOG_LEVEL=INFO|DEBUG`.
- On failure, capture:
  - log files
  - checkpoint.json
  - output files
  - SSE event trace
  - test case config

## Failure Artifacts

On any failure, the harness should produce a diagnostics package:

```
artifacts/<case-id>/
  logs/
  checkpoints/
  outputs/
  sse_events.jsonl
  case.yaml
  summary.json
```

This must be uploaded in CI to simplify debugging.

## Local Execution

CI profile (no models):

```
.\.venv\Scripts\python.exe -m pytest -m integration_ci
```

Full profile (real models):

```
.\build\AnchorFlux_v3.1.1_Portable\tools\python\python.exe -m pytest -m integration_full
```

## GitHub Actions (CI profile)

Workflow sketch:

```
name: Integration CI
on: [push, pull_request]
jobs:
  integration_ci:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install deps
        run: |
          python -m venv .venv
          .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
          .\\.venv\\Scripts\\python.exe -m pip install -r requirements-dev.txt
      - name: Run integration CI
        run: .\\.venv\\Scripts\\python.exe -m pytest -m integration_ci
      - name: Upload artifacts
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: integration-ci-artifacts
          path: artifacts/
```

## Extending the Test Suite

Add a new case:
1. Create a new YAML/JSON case in `tests/integration_cases/`.
2. Add or reuse phase assertions.
3. Run `pytest -m integration_ci` locally.

Add a new phase assertion:
1. Create a new plugin in `tests/integration_harness/phases/`.
2. Register it in the harness dispatcher.
3. Add it to cases that need it.

## Troubleshooting

- If logs are too noisy, set `TEST_LOG_LEVEL=WARNING`.
- If a case is flaky, increase per-phase timeout and review SSE timing.
- If outputs are missing, inspect `artifacts/<case-id>/summary.json`.
