# Agent Execution Runbook

**Task:** implement M0-M6 in [milestones.md](milestones.md), using
[design.md](design.md) for architectural decisions and the
[research notes](../research/README.md) for evidence. This is the execution
companion, not another specification. The roadmap is currently named
`milestones.md`; do not restore the deleted older plan or rename the user's files.

Complete the implementation, not just scaffolding. WavLM and guide support are
optional for end users, but implementing/testing their opt-in paths is part of
this assignment. Deferred research in the roadmap's final section stays deferred.
Human listening and target-NLE approval are separate release gates: prepare the
evaluation materials, report those gates honestly, and continue implementing M6
without prematurely promoting a new default.

## 1. PowerShell preamble and inspection

Run this preamble in **each fresh shell/tool invocation**, then the relevant
block below. Commands require Windows PowerShell, not Bash or PowerShell 7.
`Run` turns nonzero native exit codes into errors; do not continue after failures.

```powershell
Set-Location 'D:\Code\sentence_mixer'
$ErrorActionPreference = 'Stop'
$Repo = (Get-Location).Path
$Py = Join-Path $Repo '.venv\Scripts\python.exe'
$Mm = Join-Path $Repo 'bin\micromamba.exe'
$MambaRoot = Join-Path $Repo '.mamba'
$env:MAMBA_ROOT_PREFIX = $MambaRoot
$env:MFA_ROOT_DIR = Join-Path $Repo '.build\mfa'
$env:HF_HOME = Join-Path $Repo '.build\huggingface'
$env:NLTK_DATA = Join-Path $Repo '.build\nltk'
function Run {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Exit ${LASTEXITCODE}: $Program $($Arguments -join ' ')"
    }
}
```

```powershell
Run 'git' @('--no-pager', 'status', '--short')
Run 'git' @('--no-pager', 'diff', '--stat')
Get-Command py.exe, ffmpeg.exe, ffprobe.exe, tar.exe -ErrorAction Stop
Get-ChildItem .\src, .\design, .\input
Run 'ffmpeg.exe' @('-version')
Run 'ffprobe.exe' @('-version')
```

Read the current source before editing. Preserve unrelated dirty changes,
existing CLI commands, custom dictionaries, media, and editor settings. Use
targeted patches; no blanket replacement with the obsolete example code.
Downloads of public packages/models are permitted; keep source media/transcripts
local. Do not commit unless explicitly requested.

## 2. M0: create the tooling and tests, then install

First implement `pyproject.toml` with explicit discovery for this repository's
existing module layout. Baseline dependencies are `numpy`, `scipy`, `soundfile`,
`librosa`, `praatio`, `webvtt-py`, `g2p_en`, and `matplotlib`; retain any additional
dependency still used by existing code. Add a `dev` extra with `pytest`, `ruff`,
`mypy`, and `pip-tools`, and a separate `wavlm` extra with `torch`, `transformers`,
and `huggingface-hub`. Preserve existing script/import compatibility. Configure
pytest paths and register `model` and `real_corpus` markers. Real models/network
access must not run during collection or ordinary unit tests.

Add ignore entries for `.venv`, `.mamba`, `bin`, `.build`, caches, and generated
outputs without replacing current exclusions. Create the typed contracts and M0
characterization tests. Configure lint/type checking meaningfully; do not disable
checks globally to make them pass.

These are first-time resolution commands. On subsequent runs, reuse the generated
locks rather than silently updating dependencies:

```powershell
if (-not (Test-Path -LiteralPath $Py)) {
    Run 'py.exe' @('-3.10', '-m', 'venv', '.venv')
}
Run $Py @('-m', 'pip', 'install', '-e', '.[dev]')
Run $Py @('-m', 'piptools', 'compile', '--extra', 'dev', '--output-file', 'requirements-dev.lock', 'pyproject.toml')
Run $Py @('-m', 'pip', 'install', '-r', 'requirements-dev.lock')
Run $Py @('-m', 'pip', 'install', '--no-deps', '-e', '.')
Run $Py @('.\run_mixer.py', '--help')
Run $Py @('-m', 'pytest', '-q', 'tests\test_cli_compat.py', 'tests\test_contracts.py')
```

If using `g2p_en`, explicitly provision its NLTK resources rather than allowing
unexpected downloads during inference:

```powershell
Run $Py @('-m', 'nltk.downloader', '-d', $env:NLTK_DATA, 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict')
```

## 3. M1 prerequisite: isolate and provision MFA

Use repository-local tools/environments, not an unrelated existing environment.
This uses the [official Windows micromamba bootstrap](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#windows).
Record the binary version/hash and freeze resolved dependencies after validation.
Do not replace an existing executable or environment without checking it first.

```powershell
if (-not (Test-Path -LiteralPath $Mm)) {
    New-Item -ItemType Directory -Force -Path '.\bin', '.\.build\bootstrap\micromamba' | Out-Null
    Invoke-WebRequest -UseBasicParsing `
      -Uri 'https://micro.mamba.pm/api/micromamba/win-64/latest' `
      -OutFile '.\.build\bootstrap\micromamba.tar.bz2'
    Run 'tar.exe' @('-xf', '.\.build\bootstrap\micromamba.tar.bz2', '-C', '.\.build\bootstrap\micromamba')
    Copy-Item -LiteralPath '.\.build\bootstrap\micromamba\Library\bin\micromamba.exe' -Destination $Mm
}
Run $Mm @('--version')
Get-FileHash -LiteralPath $Mm -Algorithm SHA256
if (-not (Test-Path -LiteralPath (Join-Path $MambaRoot 'envs\mfa_env\conda-meta\history'))) {
    Run $Mm @('create', '-r', $MambaRoot, '-n', 'mfa_env', '--override-channels', '-c', 'conda-forge', 'montreal-forced-aligner>=3.3,<5', '-y')
}
Run $Mm @('run', '-r', $MambaRoot, '-n', 'mfa_env', 'mfa', '--help')
foreach ($kind in @('acoustic', 'dictionary', 'g2p')) {
    Run $Mm @('run', '-r', $MambaRoot, '-n', 'mfa_env', 'mfa', 'model', 'download', $kind, 'english_us_arpa')
}
Run $Mm @('run', '-r', $MambaRoot, '-n', 'mfa_env', 'mfa', 'model', 'inspect', 'acoustic', 'english_us_arpa')
$MfaLock = Run $Mm @('env', 'export', '-r', $MambaRoot, '-n', 'mfa_env')
$MfaLock | Set-Content -Encoding UTF8 '.\environment-win-64.lock.yml'
```

The `<5` bound retains the documented legacy model-download interface; MFA 3.4
deprecates it for later removal. Verify installed help and model metadata against
the [official model documentation](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/models/index.html).
Do not confuse software and model versions. Record actual model checksums and
phone sets; do not claim that a downloaded alias equals the 2026 benchmark model.
Remove machine-specific prefix fields from the tracked environment export while
retaining exact package/build pins. Wire the Python launcher to the configured
root/executable; changing shell variables alone does not fix today's hardcoding.

If solving/downloading fails, preserve the error, try a documented compatible
configuration, and record any changed constraint. Continue independent code/unit
tests; never substitute a fake successful alignment.

## 4. Implement M1-M6 in this order

The files/tests/tools below are **deliverables to create or update**, not
pre-existing commands to blindly invoke. Implement each row, run its focused
command, fix failures, then proceed. Refer to the matching roadmap milestone for
contracts and edge cases rather than weakening them.

| Step | Concrete implementation work | Focused verification after implementation |
| --- | --- | --- |
| M1 | Fix [subtitle preprocessing](../src/vtt_preprocessor.py); add shared lexicon, source-time/speaker contracts, content-aware caches, validated ordered inventory, and configurable MFA execution. Preserve silence and actual pronunciations. | `Run $Py @('-m','pytest','-q','tests\test_vtt_preprocessor.py','tests\test_inventory_builder.py','tests\test_media_time.py','tests\test_cache.py')` |
| M2 | Add continuous basic feature extraction and target pronunciation graph modules. Implement voicing/missing-data policies, context across words, and duration/prosody statistics. | `Run $Py @('-m','pytest','-q','tests\test_features.py','tests\test_target_graph.py')` |
| M3 | Add span retrieval and global selection behind `--selector span`; retain legacy selection. Protect full runs, validate pronunciation coverage, and implement bounded coupling variants and deterministic search. | `Run $Py @('-m','pytest','-q','tests\test_span_selector.py')` |
| M4 | Implement immutable edit planning, PCM-first rendering, rational video scheduling, and master-audio XML. Add `tools\validate_mix.py --stem <output-basename>` with nonzero failure on invalid artifacts/timing/provenance. | `Run $Py @('-m','pytest','-q','tests\test_edit_plan.py','tests\test_video_assembler.py','tests\test_fcp_xml_generator.py','tests\test_validate_mix.py')` |
| M5 | Wire all commands in [run_mixer.py](../run_mixer.py), migration/reporting, and auditionable alternatives. Create `tools\evaluate.py --suite <json> --output-dir <dir>` and `evaluation\suite.json` for repeatable cases, metrics, blinded assets, and a separate assignment key. Update usage docs. | `Run $Py @('-m','pytest','-q','-m','not model and not real_corpus','tests\test_cli_compat.py','tests\integration\test_pipeline.py','tests\test_evaluation.py')` |
| M6 | Implement optional WavLM retrieval/reranking, checkpoint/layer caches, and guide prosody. Add the model-preparation helper in section 6 and real-model tests. The basic path must work without Torch. | `Run $Py @('-m','pytest','-q','tests\test_wavlm_backend.py','tests\test_prosody_guide.py')` |

Mandatory regressions include the rank-31 intact word, repeated 14 ms unit,
ordinary multiline/rolling captions, cross-word context and real silence,
pronunciation variants, no fabricated features, and no automatic word pauses.
Test exact search against brute force and long timelines with 1,000 short
contributions. Required final tests cannot be empty, permanently skipped, or
xfail placeholders. Mock-based unit tests do not replace real-media/model checks.

Implement all roadmap-required baseline behavior. If experimenting with half-phone/
diphone units, represent and test explicit coverage parts; do not just trim phones
and retain their old labels. Deferred research is not a reason to omit the
required phrase/word/phone path.

## 5. Real-media smoke and acceptance evidence

Create `evaluation\smoke_sources.json` using the roadmap's manifest schema, paths
relative to that manifest (for example `..\\input\\obi_wan_ep3.mp4` in JSON), and
verified speaker intervals/metadata. There is currently no root source manifest;
do not assume one exists or resurrect a deleted user file.

Available sources are the episode 2/3 MP4/VTT pairs under `input`. The episode 3
captions contain "hello there" around 00:05:28; that is a starting point for
inspection, not verified phonetic timing or proof of speaker purity. Do not mark
whole files single-speaker merely because of their names. If verification requires
human input, generate candidates and record that limitation rather than inventing
verification. Synthetic integration tests must still run independently.

After M1-M5 and a verified smoke manifest:

```powershell
Run $Py @('.\run_mixer.py', 'mix', '--manifest', '.\evaluation\smoke_sources.json', '--speaker', 'speaker-a', '--text', 'Hello there', '--selector', 'span', '--features', 'basic', '--mode', 'source-derived', '--fps', '30000/1001', '--render', '.\output\smoke_basic.mp4', '--plan', '.\output\smoke_basic.plan.json', '--xml', '.\output\smoke_basic.xml')
Run $Py @('.\tools\validate_mix.py', '--stem', '.\output\smoke_basic')
Run 'ffprobe.exe' @('-v', 'error', '-show_streams', '-show_format', '-of', 'json', '.\output\smoke_basic.mp4')
Run $Py @('.\tools\evaluate.py', '--suite', '.\evaluation\suite.json', '--output-dir', '.\output\evaluation')
```

The validator must check exact master samples, target/source coverage, legal
source mappings, declared processing, logical mux timing, bounded frame error,
and XML/master agreement. FFprobe metadata alone is not this validation.
Evaluation must include intact phrases, new word combinations, and subword
fallback, not just the two-word smoke case.

Create `tests\integration\test_real_corpus.py`, marked `real_corpus`, to exercise
actual MFA alignment, cached reruns, and render validation using this manifest.
Fail with an actionable prerequisite error when explicitly requested without
verified inputs; do not silently skip it or substitute the synthetic fixture.

## 6. Finish M6: real WavLM, not just an interface

Implement `tools\prepare_wavlm.py --model <id> --config <json>` to resolve and
persist an immutable Hugging Face revision, select an explicit layer profile,
download/load that revision locally, and validate it with `trust_remote_code=False`.
Reuse a saved revision on reruns; do not silently follow a moving branch. Store
weights under `HF_HOME`, not Git. Make `--features wavlm` consume the resulting
default `config\wavlm.json`, and record its identity in caches/plans.

Only now install the optional dependencies:

```powershell
Run $Py @('-m', 'piptools', 'compile', '--extra', 'dev', '--extra', 'wavlm', '--constraint', 'requirements-dev.lock', '--output-file', 'requirements-wavlm.lock', 'pyproject.toml')
Run $Py @('-m', 'pip', 'install', '-r', 'requirements-wavlm.lock')
Run $Py @('.\tools\prepare_wavlm.py', '--model', 'microsoft/wavlm-base-plus', '--config', '.\config\wavlm.json')
Run $Py @('-m', 'pytest', '-q', '-m', 'model', 'tests\integration\test_wavlm_real.py')
Run $Py @('.\run_mixer.py', 'mix', '--manifest', '.\evaluation\smoke_sources.json', '--speaker', 'speaker-a', '--text', 'Hello there', '--selector', 'span', '--features', 'wavlm', '--mode', 'source-derived', '--fps', '30000/1001', '--render', '.\output\smoke_wavlm.mp4', '--plan', '.\output\smoke_wavlm.plan.json', '--xml', '.\output\smoke_wavlm.xml')
Run $Py @('.\tools\validate_mix.py', '--stem', '.\output\smoke_wavlm')
```

Test real encoder loading, feature/source-time mapping, warm-cache reuse, and
CPU execution on a short fixture, even when no real-corpus speaker manifest is
available. Model tests must lazily import optional dependencies and fail rather
than skip when explicitly requested but incorrectly configured.
Verify that enabled embedding scores actually participate in ranking; extracting
unused vectors is not a completed WavLM backend.

Also implement/test `--guide`: validate the guide transcript, derive soft relative
prosody, and prove its waveform cannot enter output. A supplied human guide is
needed for real-guide acceptance, not for unit-testing that safety contract.
Extend the evaluation suite with basic/WavLM ablations and resource measurements.
Keep source-only output, sample-accurate coupling, and conventional acoustic
features; no vocoder or text-to-WavLM guess.

## 7. Final automated gate and handoff

```powershell
Run $Py @('-m', 'pip', 'check')
Run $Py @('-m', 'ruff', 'check', 'src', 'tests', 'tools', 'run_mixer.py')
Run $Py @('-m', 'mypy', 'src', 'tools', 'run_mixer.py')
Run $Py @('-m', 'pytest', '-q', '-m', 'not model and not real_corpus')
Run $Py @('-m', 'pytest', '-q', '-m', 'model')
Run $Py @('-m', 'pytest', '-q', '-m', 'real_corpus')
Run 'git' @('--no-pager', 'diff', '--check')
Run 'git' @('--no-pager', 'status', '--short')
```

Run real-corpus tests with the verified fixture/configuration; if unavailable,
report the exact missing prerequisite, not a fabricated pass. Record commands,
exit codes, versions, outputs, and outstanding gates in
`.build\reports\implementation-status.json`, with per-milestone states such as
`implemented`, `automated_verified`, and `human_validation_pending`.

Do not call the full roadmap release-validated until required listening and NLE
checks actually happen. Keep legacy/basic defaults until promotion gates pass,
but finish the opt-in span/WavLM/guide code and evaluation tooling. Final handoff:
changed files, milestone results, reproducible run commands, output paths,
test/lint/type results, and precisely identified remaining human/external checks.
