# Expert whitelists: run only the experts a task needs (opt-in)

An expert whitelist tells the engine which routed experts it may use. Every other expert is never chosen by the
router and never loaded into VRAM or RAM. The model's weights are not changed and nothing is trained: the engine only
picks from fewer experts. Without `--expert-whitelist` nothing changes.

The goal is a model that is specialised for a few programming languages (for example Rust, or Rust and TypeScript)
and needs less memory. Quality for the chosen languages should stay close to the full model. Quality for everything
else (other languages, prose, foreign languages) may drop a lot. That is expected.

The feature has two parts. The engine side reads a whitelist and keeps to it (sections 1 to 6). The generator
(section 7) builds a whitelist from calibration data: it runs your samples through the model, counts which experts the
router picks for which language and task, and keeps the experts that carry your languages. Before the large
collection, `analyze` measures on a small sample whether the model's experts specialise by language at all (section
7.7). Three modes decide what happens to a blocked expert the router would pick: `strict` (never), `substitute` (a
similar allowed expert takes over) and `tiered` (a second tier is read from the drive when picked). **Nothing on this
page is a measurement**: no speed or quality number exists yet for a whitelisted run or a collection on a real model,
and the CUDA and HIP builds of this change were not run on the machine it was written on (sections 6 and 7.9).

Sections: 1 the survey this feature is built on, 2 the formats, 3 what the engine does with a whitelist, 4 how to use
it, 5 the tests, 6 limits and open points of the engine side, 7 the generator.

## 1. How the engine routes and loads experts (survey)

Read from the code of engine 0.1.46 (Qwen3.8-Flash-Next family, `src/` and `include/`). This is static analysis of
the code, not a measurement.

**Routing.** Every layer is an MoE layer with the same expert count (512, the Coder 256) and top-10. The router is a
BF16 GEMV that writes one float logit per expert into a device buffer, then a top-k kernel reads it:

    p = softmax(logits)          over ALL experts (float or double exp, max subtracted)
    ids = top-k of p             strict greater, ties to the lower expert index
    w = p[ids] / max(sum(p[ids]), 2^-14)

There is no sigmoid, no bias or correction term (`exp_probs_b`) and no group-limited routing. The kernels are
`router_top10` (`src/kernels/cuda/router_top10.cu`), the fused `native_router_top10` / `_multi`
(`src/kernels/cuda/native_router.cu`) and the prompt's `route` (`src/prefill/kernels.cu`). A logit of `-INFINITY`
gives `p = 0` as long as one logit is finite. Two opt-in passes change the ids after the top-k:
`STRATA_ROUTE_RESIDENT` (swap tail ranks for resident experts) and `STRATA_ROUTE_PRIOR` (re-pick with a bonus for
resident experts). The debug hook `STRATA_DBG_ROUTE_OVERRIDE` can replace a prompt's ids from a file.

**Shared expert and dense layers.** The shared expert (`ffn_*_shexp`, with its own sigmoid gate) runs on every token
and is never routed. The supported models have no dense FFN layers and one expert count for all layers.

**Where routing runs (CUDA/HIP).**

| Path | Code | Notes |
|---|---|---|
| Decode, one token | `moe_route` (`src/core/layer.cpp`) | inside the captured per-layer and per-token graphs |
| Verify window (speculative check, also `--batch` slots) | `Verifier::record_window` (`src/core/verify.cpp`) | one captured graph per window; per-token `moe_route`, or one multi-row router launch |
| gfx906 window | `moe_route_window` (`src/core/layer.cpp`) | HIP only |
| MTP draft layer | `MtpDrafter` (`src/core/mtp.cpp`) | the draft layer's own 512 experts, all in VRAM |
| Prompt (prefill) | `prefill.cpp` + `route()` | not captured; the host reads the ids back |
| Router look-ahead | `RouterLookahead` (`src/core/expert_source.cpp`) | host prediction of the next layer; only warms pages |
| Qwen3.6-35B-A3B | `src/qwen35/q36_engine.cu` | its own engine |
| Mac and Intel | `metal/`, `sycl/` | their own trees |

**How experts are loaded.** By default (`ArenaExpertSource`) the engine reads all of `experts.bin` (or the GGUF
shards) into pinned RAM at start. `--mmap-experts` maps the file instead (`FileExpertSource`), and
`--resident-experts` adds a compact RAM copy of the experts the GPU does not hold, with an explicit offset table per
(layer, expert). The VRAM expert cache is filled from the `--expert-profile` ranking at start. During the run the
adaptive tier, the RAM tier (`STRATA_RAM_ADAPT`), the helper and peer GPUs and `--kv-grow` move experts by their
routing counts. The prompt path streams every non-resident expert of a layer to the GPU through a ring (the
"streamed walk"). The router look-ahead and the Foresight swap space (`STRATA_FS_SLOTS`) warm or copy predicted
experts.

**Pack format.** No file stores an offset per expert. The offset is always computed:
`layer_offset(l) + e * blob_bytes(l)` (`ExpertLayout`, `include/strata/kernels/cpu/expert_layout.hpp`), with the
per-layer numbers from `native_experts.txt` (native packs) or the fixed 1,382,400-byte blob (canonical packs).

**Telemetry that exists.** `--dump-routing PATH` writes every generated token's routed ids and weights per layer
(the weights are zeros on native packs; the prompt is not recorded). `--expert-profile-save` writes the adaptive
tier's learned ranking. `STRATA_LOOKAHEAD_STATS=1` scores the look-ahead's guesses.

## 2. The formats

All files are JSON (the repository's format for configs and manifests). Unknown keys are ignored; a newer
`format_version` is refused.

### 2.1 Profiles: what the specialised model is for

A profile describes the languages and tasks, not the experts. Examples: `data/expert_whitelist/profiles/`
(`rust-coding.json`, `python-coding.json`, `typescript-coding.json`). The Rust and TypeScript profiles use a small
demonstration set in `data/expert_whitelist/calibration/` (a few prompts each; for tests and examples only, far too
small for a real whitelist). The Python profile uses a larger set meant for a real run (section 4.1).

```json
{
  "format_version": 1,
  "kind": "strata-whitelist-profile",
  "name": "rust-coding",
  "target_languages": {
    "rust": {"weight": 1.0, "min_coverage": 0.95, "calibration": ["../calibration/rust.jsonl"], "sources": ["~/code/**/*.rs"]}
  },
  "tasks": {"codegen": 1.0, "compiler_errors": 0.9, "debugging": 0.8, "refactoring": 0.6, "repo_analysis": 0.5, "tool_use": 0.7},
  "base_requirements": {
    "enabled": true,
    "natural_languages": ["en", "de"],
    "items": {
      "instructions":  {"calibration": ["../calibration/base_instructions_en.jsonl"], "min_coverage": 0.90},
      "chat_template": {"calibration": ["../calibration/base_chat_template.jsonl"], "min_coverage": 0.99},
      "tool_calls":    {"calibration": ["../calibration/base_tool_calls.jsonl"], "min_coverage": 0.95},
      "shell_build":   {"calibration": ["../calibration/base_shell_build.jsonl"], "min_coverage": 0.90}
    }
  },
  "excluded": {"python": {"calibration": ["../calibration/python.jsonl"]}},
  "reference_corpus": [],
  "target_hardware": {"vram_gib": 16, "ram_gib": 64, "reserve_gib": 8},
  "limits": {"max_samples": 2000, "max_tokens_per_sample": 2048, "seed": 1},
  "chat_template": "model"
}
```

- `target_languages`: each with a weight (above 0), its own minimum coverage (0 to 1) and calibration files or
  source globs.
- `base_requirements`: what every specialised model must keep, each with its own data and coverage: instructions in
  the user's languages, the chat template, tool calls, shell and build output. On by default; `"enabled": false`
  drops them.
- `excluded`: languages and domains that are explicitly not needed. They serve as a contrast and are measured apart.
- `target_hardware`: the memory the whitelist's size is derived from.
- Paths and source globs are relative to the profile file, or to the repository root when only that matches. A
  source glob that matches no file is a warning in `prepare`.
- Calibration files are JSON lines: `"prompt"` (required), `"response"`, `"task"` and `"language"` (optional).

Several profiles combine (`check-profile rust-coding.json typescript-coding.json`). A language or task named twice
takes the larger weight and coverage and both sets of files. The hardware takes the smaller VRAM and RAM and the
larger reserve. A domain one profile excludes and another targets is not excluded.

```bash
python tools/expert_whitelist.py check-profile data/expert_whitelist/profiles/rust-coding.json data/expert_whitelist/profiles/typescript-coding.json
```

### 2.2 The whitelist manifest: which experts

```json
{
  "format_version": 1,
  "kind": "strata-expert-whitelist",
  "model_fingerprint": "fnv1a64:25d1f5de79ecddd4",
  "num_layers": 48,
  "experts_per_layer": 512,
  "top_k": 10,
  "mode": "strict",
  "substitute_threshold": 0.5,
  "layers": [
    {"layer": 0, "allowed": [1, 5, 6, 14, ...], "on_demand": [7, 9],
     "substitutes": [{"expert": 2, "with": 1, "similarity": 0.81, "rel_l2": 0.44}], "scores": {"1": 0.52}},
    ...
  ],
  "generation": {"profile": "rust-coding", "scoring": {"method": "...", "params": {}}, "calibration_tokens": 123456,
                 "created": "2026-10-21", "stats": [{"path": "...", "sha256": "..."}]}
}
```

- Every expert id is the **original** one, `(layer, expert)`, numbered as the model's router numbers it. It is the
  only id in manifests, statistics, logs and tests.
- `allowed` is the resident tier: routed and loaded. `on_demand` (tiered mode only) is routed, but read from the
  drive when the router picks it. `substitutes` (substitute mode) names an allowed replacement for a blocked expert,
  with how alike their outputs are (`similarity`, a mean cosine from -1 to 1, and optionally `rel_l2`, the mean
  relative distance; section 7.3). `substitute_threshold` is the similarity a replacement needs: one below it is not
  used, and its blocked expert falls back to the best allowed one. A replacement without a `similarity` is always
  used. `scores` are optional, for traceability.
- `experts_per_layer` is one number, or a list when layers differ. The shared expert is never listed: it is never
  routed, never filtered and not part of any budget.
- `model_fingerprint` ties the manifest to one model and one pack. It is FNV-1a 64 over a text of the layer count,
  top-k, hidden width, experts per layer and the pack's expert index (where each layer's experts start, the bytes of
  one expert). Print it with:

```bash
python tools/expert_whitelist.py fingerprint --pack <pack> --gguf <model>-00001-of-00002.gguf
```

**The engine refuses a manifest at start, with a message that names the problem, when:** the fingerprint is not this
model's, the `format_version` is unknown, `num_layers`, `top_k` or `experts_per_layer` differ from the model, a layer
or expert id is out of range, an id or a layer is listed twice, an expert is both allowed and on_demand, an MoE layer
is missing, a substitute is not allowed (or replaces an allowed expert), `substitute_threshold` or a `similarity` is not
a number from -1 to 1, or a layer has fewer routable experts than top-k. **top-k is never lowered.** The same checks
without the engine:

```bash
python tools/expert_whitelist.py validate my-whitelist.json --pack <pack> --gguf <model>-00001-of-00002.gguf
```

### 2.3 The pack's expert index (optional sidecar)

The packs store no offset per expert; the engine computes `layer_offset + expert * blob_bytes` from
`native_experts.txt` or the canonical layout. `experts.index.json` writes those numbers down for an existing pack,
without changing any file the pack already has:

```bash
python tools/iq_pack.py index <pack>
python tools/iq_pack.py index --check <pack>
```

When a pack has the file, the engine checks it against the layout it computed and refuses a stale one. A pack
without it runs exactly as before.

## 3. What the engine does with a whitelist

### 3.1 Routing, by mode

The mode is in the manifest; `--expert-whitelist-mode` overrides it. The router weights are never changed, no expert
is created and no weights are merged, in any mode.

- **strict** (the default): before the top-k, every blocked expert's logit becomes `-inf`. The softmax then runs
  over the allowed experts only, the top-k picks among them, and the weights are normalised exactly as the model
  normalises them (`w / max(sum, 2^-14)`). The allowed logits keep their bits. After the top-k a check runs. If a
  blocked id still got through, the best allowed expert not yet chosen takes its place and the token's weights are
  computed again. That can only happen when an allowed expert's probability rounds to 0 (a logit gap over about 87)
  or a debug pass rewrote the ids.
- **substitute**: the model's own top-k over all experts. Each blocked pick goes to its replacement from the
  manifest and takes its router weight along. If the replacement was already chosen, the two weights add and the
  freed rank takes the best allowed expert not chosen, with weight 0, so every token still has top-k distinct
  experts. A blocked pick without a replacement at or above `substitute_threshold` (none listed, or one below it)
  falls back to the best allowed expert (the largest logit), which takes its weight. The weights still sum to what the
  router gave. `--expert-whitelist-substitute-threshold X` overrides the manifest's threshold.
- **tiered**: strict over `allowed` and `on_demand` together. Only `allowed` is loaded at start. An `on_demand`
  expert is read from the drive when the router picks it and kept in the **on_demand cache**: a fixed budget in RAM
  (`--expert-whitelist-cache-gib`, default 1 GiB), the least recently used expert out first. An expert picked during the
  last three layer changes is never evicted, so the pointer a CPU pool works with stays valid, as the expert source's
  stage buffers keep theirs. A miss that finds no room (every entry still in use) is read from the files as without
  the cache and counted. Every read of an on_demand expert goes through the cache: the CPU pools of decode and verify
  windows take a pointer (`blob`), and the prompt path has the source copy the expert out of the cache into the
  stager's buffer (`copy_blob`, `copy_blobs`), holding the entry only while it is copied, so a prompt never pins
  entries and the least recently used ones still go. The prompt path's CPU share leaves on_demand experts to the GPU,
  so no pointer into the cache leaves the source. The VRAM cache, the RAM copy, the adaptive tiers and the look-ahead
  never take an on_demand expert. A chunk of the prompt routes first and then reads only what it routed, instead of
  streaming every expert. A blocked expert is still a hard error, also when the cache is asked for it.

Softmax versus sigmoid routing, and group-limited routing: the supported models use softmax over all experts and
have no expert groups and no bias terms. A family with sigmoid scores or a correction bias would need the mask
applied after the bias. One with group-limited routing would need groups without an allowed expert removed before
the group pick. Neither exists in this engine today.

The filter's arithmetic is one header, `include/strata/core/whitelist_route.hpp`. The GPU kernels
(`src/kernels/cuda/expert_whitelist.cu`, one thread per token) and the host tests run the same code. The on_demand
cache is `OnDemandCache` in `include/strata/core/expert_whitelist.hpp`; the engine (`FileExpertSource::blob`) and the
tests use the same class.

### 3.2 Every path keeps to it

| Path | What it does with a whitelist |
|---|---|
| Decode (`moe_route`, captured graphs) | mask before the top-k, the check after it, before the doorbell publishes the ids |
| Verify window and `--batch` slots (`Verifier::record_window`) | the same around the multi-row router; the check runs again after `STRATA_ROUTE_PRIOR` and `STRATA_ROUTE_RESIDENT` |
| gfx906 window (`moe_route_window`) | the same |
| Prompt (`prefill.cpp`) | the same, the check after the debug override; the host checks every routed id; the streamed walk skips blocked experts |
| MTP draft layer | its own whitelist (`--draft-expert-whitelist M`) or none (`none`, the default: all of its experts). With one, only its experts are read into a compact VRAM store, and the ids go through the slot map before the grouped kernels. The verify window that checks the drafts uses the main whitelist, so every emitted token keeps to it |
| CPU expert pool (decode and verify windows) | every routed id is checked before anything is read; a blocked one is a hard error |
| VRAM cache fill, layer split stages, peer and helper GPUs | the `--expert-profile` ranking keeps the resident tier only (allowed experts it does not rank follow, in layer order); the helper and peer tiers skip anything else |
| RAM copy (`--resident-experts` form) | holds allowed experts the VRAM cache does not; the start checks it expert by expert |
| Adaptive tiers (VRAM, `STRATA_RAM_ADAPT`), `--kv-grow` | they follow the routing counts, which count only the resident tier |
| Router look-ahead and read-ahead | predicts the resident tier only |
| on_demand cache (tiered) | `FileExpertSource::blob` (the CPU pools) and `copy_blob`/`copy_blobs` (the prompt path's stager) serve an on_demand expert from it; a blocked one is refused there too |
| Foresight swap space (`STRATA_FS_SLOTS`) | refused with a whitelist (not wired yet) |
| Qwen3.6 engine, Metal (Mac), SYCL (Intel) | refused at start with a message: their routers have no mask yet |

**Loading.** The default source (the RAM arena) reads every expert at start, so with a whitelist the engine takes
the mapped files instead (`--mmap-experts`). With an `--expert-profile` it adds the RAM copy of the allowed experts
the GPU does not hold (the `--resident-experts` form). That copy has an explicit offset per original expert, the
original-id-to-slot table. Without a profile there is no VRAM fill and no RAM copy, and allowed experts are read
from the mapped files when routed. Blocked experts are never read: the expert source refuses them (`blob`,
`copy_blob`, prefetch, read-ahead, the RAM tiers).

**Hard errors.** A path that still asks for a blocked expert stops the run, naming the layer and the expert:
`expert whitelist: the decode expert pool asked for expert 17 of layer 3, which the whitelist does not allow`.
Nothing is loaded silently. The one allowed exception is an on_demand expert in tiered mode.

**At start** the engine prints the manifest, profile, mode and the allowed experts in total and per layer. It prints
the estimated expert bytes (allowed and on_demand, from the pack's blob sizes). It also prints what the VRAM cache
and the RAM copy really hold, checked expert by expert.

**`--expert-whitelist-stats`** counts, per layer, the experts the router selected (on the GPU, for every path) and
the experts that ran (on the CPU, a GPU or over PCIe). In tiered mode it also counts the on_demand hits and misses
(the on_demand cache's, for every path), the bytes read per token, the cache's
evictions, the misses it had no room for, and the time the reads took (summed over the threads that read). From that
time it prints what the run's tokens/s would be without the reads, as an upper bound (a calculation: some reads
overlap with work). At the end it prints all of it, and a run that selected or ran an expert outside the whitelist
fails.

Without `--expert-whitelist` none of this runs: no kernel is launched (the captured graphs are the same), the checks
are a few atomic loads that find no whitelist, and the default source is unchanged.

## 4. How to use it

1. **A profile.** Copy one of `data/expert_whitelist/profiles/` and point it at your own calibration data: prompts
   in the target languages, the tasks you care about, the base requirements. Check it with `check-profile`.
2. **Statistics.** `tools/whitelist_gen.py prepare`, then `collect` (section 7). `collect` is the only step that runs
   the model; `estimate` says beforehand what it needs.
3. **A whitelist.** `tools/whitelist_gen.py build` writes the manifest, a report and a coverage curve (section 7.6).
   A manifest can also be written by hand or with `manifest_document()` in `tools/expert_whitelist.py`. Check it:

   ```bash
   python tools/expert_whitelist.py fingerprint --pack <pack> --gguf <model>-00001-of-00002.gguf
   python tools/expert_whitelist.py validate my-whitelist.json --pack <pack> --gguf <model>-00001-of-00002.gguf
   ```

4. **Run with it.** Add the options to the engine's arguments (the run config's `args` or `user_args`):

   ```text
   --expert-whitelist my-whitelist.json
   --expert-whitelist-mode strict|substitute|tiered     (optional: override the manifest's mode)
   --draft-expert-whitelist mtp-whitelist.json|none     (optional: the MTP draft layer's own)
   --expert-whitelist-stats                             (optional: the counters at the end)
   --expert-whitelist-cache-gib N                       (optional, tiered: the on_demand cache, default 1 GiB)
   --expert-whitelist-substitute-threshold X            (optional, substitute: override the manifest's threshold)
   ```

   A draft layer's manifest uses the draft layer's fingerprint (`fingerprint --mtp`): one layer of 512 experts.

### 4.1 The Python profile: data for a real run

`data/expert_whitelist/profiles/python-coding.json` names a data set made for a real collection run, not only for
tests:

| Part | Files | Samples | Characters |
|---|---|---|---|
| Python (target) | `calibration/python/`: code generation (90), data work (40), debugging and tracebacks (55), refactoring and review (45), tests and packaging (45), agent sessions with tool calls (35), questions in German (40) | 350 | 1,832,751 |
| Python source files (target, task `source`) | the repository's own modules and tests: `strata_control`, `strata_cli`, `serve`, `packaging`, `tools` | 325 files | |
| Base requirements | `calibration/base/`: instructions in English (45) and German (45), chat structure with system prompts (40), tool calls (45), shell and build output (45) | 220 | 794,779 |
| Contrast (excluded) | `calibration/contrast/`: code in other languages (40), prose (20), other natural languages (20) | 80 | 271,205 |

That is about 0.8 million tokens of conversations and about 0.8 million of source files (cut at 4,096 tokens each),
1.6 million in all; `prepare` holds back 10% of the samples for `eval`. These token counts are a calculation (3.6
characters per token), not a count with the model's tokenizer, which was not on the machine this was written on.

**How the data was made.** Claude (an AI model) wrote the conversations for this purpose; they are not logs of real
users. A converter turned them into these files and wrote the tool lists and tool calls exactly as
`serve/chat_template.jinja` renders them: the "# Tools" block in the system message and
`<tool_call><function=...><parameter=...>` in the assistant's turn. Every file passes `check-profile` and `prepare`.
The standard-library code and its tracebacks were run on Python 3.12 for most files; code for other libraries (pandas,
Django, FastAPI, SQLAlchemy and others) was checked by reading only.

**Limits.**

- **No thinking text.** The assistant turns have an empty thinking block. When the model thinks before it answers,
  those thinking tokens route differently, and they are not in this data. With thinking on, use
  `--whitelist-mode tiered` or a higher coverage, and check the result with `eval`.
- **One project's code style.** The source files are Strata's own code. Add your own: `"sources": [...,
  "~/projects/**/*.py"]`.
- **The hardware.** `target_hardware` is 16 GiB VRAM (an RTX 4080 class card), 64 GiB RAM and 8 GiB reserve. Set the
  RAM to what the PC has.
- **Not run.** No collection with this data has run on a GPU yet: there is no figure for its run time, its
  statistics or the quality of a whitelist made from it.

A whole run, from the repository's root (`engine.json` as in section 7.1; the model's tokenizer needs the Python
package `regex`, its chat template `jinja2`):

```bash
python tools/whitelist_gen.py prepare --profile data/expert_whitelist/profiles/python-coding.json --model engine.json --out prep/python
python tools/whitelist_gen.py estimate --model engine.json --prep prep/python --vram-budget 14G --ram-budget 48G
python tools/whitelist_gen.py collect --model engine.json --prep prep/python --out stats/python-run1
python tools/whitelist_gen.py build --stats stats/python-run1 --profile data/expert_whitelist/profiles/python-coding.json --mode coverage --out whitelists/python.json
python tools/whitelist_gen.py inspect whitelists/python.json
python tools/whitelist_gen.py eval --model engine.json --whitelist whitelists/python.json --prep prep/python
```

## 5. Tests

No model and no GPU needed:

```bash
python -m unittest tools.test_expert_whitelist
```

`tests/core/expert_whitelist_test.cpp` (CMake target `expert_whitelist_test`, built without a GPU backend too; or
`g++ -std=c++20 -O1 -Iinclude tests/core/expert_whitelist_test.cpp src/core/expert_whitelist.cpp -lpthread`)
covers the manifest's checks and the fingerprint. It runs a synthetic mini-MoE (3 layers, 16 experts, top-2) through
a prompt, generation, verify windows and speculative decoding with a one-layer draft model. It checks:

- no whitelist, and an allow-all whitelist, give bitwise the baseline's tokens and logits;
- with a whitelist, only allowed experts are selected and run, in every path;
- a non-contiguous whitelist (experts 1, 5, 6, 14) on a compact store gives exactly the masked full model's output;
- the compact load reads only the allowed experts' bytes;
- the fix-up rule holds, and a blocked request is a hard error with its layer and expert;
- **substitute mode**: the table is made from what the full model routes on the test prompt, so that every rule
  happens: a blocked pick with a replacement, one whose replacement is already chosen, one whose replacement is below
  the threshold (0.3 against 0.6), one without a similarity (always used) and one without a replacement. For every
  routed token of a prompt and 14 generated tokens, the result equals the rule recomputed from the model's own top-k
  (the replacement and its weight bit for bit, added weights, the fallback), the weights keep their sum, and the store
  is never asked for a blocked expert (the reader read the resident tier only). The threshold from the command line
  and the new manifest checks are covered too;
- **tiered mode** through the engine's `OnDemandCache` with a budget of two experts: on_demand experts are read only
  when selected, and exactly in the order a separate least-recently-used count over the router's selections predicts;
  hits, misses, evictions and bytes equal that count (in the cache and in the run's counters); the budget holds
  throughout; a blocked expert is a hard error and never read; an expert picked in the current layer is not evicted (the
  next miss finds no room and is not read); a budget below one expert is refused; four threads asking at once read
  every expert once. The copies the prompt path makes (`copy`) give the right bytes, are held only while they are
  copied (the next miss in the same layer may evict them), never evict an entry whose pointer was handed out in this
  layer, and refuse a blocked expert; threads copying and taking pointers at once over a budget of three experts
  for nine get the right bytes.

Three deliberately broken copies of the code (weights not added, the most recently used expert evicted, the
threshold ignored) each make these tests fail.

The Python tests check the same fingerprint and the same validation messages. They also check the example profiles
and calibration set, and that `iq_pack.py index` changes no file of a pack (hashes before and after).

The generator has its own tests (section 7.8): `routing_stats_test` (C++) and `tools/test_whitelist_gen.py`.

## 6. Limits and open points

These are the engine side's. The generator's are in section 7.9.

- **Not built or run on a GPU here.** The machine this was written on has no CUDA or HIP toolkit. The host code and
  the tests were compiled and run with g++ 13. The CUDA/HIP files were only checked for syntax against a stand-in
  for the CUDA headers. Before review: build CUDA and HIP, run the parity and engine tests, and check that the default
  path is byte-identical.
- **No measurement yet:** no tokens/s, memory or quality figure for a whitelisted run. The quality report per
  language comes with the generator.
- Not on Metal, Intel (SYCL) or the Qwen3.6 engine: refused at start.
- The Foresight swap space (`STRATA_FS_SLOTS`) and `--shared-expert-arena` are refused with a whitelist.
- Paths that run entirely on the GPU (every routed expert resident, the device plan) keep to the whitelist by
  construction (mask and check in the routing kernels). They have no host step that could stop the run;
  `--expert-whitelist-stats` still counts what they selected.
- Without `--expert-profile` there is no VRAM cache fill and no RAM copy; allowed experts come from the mapped files.
- In tiered mode the prompt chunks route first and read only the routed experts, which may be slower than the streamed
  walk; an on_demand expert is read one at a time through the cache (not in the merged unbuffered reads of
  `copy_blobs`). A source that wraps the expert files (the helper GPUs' `RemoteExperts`) is not recognised as having a
  cache: its prompt path takes the pointer path, whose entries stay pinned until the next decode layer, and once the
  cache is full the rest is read from the files. No measurement of the cache's effect on tokens/s exists yet; the
  counters above are what a run reports.
- Where this differs from the task as first written: the pack format has no per-expert offsets, but they are
  computed exactly from the layout, so the sidecar index is optional and only checked. The supported models have no
  sigmoid routing, bias terms, expert groups, dense layers or per-layer expert counts. The format handles
  per-layer counts, and the runtime refuses a geometry it does not have.

## 7. The generator: from calibration data to a whitelist

`tools/whitelist_gen.py` builds a whitelist from what the model does on your data. It trains nothing and changes no
weight. It runs your samples through the model, counts which experts the router picks for which language, task and
part of the conversation, and keeps the experts that carry the languages you want.

### 7.1 The stages

Each stage reads files and writes files, and each one can run on its own:

| Stage | Reads | Writes | Runs the model? |
|---|---|---|---|
| `prepare` | profiles, calibration files, the tokenizer | a prep folder: `samples.bin`, `holdout.bin`, `prep.json` | no |
| `estimate` | the model's metadata (pack, GGUF), a prep folder | a plan: memory, scratch space, reads, the mode | no |
| `analyze` | prep folders (or statistics) | `analysis.json`, `analysis.md`, the sample's statistics | yes, a small sample (no with `--stats`) |
| `collect` | an engine config, a prep folder | a statistics folder | yes: the only stage that must |
| `merge` | statistics folders | one statistics folder | no |
| `build` | statistics, profiles | the manifest, `.report.json`, `.report.md`, `.coverage.csv` | no |
| `inspect` | a manifest | a readable summary | no |
| `eval` | an engine config, a manifest, a prep folder | an eval folder | yes |

An engine config is the JSON `tools/quant_quality.py` reads: `"exe"`, `"args"` (the engine's arguments, with
`--pack` and `--native`), and optionally `"env"`, `"lib_dirs"` and `"cwd"`. The stages that do not run the model never
load weights; they read the tokenizer, the model's metadata and what other stages wrote. Every output records what it
was made from: the SHA-256 of its inputs, the model fingerprint, the parameters and the generator's version.

A whole run, for Rust:

```bash
python tools/whitelist_gen.py prepare --profile data/expert_whitelist/profiles/rust-coding.json --model engine.json --out prep/rust
python tools/whitelist_gen.py analyze --model engine.json --prep prep/rust --profile data/expert_whitelist/profiles/rust-coding.json --target-hardware vram=16G,ram=64G,reserve=8G --gguf <model>-00001-of-00002.gguf --out analysis/rust
python tools/whitelist_gen.py estimate --model engine.json --prep prep/rust --vram-budget 14G --ram-budget 48G --scratch /nvme/strata-tmp
python tools/whitelist_gen.py collect --model engine.json --prep prep/rust --out stats/rust-run1
python tools/whitelist_gen.py build --stats stats/rust-run1 --profile data/expert_whitelist/profiles/rust-coding.json --mode coverage --coverage 0.95 --out whitelists/rust-c95.json
python tools/whitelist_gen.py inspect whitelists/rust-c95.json
python tools/whitelist_gen.py eval --model engine.json --whitelist whitelists/rust-c95.json --prep prep/rust
```

More runs add up with `merge --in stats/rust-run1 stats/rust-run2 stats/ref-run1 --out stats/rust-merged`. A sweep
writes one manifest per coverage: `build ... --sweep coverage=0.99,0.97,0.95,0.90,0.80 --out whitelists/rust-{value}.json`.
Two profiles combine as in section 2.1: `build --profile rust-coding.json --profile typescript-coding.json --mode budget
--target-hardware vram=16G,ram=64G --context 32768 --gguf <model>-00001-of-00002.gguf --out whitelists/rust-ts.json`.

Sizes are written as `16G`, `512M` or bytes; `G` means GiB.

### 7.2 Calibration data and `prepare`

A calibration file has one JSON object per line:

```json
{"id": "rust-0001", "language": "rust", "task": "compiler_errors", "prompt": "...", "response": "..."}
{"id": "rust-0002", "language": "rust", "task": "codegen", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

`prompt` (with an optional `response`) or `messages` in the chat form; `id`, `language` and `task` are optional. A
line without an id gets `<file name>-<line number>`. Where a sample comes from decides its data set:

- a target language's calibration files: data set `target`, language from the line or else the profile's name for it;
- a target language's `sources` globs: every matching file is one sample (the file as the prompt, task `source`);
- a base requirement's files: data set `base`, language and task the requirement's name (`instructions`, ...);
- an excluded domain's files: data set `excluded`;
- the profile's `reference_corpus` and every `--reference FILE`: data set `reference`.

`--only reference` (or `target`, `base`, `excluded`) prepares only some of them, so a reference set can be prepared and
collected once and merged with every profile's. A language or task the profile does not weight gets weight 0 in
`build`; `prepare` and `build` both list them.

What `prepare` does:

- **Tokens.** With the model's own tokenizer: the pack's `tokenizer/` folder or the GGUF (both need the Python package
  `regex`), taken from `--model` or given with `--tokenizer`. `--tokenizer bytes` is a stand-in for tests only: such
  samples cannot run through a real model.
- **The chat template and the regions.** `"chat_template"` in the profile, or `--chat-template`: `model` renders the
  model's own template (needs `jinja2` and `serve/frontend.py`; without them the samples use ChatML, and the warning
  and `prep.json` say why), `chatml` the built-in ChatML (it leaves out the model template's empty think block),
  `none` the plain text, or a `.jinja` file. The whole conversation is rendered once, as the model reads it, and cut
  into one run of tokens per message, so each token knows its region: `system`, `prompt` (a user message), `response`
  (an assistant message) or `tool`. A message's template tokens belong to its region. Where the template renders the
  end of a message differently once more follows (consecutive tool results share one turn and one `<|im_end|>`), the
  message ends after its own text and as much of its closing as the whole conversation has.
- **Length.** Samples longer than `limits.max_tokens_per_sample` (or `--max-tokens`) are cut, the start kept.
- **Duplicates.** Two samples with the same messages: the first is kept, the others are listed in `prep.json`.
- **The holdout split.** A sample is held out when the SHA-256 of its id (with the profile's seed) falls below the
  holdout share (`--holdout`, default 0.10). It depends on the id alone, so the split is the same on every machine and
  for every run. Held-out samples go to `holdout.bin` and never into `collect`. Their ids are in `prep.json`, travel
  with every statistics run, and end up in the manifest (`generation.holdout_ids`), so `eval` measures on them only.
- **The sample count.** At most `limits.max_samples` (or `--max-samples`) per data set and language; a fixed subset by
  the hash of the id.
- **Categories.** Every token's category is `dataset|language|task|region|origin`. The origin is `prefill`: the
  samples are read teacher-forced, prompt and reference answer alike.
- **The summary.** `prep.json` counts samples and tokens per data set, language, task and region, for the training
  samples and the holdout.

`samples.bin` (and `holdout.bin`), little-endian: `SXCALIB1`, u32 version (1), u32 category count, u32 sample count,
u32 0; per category a u32 length and the UTF-8 key; per sample a u32 id length and the id, a u32 token count n, n i32
token ids, n u16 category indices. `src/core/routing_stats.cpp` reads it.

### 7.3 The statistics

Per MoE layer, routed expert and category the collector counts:

| Value | Meaning |
|---|---|
| `count` | tokens for which the router put the expert in its top-k |
| `gate_sum`, `gate_sq_sum` | the sum and the sum of squares of the expert's final, normalised router weight over those tokens |
| `contrib_sum` | the sum of weight x \|\|expert output\|\| (0 where the path cannot give it, `has_contrib` says) |
| `rank_hist` | how often the expert was at rank 0, 1, ..., k-1 |

and per category the tokens that passed the layer (the denominator). Every value is an unsigned 64-bit integer; the
sums are fixed point (weight x 2^32, contribution x 2^20). Adding integers is exact, so statistics merged in any order,
split over any number of runs, interrupted and resumed, or collected layer by layer give the same bytes. No per-token
data is kept: the size depends on the layers, experts and categories, not on the number of tokens. For Qwen3.8-Flash-
Next with 30 categories it is 8 x 30 x (1 + 512 x 14) B = 1.7 MB per layer, 83 MB for 48 layers (a calculation).

A statistics folder holds `meta.json` and one `layer_NNN.bin` per MoE layer. A layer file is a 64-byte header, then
`tokens[C]`, then `values[C][E][4 + k]` in the order count, gate_sum, gate_sq_sum, contrib_sum, rank_hist[0..k-1], all
little-endian u64:

| Bytes | Header field |
|---|---|
| 0-7 | `SXRSTAT1` |
| 8-11 | format version (1) |
| 12-15 | layer |
| 16-19 | experts E |
| 20-23 | top-k |
| 24-27 | categories C |
| 28-31 | flags (1: has contrib) |
| 32-35, 36-39 | the fixed-point bits (32, 20) |
| 40-47 | blocks done (a layer-wise run's progress; 0 in a finished folder) |
| 48-55 | FNV-1a 64 of the model fingerprint |
| 56-63 | FNV-1a 64 of the category keys (one per line) |

`meta.json` records the format, the model fingerprint and its inputs (layers, experts per layer, top-k, width, the
pack's expert index, the MoE layers), the non-expert bytes when the engine knows them, the weights' quantization, the
categories, each layer file's size and CRC-32, the tokens and samples, the run's id and mode, the date, the generator,
the run's parameters, the caller's description of the run (`collect`: the prep folder's hashes and holdout ids, the
profiles, the tokenizer, the engine command) and the ids of the samples it counted. A reader checks all of it: a
damaged or edited layer file, or a folder from another model, is refused with a message.

**The experts' similarity (optional, for the substitute mode).** For a share of the tokens (default 1%, chosen by
FNV-1a 64 of "seed:sample id:position", so every path and block layout picks the same tokens) the collector runs every
expert of the layer on the token's real expert input. For each expert i the router picked (w: its final router
weight) and every expert j of the layer it adds w x (1 + cos(y_i, y_j)) and w x min(\|y_i - y_j\| / \|y_i\|, 4), and
per expert i its weight and its count of picks. The means are the similarity of j's output to i's, weighted by how
much the router gave i: what a replacement of i by j takes over. Like the statistics, every term is a fixed-point
integer (x 2^32) rounded on its own, so the sums merge exactly. Inference only; no weight changes.

`similarity_NNN.bin` per MoE layer: a 64-byte header (`SXESIM01`, version 1, layer, experts E, the fixed-point bits 32,
the distance cap x 1000, 0, the sampled tokens, the blocks done, FNV-1a 64 of the model fingerprint, FNV-1a 64 of the
sampling's text), then row_w[E], row_n[E], cos[E][E], rel[E][E], all u64. That is 2 x 512 x 512 x 8 B = 4 MB per layer
of Qwen3.8-Flash-Next, 200 MB for 48 layers (a calculation). `meta.json` then has a `similarity` section: the share, the
seed, the hash, the sampled tokens and each file's size and CRC-32. A file of another model or another sampling is
refused.

### 7.4 Collecting: token mode and layer-wise mode

`collect` runs the engine (`strata generate`) with these options, then reads the folder back:

```text
--routing-stats-out DIR             the statistics folder (the run reads its samples, writes the folder and exits)
--routing-stats-samples FILE        a prep folder's samples.bin
--routing-stats-mode token|layerwise
--routing-stats-meta FILE           a JSON object copied into meta.json ("collect")
--routing-stats-resume              continue an interrupted run in DIR
--routing-stats-ram-budget S        the collector's own buffers (the accumulators, one sample's routing, a block's rows)
--routing-stats-vram-budget S       layer-wise, the scheduler's split form only (the engine places its own buffers)
--routing-stats-prefetch            layer-wise, the split form only: read the next layer ahead
--routing-stats-block-tokens N      layer-wise: tokens per block
--routing-stats-scratch DIR         layer-wise: where the rows between layers go (default DIR/scratch)
--routing-stats-scratch-max S       layer-wise: what the scratch files may take at once
--routing-stats-spill f32|f16|bf16  layer-wise: their precision (default f32)
--routing-stats-checkpoint-every N  token mode: samples between checkpoints (default 64)
--routing-stats-similarity F        also the experts' similarity data on a share F of the tokens (section 7.3)
--routing-stats-similarity-seed N   which tokens that share samples (default 1)
```

**What is recorded.** The router's final ids and weights of every position of every MoE layer, after the top-k and its
normalisation, on the batched prompt path. Every sample starts from an empty sequence at position 0 and is read
teacher-forced, prompt and reference answer together. The model's routing is not changed. Without
`--routing-stats-out` nothing of this runs: the prompt path checks one empty callback per layer and chunk, and
nothing is copied or waited for. The options are refused together with `--expert-whitelist` (statistics are collected
on the full model), `--serve`, `--tokens` and `STRATA_DBG_ROUTE_OVERRIDE`, on Metal and for Qwen3.6 models, and they
need the batched prompt path (`--prefill`).

**The similarity data** (section 7.3) is collected by the collector library (`src/core/expert_similarity.cpp`,
`RoutingStatsSchema::similarity`): in token mode the path reports every expert's output at each sampled position
(`RoutingSink::expert_outputs`), the sink keeps the sample's contributions until the sample is done, and the checkpoint
holds the similarity accumulators too; in layer-wise mode the split scheduler loads every expert of a layer once in a
block that samples a token (it is loaded anyway), runs it on the routed tokens and on the sampled ones, and writes
the similarity file after the statistics (a crash between the two is resumed). A run whose path gives no expert
outputs for a sampled position stops with "this path does not collect similarity data".

**The engine** (`strata generate --routing-stats-similarity F`, which `collect --similarity F` passes on; the seed with
`--routing-stats-similarity-seed`, `collect --similarity-seed`): its prompt path computes only the routed experts. So
after a layer's routing it downloads the expert input (`m.mixed`, what the router and the experts read) of the rows the
collector samples (`Prefill::moe_rows_wanted`, `on_moe_rows`), and every expert of the layer is computed on them on the
CPU (`expert_outputs_cpu`, which calls `all_expert_outputs` in `include/strata/kernels/cpu/pool.hpp`: the verify
window's kernels and pool, up to 8 rows and 96 experts a batch, in a pool of its own so it never shares one with the
prompt path's CPU share). `ChunkOutputFeeder` hands the outputs to the collector,
in token mode and in layer-wise mode. The CPU's outputs are its quantized arithmetic, close to the GPU's but not the
same bits: good enough for a similarity, which is a mean over many tokens. The cost: per sampled token every expert of
every MoE layer once on the CPU. For Qwen3.8-Flash-Next (512 experts, top-10) and 1% of the tokens that is about five
experts' work per token and layer on average, half of what the ten routed experts take (a calculation; no time has
been measured).

**Token mode** runs each sample through all layers, as the model runs. Every `--routing-stats-checkpoint-every`
samples it writes all accumulators and the done sample ids to one checkpoint file (written under a temporary name and
renamed). `--resume` continues after the last checkpoint and gives the same bytes as a run without a stop. A sample
that fails half way adds nothing.

**Layer-wise mode** is for a model that does not fit in RAM and VRAM. Token by token, such a model would read every
routed expert from the drive again for every token. The scheduler (`src/core/layerwise_calib.cpp`) turns the loops
around. The samples are cut into blocks (`--block-tokens`, or the most the RAM budget holds, or 16384 tokens when
nothing limits it; a sample is never split). For each block and each layer:

1. load the layer's dense weights, once per block;
2. per sample: the part before the experts (attention or recurrence, with this layer's KV cache only, dropped after the
   sample), reading the sample's input rows from the scratch file of the layer boundary (layer 0: the embeddings);
3. route every token of the block;
4. per expert that got at least one token, in id order: load it once and run it on all of its tokens. An expert no
   token of the block routed to is never loaded;
5. per sample: the part after the experts, writing the output rows to the next boundary's scratch file;
6. release the layer and add the block's routing to the layer's statistics.

At most two boundaries exist at a time, the input and the output of the current layer, and they are read and written
one sample at a time. F32 is the engine's residual precision, so a layer-wise run routes exactly as the token path
does; F16 and BF16 halve the scratch files and may move a few routing decisions. The scheduler refuses to start when
the budgets do not hold a layer with one expert and the longest sample (VRAM), two layer accumulators and one sample's
rows (RAM), or the scratch files (size and free space), and says which number is short. With `--prefetch` (split form)
the next layer is read from the drive into RAM while the current one computes, when the RAM budget holds it.

**Resuming.** After each layer the scheduler writes, in this order: the output boundary (renamed into place), the
layer's statistics (with the count of blocks they hold in the header), a progress record (`progress.bin`: block and
layer), and only then deletes the input boundary. A crash at any point leaves enough to continue at that block and
layer: a layer whose statistics already hold the block runs again only for its output rows. When the input it needs
is gone (scratch files on another drive whose last rename a power loss undid, or a tmpfs after a reboot), the block
starts again at layer 0; the layers that already counted it do not count it twice. A resumed run must read the same
samples with the same blocks; otherwise it is refused.

**The scratch folder.** The boundary files go into a folder of their own inside the scratch folder
(`strata-layerwise-<id>`, the id from the plan and the output folder), so two runs never share or read each other's
files. Only files with the boundaries' own names (`block_N_layer_N.rows`) are ever removed, and after a complete run
that folder; everything else in a scratch folder you name stays. The statistics stay too.

**The engine in layer-wise mode.** `Prefill::run_layer` runs one layer over one sample on the prompt path: the stage
machinery of a multi-GPU layer split with a range of one layer, the rows handed in from the boundary file and the
output rows handed back. The scheduler then orders the work, keeps the boundaries and resumes; the engine handles its
experts as its prompt path always does. What that does and does not give: the engine still loads all its dense weights
into VRAM at start, and a model that does not fit cannot be loaded at all; its prompt path stages the experts each
sample's chunk routes to, per sample, not once per block. The full layer-wise expert grouping (step 4) is the
scheduler's "split" form, implemented and tested on the synthetic model (section 7.8); a backend that loads one layer
at a time plugs in there. For the models this engine runs, which fit in 16 GB of VRAM and 64 GB of RAM, token mode is
the mode to use, and `estimate` recommends it.

**`estimate`** computes from the model's metadata and the budgets alone: the non-expert and expert bytes (from the
GGUF's tensors, or the pack's `dense.bin` spread over the layers; the PLE table `per_layer_token_embd.weight` is left
out, because the engine reads its rows from the drive), whether token mode fits (non-expert weights in VRAM,
everything in VRAM plus RAM) and how much it would read from the drive if not, and the layer-wise plan: blocks, RAM,
VRAM and scratch need, and the bytes read (the dense weights once per block, the experts a uniform router would touch,
and the bound of every expert once per block). Times assume a drive speed (`--nvme-gbps`, default 2.0 GB/s) and a
prompt speed only when you give a measured one (`--prefill-tps`): they are estimates, not measurements. It exits with 1
and names the problem when the budgets do not suffice, and recommends `--mode token` or `--mode layerwise`. `collect`
prints the same estimate before it starts the engine, stops when a layer-wise run does not fit, warns when token mode
would read experts from the drive per token, and with `--mode auto` (the default) takes the recommendation.

The weights' quantization is recorded in `meta.json` (the GGUF the pack's experts come from, or the canonical Q2_0
pack) and copied into the manifest: routing on a quantized model can differ a little from the full-precision one.

### 7.5 Merge

`merge` adds statistics element by element. It checks that every input has the same model fingerprint, model section
and quantization, and refuses a run that is in two inputs (by its id) and samples that two inputs both counted (their
tokens would count twice; `--allow-overlap` merges anyway). The categories are united and sorted. The result does not
depend on the order of the inputs or on how merges of merges are nested, byte for byte; it lists the collection runs it
holds, each with its parameters and inputs. Similarity data is added up over the inputs that have it; the `similarity`
section lists the runs it came from and the runs without it.

### 7.6 Build: scoring and selection

**Components.** Each target language of the profile and each base requirement is one component k. Every other
category is reference: excluded domains, the reference corpus, and samples in languages the profile does not target.

**Weights.** A category's weight is its language's weight, its task's weight (0 for a task the profile does not list;
every task 1 when the profile lists none) and its region's weight (`--region-weights`, default response 1.0, prompt
0.5, system 0.5, tool 0.5). Base requirements weigh `--base-weight` (default 1).

**Scores**, per layer l and expert e (g: the final router weight, 0 when not chosen):

1. The profile mass `M_p(l,e) = sum over tokens of w_lang * w_task * w_region * g`, normalised per layer to `m_p`.
   Per component the same over its own categories: `m_k`. `--use-contrib` takes weight x \|\|output\|\| instead of g
   where the statistics have it.
2. The reference mass `m_r` the same way over the reference categories.
3. The contrast `c = log((m + alpha) / (m_r + alpha))`, `alpha = 1 / (experts x 100)` unless `--alpha`.
4. The score `s = m * exp(lambda * clip(c, -C, C))`, `lambda` 0.5 (`--lambda`; 0 ranks by mass alone), C 3 (`--clip`).
5. The protected set: the experts that carry `--protect-mass` (0.8) of a component's mass, largest first. They are
   always kept.
6. Uncertain experts: fewer than `--n-min` (20) observations in a component. They are left out unless protected
   (`--include-uncertain` keeps them); the report lists them per layer.

Without reference data the contrast is off (the report says so) and the experts are ranked by mass.

**Selection**, per layer, experts in score order, ties to the lower id:

- `coverage`: add experts until the component's mass `m_k` reaches its target: `--coverage` for the target languages
  (default: each one's `min_coverage`), each base requirement's own `min_coverage`. Coverage counts mass, not score, so
  0.95 means 95% of the component's routing weight. The layer keeps the union of all components' experts, so a
  language with little data is not drowned by another's mass.
- `count N`, `fraction F`: N (or F x experts) experts per layer by the weighted mean of the components' scores.
- `budget B`: B experts (a number) or bytes (`16G`) over all layers. Every layer first gets its floor and its protected
  experts, then every component's `min_coverage`, then the rest of the budget goes to the expert with the highest
  marginal utility `sum_k w_k * m_k(l,e) * layer_weight(l) / bytes(l,e)`, one at a time. When the budget does not hold
  the minimum coverages, `build` stops and says by how much; `--allow-under-coverage` builds the best whitelist the
  budget holds and reports the shortfall per component and layer. `layer_weight` is 1 unless `--layer-weights` gives a
  file (for an ablation's result, section 7.9).

Every layer keeps at least top-k + `--margin` (2) experts. With group-limited routing (`--expert-groups`,
`--group-topk`) a group the router may pick needs ceil(top-k / group-topk) allowed experts, or the whole group is
blocked and the report says so; none of the supported models has expert groups.

**The budget from the hardware** (`--target-hardware vram=16G,ram=64G,reserve=8G[,buffer=2G]`, or the profile's
`target_hardware`):

    experts' budget = VRAM + RAM - reserve - non-expert weights - KV cache(--context) - runtime buffer

The non-expert weights are what the engine recorded in the statistics (its dense weight arena; 0 for a native-dense
run, then `--dense-bytes` or `--gguf` must say; from the GGUF without the PLE table, which the engine reads from the
drive). The KV cache is computed from the GGUF's metadata (`--gguf`): FP16 K
and V of the full-attention layers per token of context, plus the recurrent layers' state per sequence; or
`--kv-bytes-per-token` and `--state-bytes`. The runtime buffer is an assumption (2 GiB, `--runtime-buffer`). The
report shows the whole calculation and marks it as one. How the allowed experts spread over VRAM and RAM is the
engine's own placement; the whitelist only says which experts exist. When the non-expert weights and the floor do not
fit, `build` stops.

**The mode** (`--whitelist-mode`, default `strict`) is written into the manifest:

- `tiered`: the selection above is the resident tier. Then per layer and component, experts in score order past it
  until the component's mass reaches `--on-demand-coverage` (default 0.995); their union is the `on_demand` tier
  (uncertain experts too: they stay on the drive until picked). In budget mode the budget less the engine's
  on_demand cache (`--on-demand-cache`, default 1G) holds the resident tier. The report gives per layer the on_demand
  experts, their bytes, the coverage resident and with on_demand, and the bytes per token if every on_demand pick
  missed the cache (an upper bound).
- `substitute`: needs similarity data in the statistics. For every blocked expert with at least
  `--substitute-min-tokens` (20) sampled picks, the allowed expert whose output is most like its own (the highest mean
  cosine; a tie to the smaller distance, then the lower id) goes into the manifest with its similarity and distance,
  and `--substitute-threshold` (default 0.5) becomes its `substitute_threshold`. All of them are listed; the engine
  uses those at or above the threshold, so a lower threshold on the command line can use more. The report gives per
  layer the blocked experts, the ones listed, at the threshold and without data, and the share of each component's
  blocked mass a replacement in use takes.

**The manifest** is the engine's format (section 2.2). Its `generation` section holds the profile names and hashes, the
scoring method (`profile-mass-contrast-v1`) with every parameter including the defaults, the calibration tokens, the
statistics' run id and hash, the quantization, the holdout ids and the hardware calculation. The same statistics and
parameters give the same manifest, byte for byte. `build` checks it with the engine's rules before it writes it.

**The report** (`<name>.report.json` and `.report.md`): per component the coverage per layer and on average and every
layer below its `min_coverage`; the hardware calculation and whether the whitelist fits; per layer the allowed
experts, the profile mass and reference mass they cover, the uncertain experts and their bytes; the expert bytes
before and after (calculated from the pack index, not measured); warnings for nearly flat layers (normalised entropy
above 0.9: a whitelist saves little there), components with fewer than `--min-tokens` (2000) tokens, unknown tasks and
untargeted languages. `<name>.coverage.csv` has the coverage curve: per layer, the profile mass (and each component's)
of the best n experts, n = 1 ... experts.

### 7.7 analyze, inspect, eval

**`analyze`** measures, before the large collection, whether the model's routing can be specialised at all: experts
may specialise by kind of token rather than by topic, and that is not known beforehand. It collects up to `--sample`
tokens (default 4000) of each `--prep` folder, in two disjoint halves (by the hash of the sample id, collected apart and
merged into `stats/`), or reads `--stats` (once, or twice for two halves you collected). It reports numbers only, no
verdict:

- per layer and component (each target language, each base requirement, and the reference corpus): how many experts
  carry 80, 90, 95 and 99% of its routing mass, the normalised entropy and the effective number of experts (e^H); per
  component the means and the largest layer;
- the overlap of the components' expert sets, per pair, per layer and on average: the Jaccard index of their 95% sets,
  the mass overlap (the sum over the experts of the smaller of the two masses: 1 for the same routing, 0 for disjoint)
  and how much of each one's mass the other's 95% set covers;
- stability: per layer and component, the same numbers between the two halves, and the cross coverage (the mass of one
  half that the other half's 95% set covers, the smaller of the two). Layer and component pairs below
  `--stability-coverage` (0.90) are listed: a large difference between the halves means the data does not suffice;
- the projection against the experts' budget (`--target-hardware`, or `--budget`), per mode: `strict` (the floor of
  top-k + `--margin` per layer, then the most mass per byte; the coverage per component and the mass that falls on
  blocked experts), `substitute` (the same resident set, and with similarity data the share of the blocked mass a
  replacement at or above `--substitute-threshold` would take) and `tiered` (the budget less `--on-demand-cache`
  resident, the experts that reach `--on-demand-coverage` on_demand; the mass on blocked and on on_demand experts, and
  the bytes read per token if every on_demand pick missed the cache, an upper bound);
- the nearly flat layers (normalised entropy of the profile's routing above `--flat-entropy`, 0.9) and their share of
  the budget (of the strict projection's resident experts).

All bytes and the projection are calculations from the sample, not measurements. It is a feasibility check, not the
whitelist.

**`inspect`** prints a manifest: the model, the mode, the allowed experts in total and per layer, the parameters, and
with the report beside it the coverage per component and the warnings.

**`eval`** reads the held-out samples teacher-forced twice, with the full model and with the whitelist, through
`tools/quant_quality.py`'s engine run (`STRATA_LOGPOS`; the engine config needs `--expert-profile`, as there). It
compares them per language, per language and region, and per task: top-1 agreement, KL against the full model and the
perplexity of the text under each. Its numbers are measurements of that run, on that machine.

### 7.8 Tests

No model and no GPU needed:

```bash
python -m unittest tools.test_whitelist_gen
```

`tests/core/routing_stats_test.cpp` (CMake target `routing_stats_test`; or
`g++ -std=c++20 -O1 -ffp-contract=off -Iinclude tests/core/routing_stats_test.cpp src/core/routing_stats.cpp src/core/layerwise_calib.cpp src/core/expert_whitelist.cpp src/core/expert_similarity.cpp -lpthread`)
runs a synthetic mini-MoE: 4 layers (layer 2 dense), 16 experts, top-2, width 8, a causal attention with a KV cache per
layer, and a router that sends tokens 0-15 ("rust") to experts 0-3, 16-31 ("python") to 4-7 and 32-47 (prose) to 8-11.
The weights sit in a "drive" image whose reads are counted. It checks:

1. statistics: merges are exact (commutative, associative, two half runs merged = the whole run); token mode stopped
   after its 14th sample and resumed writes the same bytes as an uninterrupted run; the RAM frame is kept, and one byte less
   is refused before anything is written;
2. the language experts: 95% of the rust tokens' router weight sits on experts 0-3 in every MoE layer;
3. the layer-wise mode: with budgets that force 4 blocks of up to 367 tokens and batches of 60 rows, its statistics
   equal the token path's byte for byte; the VRAM and RAM peaks stay within the budgets (5232 of 5232 B, 57280 of
   57344 B); every layer is loaded once per block and read from the drive once per block; every expert is loaded at
   most once per block and layer, exactly the experts the block routed to; at most two boundary files exist at once;
   budgets that are too small are refused with their numbers; after a simulated abort at each of five points of a
   layer (after the routing, half way through the experts, after the boundary, after the statistics, after the
   progress record) in four places, the resumed run continues at that layer, not at layer 0, and gives the same bytes;
   a lost boundary file makes the block start again at layer 0 and still gives the same bytes; a scratch folder the
   user names keeps its other files, and one that is the output folder keeps the statistics; a failing read-ahead
   stops the run with its reason; the fused form (the engine's shape), the read-ahead and F16 / BF16 scratch files
   work too. The test also runs clean under AddressSanitizer and UndefinedBehaviorSanitizer.
4. the similarity data, on the same model with two pairs of twins in every MoE layer (expert 4 a copy of expert 0,
   expert 5 expert 1 with its output halved) and a quarter of the tokens sampled: the twins come out with a mean
   cosine of 1 and distances of 0 (4 to 0), 1 (5 to 1) and 0.5 (1 to 5); unrelated experts do not; the statistics are
   the same bytes as without the similarity data; two halves merged are the whole run; token mode resumed after an
   abort, the layer-wise split form (every expert loaded once in a block that samples a token, the RAM budget kept),
   the split form resumed after an abort between the statistics and the similarity file (and after it, and mid-layer)
   and the fused form all write the same files as token mode, byte for byte; a RAM budget that does not hold the
   sampled tokens' outputs is refused up front with its reason; a resume with another share or seed and a path that
   gives no expert outputs are refused. It writes or compares a second fixture, `stats-mini-sim`.
5. the engine's shape: a sample in chunks (1, 5 and 64 positions), each chunk layer by layer with the routed experts
   only, the sampled rows' expert input handed to `ChunkOutputFeeder` after the chunk's routing and every expert
   computed on them apart (as the engine does on the CPU): the statistics and similarity files equal token mode's,
   byte for byte, and only the sampled rows are computed, once per layer; a failed computation and expert input without
   its routing stop the run with the reason.

`src/kernels/cpu/all_expert_outputs_test.cpp` (CMake target `all_expert_outputs_test`, linked with the CPU kernels and
ggml) runs the engine's CPU computation of the similarity data, `all_expert_outputs`, on 100 synthetic native Q4_0
experts (each with its own weights) and 11 rows, so the groups of 96 experts and 8 rows both split: every output is
the single-token sequence's for that expert and row (native_quant_act, native_gu_rows, native_quant_h,
native_down_rows), bit for bit and in its place; an expert without a blob is refused by its number. With AVX-512 it
checks the canonical Q2_0 pack's kernels the same way. On the machine this was written on (an i5-3470, AVX without
AVX2, no cmake) it was built by hand against llama.cpp 3cf03257's ggml, the commit CMakeLists.txt pins; the native half
passed (1100 outputs bit for bit) and three deliberately broken copies (rows and experts swapped, the second group of
experts given the first group's blobs, every row given the first row's input) each failed it. The existing
`pool_tasks_test` passed in the same build.

In `routing_stats_test` a one-ulp change in one scratch value makes the byte comparison fail, so the check is not a
loose one. It also writes (`--write-fixture DIR`) or compares the fixture `tests/data/expert_whitelist/stats-mini`.
The Python test reads that fixture and checks that `build` finds exactly the language experts (rust: 0-3 in every MoE layer; rust and
python: 0-7; rust with the prose base requirement: 0-3 and 8-11), that build is deterministic, that its manifest passes
the engine's checks, the modes, the floor, the uncertain rule, the budget and hardware calculation, the sweep and the
report; that merge is commutative and associative byte for byte and refuses other models, the same run twice and
overlapping samples; that prepare marks regions, drops duplicates, splits the holdout deterministically and keeps it
out of the training samples; that the Python and C++ readers and writers of the binary formats agree byte for byte;
that the Python plan in `estimate` equals the C++ scheduler's; the engine command `collect` builds; and the grouping
of `eval`. For this part of the generator it also checks:

- `analyze` on synthetic statistics with a known answer (two layers of 16 experts: in layer 0 rust on experts 0-3 with
  0.4, 0.3, 0.2, 0.1 of its mass and python on 4-7; in layer 1 every expert alike): the experts for 80/90/95/99%
  (3, 3, 4, 4 and 13, 15, 16, 16), the entropy and effective experts, the overlaps (0 between the languages in layer 0,
  1 in layer 1, a quarter against the uniform reference), the flat layer and its share of the budget, the projection
  of every mode down to the resident experts, coverages, blocked and on_demand masses and bytes per token, the
  substitute share from synthetic similarity data, and the stability numbers for identical halves and for halves that
  disagree (rust on other experts in one half: Jaccard 0, cross coverage 0, listed);
- `build --whitelist-mode tiered`: the resident tier is the strict selection, the on_demand tier reaches the second
  coverage in every layer and the manifest passes the engine's checks; in budget mode the cache comes out of the budget;
- `build --whitelist-mode substitute` on the similarity fixture: the twins are found (4 to 0 and 5 to 1, similarity 1),
  every listed replacement is the most similar allowed expert, the engine's table uses exactly those at or above the
  threshold; statistics without similarity data are refused;
- the similarity files: the C++ fixture is read, the Python writer gives the same bytes, a damaged file or another
  sampling is refused, and merge adds the data commutatively.

`tools/test_expert_whitelist.py` checks the manifest's new fields against the engine's rules, with a third fixture
(`mini-substitute.json`) that the C++ test reads too.

### 7.9 Limits and open points

- **Not built or run on a GPU here, and no measurement.** The machine this was written on has no CUDA or HIP toolkit.
  The scheduler, the statistics and both tests were built and run with g++ 13 and Python 3.12. The engine's parts
  (the prompt path's routing hook, `Prefill::run_layer`, the `--routing-stats-*` options) were only checked for syntax,
  through the Metal build's CUDA stand-in (`metal/include/metal_compat`). No collection has run on a real model: there
  is no speed, memory or quality figure for it, and none for a generated whitelist. Before review: build CUDA and HIP,
  collect on a small prep in both modes and compare their statistics (they should agree within the existing tolerance;
  with F32 scratch files and the same chunking they should be the same), and check the default path is byte-identical.
- **Models larger than RAM + VRAM.** The engine can only load the models it supports, which fit the reference hardware.
  The layer-wise schedule with one load per layer and block and the experts grouped across the block is implemented and
  tested in the scheduler's split form; the engine uses the fused form (section 7.4). A loader that places one layer
  at a time is the open piece for a model that does not fit.
- **Not from the engine:** weight x \|\|output\|\| (the prompt path's grouped kernels give no per-pick output norm, so
  `has_contrib` is false and `--use-contrib` is refused on such statistics), the routing of generated tokens (the
  `generated` origin exists in the format; collection reads teacher-forced only).
- **The engine's similarity data was not run end to end.** The download of the sampled rows (prefill.cpp) and the
  wiring in generate.cpp were only checked for syntax. Its two ends are tested: the CPU computation
  (`all_expert_outputs_test`, with the real kernels: native Q4_0 experts bit for bit against each expert alone; the
  canonical Q2_0 pack's kernels need AVX-512, which the machine this was written on lacks, so that half was not run)
  and the collector's side (`ChunkOutputFeeder`: the engine's shape in miniature writes the same files as token mode,
  byte for byte).
- **Not implemented:** the ablation check (logit KL of boundary candidates on the holdout); `--layer-weights` takes its
  result when one exists.
- **Calculations, not measurements:** the hardware budget (the KV cache from the GGUF's metadata assumes FP16 K/V;
  `--kv q8` or `q4` take less), the runtime buffer (2 GiB assumed), the bytes before and after, everything `estimate`
  prints.
- The Intel (SYCL) build compiles the same `generate.cpp`; its build file now also compiles the whitelist and
  statistics sources and a stub for the whitelist's routing entry points (`sycl/src/kernels/expert_whitelist_stub.cpp`,
  which refuses a whitelist at start, as Metal does). It was not built here.
