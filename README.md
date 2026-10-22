<p align="center">
 <h1 align="center">StrataX</h1>
</p>

<p align="center"><b>English</b> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a></p>

<h3 align="center">Run a state-of-the-art model on your own gaming PC</h3>

<p align="center">Support for NVIDIA, AMD, Intel · Windows, Linux, Mac · free and open source</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?include_prereleases&label=release" alt="Latest release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/Niko1221/Strata/stargazers"><img src="https://img.shields.io/github/stars/Niko1221/Strata?style=flat" alt="GitHub stars"></a>
</p>

## New: expert whitelists (experimental)

This version of Strata can run the model with **only the experts your work needs**. You make a list of them once,
from examples of your own kind of work, and the engine then leaves all other experts out. That saves memory. The
feature is opt-in: without the new options nothing changes.

**On this page:** [what it is](#what-is-an-expert-whitelist) · [what is new](#what-is-new) ·
[status](#status-what-works-and-what-is-not-measured-yet) · [first results](#first-results-one-run) ·
[step by step](#step-by-step-make-and-use-a-whitelist) · [problems](#when-something-goes-wrong-with-a-whitelist) ·
all details in [docs/EXPERT_WHITELIST.md](docs/EXPERT_WHITELIST.md)

### What is an expert whitelist?

The model Strata runs is a *mixture of experts*. Each of its 48 layers has 512 small sub-networks, the **experts**.
For every piece of a word (a *token*) a small **router** picks the 10 experts that fit best. Only those 10 compute,
which is why such a big model can run on a PC at all. But over a long conversation nearly every expert is picked at
some point, so all of them must be at hand: about 47 GB for the IQ3_S model, in RAM and graphics memory.

Not every task needs every expert. A model that only writes Python may not need the experts the router picks for
poems, for French or for Java. An **expert whitelist** is a list of the experts the engine may use. The engine then:

- never picks an expert that is not on the list, and
- never loads it into RAM or graphics memory.

Nothing is trained and no weight is changed. The model only chooses from fewer experts.

**What you can gain:** less memory for the experts, so more of the ones you need fit into graphics memory or RAM.
**What it costs:** answers outside the whitelist's topic get worse, maybe much worse. Answers on the topic can
change a little. How much is measured with `eval` (step 9 below), not guessed.

The list is not written by hand. A **generator** (`tools/whitelist_gen.py`) runs your example conversations through
the model, counts which experts the router picks for which language and task, and keeps the experts that carry your
work.

What happens when the router would like an expert that is not on the list depends on the **mode**:

| Mode | What happens to an expert that is not on the list | Good for |
|---|---|---|
| `strict` (default) | It is never used; the router chooses among the allowed experts. | The smallest memory, the largest quality risk. |
| `tiered` | A second list (`on_demand`) is read from the drive only when the router picks one, and kept in a small RAM cache. | Less memory with quality close to the full model. Rare experts cost a drive read. |
| `substitute` | A similar allowed expert takes over (similarity measured during the collection). | Experiments; needs `collect --similarity`. |

### What is new

- **The engine** (`strata`) reads a whitelist: `--expert-whitelist FILE`, `--expert-whitelist-mode
  strict|substitute|tiered`, `--draft-expert-whitelist` (the MTP draft layer), `--expert-whitelist-stats` (counters
  at the end), `--expert-whitelist-cache-gib` (tiered: the RAM cache, default 1 GiB) and
  `--expert-whitelist-substitute-threshold`. Every path keeps to it: decode, speculative checks, the prompt path, the
  expert caches and the look-ahead.
- **The engine records routing statistics** (`--routing-stats-*`): it reads prepared calibration samples and counts,
  per layer, expert and category, how often and how strongly each expert was picked. Token mode, a layer-wise mode for
  models larger than RAM and VRAM, and optional expert similarity data.
- **The generator** `tools/whitelist_gen.py` with the stages `prepare`, `analyze`, `estimate`, `collect`, `merge`,
  `build`, `inspect` and `eval`.
- **A checker** `tools/expert_whitelist.py`: `check-profile`, `fingerprint` and `validate`.
- **A ready Python profile** with data for a real run (`data/expert_whitelist/profiles/python-coding.json`): 650
  example conversations (code, debugging, tests, data work, agent sessions with tool calls, German questions, plus
  general abilities every model must keep) and the repository's own Python files.

### Status: what works and what is not measured yet

- **Experimental and opt-in.** The ready-made installers and engines do not contain it. Build the engine from this
  source tree (step 1).
- **Supported:** the Qwen3.8-Flash-Next family on NVIDIA cards (CUDA). The HIP (AMD) build has the code but was not
  built or run. Mac (Metal), Intel (SYCL) and the Qwen3.6-35B-A3B engine refuse a whitelist at start.
- **What has run:** the engine was built for an RTX 4080 (sm_89, CUDA 13.0), `collect` ran on the IQ3_S model, and
  `build` made whitelists from it (next section).
- **Not measured yet:** the quality of a whitelisted model (`eval`), its speed, and the `tiered` mode's drive reads on
  a GPU. Until those numbers exist, treat every whitelist as an experiment.
- **The Python data was written by an AI model (Claude) for this purpose.** It is not taken from real users. It
  contains no thinking text, so the tokens the model writes while it thinks are not part of the calibration.

### First results (one run)

Measured on: Qwen3.8-Flash-Next IQ3_S, one RTX 4080 (16 GB), Linux, engine 0.1.46.1 built from this tree, the
`python-coding` profile with a control vector active (the server's own configuration). `collect` read 872 samples,
1,496,785 tokens, in token mode. The counts and sizes below are what `build` computed from that run. They are not
speed or quality measurements.

| Whitelist | Experts kept (resident) | Expert memory | Saved |
|---|---|---|---|
| none (full model) | 24,576 | 46.84 GiB | |
| `strict`, Python 95 %, general abilities 90-99 % | 21,195 | 40.25 GiB | 6.6 GiB (14 %) |
| `tiered`, Python 90 % resident, up to 99.5 % from the drive, general abilities 85-95 % | 18,108 | 34.33 GiB | 12.5 GiB (27 %) |

**What this shows:** for Python work, this model spreads its routing over almost all experts. In most layers (at least 34 of 48) the
spread is nearly even (normalised entropy 0.90 to 0.95, where 1.0 is perfectly even). So a whitelist that keeps
95 % of the routing keeps most experts. Larger savings need lower coverage or the `tiered` mode, and only `eval` can
say what that costs.

### Step by step: make and use a whitelist

You need:

- a PC with an NVIDIA card that already runs Strata with a Qwen3.8-Flash-Next model (a pack and its GGUF files),
- **the whole repository** on that PC (the generator uses `serve/`, `tools/` and `data/`),
- Python 3.10 or newer with the packages `regex` and `jinja2` (`python3 -m pip install regex jinja2`, or on
  Ubuntu `sudo apt install python3-regex python3-jinja2`),
- for building the engine: the CUDA toolkit 13.0 or newer, CMake and Ninja, and a driver 580 or newer,
- a few hundred MB of free disk for the samples and statistics, and time: the collection reads every sample once.

Run every command from the repository's folder.

**1. Build an engine with whitelist support.** The release engines do not have it. For an RTX 40 card (sm_89; RTX 30
is 86, RTX 50 is 120, RTX 20 is 75):

```bash
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
      -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
cmake --build build --target strata strata_kq
```

Check that it has the new options (the number must be above 0):

```bash
build/strata --help 2>&1 | grep -c expert-whitelist
```

**2. Write `engine.json`.** It tells the generator how to start the engine. Take the model's arguments from the start
configuration Strata already uses. A `setup.py` install writes one per model, for example `strata-iq3_s.json`. This
copies its arguments and points `"exe"` at the new engine:

```bash
python3 - <<'EOF'
import json, pathlib
eng = pathlib.Path("build").resolve()
cfg = json.loads(pathlib.Path("strata-iq3_s.json").read_text())
out = {"exe": str(eng / "strata"), "lib_dirs": [str(eng)], "args": cfg["args"], "cwd": cfg.get("cwd", ".")}
if cfg.get("env"):
    out["env"] = cfg["env"]
pathlib.Path("engine.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
EOF
```

Without such a file, write it by hand. The important parts are `--pack` (the folder with `native_experts.txt`) and
`--native` (the first GGUF file):

```json
{"exe": "/path/to/build/strata", "lib_dirs": ["/path/to/build"],
 "args": ["--pack", "/path/to/packs/iq3_s", "--native", "/path/to/Qwen3.8-Flash-Next-...-00001-of-00002.gguf",
          "--prefill", "auto"]}
```

Everything in `args` also applies to the collection: a control vector there shapes the statistics too. That is right
when you always run the model with it.

**3. Pick a profile.** A profile says which languages and tasks the whitelist is for. `python-coding.json` is ready
for a real run. The Rust and TypeScript profiles use a tiny demonstration set: copy one and point it at your own data
first (format in [docs/EXPERT_WHITELIST.md](docs/EXPERT_WHITELIST.md), section 2.1). Set `target_hardware` in the
profile to your PC (graphics memory and RAM in GiB). Check it:

```bash
python3 tools/expert_whitelist.py check-profile data/expert_whitelist/profiles/python-coding.json
```

**4. Prepare the samples.** This turns the conversations into the model's tokens, with its own chat template. Nothing
runs on the graphics card:

```bash
python3 tools/whitelist_gen.py prepare --profile data/expert_whitelist/profiles/python-coding.json --model engine.json --out prep/python
```

There must be no warning about the chat template. 10 % of the samples are held back for `eval`.

**5. Check what the collection needs.** `estimate` loads no model. Give it your graphics memory and RAM, each minus a
little room:

```bash
python3 tools/whitelist_gen.py estimate --model engine.json --prep prep/python --vram-budget 14G --ram-budget 56G --scratch /path/to/ssd/strata-tmp
```

For the models Strata runs, it should say `recommended: --mode token`.

**6. Stop the running Strata server.** The collection starts its own engine, and two engines do not fit into memory
together. Use `strata launch --stop`, or end the `serve/server.py` process (`kill <PID>`, not `-9`). Check that the
graphics card is free (the output must be empty):

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader
```

**7. Collect the statistics.** This is the long step: the engine reads every sample once. Start it so it keeps
running when you close the terminal:

```bash
nohup python3 tools/whitelist_gen.py collect --model engine.json --prep prep/python --out stats/python-run1 --mode token > collect.log 2>&1 &
```

Every 64 samples it writes a checkpoint (`grep checkpoint collect.log`). If it stops, run the same command with
`--resume`. When it is done, the log ends with `routing stats: ... samples ... in ... s`. You can start the Strata
server again: the next steps do not need the graphics card, except `eval`.

**8. Build the whitelist.** No graphics card needed; the statistics are reused, so try several versions. Add `--gguf`
with your first GGUF file so the report can compute the memory.

The plain version (`strict`, the coverages in the profile):

```bash
python3 tools/whitelist_gen.py build --stats stats/python-run1 --profile data/expert_whitelist/profiles/python-coding.json --mode coverage --gguf /path/to/model-00001-of-00002.gguf --out whitelists/python.json
```

A smaller version that stays closer to the full model (`tiered`: 90 % in memory, the rest up to 99.5 % read from the
drive when needed):

```bash
python3 tools/whitelist_gen.py build --stats stats/python-run1 --profile data/expert_whitelist/profiles/python-coding.json --mode coverage --coverage 0.90 --whitelist-mode tiered --on-demand-coverage 0.995 --gguf /path/to/model-00001-of-00002.gguf --out whitelists/python-tiered.json
```

The first line of the output names the size, for example `21195 of 24576 experts, 40.25 GiB of 46.84 GiB`. Next to
the whitelist, `build` writes a report (`.report.md`): the coverage per layer and per part of the profile, the memory
before and after, and warnings. To compare several sizes at once, use `--sweep coverage=0.95,0.90,0.85,0.80` with
`{value}` in `--out`. For a fixed size, use `--mode budget --budget 30G`. The general abilities (instructions, chat
structure, tool calls, shell output) keep their own coverages from the profile; lower them in a copy of the profile
to save more.

**9. Measure the quality.** `eval` runs the held-back samples twice, with the full model and with the whitelist, and
compares them per language, task and part of the conversation: how often the most likely next token stays the same,
the KL divergence and the perplexity. Stop the Strata server first (step 6):

```bash
nohup python3 tools/whitelist_gen.py eval --model engine.json --whitelist whitelists/python-tiered.json --prep prep/python --out evals/python-tiered > eval.log 2>&1 &
```

A top-1 match close to 100 % and a KL close to 0 on the Python answers mean the whitelist hardly changes the model.
If the Python part drops clearly, raise the coverage and build again.

**10. Use it.** Check it against the model, then add it to the engine's arguments:

```bash
python3 tools/expert_whitelist.py validate whitelists/python-tiered.json --pack /path/to/packs/iq3_s --gguf /path/to/model-00001-of-00002.gguf
```

In the start configuration (for example `strata-iq3_s.json`), set `"exe"` to the engine from step 1 and add to
`"args"`:

```text
"--expert-whitelist", "/path/to/whitelists/python-tiered.json"
```

Optional: `"--expert-whitelist-stats"` prints at the end, per layer, which experts were selected and run, and
`"--expert-whitelist-cache-gib", "2"` gives the `tiered` mode a larger RAM cache. Start Strata as usual. To go back to
the full model, remove these arguments.

### When something goes wrong with a whitelist

| Message | Cause | What to do |
|---|---|---|
| `prepare: warning: the model's chat template could not be rendered here (...)` | `jinja2` is missing for this Python, or `serve/frontend.py` is not there. The message names which. | Install `jinja2` for the Python you run, or copy the whole repository. Run `prepare` again. |
| `the chat template does not render a conversation as a prefix of the longer one` | An older `tools/whitelist_gen.py` (parallel tool calls). | Update the repository. |
| `prepare: warning: the python source glob ... matches no file` | The profile reads Python files from folders that are not on this PC. | Harmless (they are only example text), or copy those folders, or add your own code under `"sources"`. |
| `neither native_experts.txt nor manifest.json: not a Strata pack` | `--pack` in `engine.json` points at the wrong folder. | Find the pack: `find / -name native_experts.txt 2>/dev/null`. |
| `estimate` reports about 31 GiB of non-expert weights and recommends `layerwise` | An older `tools/whitelist_gen.py` counted the PLE table, which the engine reads from the drive. | Update the repository; it should then report about 4 GiB and recommend `token`. |
| The collection stops or the PC runs out of memory | The Strata server is still running next to the collection. | Stop it first (step 6). |
| Many warnings `the profile's routing is nearly flat` | A property of the model: it uses almost all experts for this work. | Expected. A whitelist saves less there; try `tiered`, a lower coverage or a budget, then check with `eval`. |
| The engine does not know `--routing-stats-out` or `--expert-whitelist` | `engine.json` or the start configuration points at a release engine. | Use the engine from step 1. |

Everything else (the formats, how each mode routes, the generator's scoring, all options and the open points) is in
**[docs/EXPERT_WHITELIST.md](docs/EXPERT_WHITELIST.md)**.

---

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, a 125-billion-parameter model that
normally needs a server, on a normal PC. It chats, writes code, reads pictures and works with your apps and coding
agents. Your chats and files stay on your PC.

**New here?** Check [what you need](#what-you-need), then [install the app](#option-a-the-strata-app).

- **Runs on a gaming PC.** 32 GB of RAM and graphics memory together is enough to start.
- **Works with your tools.** It speaks the OpenAI and Anthropic APIs, so Claude Code, Codex CLI, OpenCode and other agents connect to it.
- **Reads pictures.** They are on by default.
- **Free.** MIT license, no account, no cloud.
- **Private.** No telemetry. What Strata downloads, when it uses the internet and how it checks downloads is in [docs/SAFETY.md](docs/SAFETY.md).

Version 0.2.1 · engine 0.1.46

## What you need

**You need at least 32 GB of RAM and graphics memory (VRAM) together.** On a Mac or a Ryzen AI Max, that is the unified
memory. More memory gives a bigger model or a faster one.

- A 32 GB PC with a 12-16 GB card runs the Coder.
- 64 GB of RAM runs every size.

| Your computer | Details |
| --- | --- |
| **NVIDIA** GeForce RTX 20, 30, 40, 50 series (12 GB or more) | Install the app or the Terminal. |
| **AMD** Radeon RX 6800 and newer, Radeon AI PRO R9700 (12 GB or more) | [AMD cards](docs/AMD_HIP.md) |
| **Intel** Arc | [Intel Arc](docs/INTEL_ARC.md), [Intel engine](docs/INTEL.md) |
| **Apple Silicon Mac** (M1 or newer, macOS 15 or newer) | [Mac](docs/MAC.md) |
| **AMD Ryzen AI Max (Strix Halo)** and other unified-memory systems | [Strix Halo](docs/STRIX_HALO.md) |
| Older cards and processors | [Older GPUs](docs/OLDER_GPUS.md), [Older CPUs](docs/INSTALL.md#older-cpus-experimental) |

Also needed:

- about 80 GB of free disk (an SSD makes the first start much faster)
- Windows 10 / 11, Linux or macOS
- a current graphics driver

Two or three cards of the same make can share a model ([multi-GPU](docs/MULTI_GPU.md)). Strata never splits a model across
NVIDIA and AMD cards. Full list: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Install

Pick one option. They all start the same engine and use the same models.

### Option A: the Strata app

The easiest way. The app has the engine inside: install it, pick a model, chat. Download the file for your system and
graphics card from the [releases page](https://github.com/Niko1221/Strata/releases/latest). Not sure which one? See the
line under the table.

| System | Your graphics card | File to download |
| --- | --- | --- |
| **Windows 10 / 11** | NVIDIA RTX, driver 580 or newer | [`Strata-0.2.1-windows-cuda13.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.1-windows-cuda13.exe) |
| | NVIDIA, older card or driver older than 580 | [`Strata-0.2.1-windows-cuda12.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.1-windows-cuda12.exe) |
| | AMD Radeon RX 9000 series or Radeon AI PRO R9700 | [`Strata-0.2.1-windows-hip.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.1-windows-hip.exe) |
| | AMD Radeon RX 7000 series or Ryzen AI Max (Strix Halo) | [`Strata-0.2.1-windows-hip-rdna3.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.1-windows-hip-rdna3.exe) |
| **Linux** | NVIDIA RTX, driver 580 or newer | [`Strata-0.2.1-linux-nvidia-cuda13.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.1-linux-nvidia-cuda13.AppImage) |
| | NVIDIA, older card or driver older than 580 | [`Strata-0.2.1-linux-nvidia-cuda12.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.1-linux-nvidia-cuda12.AppImage) |
| | AMD Radeon RX 9000 series or Radeon AI PRO R9700 | [`Strata-0.2.1-linux-amd-rx9070.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.1-linux-amd-rx9070.AppImage) |
| | AMD Radeon RX 7000 series | [`Strata-0.2.1-linux-amd-rx7900.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.1-linux-amd-rx7900.AppImage) |
| | AMD Ryzen AI Max (Strix Halo) | [`Strata-0.2.1-linux-amd-strixhalo.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.1-linux-amd-strixhalo.AppImage) |
| **Mac** (Apple Silicon, macOS 15 or newer) | built in | [`Strata-0.2.1-mac.dmg`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.1-mac.dmg) |

**If you are unsure:** for NVIDIA, take `cuda13` if your driver is version 580 or newer, otherwise `cuda12`. `cuda12` is
for older cards and older drivers. The `hip` and `amd` files are for AMD cards. Other AMD cards (RX 6000) have no
ready-made app yet: see [AMD cards](docs/AMD_HIP.md).

Then:

- **Windows:** run the `.exe`. It installs for your user and needs no administrator rights. If Windows says "Windows
  protected your PC", choose **More info**, then **Run anyway** (the installer is not code-signed yet).
- **Linux:** make the file runnable (`chmod +x`) and start it. It needs FUSE 2: `sudo apt install libfuse2` (Ubuntu 22.04)
  or `libfuse2t64` (Ubuntu 24.04 and newer).
- **Mac:** open the `.dmg` and drag Strata onto Applications.

On the first screen the app reads your PC and recommends a model that fits. Press install. The model is a big download
(about 40-70 GB) and continues where it stopped if it is interrupted. Closing the window stops Strata and the engine.

### Option B: the Terminal (one line)

The Terminal is the `strata` command, for scripts, servers and AI agents.

Linux and Mac:

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

The script checks your PC, downloads the Terminal, verifies its SHA-256 and unpacks it into your home folder. It needs no
administrator rights. On Windows, open a new terminal window afterwards. Then start Strata:

```sh
strata launch
```

It asks one question at a time: model, context, pictures, who may connect. Then it downloads the engine for your graphics
card (signed, and test-started before use) and the model, and starts serving. For scripts and AI agents, use flags:

```sh
strata launch --model q2_0 --yes --no-tui
```

To read the install script before it runs, download it first with `curl -fsSL <address> -o install.sh`, read it, then run
`sh install.sh`. All screens and flags: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md).

### Option C: let your AI set it up

Paste this into Claude Code, Cursor, Codex, GitHub Copilot or another coding assistant:

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

It checks your graphics card, RAM and disk, picks the model that fits, installs and starts it, and tells you how to
connect your apps. AI tools can also install, start and stop Strata through its [MCP server](docs/MCP_SERVER.md).

Prefer to build it yourself? See [Build from source](#build-from-source).

## Models

The app and `strata launch` recommend a model for your memory. It is one model in several sizes, compressed more or less.
Smaller sizes are faster. Larger sizes are a bit smarter.

### Low on memory? Try Gyro

[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) (by agentionai) is the smallest way to run this model.

| Size | Memory it needs (RAM + VRAM) | Download | What it is |
| --- | ---: | ---: | --- |
| **Gyro-S** | 27.6 GB | 58.5 GB | For PCs that fit nothing else. Quality is lower than IQ2_XS. |
| **Gyro-M** | 35 GB | 92.0 GB | About IQ3_XXS quality with about 20% less GPU memory. It writes answers 17-21% slower than Q2_0 on a 12 GB card and on an R9700. |

Gyro runs on NVIDIA and AMD cards (AMD is experimental), not on Intel Arc or the Mac. Details and measurements:
[docs/GYRO.md](docs/GYRO.md). Start it with `strata launch --model gyro-s`.

### The other sizes

| Your memory (RAM + VRAM) | Take | Notes |
| --- | --- | --- |
| **32 GB** | **Gyro-S**, or the **[Coder](docs/MODELS.md#coder)** for code | The Coder needs 32 GB of RAM and keeps half of the experts, so it is weaker outside code |
| **40 GB** | **[Swift 1.5](docs/MODELS.md#swift-15)** IQ2_XS (39.2 GB), or **Gyro-M** (35 GB) | Swift thinks for a shorter time before it answers. Gyro-M is close to IQ3_XXS in quality, but it writes answers 17-21% slower than Q2_0 |
| **48 GB** | **IQ3_XXS** (47.0 GB) | Slower, better |
| **64 GB** | **IQ3_S** (54.8 GB), the original or Swift 1.5 | The slowest, the best: it matches the full model on the published tests |
| **96 GB or more** | Unsloth's **UD-IQ4_XS** (about 72 GB of RAM) to **Q8_0** (about 136 GB), 4 to 8 bit | Closer to the full model. With less RAM, Strata reads part of them from the SSD, which is much slower |

Less RAM than a row says can still work: Strata switches to its
[low-RAM mode](docs/MODELS.md#a-big-graphics-card-and-little-ram) by itself. Expect it to be slower, and much slower
with a small graphics card.

More choices:

- **[Swift 1.5](docs/MODELS.md#swift-15)** is a fine-tune that thinks for a shorter time before it answers, at about the same quality and RAM.
- **[NVFP4](docs/NVFP4.md)** is NVIDIA's 4-bit version for NVIDIA cards. It needs about 80 GB of RAM to keep every expert in RAM.
- **[Qwen3.6-35B-A3B](docs/QWEN36.md)** is a smaller, different model that reads pictures and can answer several requests at once.
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs)** is set up by hand.

Every size, download and what fits where: [docs/MODELS.md](docs/MODELS.md).

## How fast is it?

Based on a 12 GB graphics card (RTX 5070, Ryzen 5 7600, 64 GB RAM). See results from other cards:
[speed of each model](docs/MODELS.md#how-fast-is-each-size), [community results](docs/COMMUNITY_BENCHMARKS.md).

| Model | Answers appear at | Reads your prompt at |
| --- | --- | --- |
| **Q2_0** | `████████████` 94 tokens/s | `████████████` 2,650 tokens/s |
| **IQ2_XS** | `██████████░░` 79 tokens/s | `█████████░░░` 2,090 tokens/s |
| **IQ3_XXS** | `████████░░░░` 62 tokens/s | `████████░░░░` 1,750 tokens/s |
| **IQ3_S** | `███████░░░░░` 53 tokens/s | `███████░░░░░` 1,620 tokens/s |
| **Coder** | `███████░░░░░` 55 tokens/s | `██████████░░` 2,180 tokens/s |

A token is about ¾ of a word, so 60 tokens per second is faster than you can read. "Reads your prompt" is how fast Strata
takes in a long document, code or chat history.

Measured on a 16 GB AMD card (RX 9070 XT, Ryzen 9 3900X, 47 GB RAM):

| Model | Answers appear at | Reads your prompt at |
| --- | --- | --- |
| **Q2_0** | `████████░░░░` 60 tokens/s | `█████░░░░░░░` 1,160 tokens/s |
| **IQ2_XS** | `███████░░░░░` 52 tokens/s | `█████░░░░░░░` 1,110 tokens/s |
| **Coder** | `██████░░░░░░` 44 tokens/s | `██████░░░░░░` 1,420 tokens/s |

The bars use one scale for both tables. A card with more VRAM is faster: an RTX 3090 (24 GB) should write about 100-140
tokens per second. The engine version behind each number is in [DETAILS.md](docs/DETAILS.md#speed-measured).

## Using it

### In the app

The app has a **Chat**, a **Monitor** of the model and your GPU, CPU and RAM, a **Models** page and **Settings**. The chat
can work in a folder on your PC. In **Manual** mode it asks before it changes a file or runs a command. Type
`/loop 10m <what to do>` to repeat a message every 10 minutes while the chat is open.

The chat can search the web with DuckDuckGo (or Bing), with no account. You can add a search key under **Settings, Tools**.
Search is the one feature that sends something off your PC, the search words.

To run two models, turn on **Run multiple models** and give each model its own card of the same make.

<p align="center"><img src="docs/media/app-chat.png" width="900" alt="The Strata app: a chat where the agent built a small voxel pagoda garden, with the page open in the preview on the right"><br>
<sub>A chat in the Strata app: the agent wrote a small voxel pagoda garden as one HTML file, and the page opens in the preview next to the conversation</sub></p>

### In the Terminal

<p align="center"><img src="docs/media/terminal-monitor.png" width="900" alt="The Strata Terminal monitor overview while the model writes a reply"><br>
<sub><code>strata monitor</code> while the model writes a reply: speed, GPU, memory and the recent requests</sub></p>

```sh
strata launch                                  # guided screens: model, context, pictures, access, then serve
strata launch --model q2_0 --yes --no-tui     # no questions: serve, print the address, return
strata launch --plan --model q2_0              # show what would happen; nothing changes
strata launch --status                         # what is being served now
strata launch --stop                           # stop it
strata monitor --url http://127.0.0.1:8080     # watch speed, GPU, memory and requests
strata update --check                          # is there a new engine? (strata update installs it)
strata doctor                                  # check the installed engine files
```

`strata --help` lists everything. Starting and serving never download anything. Only the install, `strata launch` (when
you choose a model) and `strata update` do. To reach Strata from your phone or another PC, use
`strata launch --access lan`. It makes an API key and will not start on a network without one. No browser opens unless
you add `--open`. Guide: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md). Whole client: [docs/TERMINAL.md](docs/TERMINAL.md).

### With AI agents

An agent installs Strata with the one-line command in Option B and runs
`strata launch --model NAME --yes --no-tui`. The steps, flags and exit codes are in [docs/AI_SETUP.md](docs/AI_SETUP.md),
and the repository rules are in [AGENTS.md](AGENTS.md).

### Connect your other apps

| App type | Base URL |
| --- | --- |
| OpenAI-compatible (most apps and agents) | `http://127.0.0.1:8080/v1` (any API key, any model name) |
| Anthropic API (Claude Code and others) | `http://127.0.0.1:8080`, without `/v1`. These apps add `/v1/messages` themselves. Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` |
| OpenAI Responses API (Codex CLI) | `http://127.0.0.1:8080/v1/responses` ([setup](docs/DETAILS.md#the-responses-api-and-codex-cli)) |

`strata connect` prints the exact settings for OpenCode, Codex CLI, Claude Code, oh-my-pi, pi, Hermes Agent and OpenClaw.
`strata connect opencode --write` writes the file after asking and keeps a backup.

### Pictures

Pictures are on by default for models that can read them. Attach or paste a picture in the chat, or send it from your app.
The picture reader runs on the processor when your graphics card has less than 24 GB (a few seconds per picture, nothing
taken from the card's memory) and on the graphics card from 24 GB. The rule is the same for NVIDIA and AMD. Turn pictures
off per model, or choose the reader with `--vision off`, `--vision gpu` or `--vision cpu`.

The numbers behind the rule: on a Radeon AI PRO R9700 the card's reader took 57 ms for a 260-token picture and the
processor's took 3.1 s. On 16 GB cards (Tesla P100 and RTX A4000, Linux) the card's reader cost 3-5% of text speed,
because it holds about 1.4 GB that the model would use otherwise. On a Mac the reader runs on the Mac's GPU whatever its memory (0.45 s for a 260-token picture on an M4 with 32 GB, about 1.2 GB of the shared
memory); the processor reader needs about 50 s per picture there, so it runs only when you ask for it with `--vision cpu`.

### Good to know

- Choose the thinking level (**off, low, medium or high**) in the chat menu or in your app's "reasoning effort". Off is fastest.
- Strata answers one request at a time by default. For several at once, use `--parallel N` in `strata launch` or
  `"parallel": 2` in the config ([BATCHING.md](docs/BATCHING.md)). On a 12 GB card this makes each answer slower.
- Strata reads the first message of a chat in full, about 1 minute per 30,000 tokens. Follow-up messages start in seconds.
- Where your chats are stored and the whole API: [docs/INSTALL.md](docs/INSTALL.md#where-things-are-stored), [docs/DETAILS.md](docs/DETAILS.md#using-it).

## Build from source

<details>
<summary>Windows, Linux and Mac from the repository</summary>

[Download Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) and unzip it, or `git clone` it.
**Windows:** double-click `START-HERE.bat`. **Linux:** run `./setup.sh` in the Strata folder.

**Mac (Apple Silicon, macOS 15 or newer):** install Apple's Command Line Tools once with `xcode-select --install`
(Xcode itself is not needed). Open Terminal in the Strata folder and run `./setup.sh`. It builds the Mac engine on your
Mac (about 7 minutes), asks the same questions, downloads the model and starts it. Next time, `./setup.sh` starts it
right away. Which size fits your Mac: [docs/MAC.md](docs/MAC.md).

The installer finds your card and sets up the right engine. It asks which model and size, how much context (how much text
the model keeps in mind) and whether it should read pictures. Press Enter for the recommended answer. Then it downloads the
model (about 70 GB) and starts it. If the download stops, run it again and it continues. Your browser opens the Strata
page at `http://127.0.0.1:8080`.

> While the model starts, your PC can be slow or stop responding for 1-3 minutes (longest the first time). Strata loads
> 35-55 GB into your RAM and locks part of it for the graphics card. Wait, and don't close the window.

Next time, run `START-HERE.bat` (or `./setup.sh`) again. It starts right away and downloads nothing twice. Close its
window to stop the model. `UPDATE.bat` (`./update.sh`) updates Strata without starting it. Updating, Docker, several cards,
where the files go and every option: [docs/INSTALL.md](docs/INSTALL.md).

To measure your PC, run `BENCH.bat` (`./bench.sh`). It takes 5-10 minutes and writes a report you can share
([how](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

**Optional: more speed on one card when the whole model fits in RAM.** Two opt-in settings by RubenTorney (#1904, #1905).
They help when you run one graphics card and your RAM holds every expert, for example Q2_0 or IQ3_XXS on a 64 GB PC. Open
`strata-<model>.json` in the Strata data folder ([where](docs/INSTALL.md#where-things-are-stored)) and add:

- `"--adapt-async", "1"` to the `"args"` list. The expert cache then updates on a helper thread instead of pausing the
  answer. Measured on an RTX 5070 (Windows, Q2_0): +4-5% decode, +7-10% on code and long documents. Answers can differ
  slightly from the default and stay the same from run to run.
- `"STRATA_PIN_BY_NEED": "1"` in the `"env"` block (Windows). Windows can lock only about half the RAM for the graphics
  card, and this locks the experts the card needs first. Measured on the RTX 5070: +4.4% prompt reading.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Restart Strata after the change. To undo, remove the two entries.

</details>

## How does it work?

Models like this one usually run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. Strata makes the model fit by sharing the work across your whole PC. Picture a library.

- The model is a team of 24,576 small specialists ("experts"), like 24,576 reference books. Each word needs only 10 of them.
- **Your graphics card is the desk.** It is small and fast. The roughly 4,000 books used most often lie open on it.
- **Your RAM is the shelves in the reading room.** All 24,576 books are there. When a word needs a book that is not on the
  desk, your processor reads it from the shelf at once, while the card keeps working.
- **Your SSD is the archive.** It holds a 29 GB lookup table. The model reads a few rows of it per word, so it does not
  need to sit in RAM.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

A small helper also guesses the next few words. The big model checks all of them at once and keeps the right ones. You get
the same answer, 1.6-1.8x sooner. Long texts are read in big pieces, up to 8,192 tokens at a time, at over 1,000 tokens
per second.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

The longer explanation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Every part and its numbers:
[the details](docs/DETAILS.md#how-it-works) and the [paper](docs/paper/Strata-Paper.pdf).

## Something went wrong?

- **My PC froze the first time Strata started.** It loads the model into RAM. Wait, and don't close the window. Still
  frozen after 10 minutes? Restart the PC, close other programs and try again, or pick a smaller size.
- **It stopped while downloading or installing.** Start the app or `strata launch` again, or run `START-HERE.bat` (or
  `./setup.sh`) again. It continues where it stopped.
- **It is very slow, the disk light keeps blinking, or it says "the engine stopped unexpectedly".** Your PC does not have
  enough free RAM. Close other programs (browsers use a lot), or pick a smaller size such as Q2_0 or IQ2_XS.
- **It says there is no engine for this PC.** In the app, open **Settings, Engine** and press **Update engine**. In the
  Terminal, run `strata update`. `strata doctor` shows which engine files are installed.
- **It says port 8080 is already in use.** Strata is already running. Look for its window.

More problems and fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Still stuck? Open an
[issue](https://github.com/Niko1221/Strata/issues). In the app, attach the file from **Settings, Engine, Diagnostics, View
logs, Save logs**. In the Terminal, paste the output of `strata doctor`. From source, attach `strata-<model>.log` from the
Strata folder. For a security problem, report it privately: [SECURITY.md](SECURITY.md).

## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. It was compressed by
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5), agentionai (Gyro)
and Unsloth. Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata is open source under the [MIT License](LICENSE). A few parts and
every model have their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).

## Contributing

Found a bug, or want to send a change or a benchmark? Read [CONTRIBUTING.md](CONTRIBUTING.md) first. It says what to
include so that others can check your work.

## Support Strata

Strata is free and MIT-licensed. I build it and test it on whatever hardware I can get. If it runs well on your PC, a coffee helps
me keep testing new cards and fixing what you report. No pressure. Using it and reporting what you find helps just as much.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>

Interested in sponsoring Strata? Contact me via [GitHub](https://github.com/Niko1221).
