"""tools/whitelist_gen.py - the expert whitelist generator: from calibration data to a whitelist manifest.

docs/EXPERT_WHITELIST.md section 7 explains every stage.  Each one reads and writes files, and each can run alone:

    prepare   --profile P [--profile P2] --out PREP      tokenized, deduplicated samples and a fixed holdout split
    estimate  --model CFG|--pack PACK --prep PREP ...      what a collection needs (memory, scratch space, reads), and
                                                           which mode (token or layerwise) fits; nothing is run
    analyze   --model CFG --prep PREP [--prep ...] --out D how specialised the experts are, on a small sample
    collect   --model CFG --prep PREP --out STATS          the routing statistics (the only stage that runs the model)
    merge     --in STATS [STATS ...] --out STATS           statistics of independent runs, added up exactly
    build     --stats STATS --profile P --out W.json       the whitelist manifest, a report, a coverage curve
    inspect   W.json                                       a readable summary of a manifest
    eval      --model CFG --whitelist W.json --prep PREP   a short quality check on the holdout, with the runtime

CFG is an engine config (JSON: "exe", "args", "env", "lib_dirs", "cwd"), the form tools/quant_quality.py reads.  The
stages that do not run the model never load weights: they read the tokenizer, the model's metadata and the files
other stages wrote.  Every output records the inputs (hashes), the model fingerprint and the parameters it came from.
Only the standard library is used; the model's own tokenizer needs the `regex` package (tools/strata_tokenizer.py) and
the model's chat template `jinja2` (serve/frontend.py).
"""
from __future__ import annotations

import argparse
import array
import hashlib
import heapq
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import glob
import zlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import expert_whitelist as W  # noqa: E402

GENERATOR = "whitelist_gen/1"
PREP_KIND = "strata-whitelist-prep"
STATS_KIND = "strata-routing-stats"
SAMPLES_MAGIC = b"SXCALIB1"
LAYER_MAGIC = b"SXRSTAT1"
GATE_FRAC_BITS = 32
CONTRIB_FRAC_BITS = 20
U64_MAX = (1 << 64) - 1
REGION_OF_ROLE = {"system": "system", "user": "prompt", "assistant": "response", "tool": "tool"}
DEFAULT_REGION_WEIGHTS = {"response": 1.0, "prompt": 0.5, "system": 0.5, "tool": 0.5}
DATASETS = ("target", "base", "excluded", "reference")
SCORING_METHOD = "profile-mass-contrast-v1"
DEFAULT_BLOCK_TOKENS = 16384   # include/strata/core/layerwise_calib.hpp kDefaultBlockTokens
SIM_MAGIC = b"SXESIM01"        # include/strata/core/expert_similarity.hpp
SIM_FRAC_BITS = 32
SIM_REL_CAP_MILLI = 4000       # kSimilarityRelCap = 4.0
SIM_WEIGHTING = "the final router weight of the picked expert"
DEFAULT_ON_DEMAND_CACHE = 1 << 30   # the engine's --expert-whitelist-cache default


class GenError(W.WhitelistError):
    """A stage that cannot go on; the message says why and what to do."""


# ================================ SMALL HELPERS ================================

def parse_size(s) -> int:
    """'14G', '512M', '1.5T', '2GiB' or plain bytes.  K, M, G and T are binary (GiB): the engine's own unit."""
    if isinstance(s, int):
        return s
    t = str(s).strip().upper().replace("IB", "").replace("B", "")
    mult = 1
    if t and t[-1] in "KMGT":
        mult = 1 << (10 * ("KMGT".index(t[-1]) + 1))
        t = t[:-1]
    try:
        v = float(t)
    except ValueError:
        raise GenError(f"not a size: {s!r} (examples: 16G, 512M, 1.5T)") from None
    if v < 0:
        raise GenError(f"a size cannot be negative: {s!r}")
    return int(v * mult)


def fmt_bytes(b: float) -> str:
    for unit, div in (("TiB", 1 << 40), ("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if abs(b) >= div:
            return f"{b / div:.2f} {unit}"
    return f"{int(b)} B"


def parse_hardware(s: str | None) -> dict:
    """'vram=16G,ram=64G,reserve=8G[,buffer=2G]' -> bytes."""
    out = {}
    for part in (s or "").split(","):
        if not part.strip():
            continue
        k, sep, v = part.partition("=")
        k = k.strip().lower()
        if not sep or k not in ("vram", "ram", "reserve", "buffer"):
            raise GenError(f"--target-hardware takes vram=, ram=, reserve= and buffer=, not {part!r}")
        out[k] = parse_size(v)
    return out


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def atomic_write(path, data: bytes | str) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data.encode("utf-8") if isinstance(data, str) else data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, obj) -> None:
    atomic_write(path, json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def fnv_hex(text: str) -> str:
    return "fnv1a64:%016x" % W.fnv1a64(text.encode())


def hash_fraction(seed, text: str) -> float:
    """A number in [0, 1) from a seed and a text: the same on every machine (the holdout split, the subsets)."""
    return int(hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()[:16], 16) / float(1 << 64)


def log(msg: str) -> None:
    print(msg, flush=True)


# ================================ TOKENIZERS AND THE CHAT TEMPLATE ================================

class ByteTokenizer:
    """A stand-in for tests and examples, never for a real whitelist: one token per UTF-8 byte, and the three ChatML
    literals as tokens 256, 257, 258."""
    SPECIAL = {"<|im_start|>": 256, "<|im_end|>": 257, "<|endoftext|>": 258}

    def encode(self, text: str, parse_special: bool = True) -> list[int]:
        if not parse_special:
            return list(text.encode("utf-8"))
        out, i = [], 0
        while i < len(text):
            for lit, tid in self.SPECIAL.items():
                if text.startswith(lit, i):
                    out.append(tid)
                    i += len(lit)
                    break
            else:
                j = min((p for p in (text.find(lit, i) for lit in self.SPECIAL) if p >= 0), default=len(text))
                out += list(text[i:j].encode("utf-8"))
                i = j
        return out


def model_paths(cfg: dict | None) -> tuple[str | None, str | None]:
    """The pack directory and the first GGUF shard an engine config names (--pack, --native)."""
    if not cfg:
        return None, None
    args = [str(a) for a in cfg.get("args", [])]
    pack = gguf = None
    for i, a in enumerate(args[:-1]):
        if a == "--pack":
            pack = args[i + 1]
        elif a == "--native":
            gguf = args[i + 1]
    base = pathlib.Path(cfg.get("cwd") or ".")
    fix = (lambda p: str(p if pathlib.Path(p).is_absolute() else base / p) if p else None)
    return fix(pack), fix(gguf)


def load_model_config(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        cfg = json.loads(pathlib.Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as e:
        raise GenError(f"--model {path}: not a readable engine config ({e})") from None
    if not isinstance(cfg, dict) or "exe" not in cfg or not isinstance(cfg.get("args", []), list):
        raise GenError(f"--model {path}: an engine config needs \"exe\" and \"args\" (tools/quant_quality.py's form)")
    return cfg


def load_tokenizer(spec: str | None, cfg: dict | None):
    """(tokenizer, description).  `bytes`, a pack's tokenizer/ directory, a GGUF, or what the engine config names."""
    if spec is None and cfg is not None:
        pack, gguf = model_paths(cfg)
        if pack and (pathlib.Path(pack) / "tokenizer" / "vocab.json").exists():
            spec = str(pathlib.Path(pack) / "tokenizer")
        elif gguf:
            spec = gguf
    if spec is None:
        raise GenError("no tokenizer: give --tokenizer (a pack's tokenizer/ folder or a GGUF) or --model")
    if spec == "bytes":
        return ByteTokenizer(), {"kind": "bytes", "fingerprint": "bytes-v1", "note": "test tokenizer, not the model's"}
    try:
        import strata_tokenizer as ST  # noqa: E402  (needs the regex package)
    except ImportError as e:
        raise GenError(f"the model's tokenizer needs the Python package `{e.name}` (pip install {e.name})") from None
    p = pathlib.Path(spec)
    if p.is_dir():
        vocab = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
        toks = [None] * len(vocab)
        for t, i in vocab.items():
            toks[i] = t
        merges = (p / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((p / "token_type.json").read_text()) if (p / "token_type.json").exists() else None
        h = hashlib.sha256()
        for name in ("vocab.json", "merges.txt", "token_type.json"):
            if (p / name).exists():
                h.update((p / name).read_bytes())
        return ST.Tokenizer(toks, merges, types), {"kind": "bpe", "path": str(p), "fingerprint": "sha256:" + h.hexdigest()}
    if p.suffix == ".gguf" and p.exists():
        tok = ST.Tokenizer.from_gguf(p)
        h = hashlib.sha256("\n".join(tok.tokens).encode("utf-8"))
        h.update("\n".join(f"{a} {b}" for (a, b), _ in sorted(tok.ranks.items(), key=lambda kv: kv[1])).encode("utf-8"))
        return tok, {"kind": "bpe", "path": str(p), "fingerprint": "sha256:" + h.hexdigest()}
    raise GenError(f"--tokenizer {spec}: not 'bytes', a tokenizer folder (vocab.json, merges.txt) or a .gguf")


class Template:
    """How a sample's messages become text, message by message, so every token knows its region."""

    def __init__(self, mode: str, cfg: dict | None):
        self.mode, self.path, self.jinja = mode, None, None
        self.unavailable = None              # why the model's template fell back to ChatML (the import error)
        if mode in ("chatml", "none"):
            return
        if mode == "model":
            pack, _ = model_paths(cfg)
            cand = [pathlib.Path(pack) / "tokenizer" / "chat_template.jinja"] if pack else []
            cand.append(REPO / "serve" / "chat_template.jinja")
            self.path = next((str(c) for c in cand if c.exists()), None)
        else:
            self.path = mode
            if not pathlib.Path(mode).exists():
                raise GenError(f"chat template {mode}: no such file")
        try:
            sys.path.insert(0, str(REPO / "serve"))
            import frontend  # noqa: E402  (needs jinja2)
            self.jinja = frontend.ChatTemplate(self.path)
        except ImportError as e:
            fe = REPO / "serve" / "frontend.py"
            if not fe.exists():                # tools/ copied without serve/: no pip package provides this file
                why = f"{fe} is missing: copy the repository's serve folder beside tools"
            elif getattr(e, "name", None) in ("jinja2", "markupsafe"):
                why = f"{e}: install it for this Python ({sys.executable} -m pip install jinja2)"
            else:
                why = f"{e} (Python {sys.executable})"
            if mode != "model":
                raise GenError(f"a Jinja chat template needs serve/frontend.py and jinja2: {why}") from None
            self.mode, self.path = "chatml", None   # the model's template unavailable: ChatML, said in prep.json
            self.unavailable = why

    def describe(self) -> dict:
        d = {"mode": self.mode}
        if self.unavailable:
            d["fallback"] = self.unavailable
        if self.path:
            d.update(path=self.path, sha256=sha256_file(self.path))
        return d

    def _split(self, messages: list[dict]) -> list[str]:
        """The whole conversation as the template renders it, cut into one piece per message.  A message ends where
        the conversation up to it ends, when that is a prefix of the whole.  Some templates render the end of a
        message differently once more follows (consecutive tool results share one turn and one <|im_end|>; earlier
        turns lose their thinking block): there the message ends after its own content and as much of its closing
        as the whole conversation has."""
        full = self.jinja.render(messages, add_generation_prompt=False)
        cuts, cur = [], 0
        for i, m in enumerate(messages):
            part = self.jinja.render(messages[:i + 1], add_generation_prompt=False)
            if full.startswith(part):
                end = len(part)
            else:
                needle = m["content"].strip()
                at, pat = (full.find(needle, cur), part.rfind(needle)) if needle else (-1, -1)
                if at < 0 or pat < 0:
                    raise GenError(f"the chat template does not render message {i + 1} ({m['role']}) where the "
                                   "whole conversation has it; use \"chat_template\": \"chatml\"")
                end, tail = at + len(needle), part[pat + len(needle):]
                end += len(os.path.commonprefix([tail, full[end:]]))
            end = max(end, cur)
            cuts.append(end)
            cur = end
        cuts[-1] = len(full)
        return [full[a:b] for a, b in zip([0] + cuts[:-1], cuts)]

    def spans(self, messages: list[dict]) -> list[tuple[str, str]]:
        """(region, text) per message, in order; the text includes the template's scaffolding of that message."""
        out = []
        if self.jinja is not None:
            return [(REGION_OF_ROLE[m["role"]], t) for m, t in zip(messages, self._split(messages))]
        for i, m in enumerate(messages):
            if self.mode == "chatml":
                out.append((REGION_OF_ROLE[m["role"]], f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"))
            else:
                out.append((REGION_OF_ROLE[m["role"]], m["content"] + ("\n\n" if i + 1 < len(messages) else "")))
        return out


# ================================ CALIBRATION ROWS AND SAMPLE FILES ================================

def row_messages(row: dict, where: str) -> list[dict]:
    if "messages" in row:
        msgs = row["messages"]
        if not isinstance(msgs, list) or not msgs:
            raise GenError(f'{where}: "messages" must be a non-empty list')
        for m in msgs:
            if not isinstance(m, dict) or m.get("role") not in REGION_OF_ROLE or not isinstance(m.get("content"), str):
                raise GenError(f'{where}: each message needs "role" (system, user, assistant, tool) and a "content" '
                               "string")
        return [{"role": m["role"], "content": m["content"]} for m in msgs]
    msgs = [{"role": "user", "content": row["prompt"]}]
    if row.get("response"):
        msgs.append({"role": "assistant", "content": row["response"]})
    return msgs


def write_samples(path, categories: list[str], samples: list[dict]) -> None:
    """samples.bin (src/core/routing_stats.cpp read_calib_samples): little-endian, see docs 7.2."""
    b = bytearray(SAMPLES_MAGIC)
    b += (1).to_bytes(4, "little") + len(categories).to_bytes(4, "little") + len(samples).to_bytes(4, "little")
    b += (0).to_bytes(4, "little")
    for k in categories:
        kb = k.encode("utf-8")
        b += len(kb).to_bytes(4, "little") + kb
    for s in samples:
        ib = s["id"].encode("utf-8")
        b += len(ib).to_bytes(4, "little") + ib + len(s["tokens"]).to_bytes(4, "little")
        t = array.array("i", s["tokens"])
        c = array.array("H", s["cats"])
        if sys.byteorder != "little":
            t.byteswap()
            c.byteswap()
        b += t.tobytes() + c.tobytes()
    atomic_write(path, bytes(b))


def read_samples(path) -> tuple[list[str], list[dict]]:
    data = pathlib.Path(path).read_bytes()
    if data[:8] != SAMPLES_MAGIC:
        raise GenError(f"{path}: not a prepared sample file")
    u32 = lambda at: int.from_bytes(data[at:at + 4], "little")  # noqa: E731
    if u32(8) != 1:
        raise GenError(f"{path}: sample file version {u32(8)}")
    n_cat, n_s, p = u32(12), u32(16), 24
    cats = []
    for _ in range(n_cat):
        n = u32(p)
        cats.append(data[p + 4:p + 4 + n].decode("utf-8"))
        p += 4 + n
    samples = []
    for _ in range(n_s):
        n = u32(p)
        sid = data[p + 4:p + 4 + n].decode("utf-8")
        p += 4 + n
        nt = u32(p)
        p += 4
        t = array.array("i")
        t.frombytes(data[p:p + 4 * nt])
        p += 4 * nt
        c = array.array("H")
        c.frombytes(data[p:p + 2 * nt])
        p += 2 * nt
        if sys.byteorder != "little":
            t.byteswap()
            c.byteswap()
        samples.append({"id": sid, "tokens": list(t), "cats": list(c)})
    if p != len(data):
        raise GenError(f"{path}: {len(data) - p} bytes after the last sample")
    return cats, samples


def split_key(key: str) -> dict:
    f = key.split("|")
    if len(f) != 5:
        raise GenError(f"category {key!r} is not dataset|language|task|region|origin")
    return dict(zip(("dataset", "language", "task", "region", "origin"), f))


def clean_field(v: str) -> str:
    return str(v).replace("|", "/").strip() or "general"


# ================================ PREPARE ================================

def prepare(profiles: list[str], out: str, tokenizer: str | None = None, model: str | None = None,
            holdout: float = 0.10, reference: list[str] | None = None, only: list[str] | None = None,
            chat_template: str | None = None, max_tokens: int | None = None, max_samples: int | None = None,
            seed: int | None = None) -> dict:
    cfg = load_model_config(model)
    prof = W.combine_profiles([W.load_profile(p) for p in profiles]) if profiles else None
    if prof is None and not reference:
        raise GenError("prepare needs --profile (or --reference files for a reference-only set)")
    if not 0.0 <= holdout < 1.0:
        raise GenError("--holdout must be in [0, 1)")
    limits = prof["limits"] if prof else {"max_samples": 2000, "max_tokens_per_sample": 2048, "seed": 0}
    max_tokens = max_tokens or limits["max_tokens_per_sample"]
    max_samples = max_samples or limits["max_samples"]
    seed = limits["seed"] if seed is None else seed
    tok, tok_info = load_tokenizer(tokenizer, cfg)
    tmpl = Template(chat_template or (prof["chat_template"] if prof else "none"), cfg)
    only = set(only or DATASETS)
    # ---- where every sample comes from: (dataset, language, default task, file)
    sources: list[tuple[str, str, str, str]] = []
    globs: list[tuple[str, str]] = []
    if prof:
        for lang, spec in prof["target_languages"].items():
            sources += [("target", lang, "general", f) for f in spec["calibration"]]
            globs += [(lang, g) for g in spec["sources"]]
        for item, spec in prof["base_requirements"]["items"].items():
            sources += [("base", item, item, f) for f in spec["calibration"]]
        for ex, spec in prof["excluded"].items():
            sources += [("excluded", ex, "excluded", f) for f in spec["calibration"]]
        sources += [("reference", "reference", "reference", f) for f in prof["reference_corpus"]]
    sources += [("reference", "reference", "reference", str(pathlib.Path(f).resolve())) for f in reference or []]
    sources = [s for s in sources if s[0] in only]
    rows, inputs, warnings = [], [], []
    seen_ids: dict[str, str] = {}
    unknown_lang: dict[str, int] = {}
    unknown_task: dict[str, int] = {}
    targets = set(prof["target_languages"]) if prof else set()
    tasks = prof["tasks"] if prof else {}
    for dataset, lang, task, path in sources:
        inputs.append({"path": path, "sha256": sha256_file(path), "dataset": dataset, "language": lang})
        stem = pathlib.Path(path).stem
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                where = f"{path}:{n}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise GenError(f"{where}: {e.msg}") from None
                has_prompt = isinstance(row, dict) and isinstance(row.get("prompt"), str) and bool(row["prompt"])
                if not isinstance(row, dict) or (not has_prompt and "messages" not in row):
                    raise GenError(f'{where}: each line needs a non-empty "prompt" or a "messages" list')
                for key in ("id", "response", "task", "language"):
                    if key in row and not isinstance(row[key], str):
                        raise GenError(f'{where}: "{key}" must be a string')
                sid = clean_field(row.get("id") or f"{stem}-{n:05d}")
                if sid in seen_ids:
                    raise GenError(f'{where}: the id "{sid}" is used twice (also {seen_ids[sid]})')
                seen_ids[sid] = where
                language = lang if dataset in ("base", "excluded") else clean_field(row.get("language") or lang)
                t = clean_field(row.get("task") or task)
                if dataset == "target" and language not in targets:
                    unknown_lang[language] = unknown_lang.get(language, 0) + 1
                if dataset == "target" and tasks and t not in tasks:
                    unknown_task[t] = unknown_task.get(t, 0) + 1
                rows.append({"id": sid, "dataset": dataset, "language": language, "task": t,
                             "messages": row_messages(row, where), "where": where})
    for lang, pattern in globs:   # source files of a target language: each file one prompt-only sample, task "source"
        files = sorted(pathlib.Path(p) for p in glob.glob(os.path.expanduser(pattern), recursive=True))
        files = [p for p in files if p.is_file()]
        if not files:
            warnings.append(f"the {lang} source glob {pattern} matches no file")
        files.sort(key=lambda p: hash_fraction(seed, str(p)))
        used = 0
        for p in files:
            if used >= max_samples:
                break
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if not text.strip():
                continue
            sid = clean_field(f"{lang}-source-{hashlib.sha256(str(p).encode()).hexdigest()[:12]}")
            if sid in seen_ids:
                continue
            seen_ids[sid] = str(p)
            rows.append({"id": sid, "dataset": "target", "language": lang, "task": "source",
                         "messages": [{"role": "user", "content": text}], "where": str(p)})
            inputs.append({"path": str(p), "sha256": sha256_file(p), "dataset": "target", "language": lang})
            used += 1
    if not rows:
        raise GenError("no calibration samples: the profile names no calibration file with lines in it")
    # ---- duplicates (by content), the holdout split (by id), the cap per data set and language (a fixed subset)
    first: dict[str, str] = {}
    dups = []
    kept = []
    for r in rows:
        h = hashlib.sha256(canonical(r["messages"]).encode()).hexdigest()
        if h in first:
            dups.append({"id": r["id"], "same_as": first[h]})
            continue
        first[h] = r["id"]
        kept.append(r)
    train, held = [], []
    for r in kept:
        (held if hash_fraction(f"holdout:{seed}", r["id"]) < holdout else train).append(r)
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in train:
        groups.setdefault((r["dataset"], r["language"]), []).append(r)
    over = 0
    keep_ids = set()
    for key, rs in groups.items():
        rs = sorted(rs, key=lambda r: hash_fraction(f"subset:{seed}", r["id"]))
        over += max(0, len(rs) - max_samples)
        keep_ids.update(r["id"] for r in rs[:max_samples])
    train = [r for r in train if r["id"] in keep_ids]
    # ---- tokens and their regions, cut to the longest sample the profile allows
    keys: set[str] = set()
    truncated = 0

    def tokenize(r):
        nonlocal truncated
        toks, regs = [], []
        for region, text in tmpl.spans(r["messages"]):
            ids = tok.encode(text, parse_special=tmpl.mode != "none")
            toks += ids
            regs += [region] * len(ids)
        if len(toks) > max_tokens:
            truncated += 1
            toks, regs = toks[:max_tokens], regs[:max_tokens]
        if not toks:
            raise GenError(f"{r['where']}: the sample has no tokens")
        r["tokens"] = toks
        r["keys"] = [f"{r['dataset']}|{r['language']}|{r['task']}|{g}|prefill" for g in regs]
        keys.update(r["keys"])

    for r in train + held:
        tokenize(r)
    categories = sorted(keys)
    index = {k: i for i, k in enumerate(categories)}
    if len(categories) > 65535:
        raise GenError(f"{len(categories)} categories (dataset x language x task x region): at most 65535")
    for r in train + held:
        r["cats"] = [index[k] for k in r.pop("keys")]
    outp = pathlib.Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    write_samples(outp / "samples.bin", categories, train)
    write_samples(outp / "holdout.bin", categories, held)

    def summary(rs):
        acc: dict[tuple, list[int]] = {}
        for r in rs:
            seen = set()
            for c in r["cats"]:
                k = tuple(split_key(categories[c])[f] for f in ("dataset", "language", "task", "region"))
                a = acc.setdefault(k, [0, 0])
                a[1] += 1
                if k not in seen:
                    a[0] += 1
                    seen.add(k)
        return [{"dataset": k[0], "language": k[1], "task": k[2], "region": k[3], "samples": v[0], "tokens": v[1]}
                for k, v in sorted(acc.items())]

    if unknown_lang:
        warnings.append("target samples in languages the profile does not target (weight 0 in build, they count as "
                        "reference): " + ", ".join(f"{k} ({v})" for k, v in sorted(unknown_lang.items())))
    if unknown_task:
        warnings.append("tasks the profile does not weight (weight 0 in build): " +
                        ", ".join(f"{k} ({v})" for k, v in sorted(unknown_task.items())))
    if tmpl.mode == "chatml" and (chat_template or (prof and prof["chat_template"])) == "model":
        warnings.append(f"the model's chat template could not be rendered here ({tmpl.unavailable}): the samples use "
                        "plain ChatML; fix it and run prepare again")
    if tok_info["kind"] == "bytes":
        warnings.append("the byte tokenizer is a test stand-in: these samples cannot be run through a real model")
    fingerprint = None
    pack, gguf = model_paths(cfg)
    if pack:
        try:
            fingerprint = W.pack_model_info(pack, gguf=gguf if gguf and pathlib.Path(gguf).exists() else None).fingerprint()
        except (W.WhitelistError, OSError, ValueError):
            fingerprint = None
    doc = {
        "format_version": 1, "kind": PREP_KIND, "generator": GENERATOR,
        "model_fingerprint": fingerprint,
        "profiles": [{"name": W.load_profile(p)["name"], "path": str(p), "sha256": sha256_file(p)} for p in profiles],
        "profile_name": prof["name"] if prof else "",
        "inputs": inputs,
        "tokenizer": tok_info,
        "chat_template": tmpl.describe(),
        "params": {"max_tokens_per_sample": max_tokens, "max_samples": max_samples, "holdout_fraction": holdout,
                   "seed": seed, "datasets": sorted(only)},
        "categories": categories,
        "samples": {"file": "samples.bin", "sha256": sha256_file(outp / "samples.bin"), "count": len(train),
                    "tokens": sum(len(r["tokens"]) for r in train)},
        "holdout": {"file": "holdout.bin", "sha256": sha256_file(outp / "holdout.bin"), "count": len(held),
                    "tokens": sum(len(r["tokens"]) for r in held), "ids": sorted(r["id"] for r in held)},
        "summary": {"train": summary(train), "holdout": summary(held)},
        "dropped": {"duplicates": dups, "over_max_samples": over, "truncated": truncated},
        "unknown": {"languages": unknown_lang, "tasks": unknown_task},
        "warnings": warnings,
    }
    write_json(outp / "prep.json", doc)
    return doc


def read_prep(path: str) -> dict:
    p = pathlib.Path(path)
    try:
        doc = json.loads((p / "prep.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise GenError(f"{path}: no prep.json ({e}); run `prepare` first") from None
    if doc.get("kind") != PREP_KIND or doc.get("format_version") != 1:
        raise GenError(f"{path}/prep.json: not a prepared data set of this version")
    for part in ("samples", "holdout"):
        f = p / doc[part]["file"]
        if not f.exists() or sha256_file(f) != doc[part]["sha256"]:
            raise GenError(f"{f}: missing or changed since prepare wrote it")
    doc["dir"] = str(p)
    return doc


# ================================ STATISTICS ================================

class LayerData:
    """One layer file: tokens[c] and values[(c * E + e) * (4 + k) + j], j = count, gate, gate^2, contrib, rank 0..k-1."""

    def __init__(self, layer: int, n_expert: int, top_k: int, n_cat: int, tokens=None, values=None):
        self.layer, self.E, self.k, self.C = layer, n_expert, top_k, n_cat
        self.W = 4 + top_k
        self.tokens = tokens if tokens is not None else array.array("Q", [0] * n_cat)
        self.values = values if values is not None else array.array("Q", bytes(8 * n_cat * n_expert * self.W))

    def col(self, c: int, j: int):
        base = c * self.E * self.W
        return self.values[base + j: base + self.E * self.W: self.W]


class SimData:
    """One layer's similarity accumulators (src/core/expert_similarity.cpp): row_w[E], row_n[E], cos[E][E], rel[E][E].
    cos holds w x (1 + cosine), rel w x min(relative L2 distance, 4), both x 2^32, w the picked expert's router weight."""

    def __init__(self, layer: int, n_expert: int, tokens: int = 0, values=None):
        self.layer, self.E, self.tokens = layer, n_expert, tokens
        self.values = values if values is not None else array.array("Q", bytes(8 * (2 * n_expert + 2 * n_expert ** 2)))

    def row_weight(self, i: int) -> int:
        return self.values[i]

    def row_tokens(self, i: int) -> int:
        return self.values[self.E + i]

    def mean_cos(self, i: int, j: int) -> float | None:
        w = self.values[i]
        return None if w == 0 else self.values[2 * self.E + i * self.E + j] / w - 1.0

    def mean_rel(self, i: int, j: int) -> float | None:
        w = self.values[i]
        return None if w == 0 else self.values[2 * self.E + self.E * self.E + i * self.E + j] / w


class Stats:
    def __init__(self, path: str, meta: dict, model: W.ModelInfo, layers: dict[int, LayerData],
                 similarity: dict[int, SimData] | None = None):
        self.path, self.meta, self.model, self.layers = path, meta, model, layers
        self.categories = [c["key"] for c in meta["categories"]]
        self.has_contrib = bool(meta.get("has_contrib"))
        self.similarity = similarity

    @property
    def fingerprint(self) -> str:
        return self.meta["model_fingerprint"]


def model_from_meta(m: dict) -> W.ModelInfo:
    epl = m["experts_per_layer"]
    if isinstance(epl, int):
        epl = [epl] * m["num_layers"]
    return W.ModelInfo(m["num_layers"], list(epl), m["top_k"], m["hidden"], list(m["layer_offset"]),
                       list(m["blob_bytes"]), "statistics")


def schema_fnv(keys: list[str]) -> int:
    return W.fnv1a64("".join(k + "\n" for k in keys).encode())


def layer_header(layer: int, ld: LayerData, has_contrib: bool, fp: str, keys: list[str]) -> bytes:
    le = lambda v, n: int(v).to_bytes(n, "little")  # noqa: E731
    return (LAYER_MAGIC + le(1, 4) + le(layer, 4) + le(ld.E, 4) + le(ld.k, 4) + le(ld.C, 4) + le(1 if has_contrib else 0, 4)
            + le(GATE_FRAC_BITS, 4) + le(CONTRIB_FRAC_BITS, 4) + le(0, 8) + le(W.fnv1a64(fp.encode()), 8)
            + le(schema_fnv(keys), 8))


def similarity_sampled(fraction: float, seed: int, sample_id: str, pos: int) -> bool:
    """Whether a position is sampled for the similarity data (src/core/expert_similarity.cpp similarity_sampled):
    FNV-1a 64 of "<seed>:<id>:<pos>" below fraction x 2^64."""
    if fraction <= 0:
        return False
    if fraction >= 1:
        return True
    return (W.fnv1a64(f"{seed}:{sample_id}:{pos}".encode()) >> 11) < fraction * 9007199254740992.0


def config_fnv_int(text) -> int:
    """The header's configuration hash from meta.json's "config_fnv1a64" ("mixed" after a merge of different ones: 0)."""
    return int(text.split(":", 1)[1], 16) if isinstance(text, str) and text.startswith("fnv1a64:") else 0


def sim_header(layer: int, sd: SimData, fp: str, config_fnv: int) -> bytes:
    le = lambda v, n: int(v).to_bytes(n, "little")  # noqa: E731
    return (SIM_MAGIC + le(1, 4) + le(layer, 4) + le(sd.E, 4) + le(SIM_FRAC_BITS, 4) + le(SIM_REL_CAP_MILLI, 4) + le(0, 4)
            + le(sd.tokens, 8) + le(0, 8) + le(W.fnv1a64(fp.encode()), 8) + le(config_fnv, 8))


def read_similarity(p: pathlib.Path, meta: dict, model: W.ModelInfo) -> dict[int, SimData]:
    sm = meta["similarity"]
    if sm.get("frac_bits") != SIM_FRAC_BITS or sm.get("rel_cap") != SIM_REL_CAP_MILLI // 1000:
        raise GenError(f"{p}/meta.json: the similarity data has another scale")
    out = {}
    for f in sm["files"]:
        l = f["layer"]
        data = (p / f["file"]).read_bytes()
        if len(data) != f["bytes"] or "%08x" % zlib.crc32(data) != f["crc32"]:
            raise GenError(f"{p / f['file']}: the file changed or is damaged (size or CRC-32 differ from meta.json)")
        E = model.experts_per_layer[l]
        sd = SimData(l, E, int.from_bytes(data[32:40], "little"), array.array("Q"))
        if data[:32] != sim_header(l, sd, meta["model_fingerprint"], 0)[:32] or \
                data[48:64] != sim_header(l, sd, meta["model_fingerprint"], config_fnv_int(sm.get("config_fnv1a64")))[48:64]:
            raise GenError(f"{p / f['file']}: the header does not match meta.json (layer, geometry, model or sampling)")
        body = data[64:]
        if len(body) != 8 * (2 * E + 2 * E * E):
            raise GenError(f"{p / f['file']}: wrong size")
        sd.values.frombytes(body)
        if sys.byteorder != "little":
            sd.values.byteswap()
        out[l] = sd
    moe = [l for l in range(model.n_layers) if model.experts_per_layer[l] > 0]
    if sorted(out) != moe:
        raise GenError(f"{p}: the similarity files do not cover every MoE layer")
    return out


def read_stats(path: str) -> Stats:
    p = pathlib.Path(path)
    try:
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise GenError(f"{path}: no complete statistics (meta.json: {e}); an interrupted collect resumes with --resume") \
            from None
    if meta.get("kind") != STATS_KIND or meta.get("format_version") != 1:
        raise GenError(f"{path}/meta.json: not routing statistics of this version")
    model = model_from_meta(meta["model"])
    if model.fingerprint() != meta["model_fingerprint"]:
        raise GenError(f"{path}/meta.json: the model section does not give the recorded fingerprint")
    fpb = meta.get("fixed_point", {})
    if fpb.get("gate_frac_bits") != GATE_FRAC_BITS or fpb.get("contrib_frac_bits") != CONTRIB_FRAC_BITS:
        raise GenError(f"{path}: another fixed-point scale")
    keys = [c["key"] for c in meta["categories"]]
    if fnv_hex("".join(k + "\n" for k in keys)) != meta["schema_fnv1a64"]:
        raise GenError(f"{path}/meta.json: the category list does not match its hash")
    layers = {}
    for lf in meta["layer_files"]:
        l = lf["layer"]
        data = (p / lf["file"]).read_bytes()
        if len(data) != lf["bytes"] or "%08x" % zlib.crc32(data) != lf["crc32"]:
            raise GenError(f"{p / lf['file']}: the file changed or is damaged (size or CRC-32 differ from meta.json)")
        E = model.experts_per_layer[l]
        ld = LayerData(l, E, model.top_k, len(keys), array.array("Q"), array.array("Q"))
        want = layer_header(l, ld, bool(meta.get("has_contrib")), meta["model_fingerprint"], keys)
        if data[:40] != want[:40] or data[48:64] != want[48:64]:
            raise GenError(f"{p / lf['file']}: the header does not match meta.json (layer, geometry or model)")
        body = data[64:]
        if len(body) != 8 * (len(keys) + len(keys) * E * ld.W):
            raise GenError(f"{p / lf['file']}: wrong size")
        ld.tokens.frombytes(body[:8 * len(keys)])
        ld.values.frombytes(body[8 * len(keys):])
        if sys.byteorder != "little":
            ld.tokens.byteswap()
            ld.values.byteswap()
        layers[l] = ld
    if sorted(layers) != [l for l in range(model.n_layers) if model.experts_per_layer[l] > 0]:
        raise GenError(f"{path}: the layer files do not cover every MoE layer")
    sim = read_similarity(p, meta, model) if meta.get("similarity") else None
    return Stats(str(p), meta, model, layers, sim)


def write_stats(path: str, meta: dict, layers: dict[int, LayerData], similarity: dict[int, SimData] | None = None) -> None:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    keys = [c["key"] for c in meta["categories"]]
    files = []
    for l in sorted(layers):
        ld = layers[l]
        t, v = array.array("Q", ld.tokens), array.array("Q", ld.values)
        if sys.byteorder != "little":
            t.byteswap()
            v.byteswap()
        data = layer_header(l, ld, bool(meta.get("has_contrib")), meta["model_fingerprint"], keys) + t.tobytes() + v.tobytes()
        name = "layer_%03d.bin" % l
        atomic_write(p / name, data)
        files.append({"layer": l, "file": name, "bytes": len(data), "crc32": "%08x" % zlib.crc32(data)})
    meta = dict(meta, layer_files=files, schema_fnv1a64=fnv_hex("".join(k + "\n" for k in keys)))
    if similarity:
        sm = dict(meta["similarity"])
        cfg = config_fnv_int(sm.get("config_fnv1a64"))
        sfiles = []
        for l in sorted(similarity):
            sd = similarity[l]
            v = array.array("Q", sd.values)
            if sys.byteorder != "little":
                v.byteswap()
            data = sim_header(l, sd, meta["model_fingerprint"], cfg) + v.tobytes()
            name = "similarity_%03d.bin" % l
            atomic_write(p / name, data)
            sfiles.append({"layer": l, "file": name, "bytes": len(data), "crc32": "%08x" % zlib.crc32(data)})
        sm["files"] = sfiles
        meta["similarity"] = sm
    else:
        meta.pop("similarity", None)
    write_json(p / "meta.json", meta)


def leaf_runs(st: Stats) -> list[dict]:
    """The collection runs a statistics directory holds: itself, or what a merge put together."""
    if st.meta.get("mode") == "merged":
        return [dict(r) for r in st.meta["runs"]]
    r = dict(st.meta["runs"][0])
    r["collect"] = st.meta.get("collect", {})
    r["run"] = st.meta.get("run", {})
    return [r]


def merge(inputs: list[str], out: str, allow_overlap: bool = False) -> dict:
    """Adds statistics up, element by element (exact: the values are integers).  The order of the inputs, and how a
    merge of merges is nested, do not change a byte of the result."""
    if not inputs:
        raise GenError("merge needs at least one --in")
    sts = [read_stats(i) for i in inputs]
    ref = sts[0]
    for st in sts[1:]:
        if st.fingerprint != ref.fingerprint:
            raise GenError(f"{st.path} was collected on another model or pack (fingerprint {st.fingerprint}, "
                           f"{ref.path} has {ref.fingerprint}): statistics of different models cannot be merged")
        if st.meta["model"] != ref.meta["model"]:
            raise GenError(f"{st.path}: the model section differs from {ref.path}")
    quant = sorted({st.meta.get("quantization", "") for st in sts})
    if len(quant) > 1:
        raise GenError("the statistics were collected on different weights (quantization " + ", ".join(quant) +
                       "): merge only runs on the same pack")
    runs: dict[str, dict] = {}
    for st in sts:
        for r in leaf_runs(st):
            if r["run_id"] in runs:
                raise GenError(f"the run {r['run_id']} is in more than one input ({st.path}): it would count twice")
            runs[r["run_id"]] = r
    done: dict[str, str] = {}
    overlap = []
    for st in sts:
        for sid in st.meta.get("samples_done", []):
            if sid in done:
                overlap.append(sid)
            done[sid] = st.path
    if overlap and not allow_overlap:
        raise GenError(f"{len(overlap)} sample ids are in more than one input (e.g. {sorted(overlap)[0]!r}): their "
                       "tokens would count twice; --allow-overlap merges anyway")
    keys = sorted({k for st in sts for k in st.categories})
    index = {k: i for i, k in enumerate(keys)}
    model = ref.model
    contrib = all(st.has_contrib for st in sts)
    layers = {}
    for l in ref.layers:
        E, k = model.experts_per_layer[l], model.top_k
        acc = LayerData(l, E, k, len(keys))
        tok = [0] * len(keys)
        val = [0] * (len(keys) * E * acc.W)
        for st in sts:
            ld = st.layers[l]
            n = E * acc.W
            for c, key in enumerate(st.categories):
                ci = index[key]
                tok[ci] += ld.tokens[c]
                dst = ci * n
                val[dst:dst + n] = [x + y for x, y in zip(val[dst:dst + n], ld.values[c * n:(c + 1) * n])]
        if any(v > U64_MAX for v in tok) or any(v > U64_MAX for v in val):
            raise GenError(f"layer {l}: a sum passes 2^64; merge fewer runs at once")
        if not contrib:
            for i in range(3, len(val), acc.W):
                val[i] = 0
        acc.tokens = array.array("Q", tok)
        acc.values = array.array("Q", val)
        layers[l] = acc
    # the similarity data: the sum over the inputs that have it (it does not depend on the categories)
    similarity = None
    sim_meta = None
    with_sim = [st for st in sts if st.similarity]
    if with_sim:
        similarity = {}
        for l in ref.layers:
            E = model.experts_per_layer[l]
            vals = [0] * (2 * E + 2 * E * E)
            tok = 0
            for st in with_sim:
                sd = st.similarity[l]
                tok += sd.tokens
                vals = [x + y for x, y in zip(vals, sd.values)]
            if tok > U64_MAX or any(v > U64_MAX for v in vals):
                raise GenError(f"layer {l}: a similarity sum passes 2^64; merge fewer runs at once")
            similarity[l] = SimData(l, E, tok, array.array("Q", vals))
        sims = [st.meta["similarity"] for st in with_sim]
        fractions, seeds = {x.get("fraction") for x in sims}, {x.get("seed") for x in sims}
        cfgs = sorted({x.get("config_fnv1a64", "") for x in sims})
        sim_meta = {"fraction": fractions.pop() if len(fractions) == 1 else None,   # None: the runs sampled differently
                    "seed": seeds.pop() if len(seeds) == 1 else None,
                    "config_fnv1a64": cfgs[0] if len(cfgs) == 1 else "mixed",
                    "frac_bits": SIM_FRAC_BITS, "rel_cap": SIM_REL_CAP_MILLI // 1000, "weighting": SIM_WEIGHTING,
                    "tokens": similarity[min(similarity)].tokens,
                    "runs": sorted(r["run_id"] for st in with_sim for r in leaf_runs(st)),
                    "runs_without": sorted(r["run_id"] for st in sts if not st.similarity for r in leaf_runs(st)),
                    "files": []}
    leaves = [runs[r] for r in sorted(runs)]
    run_id = fnv_hex("\n".join(sorted(runs)) + "\n")
    meta = {
        "format_version": 1, "kind": STATS_KIND,
        "model_fingerprint": ref.fingerprint, "model": ref.meta["model"],
        # the non-expert bytes depend on the engine's options, not on the model: the largest any run recorded
        "dense_bytes": max(st.meta.get("dense_bytes", 0) for st in sts), "quantization": quant[0],
        "fixed_point": {"gate_frac_bits": GATE_FRAC_BITS, "contrib_frac_bits": CONTRIB_FRAC_BITS},
        "has_contrib": contrib,
        "categories": [dict(split_key(k), key=k) for k in keys],
        "schema_fnv1a64": "", "layer_files": [],
        "tokens": sum(st.meta["tokens"] for st in sts),
        "samples": sum(st.meta["samples"] for st in sts),
        "run_id": run_id, "runs": leaves, "mode": "merged",
        "created": max(st.meta.get("created", "") for st in sts),
        "generator": GENERATOR + " merge", "run": {}, "collect": {},
        "samples_done": sorted(done),
    }
    if sim_meta:
        meta["similarity"] = sim_meta
    write_stats(out, meta, layers, similarity)
    return meta


# ================================ BUILD: SCORING AND SELECTION ================================

class Profile:
    """The weights build scores with: per category its component (a target language or a base requirement) or the
    reference, and its weight; unknown values weigh 0 and are reported."""

    def __init__(self, prof: dict | None, region_weights: dict, base_weight: float = 1.0):
        self.prof = prof
        self.region_w = region_weights
        self.base_weight = base_weight
        self.targets = dict(prof["target_languages"]) if prof else {}
        self.base = dict(prof["base_requirements"]["items"]) if prof else {}
        self.tasks = dict(prof["tasks"]) if prof else {}
        self.unknown: dict[str, set] = {"tasks": set(), "regions": set(), "languages": set()}

    def components(self) -> list[str]:
        return [f"lang:{k}" for k in sorted(self.targets)] + [f"base:{k}" for k in sorted(self.base)]

    def min_coverage(self, comp: str) -> float:
        kind, name = comp.split(":", 1)
        spec = (self.targets if kind == "lang" else self.base).get(name)
        return spec["min_coverage"] if spec else 0.95   # analyze without a profile

    def comp_weight(self, comp: str) -> float:
        kind, name = comp.split(":", 1)
        if kind == "lang":
            return self.targets[name]["weight"] if name in self.targets else 1.0
        return self.base_weight

    def role(self, cat: dict) -> tuple[str | None, float]:
        """(component or 'reference', weight) of one category."""
        rw = self.region_w.get(cat["region"])
        if rw is None:
            self.unknown["regions"].add(cat["region"])
            rw = 0.0
        if self.prof is None:   # analyze without a profile: every target language and base item, all weights 1
            if cat["dataset"] == "target":
                return f"lang:{cat['language']}", rw
            if cat["dataset"] == "base":
                return f"base:{cat['language']}", rw
            return "reference", rw
        if cat["language"] in self.targets and cat["dataset"] in ("target", "excluded", "reference"):
            tw = 1.0 if not self.tasks else self.tasks.get(cat["task"], 0.0)
            if self.tasks and cat["task"] not in self.tasks:
                self.unknown["tasks"].add(cat["task"])
            return f"lang:{cat['language']}", tw * rw
        if cat["dataset"] == "base" and cat["language"] in self.base:
            return f"base:{cat['language']}", rw
        if cat["dataset"] == "target":
            self.unknown["languages"].add(cat["language"])
        return "reference", rw


def gate_scale(use_contrib: bool) -> float:
    return float(1 << (CONTRIB_FRAC_BITS if use_contrib else GATE_FRAC_BITS))


class LayerScores:
    """Per layer: each component's mass m^k (normalised), the overall profile mass m_p, the reference mass m_r, the
    counts for the uncertainty rule, the scores s^k and the combined score."""

    def __init__(self, st: Stats, l: int, prof: Profile, comps: list[str], p: dict):
        ld = st.layers[l]
        E = ld.E
        self.E = E
        j = 3 if p["use_contrib"] else 1
        scale = gate_scale(p["use_contrib"])
        raw = {c: [0.0] * E for c in comps}
        cnt = {c: [0] * E for c in comps}
        tok = {c: 0 for c in comps}
        mp = [0.0] * E
        mr = [0.0] * E
        self.ref_tokens = 0
        for ci, key in enumerate(st.categories):
            cat = split_key(key)
            role, w = prof.role(cat)
            g = ld.col(ci, j)
            if role == "reference":
                self.ref_tokens += ld.tokens[ci]
                if w > 0:
                    for e in range(E):
                        mr[e] += w * g[e] / scale
                continue
            if role not in raw:
                continue
            n = ld.col(ci, 0)
            tok[role] += ld.tokens[ci]
            for e in range(E):
                cnt[role][e] += n[e]
            if w <= 0:
                continue
            lw = prof.comp_weight(role)
            for e in range(E):
                v = w * g[e] / scale
                raw[role][e] += v
                mp[e] += lw * v
        self.tokens = tok
        self.count = cnt
        norm = lambda v: [x / s for x in v] if (s := sum(v)) > 0 else [0.0] * len(v)  # noqa: E731
        self.m = {c: norm(raw[c]) for c in comps}
        self.mp = norm(mp)
        self.mr = norm(mr)
        self.has_ref = sum(mr) > 0
        alpha = p["alpha"] if p["alpha"] is not None else 1.0 / (E * 100.0)
        lam = p["lambda"] if self.has_ref else 0.0
        C = p["clip"]

        def score(m):
            out = []
            for e in range(E):
                c = math.log((m[e] + alpha) / (self.mr[e] + alpha)) if self.has_ref else 0.0
                out.append(m[e] * math.exp(lam * max(-C, min(C, c))))
            return out

        self.s = {c: score(self.m[c]) for c in comps}
        wsum = sum(prof.comp_weight(c) for c in comps) or 1.0
        self.combined = [sum(prof.comp_weight(c) * self.s[c][e] for c in comps) / wsum for e in range(E)]
        self.combined_mass = [sum(prof.comp_weight(c) * self.m[c][e] for c in comps) / wsum for e in range(E)]

    def order(self, scores: list[float]) -> list[int]:
        """Expert ids by score, highest first; a tie goes to the lower id."""
        return sorted(range(self.E), key=lambda e: (-scores[e], e))

    def protect(self, comp: str, protect_mass: float) -> list[int]:
        m = self.m[comp]
        if sum(m) <= 0 or protect_mass <= 0:
            return []
        out, acc = [], 0.0
        for e in sorted(range(self.E), key=lambda e: (-m[e], e)):
            if acc >= protect_mass - 1e-12:
                break
            out.append(e)
            acc += m[e]
        return out


def select_layer(sc: LayerScores, comps: list[str], prof: Profile, p: dict, targets: dict[str, float]) -> dict:
    """One layer in coverage, count or fraction mode (budget mode starts from this layer's floor; see build)."""
    E = sc.E
    chosen: set[int] = set()
    uncertain: dict[str, list[int]] = {}
    short: dict[str, float] = {}
    for c in comps:
        prot = set(sc.protect(c, p["protect_mass"]))
        chosen |= prot
        unc = [e for e in range(E) if sc.count[c][e] < p["n_min"] and sc.m[c][e] > 0]
        uncertain[c] = unc
        if p["mode"] != "coverage":
            continue
        tgt = targets[c]
        if sum(sc.m[c]) <= 0:
            continue   # no data for this component in this layer (reported by build)
        acc = sum(sc.m[c][e] for e in chosen)
        for e in sc.order(sc.s[c]):
            if acc >= tgt - 1e-12:
                break
            if e in chosen:
                continue
            if e in unc and not p["include_uncertain"]:
                continue
            if sc.m[c][e] <= 0:
                break
            chosen.add(e)
            acc += sc.m[c][e]
        if acc < tgt - 1e-12:
            short[c] = acc
    if p["mode"] in ("count", "fraction"):
        n = p["count"] if p["mode"] == "count" else int(math.ceil(p["fraction"] * E))
        unc_all = set().union(*uncertain.values()) if uncertain else set()
        for e in sc.order(sc.combined):
            if len(chosen) >= n:
                break
            if e in chosen or (e in unc_all and not p["include_uncertain"]):
                continue
            chosen.add(e)
    return {"chosen": chosen, "uncertain": uncertain, "short": short}


def apply_floor(sc: LayerScores, chosen: set[int], floor: int, exclude: set[int] | None = None) -> list[int]:
    """At least top_k + margin experts: the best by the combined score (then by id) fill up, never from `exclude`."""
    added = []
    for e in sc.order(sc.combined):
        if len(chosen) >= min(floor, sc.E):
            break
        if e not in chosen and (exclude is None or e not in exclude):
            chosen.add(e)
            added.append(e)
    return added


def apply_groups(chosen: set[int], E: int, n_group: int, group_topk: int, top_k: int) -> list[int]:
    """Group-limited routing: a group the router may pick needs ceil(top_k / group_topk) allowed experts, or the whole
    group is blocked.  Returns the blocked groups."""
    if n_group <= 1:
        return []
    size = E // n_group
    need = -(-top_k // max(group_topk, 1))
    blocked = []
    for g in range(n_group):
        ids = [e for e in range(g * size, (g + 1) * size) if e in chosen]
        if 0 < len(ids) < need:
            for e in ids:
                chosen.discard(e)
            blocked.append(g)
    return blocked


def kv_from_gguf(gguf: str | None) -> tuple[int, int, str]:
    """(KV bytes per token, fixed state bytes per sequence, how) from the model's metadata: FP16 K/V of the full
    attention layers, and the recurrent layers' state.  A calculation from the metadata, not the engine's allocation."""
    if not gguf:
        return 0, 0, ""
    import gguf_reader  # noqa: E402
    meta = gguf_reader.GGUFFile(pathlib.Path(gguf)).metadata
    arch = meta.get("general.architecture", "")
    g = lambda k, d=0: meta.get(f"{arch}.{k}", d)  # noqa: E731
    n_layers = int(g("block_count"))
    interval = int(g("full_attention_interval", 1) or 1)
    n_full = n_layers // interval
    kv_heads = int(g("attention.head_count_kv", 0) or 0)
    head = int(g("attention.key_length", 0) or 0)
    per_tok = n_full * kv_heads * head * 2 * 2
    st = int(g("ssm.state_size", 0) or 0)
    vh = int(g("ssm.time_step_rank", 0) or 0)
    conv = int(g("ssm.conv_kernel", 0) or 0)
    kh = int(g("ssm.group_count", 0) or 0)
    channels = 2 * st * kh + st * vh
    state = (n_layers - n_full) * (st * vh * st + channels * max(conv - 1, 0)) * 4
    how = (f"{n_full} full-attention layers x {kv_heads} K/V heads x {head} x 2 (K and V) x 2 B (FP16) per token; "
           f"{n_layers - n_full} recurrent layers' state {fmt_bytes(state)} per sequence")
    return per_tok, state, how


def gguf_shards(gguf: str) -> list[pathlib.Path]:
    g = pathlib.Path(gguf)
    if "-00001-of-" in g.name:
        return sorted(g.parent.glob(g.name.split("-00001-of-")[0] + "-*-of-*.gguf"))
    return [g]


# The n-gram (PLE) table: the engine reads its rows from the drive per token (--ple-io direct, the default), so it takes
# neither VRAM nor RAM. In Qwen3.8-Flash-Next's GGUFs it is most of the non-expert bytes (51.2e9 values).
DRIVE_TENSORS = ("per_layer_token_embd.weight",)


def gguf_dense_bytes(gguf: str) -> int:
    """The bytes of every tensor that is not a routed expert's and not read from the drive, over all shards (from the
    GGUF headers alone)."""
    import gguf_reader  # noqa: E402
    total = 0
    for sh in gguf_shards(gguf):
        for t in gguf_reader.GGUFFile(sh).tensors:
            if "_exps" not in t.name and t.name not in DRIVE_TENSORS:
                total += t.expected_bytes() or t.elements * 2
    return total


def hardware_budget(st: Stats, hw: dict, context: int, p: dict) -> dict:
    """Budget_experts = VRAM + RAM - reserve - non-expert weights - KV cache(context) - runtime buffer."""
    dense, dense_how = p.get("dense_bytes"), "given on the command line"
    if not dense and st.meta.get("dense_bytes"):
        dense, dense_how = st.meta["dense_bytes"], "the engine's dense weight arena, recorded by collect"
    if not dense and p.get("gguf"):
        dense, dense_how = gguf_dense_bytes(p["gguf"]), "the GGUF's non-expert tensors"
    if not dense:
        raise GenError("the statistics do not say how big the model's non-expert weights are: give --gguf (the first "
                       "shard) or --dense-bytes")
    kv_tok, state, how = p.get("kv_bytes_per_token"), p.get("state_bytes"), ""
    if kv_tok is None:
        kv_tok, state2, how = kv_from_gguf(p.get("gguf"))
        state = state if state is not None else state2
        if not how:
            raise GenError("the KV cache needs the model's metadata: give --gguf (the first shard) or "
                           "--kv-bytes-per-token")
    state = state or 0
    kv = kv_tok * context + state
    buffer = hw.get("buffer", p["runtime_buffer"])
    total = hw.get("vram", 0) + hw.get("ram", 0)
    budget = total - hw.get("reserve", 0) - dense - kv - buffer
    return {"vram": hw.get("vram", 0), "ram": hw.get("ram", 0), "reserve": hw.get("reserve", 0),
            "non_expert_weights": dense, "non_expert_how": dense_how, "context": context,
            "kv_bytes_per_token": kv_tok, "state_bytes": state,
            "kv_cache": kv, "kv_how": how or "given on the command line", "runtime_buffer": buffer,
            "expert_budget": budget,
            "note": "a calculation from the model's metadata and these inputs, not a measurement; how the allowed "
                    "experts are spread over VRAM and RAM is the runtime's own placement"}


def build(stats: str, profiles: list[str], out: str, mode: str = "coverage", coverage: float | None = None,
          count: int | None = None, fraction: float | None = None, budget: str | None = None,
          target_hardware: str | None = None, context: int = 32768, **kw) -> dict:
    st = read_stats(stats)
    prof_doc = W.combine_profiles([W.load_profile(p) for p in profiles]) if profiles else None
    if prof_doc is None:
        raise GenError("build needs --profile")
    p = {
        "mode": mode, "coverage": coverage, "count": count, "fraction": fraction,
        "lambda": kw.get("lam", 0.5), "clip": kw.get("clip", 3.0), "alpha": kw.get("alpha"),
        "protect_mass": kw.get("protect_mass", 0.8), "n_min": kw.get("n_min", 20),
        "include_uncertain": kw.get("include_uncertain", False), "margin": kw.get("margin", 2),
        "region_weights": dict(DEFAULT_REGION_WEIGHTS, **(kw.get("region_weights") or {})),
        "base_weight": kw.get("base_weight", 1.0), "use_contrib": kw.get("use_contrib", False),
        "allow_under_coverage": kw.get("allow_under_coverage", False),
        "layer_weights": {int(k): float(v) for k, v in (kw.get("layer_weights") or {}).items()},
        "expert_groups": kw.get("expert_groups", 0), "group_topk": kw.get("group_topk", 0),
        "runtime_buffer": kw.get("runtime_buffer", 2 << 30), "min_tokens": kw.get("min_tokens", 2000),
        "min_task_tokens": kw.get("min_task_tokens", 500),
        "flat_entropy": kw.get("flat_entropy", 0.9),
        "dense_bytes": kw.get("dense_bytes"), "kv_bytes_per_token": kw.get("kv_bytes_per_token"),
        "state_bytes": kw.get("state_bytes"), "gguf": kw.get("gguf"),
        "whitelist_mode": kw.get("whitelist_mode") or "strict",
        "on_demand_coverage": kw.get("on_demand_coverage", 0.995),
        "on_demand_cache": kw.get("on_demand_cache", DEFAULT_ON_DEMAND_CACHE),
        "substitute_threshold": kw.get("substitute_threshold", 0.5),
        "substitute_min_tokens": kw.get("substitute_min_tokens", 20),
    }
    if mode not in ("coverage", "count", "fraction", "budget"):
        raise GenError(f"unknown --mode {mode} (coverage, count, fraction or budget)")
    wl_mode = p["whitelist_mode"]
    if wl_mode not in W.MODES:
        raise GenError(f"unknown --whitelist-mode {wl_mode} (strict, substitute or tiered)")
    if wl_mode == "tiered" and not 0 < p["on_demand_coverage"] <= 1:
        raise GenError("--on-demand-coverage must be in (0, 1]")
    if wl_mode == "substitute" and not st.similarity:
        raise GenError("--whitelist-mode substitute needs the experts' similarity data in the statistics (collected with "
                       "the similarity share on, docs 7.3); these statistics have none")
    if wl_mode == "substitute" and not -1 <= p["substitute_threshold"] <= 1:
        raise GenError("--substitute-threshold must be from -1 to 1 (a cosine similarity)")
    if p["use_contrib"] and not st.has_contrib:
        raise GenError("--use-contrib: these statistics have no weight x ||output|| (the path that collected them "
                       "cannot give it)")
    if mode == "count" and not (count and count > 0):
        raise GenError("--mode count needs --count N")
    if mode == "fraction" and not (fraction and 0 < fraction <= 1):
        raise GenError("--mode fraction needs --fraction F in (0, 1]")
    if coverage is not None and not 0 < coverage <= 1:
        raise GenError("--coverage must be in (0, 1]")
    prof = Profile(prof_doc, p["region_weights"], p["base_weight"])
    comps = prof.components()
    model = st.model
    top_k = model.top_k
    floor = top_k + p["margin"]
    # the coverage each component must reach: --coverage for the target languages (default: their min_coverage), the
    # profile's min_coverage for every base requirement
    targets = {}
    for c in comps:
        kind = c.split(":", 1)[0]
        targets[c] = coverage if (kind == "lang" and coverage is not None) else prof.min_coverage(c)
    hw = None
    if target_hardware or (mode == "budget" and budget is None):
        spec = parse_hardware(target_hardware) if target_hardware else {}
        th = prof_doc.get("target_hardware") or {}
        for key, gib in (("vram", "vram_gib"), ("ram", "ram_gib"), ("reserve", "reserve_gib")):
            if key not in spec and gib in th:
                spec[key] = int(float(th[gib]) * (1 << 30))
        if not spec.get("vram") and not spec.get("ram"):
            raise GenError("--mode budget needs --budget or --target-hardware (or the profile's target_hardware)")
        hw = hardware_budget(st, spec, context, p)
    moe = sorted(st.layers)
    if not moe:
        raise GenError("the model has no MoE layer: there is nothing to whitelist")
    scores = {l: LayerScores(st, l, prof, comps, p) for l in moe}
    sel = {}
    report_layers = []
    warnings = []
    violations = []
    for l in moe:
        sc = scores[l]
        r = select_layer(sc, comps, prof, dict(p, mode="coverage" if mode == "budget" else mode),
                         {c: (prof.min_coverage(c) if mode == "budget" else targets[c]) for c in comps})
        sel[l] = r
    # ---- budget mode: the floor and the protected experts first, then the minimum coverages, then the best marginal
    # utility per byte across all layers
    budget_info = None
    if mode == "budget":
        bytes_of = lambda l: model.blob_bytes[l]  # noqa: E731
        if budget is not None:
            b = str(budget)
            limit_bytes = parse_size(b) if b[-1:].upper() in "KMGTB" else None
            limit_experts = int(b) if limit_bytes is None else None
        else:
            limit_bytes, limit_experts = hw["expert_budget"], None
        if wl_mode == "tiered" and limit_bytes is not None:   # the on_demand cache is RAM beside the resident experts
            limit_bytes -= p["on_demand_cache"]
        start = {}
        for l in moe:
            sc = scores[l]
            base = set()
            for c in comps:
                base |= set(sc.protect(c, p["protect_mass"]))
            apply_floor(sc, base, floor)
            start[l] = base
        used = lambda s: (sum(bytes_of(l) * len(s[l]) for l in moe) if limit_bytes is not None  # noqa: E731
                          else sum(len(s[l]) for l in moe))
        limit = limit_bytes if limit_bytes is not None else limit_experts
        if limit is None or limit <= 0 or used(start) > limit:
            raise GenError(f"the budget ({fmt_bytes(limit) if limit_bytes is not None else limit} experts) does not "
                           f"hold even the floor of {floor} experts per layer and the protected experts "
                           f"({fmt_bytes(used(start)) if limit_bytes is not None else used(start)})" +
                           (f"; the hardware leaves {fmt_bytes(hw['expert_budget'])} for experts" if hw else ""))
        cov = {l: start[l] | sel[l]["chosen"] for l in moe}
        if used(cov) <= limit:
            chosen = cov
        else:
            need = used(cov)
            msg = (f"the budget ({fmt_bytes(limit) if limit_bytes is not None else limit}) does not hold every "
                   f"component's min_coverage ({fmt_bytes(need) if limit_bytes is not None else need} needed)")
            if not p["allow_under_coverage"]:
                raise GenError(msg + "; --allow-under-coverage builds the best whitelist the budget holds and reports "
                               "the shortfall per component and layer")
            warnings.append(msg)
            chosen = {l: set(start[l]) for l in moe}
        heap = []
        unc = {}
        for l in moe:
            sc = scores[l]
            u = set()
            for c in comps:
                u |= {e for e in range(sc.E) if sc.count[c][e] < p["n_min"] and sc.m[c][e] > 0}
            unc[l] = u
            lw = p["layer_weights"].get(l, 1.0)
            for e in range(sc.E):
                if e in chosen[l] or (e in u and not p["include_uncertain"]):
                    continue
                util = sum(prof.comp_weight(c) * sc.m[c][e] for c in comps) * lw / (bytes_of(l) if limit_bytes is not None
                                                                                 else 1)
                if util > 0:
                    heapq.heappush(heap, (-util, l, e))
        total = used(chosen)
        while heap:
            _, l, e = heapq.heappop(heap)
            cost = bytes_of(l) if limit_bytes is not None else 1
            if total + cost > limit:
                continue
            chosen[l].add(e)
            total += cost
        budget_info = {"limit": limit, "unit": "bytes" if limit_bytes is not None else "experts", "used": total}
    else:
        chosen = {l: set(sel[l]["chosen"]) for l in moe}
    # ---- the floor, groups, and the report per layer and component
    allowed = {}
    blocked_groups = {}
    for l in moe:
        sc = scores[l]
        c = chosen[l]
        added = apply_floor(sc, c, floor)
        groups: list[int] = []
        while True:   # block thin groups, refill the floor from the open ones, until nothing changes
            new = apply_groups(c, sc.E, p["expert_groups"], p["group_topk"], top_k)
            if not new:
                break
            groups += new
            size = sc.E // p["expert_groups"]
            apply_floor(sc, c, floor, {e for g in groups for e in range(g * size, (g + 1) * size)})
        if groups:
            blocked_groups[l] = sorted(groups)
        allowed[l] = sorted(c)
        cov = {k: sum(sc.m[k][e] for e in c) for k in comps}
        for k in comps:
            if sum(sc.m[k]) > 0 and cov[k] < prof.min_coverage(k) - 1e-9:
                violations.append({"layer": l, "component": k, "coverage": cov[k], "min_coverage": prof.min_coverage(k)})
        ent = -sum(x * math.log(x) for x in sc.mp if x > 0) / math.log(sc.E) if sc.E > 1 else 0.0
        report_layers.append({
            "layer": l, "allowed": len(c), "experts": sc.E, "floor_added": added,
            "profile_mass": sum(sc.mp[e] for e in c), "reference_mass": sum(sc.mr[e] for e in c) if sc.has_ref else None,
            "coverage": cov, "uncertain": {k: len(v) for k, v in sel[l]["uncertain"].items()},
            "uncertain_ids": {k: v for k, v in sel[l]["uncertain"].items() if v},
            "bytes": len(c) * model.blob_bytes[l], "entropy": ent, "blocked_groups": blocked_groups.get(l, []),
        })
        if ent > p["flat_entropy"]:
            warnings.append(f"layer {l}: the profile's routing is nearly flat (normalised entropy {ent:.2f}): a "
                            "whitelist saves little there without losing coverage")
    if mode == "coverage" and coverage is not None:
        for c in comps:
            if c.startswith("lang:") and coverage < prof.min_coverage(c) - 1e-12:
                warnings.append(f"{c}: --coverage {coverage} is below the profile's min_coverage "
                                f"{prof.min_coverage(c)} (a sweep point, reported per layer below)")
    if mode != "budget" and violations and mode == "coverage" and coverage is None:
        warnings.append(f"{len(violations)} layer/component pairs miss their min_coverage (uncertain experts excluded?)")
    # tokens per component and per task: too few is a warning
    comp_tokens = {k: scores[moe[0]].tokens[k] for k in comps}
    for k, n in comp_tokens.items():
        if n == 0:
            warnings.append(f"{k}: the statistics hold no calibration data for it: it is not covered at all (collect "
                            "a prep that includes it and merge)")
        elif n < p["min_tokens"]:
            warnings.append(f"{k}: only {n} calibration tokens (fewer than {p['min_tokens']}): its experts are poorly "
                            "observed")
    task_tokens: dict[tuple[str, str], int] = {}
    ld0 = st.layers[moe[0]]
    for ci, key in enumerate(st.categories):
        cat = split_key(key)
        role, w = prof.role(cat)
        if role != "reference" and w > 0:
            task_tokens[(role, cat["task"])] = task_tokens.get((role, cat["task"]), 0) + ld0.tokens[ci]
    for (role, task), n in sorted(task_tokens.items()):
        if n < p["min_task_tokens"]:
            warnings.append(f"{role}, task {task}: only {n} calibration tokens (fewer than {p['min_task_tokens']})")
    if not any(scores[l].has_ref for l in moe):
        warnings.append("no reference data (excluded domains or a reference corpus): the contrast term is off, the "
                        "experts are ranked by profile mass alone")
    if prof.unknown["languages"]:
        warnings.append("calibration data in languages this profile does not target (counted as reference): " +
                        ", ".join(sorted(prof.unknown["languages"])))
    for what in ("tasks", "regions"):
        if prof.unknown[what]:
            warnings.append(f"{what} the profile does not weight (weight 0): " + ", ".join(sorted(prof.unknown[what])))
    # ---- the tiered mode's on_demand tier, the substitute mode's replacement table
    on_demand, substitutes, mode_report = {}, {}, None
    if wl_mode == "tiered":
        on_demand = select_on_demand(scores, comps, allowed, p["on_demand_coverage"])
        mode_report = tiered_report(scores, comps, model, allowed, on_demand, p)
    elif wl_mode == "substitute":
        substitutes = replacement_table(st, moe, allowed, p["substitute_min_tokens"])
        mode_report = substitute_report(scores, comps, allowed, substitutes, p)
    # ---- the manifest
    total_bytes = sum(model.blob_bytes[l] * model.experts_per_layer[l] for l in moe)
    kept_bytes = sum(model.blob_bytes[l] * len(allowed[l]) for l in moe)
    holdout = sorted({i for r in leaf_runs(st) for i in r.get("collect", {}).get("prep", {}).get("holdout_ids", [])})
    params = {k: v for k, v in p.items() if k not in ("gguf",)}
    params["alpha"] = p["alpha"] if p["alpha"] is not None else "1/(experts*100)"
    params["targets"] = targets
    params["floor"] = floor
    params["layer_weights"] = {str(k): v for k, v in sorted(p["layer_weights"].items())}
    for k in ("on_demand_coverage", "on_demand_cache", "substitute_threshold", "substitute_min_tokens"):
        if not (wl_mode == "tiered" and k.startswith("on_demand")) and not (wl_mode == "substitute" and k.startswith("subst")):
            params.pop(k)
    generation = {
        "profile": prof_doc["name"],
        "profiles": [{"path": str(x), "sha256": sha256_file(x)} for x in profiles],
        "scoring": {"method": SCORING_METHOD, "params": params},
        "calibration_tokens": sum(comp_tokens.values()),
        "created": st.meta.get("created", ""),
        "generator": GENERATOR,
        "stats": [{"path": str(stats), "run_id": st.meta["run_id"],
                   "meta_sha256": sha256_file(pathlib.Path(stats) / "meta.json")}],
        "quantization": st.meta.get("quantization", ""),
        "holdout_ids": holdout,
        "ablation": None,
    }
    if hw:
        generation["target_hardware"] = hw
    if wl_mode == "substitute":
        sm = st.meta["similarity"]
        generation["similarity"] = {k: sm.get(k) for k in ("fraction", "seed", "config_fnv1a64", "tokens", "weighting")}
    scores_out = {l: {e: round(scores[l].combined[e], 8) for e in allowed[l]} for l in moe}
    doc = W.manifest_document(model, allowed, mode=wl_mode, on_demand=on_demand or None, substitutes=substitutes or None,
                              scores=scores_out, generation=generation,
                              substitute_threshold=p["substitute_threshold"] if wl_mode == "substitute" else None)
    W.validate_manifest(json.dumps(doc), model)   # the engine's own checks, before anything is written
    report = {
        "kind": "strata-whitelist-report", "manifest": str(out), "mode": mode, "whitelist_mode": wl_mode,
        "params": params,
        "components": {k: {"min_coverage": prof.min_coverage(k), "target": targets[k], "weight": prof.comp_weight(k),
                           "tokens": comp_tokens[k],
                           "coverage_per_layer": {str(r["layer"]): r["coverage"][k] for r in report_layers},
                           "coverage_mean": sum(r["coverage"][k] for r in report_layers) / len(report_layers)}
                       for k in comps},
        "violations": violations, "budget": budget_info, "target_hardware": hw,
        "layers": report_layers,
        "totals": {"allowed": sum(len(v) for v in allowed.values()),
                   "experts": sum(model.experts_per_layer[l] for l in moe),
                   "expert_bytes_before": total_bytes, "expert_bytes_after": kept_bytes,
                   "bytes_note": "calculated from the pack index (blob bytes per expert), not measured"},
        "warnings": warnings,
    }
    if mode_report is not None:
        report[wl_mode] = mode_report
    if hw:
        report["fits"] = kept_bytes + (p["on_demand_cache"] if wl_mode == "tiered" else 0) <= hw["expert_budget"]
    if violations and mode == "budget" and not p["allow_under_coverage"]:
        raise GenError("internal: budget mode left a min_coverage unmet")
    outp = pathlib.Path(out)
    write_json(outp, doc)
    base = outp.with_suffix("")
    write_json(str(base) + ".report.json", report)
    atomic_write(str(base) + ".report.md", report_markdown(report, model))
    atomic_write(str(base) + ".coverage.csv", coverage_csv(scores, comps, moe))
    return report


def select_on_demand(scores: dict, comps: list[str], allowed: dict[int, list[int]], coverage: float) -> dict[int, list[int]]:
    """The tiered mode's second tier: per layer and component, experts in score order past the allowed ones until the
    component's mass reaches the second coverage; the union over the components.  Uncertain experts count too (they
    stay on the drive until the router picks them)."""
    out = {}
    for l, sc in scores.items():
        base = set(allowed[l])
        od: set[int] = set()
        for c in comps:
            m = sc.m[c]
            if sum(m) <= 0:
                continue
            acc = sum(m[e] for e in base | od)
            for e in sc.order(sc.s[c]):
                if acc >= coverage - 1e-12 or m[e] <= 0:
                    break
                if e in base or e in od:
                    continue
                od.add(e)
                acc += m[e]
        out[l] = sorted(od)
    return out


def tiered_report(scores: dict, comps: list[str], model: W.ModelInfo, allowed: dict, on_demand: dict, p: dict) -> dict:
    per_layer, per_tok = [], 0.0
    cov_r = {c: [] for c in comps}
    cov_a = {c: [] for c in comps}
    for l, sc in scores.items():
        tok = sum(sc.tokens[c] for c in comps)
        picks = sum(sc.count[c][e] for c in comps for e in on_demand[l]) / tok if tok else 0.0
        per_tok += picks * model.blob_bytes[l]
        row = {"layer": l, "on_demand": len(on_demand[l]), "bytes": len(on_demand[l]) * model.blob_bytes[l],
               "picks_per_token": picks, "coverage": {}}
        for c in comps:
            if sum(sc.m[c]) <= 0:
                continue
            r = sum(sc.m[c][e] for e in allowed[l])
            a = r + sum(sc.m[c][e] for e in on_demand[l])
            row["coverage"][c] = {"resident": r, "with_on_demand": a}
            cov_r[c].append(r)
            cov_a[c].append(a)
        per_layer.append(row)
    return {"on_demand_coverage": p["on_demand_coverage"], "cache": p["on_demand_cache"],
            "on_demand_experts": sum(len(v) for v in on_demand.values()),
            "on_demand_bytes": sum(len(on_demand[l]) * model.blob_bytes[l] for l in on_demand),
            "coverage_mean": {c: {"resident": mean(cov_r[c]), "with_on_demand": mean(cov_a[c])} for c in comps},
            "on_demand_bytes_per_token_max": per_tok, "layers": per_layer,
            "note": "calculated from the statistics: the bytes per token if every on_demand pick missed the cache"}


def replacement_table(st: Stats, moe: list[int], allowed: dict[int, list[int]], min_tokens: int) -> dict[int, list[tuple]]:
    """Per layer and blocked expert with enough similarity data: the most similar allowed expert, its mean cosine
    (clipped to [-1, 1]) and relative L2 distance.  All of them go into the manifest; the threshold decides at run time
    which are used (docs 3.1)."""
    out = {}
    for l in moe:
        a = set(allowed[l])
        rows = []
        for e in range(st.model.experts_per_layer[l]):
            if e in a:
                continue
            b = best_substitute(st.similarity.get(l), e, a, min_tokens)
            if b is not None:
                rows.append((e, b[0], max(-1.0, min(1.0, b[1])), b[2]))
        out[l] = rows
    return out


def substitute_report(scores: dict, comps: list[str], allowed: dict, substitutes: dict, p: dict) -> dict:
    thr = p["substitute_threshold"]
    per_layer = []
    share = {c: [] for c in comps}
    for l, sc in scores.items():
        a = set(allowed[l])
        used = {e for e, _, sim, _ in substitutes[l] if sim >= thr}
        blocked = [e for e in range(sc.E) if e not in a]
        row = {"layer": l, "blocked": len(blocked), "listed": len(substitutes[l]), "at_threshold": len(used),
               "without_data": len(blocked) - len(substitutes[l]), "blocked_mass_with_substitute": {}}
        for c in comps:
            bm = sum(sc.m[c][e] for e in blocked)
            if sum(sc.m[c]) > 0 and bm > 0:
                v = sum(sc.m[c][e] for e in used) / bm
                row["blocked_mass_with_substitute"][c] = v
                share[c].append(v)
        per_layer.append(row)
    return {"threshold": thr, "min_tokens": p["substitute_min_tokens"],
            "listed": sum(r["listed"] for r in per_layer), "at_threshold": sum(r["at_threshold"] for r in per_layer),
            "without_data": sum(r["without_data"] for r in per_layer),
            "blocked_mass_with_substitute_mean": {c: mean(v) for c, v in share.items()}, "layers": per_layer,
            "note": "every blocked expert with enough similarity data is listed with its most similar allowed expert; "
                    "the engine uses the ones at or above the threshold"}


def coverage_csv(scores: dict, comps: list[str], moe: list[int]) -> str:
    """The coverage curve: per layer, the profile mass (and each component's) of the n best experts by combined score."""
    lines = ["layer,experts,profile_mass," + ",".join(c.replace(",", "_") for c in comps)]
    for l in moe:
        sc = scores[l]
        acc = 0.0
        accs = {c: 0.0 for c in comps}
        for n, e in enumerate(sc.order(sc.combined), 1):
            acc += sc.mp[e]
            for c in comps:
                accs[c] += sc.m[c][e]
            lines.append(f"{l},{n},{acc:.6f}," + ",".join(f"{accs[c]:.6f}" for c in comps))
    return "\n".join(lines) + "\n"


def report_markdown(r: dict, model: W.ModelInfo) -> str:
    t = r["totals"]
    out = [f"# Whitelist report: {pathlib.Path(r['manifest']).name}", "",
           f"Selection {r['mode']}, whitelist mode {r.get('whitelist_mode', 'strict')}. Allowed {t['allowed']} of "
           f"{t['experts']} experts; expert weights "
           f"{fmt_bytes(t['expert_bytes_after'])} of {fmt_bytes(t['expert_bytes_before'])} "
           f"({t['bytes_note']}).", ""]
    if r.get("target_hardware"):
        h = r["target_hardware"]
        out += ["## Target hardware (a calculation, not a measurement)", "",
                f"VRAM {fmt_bytes(h['vram'])} + RAM {fmt_bytes(h['ram'])} - reserve {fmt_bytes(h['reserve'])} - "
                f"non-expert weights {fmt_bytes(h['non_expert_weights'])} ({h['non_expert_how']}) - KV cache "
                f"{fmt_bytes(h['kv_cache'])} "
                f"({h['context']} tokens; {h['kv_how']}) - runtime buffer {fmt_bytes(h['runtime_buffer'])} = "
                f"**{fmt_bytes(h['expert_budget'])} for experts**.", "",
                f"The whitelist's experts take {fmt_bytes(t['expert_bytes_after'])}: "
                f"{'it fits' if r.get('fits') else 'it does NOT fit'}.", ""]
    out += ["## Coverage per component", "",
            "| component | weight | tokens | target | min_coverage | mean coverage | lowest layer |",
            "|---|---|---|---|---|---|---|"]
    for k, c in r["components"].items():
        low = min(c["coverage_per_layer"].items(), key=lambda kv: kv[1]) if c["coverage_per_layer"] else ("-", 0)
        out.append(f"| {k} | {c['weight']} | {c['tokens']} | {c['target']} | {c['min_coverage']} | "
                   f"{c['coverage_mean']:.4f} | {low[0]} ({low[1]:.4f}) |")
    if r["violations"]:
        out += ["", "## Below min_coverage", "", "| layer | component | coverage | min |", "|---|---|---|---|"]
        out += [f"| {v['layer']} | {v['component']} | {v['coverage']:.4f} | {v['min_coverage']} |" for v in r["violations"]]
    out += ["", "## Per layer", "", "| layer | allowed | profile mass | reference mass | uncertain | bytes |",
            "|---|---|---|---|---|---|"]
    for L in r["layers"]:
        rm = "-" if L["reference_mass"] is None else f"{L['reference_mass']:.4f}"
        out.append(f"| {L['layer']} | {L['allowed']}/{L['experts']} | {L['profile_mass']:.4f} | {rm} | "
                   f"{sum(L['uncertain'].values())} | {fmt_bytes(L['bytes'])} |")
    if r.get("tiered"):
        t = r["tiered"]
        out += ["", "## Tiered: the on_demand tier", "",
                f"{t['on_demand_experts']} experts on_demand ({fmt_bytes(t['on_demand_bytes'])} on the drive) for a second "
                f"coverage of {t['on_demand_coverage']}; an on_demand cache of {fmt_bytes(t['cache'])}; at most "
                f"{fmt_bytes(t['on_demand_bytes_per_token_max'])} read per token ({t['note']}).", "",
                "| component | coverage resident | with on_demand |", "|---|---|---|"]
        out += [f"| {c} | {v['resident']:.4f} | {v['with_on_demand']:.4f} |" for c, v in t["coverage_mean"].items()
                if v["resident"] is not None]
    if r.get("substitute"):
        t = r["substitute"]
        out += ["", "## Substitute: the replacement table", "",
                f"{t['listed']} blocked experts listed with their most similar allowed expert, {t['at_threshold']} at or "
                f"above the threshold {t['threshold']} (used), {t['without_data']} without enough similarity data "
                f"(fewer than {t['min_tokens']} sampled picks).", "",
                "| component | share of the blocked mass with a replacement in use |", "|---|---|"]
        out += [f"| {c} | {v:.4f} |" for c, v in t["blocked_mass_with_substitute_mean"].items() if v is not None]
    if r["warnings"]:
        out += ["", "## Warnings", ""] + [f"- {w}" for w in r["warnings"]]
    return "\n".join(out) + "\n"


# ================================ INSPECT ================================

def inspect(path: str) -> str:
    doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if doc.get("kind") != W.MANIFEST_KIND:
        raise GenError(f"{path}: not an expert whitelist manifest")
    layers = doc["layers"]
    epl = doc["experts_per_layer"]
    per = [len(L["allowed"]) for L in layers]
    od = sum(len(L.get("on_demand", [])) for L in layers)
    total = sum((epl if isinstance(epl, int) else epl[L["layer"]]) for L in layers)
    gen = doc.get("generation", {})
    lines = [f"{path}",
             f"  model fingerprint {doc['model_fingerprint']}, {doc['num_layers']} layers ({len(layers)} with experts), "
             f"top-{doc['top_k']}, mode {doc.get('mode', 'strict')}",
             f"  allowed {sum(per)} of {total} experts ({100.0 * sum(per) / max(total, 1):.1f}%); per layer min "
             f"{min(per)}, median {sorted(per)[len(per) // 2]}, max {max(per)}" + (f"; on_demand {od}" if od else ""),
             f"  profile {gen.get('profile', '?')}, scoring {gen.get('scoring', {}).get('method', '?')}, "
             f"{gen.get('calibration_tokens', '?')} calibration tokens, quantization {gen.get('quantization') or '?'}",
             ]
    sp = gen.get("scoring", {}).get("params", {})
    if sp:
        keys = ("mode", "coverage", "count", "fraction", "lambda", "clip", "protect_mass", "n_min", "margin")
        lines.append("  parameters: " + ", ".join(f"{k}={sp[k]}" for k in keys if k in sp and sp[k] is not None))
    if doc.get("mode") == "substitute":
        listed = sum(len(L.get("substitutes", [])) for L in layers)
        thr = doc.get("substitute_threshold")
        used = sum(1 for L in layers for x in L.get("substitutes", [])
                   if thr is None or x.get("similarity") is None or x["similarity"] >= thr)
        lines.append(f"  substitutes: {listed} listed, {used} used" + (f" (similarity threshold {thr})" if thr is not None
                                                                        else ""))
    hw = gen.get("target_hardware")
    if hw:
        lines.append(f"  target hardware: {fmt_bytes(hw['expert_budget'])} for experts (a calculation)")
    if gen.get("holdout_ids"):
        lines.append(f"  holdout: {len(gen['holdout_ids'])} sample ids recorded for eval")
    rep = pathlib.Path(path).with_suffix("").as_posix() + ".report.json"
    if pathlib.Path(rep).exists():
        r = json.loads(pathlib.Path(rep).read_text())
        for k, c in r["components"].items():
            lines.append(f"  {k}: mean coverage {c['coverage_mean']:.4f} (min_coverage {c['min_coverage']}, "
                         f"{c['tokens']} tokens)")
        t = r["totals"]
        lines.append(f"  expert weights {fmt_bytes(t['expert_bytes_after'])} of {fmt_bytes(t['expert_bytes_before'])} "
                     "(calculated from the pack index)")
        for w in r["warnings"][:8]:
            lines.append(f"  warning: {w}")
    lines.append("  layer: allowed experts")
    for L in layers:
        ids = L["allowed"]
        shown = ", ".join(map(str, ids[:24])) + (", ..." if len(ids) > 24 else "")
        lines.append(f"    {L['layer']:3d}: {len(ids):4d}  [{shown}]")
    return "\n".join(lines) + "\n"


# ================================ MODEL METADATA AND ESTIMATE ================================

def model_numbers(cfg: dict | None, pack: str | None, gguf: str | None) -> dict:
    """What estimate needs, from the pack's expert index and the GGUF's metadata (no weights are read)."""
    if cfg is not None:
        p2, g2 = model_paths(cfg)
        pack, gguf = pack or p2, gguf or g2
    if not pack:
        raise GenError("estimate needs the model: --model CFG or --pack PACK (and --gguf for the dense weights)")
    info = W.pack_model_info(pack, gguf=gguf)
    n = {"layers": info.n_layers, "experts_per_layer": info.experts_per_layer, "top_k": info.top_k,
         "hidden": info.hidden, "blob_bytes": info.blob_bytes, "fingerprint": info.fingerprint(),
         "row_floats": info.hidden, "dense_per_layer": [0] * info.n_layers, "dense_global": 0, "dense_how": ""}
    if gguf:
        import gguf_reader  # noqa: E402
        f = gguf_reader.GGUFFile(pathlib.Path(gguf))
        arch = f.metadata.get("general.architecture", "")
        hc = int(f.metadata.get(f"{arch}.hyper_connection.count", 0) or 0)
        n["row_floats"] = info.hidden * max(hc, 1)
        for sh in gguf_shards(gguf):
            g = f if sh == pathlib.Path(gguf) else gguf_reader.GGUFFile(sh)
            for t in g.tensors:
                if "_exps" in t.name or t.name in DRIVE_TENSORS:
                    continue
                b = t.expected_bytes() or t.elements * 2
                if t.name.startswith("blk."):
                    li = int(t.name.split(".")[1])
                    if li < info.n_layers:
                        n["dense_per_layer"][li] += b
                        continue
                n["dense_global"] += b
        n["dense_how"] = "the GGUF's non-expert tensors"
    else:
        d = pathlib.Path(pack) / "dense.bin"
        if d.exists():
            per = d.stat().st_size // max(info.n_layers, 1)
            n["dense_per_layer"] = [per] * info.n_layers
            n["dense_how"] = "the pack's dense.bin spread evenly over the layers (give --gguf for the real split)"
    n["dense_bytes"] = sum(n["dense_per_layer"]) + n["dense_global"]
    n["expert_bytes"] = sum(b * e for b, e in zip(info.blob_bytes, info.experts_per_layer))
    return n


def plan_layerwise(n: dict, lengths: list[int], n_cat: int, vram: int, ram: int, block_tokens: int, scratch_max: int,
                   spill_bytes: int, prefetch: bool) -> dict:
    """The scheduler's plan (src/core/layerwise_calib.cpp plan_layerwise, split form), from the numbers alone."""
    R, X, K = n["row_floats"], n["hidden"], n["top_k"]
    longest, total = max(lengths), sum(lengths)
    acc = max((8 * n_cat + 8 * n_cat * e * (4 + K) for e in n["experts_per_layer"] if e > 0), default=0)
    pf = max(n["dense_per_layer"]) if prefetch else 0
    fixed = 2 * acc + longest * R * (8 + spill_bytes) + pf
    per_tok = (R + 2 * X) * 4 + K * 8
    problems = []
    max_block = None
    if ram:
        if fixed >= ram:
            problems.append(f"the RAM budget of {fmt_bytes(ram)} is below the {fmt_bytes(fixed)} every block needs")
            max_block = 0
        else:
            max_block = (ram - fixed) // per_tok
    if scratch_max:
        by = scratch_max // (2 * R * spill_bytes)
        max_block = by if max_block is None else min(max_block, by)
    B = block_tokens or (max_block if max_block is not None else max(min(total, DEFAULT_BLOCK_TOKENS), longest))
    if block_tokens and max_block is not None and block_tokens > max_block:
        problems.append(f"a block of {block_tokens} tokens needs more than the budgets hold ({max_block} tokens)")
    if longest > B:
        problems.append(f"the longest sample has {longest} tokens, more than a block ({B}): raise the RAM budget or "
                        "shorten the samples")
    B = max(B, longest)
    blocks, cur, largest = [], 0, 0
    for ln in lengths:
        if cur and cur + ln > B:
            blocks.append(cur)
            largest = max(largest, cur)
            cur = 0
        cur += ln
    if cur:
        blocks.append(cur)
        largest = max(largest, cur)
    vram_need = 0
    for l, E in enumerate(n["experts_per_layer"]):
        lb, eb = n["dense_per_layer"][l], (n["blob_bytes"][l] if E > 0 else 0)
        vram_need = max(vram_need, lb + max(longest * (R + 2 * X) * 4, eb + 2 * X * 4 + K * 8 if E else 0))
    if vram and vram_need > vram:
        problems.append(f"the VRAM budget of {fmt_bytes(vram)} is below the {fmt_bytes(vram_need)} the largest layer "
                        "needs with one expert and the longest sample")
    scratch = 2 * largest * R * spill_bytes
    if scratch_max and scratch > scratch_max:
        problems.append(f"the scratch files need {fmt_bytes(scratch)}, more than --scratch-max")
    dense_read = expert_upper = expert_exp = 0
    for toks in blocks:
        for l, E in enumerate(n["experts_per_layer"]):
            dense_read += n["dense_per_layer"][l]
            if E:
                expert_upper += E * n["blob_bytes"][l]
                expert_exp += E * (1 - (1 - K / E) ** toks) * n["blob_bytes"][l]
    return {"block_tokens": B, "blocks": len(blocks), "max_block_tokens": max_block, "ram_need": fixed + per_tok * largest,
            "vram_need": vram_need, "scratch_need": scratch, "dense_read": dense_read, "expert_read_upper": expert_upper,
            "expert_read_expected": int(expert_exp), "problems": problems}


def estimate(prep: str, model: str | None = None, pack: str | None = None, gguf: str | None = None,
             vram_budget: str | None = None, ram_budget: str | None = None, scratch: str | None = None,
             scratch_max: str | None = None, block_tokens: int = 0, spill: str = "f32", prefetch: bool = False,
             nvme_gbps: float = 2.0, prefill_tps: float | None = None, holdout: bool = False) -> dict:
    cfg = load_model_config(model)
    n = model_numbers(cfg, pack, gguf)
    if not any(e > 0 for e in n["experts_per_layer"]):
        raise GenError("the model has no MoE layer: there are no experts to collect statistics for")
    pr = read_prep(prep)
    cats, samples = read_samples(pathlib.Path(prep) / ("holdout.bin" if holdout else "samples.bin"))
    if not samples:
        raise GenError(f"{prep}: no samples")
    lengths = [len(s["tokens"]) for s in samples]
    vram, ram = parse_size(vram_budget or 0), parse_size(ram_budget or 0)
    sb = {"f32": 4, "f16": 2, "bf16": 2}[spill]
    lw = plan_layerwise(n, lengths, len(cats), vram, ram, block_tokens, parse_size(scratch_max or 0), sb, prefetch)
    tokens = sum(lengths)
    K = n["top_k"]
    moe = [l for l, e in enumerate(n["experts_per_layer"]) if e > 0]
    # token mode: the model's weights in VRAM and RAM; what does not fit is read from the drive for the experts routed
    model_bytes = n["dense_bytes"] + n["expert_bytes"]
    fits = (not vram or n["dense_bytes"] <= vram) and (not (vram or ram) or model_bytes <= vram + ram)
    miss = 0.0 if fits or not (vram or ram) else max(0.0, 1.0 - max(vram + ram - n["dense_bytes"], 0) / n["expert_bytes"])
    token_reads = tokens * sum(K * n["blob_bytes"][l] for l in moe) * miss
    out = {
        "model": {k: n[k] for k in ("fingerprint", "layers", "top_k", "hidden", "row_floats", "dense_bytes",
                                    "expert_bytes", "dense_how")},
        "prep": {"dir": prep, "samples": len(samples), "tokens": tokens, "longest": max(lengths),
                 "profile": pr.get("profile_name", "")},
        "budgets": {"vram": vram, "ram": ram, "scratch_max": parse_size(scratch_max or 0), "spill": spill},
        "token_mode": {"fits": fits, "drive_reads": int(token_reads),
                       "note": "the whole model in VRAM + RAM: no expert is read twice from the drive" if fits else
                       "the model does not fit: every routed expert that is not in RAM is read from the drive again for "
                       "every token"},
        "layerwise": lw,
        "assumptions": {"nvme_gbps": nvme_gbps, "prefill_tps": prefill_tps,
                        "note": "times are estimates from these assumptions, not measurements"},
    }
    io_lw = (lw["dense_read"] + lw["expert_read_expected"]) / (nvme_gbps * 1e9)
    io_tok = token_reads / (nvme_gbps * 1e9)
    out["layerwise"]["read_seconds"] = io_lw
    out["token_mode"]["read_seconds"] = io_tok
    if prefill_tps:
        out["compute_seconds"] = tokens / prefill_tps
    out["recommended_mode"] = "token" if fits else "layerwise"
    if scratch:
        try:
            free = shutil.disk_usage(scratch).free
        except OSError:
            free = None
        out["layerwise"]["scratch_free"] = free
        if free is not None and free < lw["scratch_need"]:
            lw["problems"].append(f"{scratch} has {fmt_bytes(free)} free, the scratch files need "
                                  f"{fmt_bytes(lw['scratch_need'])}")
    out["ok"] = fits or not lw["problems"]
    return out


def estimate_text(e: dict) -> str:
    m, lw, t = e["model"], e["layerwise"], e["token_mode"]
    lines = [f"model {m['fingerprint']}: {m['layers']} layers, top-{m['top_k']}, non-expert weights "
             f"{fmt_bytes(m['dense_bytes'])}" + (f" ({m['dense_how']})" if m["dense_how"] else "") +
             f", experts {fmt_bytes(m['expert_bytes'])}",
             f"samples: {e['prep']['samples']} ({e['prep']['tokens']} tokens, longest {e['prep']['longest']})",
             f"token mode: {'fits' if t['fits'] else 'does not fit'} the budgets; drive reads about "
             f"{fmt_bytes(t['drive_reads'])} (~{t['read_seconds']:.0f} s at {e['assumptions']['nvme_gbps']} GB/s)",
             f"layerwise: {lw['blocks']} blocks of up to {lw['block_tokens']} tokens; RAM {fmt_bytes(lw['ram_need'])}, "
             f"VRAM {fmt_bytes(lw['vram_need'])}, scratch {fmt_bytes(lw['scratch_need'])}; reads "
             f"{fmt_bytes(lw['dense_read'] + lw['expert_read_expected'])} expected (at most "
             f"{fmt_bytes(lw['dense_read'] + lw['expert_read_upper'])}), ~{lw['read_seconds']:.0f} s",
             ]
    if "compute_seconds" in e:
        lines.append(f"compute: ~{e['compute_seconds']:.0f} s at {e['assumptions']['prefill_tps']} tokens/s (given)")
    for p in lw["problems"]:
        lines.append(f"layerwise does not fit: {p}")
    lines.append(f"recommended: --mode {e['recommended_mode']}" + ("" if e["ok"] else "  (the budgets are too small: see "
                                                                                     "above)"))
    lines.append("(estimates from the model's metadata and the assumptions above, not measurements)")
    return "\n".join(lines) + "\n"


# ================================ COLLECT (THE ENGINE) ================================

DROP_FLAG = {"--serve", "--expert-whitelist-stats", "--resident-experts", "--resident-cpu-experts"}
DROP_VALUE = {"--tokens", "--max-new", "--expert-whitelist", "--expert-whitelist-mode", "--draft-expert-whitelist"}


def engine_command(cfg: dict, samples: str, out: str, mode: str, meta_file: str, resume: bool, p: dict) -> list[str]:
    args, i, base = [], 0, [str(a) for a in cfg.get("args", [])]
    while i < len(base):
        if base[i] in DROP_FLAG:
            i += 1
            continue
        if base[i] in DROP_VALUE:
            i += 2
            continue
        args.append(base[i])
        i += 1
    cmd = [cfg["exe"], *args, "--routing-stats-out", out, "--routing-stats-samples", samples,
           "--routing-stats-mode", mode, "--routing-stats-meta", meta_file]
    for key, flag in (("vram_budget", "--routing-stats-vram-budget"), ("ram_budget", "--routing-stats-ram-budget"),
                      ("block_tokens", "--routing-stats-block-tokens"), ("scratch", "--routing-stats-scratch"),
                      ("scratch_max", "--routing-stats-scratch-max"), ("spill", "--routing-stats-spill"),
                      ("checkpoint_every", "--routing-stats-checkpoint-every"),
                      ("similarity", "--routing-stats-similarity"), ("similarity_seed", "--routing-stats-similarity-seed")):
        if p.get(key) not in (None, "", 0):
            cmd += [flag, str(p[key])]
    if p.get("prefetch"):
        cmd.append("--routing-stats-prefetch")
    if resume:
        cmd.append("--routing-stats-resume")
    return cmd


def collect(model: str, prep: str, out: str, mode: str = "auto", resume: bool = False, print_command: bool = False,
            samples_file: str | None = None, **p) -> int:
    cfg = load_model_config(model)
    pr = read_prep(prep)
    samples = samples_file or str(pathlib.Path(prep) / "samples.bin")
    # the estimate first: what the run needs, and the mode when --mode auto (it needs the model's pack)
    try:
        e = estimate(prep, model=model, vram_budget=p.get("vram_budget"), ram_budget=p.get("ram_budget"),
                     scratch=p.get("scratch"), scratch_max=p.get("scratch_max"), block_tokens=p.get("block_tokens") or 0,
                     spill=p.get("spill") or "f32")
    except (GenError, W.WhitelistError, OSError) as ex:
        if mode == "auto":
            raise GenError(f"--mode auto needs the estimate, which failed: {ex}; give --mode token or layerwise") from None
        e = None
        log(f"collect: no estimate ({ex})")
    if e is not None:
        for line in estimate_text(e).splitlines():
            log(f"collect: {line}")
        if mode == "auto":
            mode = e["recommended_mode"]
            log(f"collect: --mode auto chose {mode}")
        if mode == "layerwise" and e["layerwise"]["problems"]:
            raise GenError("the layer-wise run does not fit the budgets: " + "; ".join(e["layerwise"]["problems"]))
        if mode == "token" and not e["token_mode"]["fits"]:
            log("collect: warning: the model does not fit the budgets: token mode reads routed experts from the drive "
                "again for every token (--mode layerwise avoids that)")
    if mode not in ("token", "layerwise"):
        raise GenError("--mode is token, layerwise or auto")
    if p.get("similarity") is not None:
        if not 0 < p["similarity"] <= 1:
            raise GenError("--similarity takes a share above 0 and up to 1 (0.01: 1% of the tokens)")
        _, ss = read_samples(samples)
        seed = p.get("similarity_seed") or 1
        n = sum(similarity_sampled(p["similarity"], seed, x["id"], i) for x in ss for i in range(len(x["tokens"])))
        log(f"collect: similarity data on {n} sampled tokens: every expert of every MoE layer is computed on the CPU for "
            "each (one routed expert's work per expert and layer per token)")
    outp = pathlib.Path(out)
    meta = {"generator": GENERATOR,
            "prep": {"dir": str(prep), "prep_sha256": sha256_file(pathlib.Path(prep) / "prep.json"),
                     "samples": samples, "samples_sha256": sha256_file(samples),
                     "holdout_sha256": pr["holdout"]["sha256"], "holdout_ids": pr["holdout"]["ids"]},
            "profiles": pr.get("profiles", []), "tokenizer": pr.get("tokenizer", {}),
            "chat_template": pr.get("chat_template", {}), "seed": pr["params"]["seed"],
            "params": {k: v for k, v in sorted(p.items()) if v not in (None, "", False, 0)},
            "mode": mode, "engine": {"exe": cfg["exe"], "args": cfg.get("args", [])}}
    outp.mkdir(parents=True, exist_ok=True)
    meta_file = outp / "collect.json"
    write_json(meta_file, meta)
    cmd = engine_command(cfg, samples, str(outp), mode, str(meta_file), resume, p)
    if print_command:
        print(" ".join(cmd))
        return 0
    env = dict(os.environ)
    env.update(cfg.get("env", {}))
    if cfg.get("lib_dirs"):
        env["LD_LIBRARY_PATH"] = os.pathsep.join(cfg["lib_dirs"] + [env.get("LD_LIBRARY_PATH", "")])
    log("collect: " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=cfg.get("cwd"), env=env)
    if r.returncode != 0:
        raise GenError(f"the engine stopped with exit code {r.returncode}; an interrupted run continues with --resume")
    st = read_stats(str(outp))
    log(f"collect: {st.meta['samples']} samples, {st.meta['tokens']} tokens, {len(st.layers)} MoE layers -> {out}")
    return 0


# ================================ ANALYZE ================================

ANALYZE_LEVELS = (0.8, 0.9, 0.95, 0.99)


def ranked(m: list[float]) -> list[int]:
    """Expert ids by mass, largest first; a tie goes to the lower id."""
    return sorted(range(len(m)), key=lambda e: (-m[e], e))


def experts_for(m: list[float], level: float) -> int:
    """How many experts, largest mass first, carry `level` of the normalised distribution `m`."""
    acc = 0.0
    for n, e in enumerate(ranked(m), 1):
        acc += m[e]
        if acc >= level - 1e-12:
            return n
    return len(m)


def top_set(m: list[float], level: float) -> set[int]:
    return set(ranked(m)[:experts_for(m, level)])


def entropy(m: list[float]) -> tuple[float, float]:
    """(normalised entropy in [0, 1], effective experts exp(H)) of a normalised distribution."""
    h = -sum(x * math.log(x) for x in m if x > 0)
    return (h / math.log(len(m)) if len(m) > 1 else 0.0), math.exp(h)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def mass_overlap(a: list[float], b: list[float]) -> float:
    """The sum over the experts of min(a, b): 1 for the same distribution, 0 for disjoint ones."""
    return sum(min(x, y) for x, y in zip(a, b))


def mean(v: list[float]) -> float | None:
    return sum(v) / len(v) if v else None


def subsample(preps: list[str], tokens_per_prep: int) -> tuple[list[str], list[dict], int]:
    """Up to N tokens of each prep (a fixed subset by the hash of the id), the category tables united; and the seed."""
    keys: set[str] = set()
    chosen = []
    seed = 0
    for prep in preps:
        cats, samples = read_samples(pathlib.Path(prep) / "samples.bin")
        seed = read_prep(prep)["params"]["seed"]
        samples.sort(key=lambda s: hash_fraction(f"analyze:{seed}", s["id"]))
        n = 0
        for s in samples:
            if tokens_per_prep and n >= tokens_per_prep:
                break
            chosen.append((cats, s))
            keys.update(cats[c] for c in s["cats"])
            n += len(s["tokens"])
    allk = sorted(keys)
    idx = {k: i for i, k in enumerate(allk)}
    seen = set()
    rows = []
    for cats, s in chosen:
        if s["id"] in seen:
            continue
        seen.add(s["id"])
        rows.append({"id": s["id"], "tokens": s["tokens"], "cats": [idx[cats[c]] for c in s["cats"]]})
    return allk, rows, seed


def write_sub_prep(d: pathlib.Path, preps: list[str], keys: list[str], rows: list[dict]) -> str:
    """A prep folder holding `rows` (no holdout): what collect reads."""
    d.mkdir(parents=True, exist_ok=True)
    write_samples(d / "samples.bin", keys, rows)
    write_samples(d / "holdout.bin", keys, [])
    doc = dict(read_prep(preps[0]))
    doc.pop("dir", None)
    doc.update(samples={"file": "samples.bin", "sha256": sha256_file(d / "samples.bin"), "count": len(rows),
                        "tokens": sum(len(r["tokens"]) for r in rows)},
               holdout={"file": "holdout.bin", "sha256": sha256_file(d / "holdout.bin"), "ids": [], "count": 0, "tokens": 0},
               categories=keys, inputs=[{"prep": x, "sha256": sha256_file(pathlib.Path(x) / "prep.json")} for x in preps])
    write_json(d / "prep.json", doc)
    return str(d)


def fill_budget(scores: dict, model: W.ModelInfo, budget: int, floor: int,
                layer_weights: dict | None = None) -> tuple[dict[int, set[int]] | None, int]:
    """The resident experts a byte budget holds: per layer the floor (the best by the profile's mass), then the expert
    with the highest mass per byte across all layers, one at a time (build's budget mode without the coverages).
    (None, bytes) when the floor alone does not fit."""
    chosen, used = {}, 0
    for l, sc in scores.items():
        chosen[l] = set(ranked(sc.combined_mass)[:min(floor, sc.E)])
        used += len(chosen[l]) * model.blob_bytes[l]
    if used > budget:
        return None, used
    heap = [(-(sc.combined_mass[e] * (layer_weights or {}).get(l, 1.0) / model.blob_bytes[l]), l, e)
            for l, sc in scores.items() for e in range(sc.E) if e not in chosen[l] and sc.combined_mass[e] > 0]
    heapq.heapify(heap)
    while heap:
        _, l, e = heapq.heappop(heap)
        if used + model.blob_bytes[l] <= budget:
            chosen[l].add(e)
            used += model.blob_bytes[l]
    return chosen, used


def coverage_of(scores: dict, comps: list[str], sets: dict[int, set[int]]) -> dict:
    """Per component: the mass the sets cover, per layer, on average and in the lowest layer."""
    out = {}
    for c in comps:
        per = {l: sum(sc.m[c][e] for e in sets[l]) for l, sc in scores.items() if sum(sc.m[c]) > 0}
        out[c] = {"mean": mean(list(per.values())), "min": min(per.values()) if per else None,
                  "per_layer": {str(l): v for l, v in per.items()}}
    return out


def best_substitute(sd: SimData | None, e: int, allowed: set[int], min_tokens: int) -> tuple[int, float, float] | None:
    """The allowed expert whose output is most like blocked expert e's (mean cosine; a tie goes to the smaller relative
    L2 distance, then the lower id): (expert, cosine, relative L2), or None without enough data."""
    if sd is None or sd.row_tokens(e) < max(min_tokens, 1) or not allowed:
        return None
    j = min(allowed, key=lambda j: (-sd.mean_cos(e, j), sd.mean_rel(e, j), j))
    return j, sd.mean_cos(e, j), sd.mean_rel(e, j)


def project_modes(st: Stats, scores: dict, comps: list[str], budget: int, p: dict) -> dict:
    """What the budget holds resident in each mode (docs 3.1): the coverage of every component and the routing mass
    that would fall on blocked or on_demand experts.  Calculations from the sample's routing, not measurements."""
    model = st.model
    floor = model.top_k + p.get("margin", 2)
    out = {"budget": budget, "floor": floor,
           "note": "a calculation from the sample's routing and the pack's blob sizes, not a measurement"}
    res, used = fill_budget(scores, model, budget, floor)
    if res is None:
        out["problem"] = (f"the budget of {fmt_bytes(budget)} does not hold the floor of {floor} experts per layer "
                          f"({fmt_bytes(used)})")
        return out
    cov = coverage_of(scores, comps, res)
    out["strict"] = {"resident_experts": sum(len(v) for v in res.values()), "resident_bytes": used, "coverage": cov,
                     "blocked_mass": {c: (1.0 - v["mean"]) if v["mean"] is not None else None for c, v in cov.items()},
                     "resident": {str(l): sorted(v) for l, v in res.items()}}
    # substitute: the same resident set; how much of the blocked mass has a replacement at or above the threshold
    sub = {"resident_experts": out["strict"]["resident_experts"], "resident_bytes": used, "coverage": cov,
           "blocked_mass": out["strict"]["blocked_mass"]}
    thr, min_tok = p.get("substitute_threshold", 0.5), p.get("substitute_min_tokens", 20)
    if st.similarity:
        share: dict[str, list[float]] = {c: [] for c in comps}
        n_sub = n_blocked = 0
        for l, sc in scores.items():
            have = set()
            for e in range(sc.E):
                if e in res[l]:
                    continue
                n_blocked += 1
                b = best_substitute(st.similarity.get(l), e, res[l], min_tok)
                if b is not None and b[1] >= thr - 1e-12:
                    have.add(e)
            n_sub += len(have)
            for c in comps:
                blocked = sum(sc.m[c][e] for e in range(sc.E) if e not in res[l])
                if sum(sc.m[c]) > 0 and blocked > 0:
                    share[c].append(sum(sc.m[c][e] for e in have) / blocked)
        sub.update(threshold=thr, min_tokens=min_tok, blocked_experts=n_blocked, with_substitute=n_sub,
                   blocked_mass_with_substitute={c: mean(v) for c, v in share.items()})
    else:
        sub["note"] = ("no similarity data in these statistics: the share of the blocked mass a replacement would take "
                       "needs it (docs 7.3)")
    out["substitute"] = sub
    # tiered: the budget less the on_demand cache resident, then the experts that reach the second coverage on_demand
    cache, cov2 = p.get("on_demand_cache", DEFAULT_ON_DEMAND_CACHE), p.get("on_demand_coverage", 0.995)
    res2, used2 = fill_budget(scores, model, budget - cache, floor) if budget > cache else (None, 0)
    if res2 is None:
        out["tiered"] = {"problem": f"the budget less the on_demand cache ({fmt_bytes(cache)}) does not hold the floor"}
        return out
    od = {}
    for l, sc in scores.items():
        od[l] = set()
        for c in comps:
            m = sc.m[c]
            acc = sum(m[e] for e in res2[l] | od[l])
            for e in ranked(m):
                if acc >= cov2 - 1e-12 or m[e] <= 0:
                    break
                if e in res2[l] or e in od[l]:
                    continue
                od[l].add(e)
                acc += m[e]
    cov_r = coverage_of(scores, comps, res2)
    cov_all = coverage_of(scores, comps, {l: res2[l] | od[l] for l in scores})
    picks, per_tok = {}, 0.0
    for l, sc in scores.items():
        tok = sum(sc.tokens[c] for c in comps)
        picks[l] = sum(sc.count[c][e] for c in comps for e in od[l]) / tok if tok else 0.0
        per_tok += picks[l] * model.blob_bytes[l]
    out["tiered"] = {
        "cache": cache, "on_demand_coverage": cov2, "resident_experts": sum(len(v) for v in res2.values()),
        "resident_bytes": used2, "on_demand_experts": sum(len(v) for v in od.values()),
        "on_demand_bytes": sum(len(od[l]) * model.blob_bytes[l] for l in od),
        "coverage_resident": cov_r, "coverage_with_on_demand": cov_all,
        "on_demand_mass": {c: (cov_all[c]["mean"] - cov_r[c]["mean"]) if cov_r[c]["mean"] is not None else None
                           for c in comps},
        "blocked_mass": {c: (1.0 - cov_all[c]["mean"]) if cov_all[c]["mean"] is not None else None for c in comps},
        "on_demand_picks_per_token": {str(l): v for l, v in picks.items()},
        "on_demand_bytes_per_token_max": per_tok,
        "bytes_note": "every on_demand pick read from the drive, no cache hit: an upper bound",
    }
    return out


def analyze_stats(st: Stats, profiles: list[str], hw: dict | None, context: int, p: dict,
                  halves: tuple[Stats, Stats] | None = None) -> dict:
    """The feasibility numbers of docs 7.7 (spec 4.4a).  Numbers only: no verdict."""
    prof_doc = W.combine_profiles([W.load_profile(x) for x in profiles]) if profiles else None
    prof = Profile(prof_doc, dict(DEFAULT_REGION_WEIGHTS), 1.0)
    if prof_doc is None:   # no profile: the components are what the statistics hold
        comps = sorted({f"lang:{c['language']}" for c in st.meta["categories"] if c["dataset"] == "target"} |
                       {f"base:{c['language']}" for c in st.meta["categories"] if c["dataset"] == "base"})
    else:
        comps = prof.components()
    pp = {"use_contrib": False, "alpha": None, "lambda": 0.5, "clip": 3.0}
    moe = sorted(st.layers)
    model = st.model
    scores = {l: LayerScores(st, l, prof, comps, pp) for l in moe}
    has_ref = any(scores[l].has_ref for l in moe)
    names = comps + (["reference"] if has_ref else [])
    dist = lambda sc, c: sc.mr if c == "reference" else sc.m[c]  # noqa: E731
    res: dict = {"levels": list(ANALYZE_LEVELS), "components": {}, "layers": [], "overlap": {}}
    acc = {c: {"n": {lv: [] for lv in ANALYZE_LEVELS}, "ent": [], "eff": []} for c in names}
    pair_acc: dict[str, list[dict]] = {}
    for l in moe:
        sc = scores[l]
        row: dict = {"layer": l, "experts": sc.E, "components": {}, "overlap": {}}
        sets = {}
        for c in names:
            m = dist(sc, c)
            if sum(m) <= 0:
                continue
            n_at = {lv: experts_for(m, lv) for lv in ANALYZE_LEVELS}
            ent, eff = entropy(m)
            row["components"][c] = {"experts_for": {str(lv): n for lv, n in n_at.items()}, "entropy": ent,
                                    "effective_experts": eff}
            sets[c] = set(ranked(m)[:n_at[0.95]])
            for lv in ANALYZE_LEVELS:
                acc[c]["n"][lv].append(n_at[lv])
            acc[c]["ent"].append(ent)
            acc[c]["eff"].append(eff)
        if sum(sc.combined_mass) > 0:
            ent, eff = entropy(sc.combined_mass)
            row["profile"] = {"experts_for": {str(lv): experts_for(sc.combined_mass, lv) for lv in ANALYZE_LEVELS},
                              "entropy": ent, "effective_experts": eff}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if a not in sets or b not in sets:
                    continue
                ov = {"jaccard_95": jaccard(sets[a], sets[b]), "mass_overlap": mass_overlap(dist(sc, a), dist(sc, b)),
                      "a95_covers_b": sum(dist(sc, b)[e] for e in sets[a]),
                      "b95_covers_a": sum(dist(sc, a)[e] for e in sets[b])}
                row["overlap"][f"{a}|{b}"] = ov
                pair_acc.setdefault(f"{a}|{b}", []).append(ov)
        row["union_95"] = sorted(set().union(*(sets[c] for c in comps if c in sets)))
        res["layers"].append(row)
    E_avg = sum(model.experts_per_layer[l] for l in moe) / len(moe)
    for c in names:
        if not acc[c]["ent"]:
            res["components"][c] = {"tokens": 0, "note": "no data"}
            continue
        ns = acc[c]["n"]
        res["components"][c] = {
            "tokens": scores[moe[0]].ref_tokens if c == "reference" else scores[moe[0]].tokens[c],
            "experts_for_mean": {str(lv): mean(v) for lv, v in ns.items()},
            "experts_for_max": {str(lv): max(v) for lv, v in ns.items()},
            "share_for_95": mean(ns[0.95]) / E_avg,
            "entropy_mean": mean(acc[c]["ent"]), "effective_experts_mean": mean(acc[c]["eff"]),
            "reference_mass_on_95_mean": (mean([o["a95_covers_b"] for o in pair_acc.get(f"{c}|reference", [])])
                                          if c != "reference" else None)}
    res["overlap"] = {k: {m: mean([o[m] for o in v]) for m in ("jaccard_95", "mass_overlap", "a95_covers_b",
                                                               "b95_covers_a")} for k, v in pair_acc.items()}
    # ---- stability: the 95% sets of two disjoint halves of the sample
    if halves is not None:
        ha, hb = ({l: LayerScores(h, l, prof, comps, pp) for l in moe} for h in halves)
        lim = p.get("stability_coverage", 0.90)
        stab: dict[str, list[dict]] = {c: [] for c in names}
        below = []
        for row in res["layers"]:
            l = row["layer"]
            row["stability"] = {}
            for c in names:
                ma, mb = dist(ha[l], c), dist(hb[l], c)
                if sum(ma) <= 0 or sum(mb) <= 0:
                    continue
                a95, b95 = top_set(ma, 0.95), top_set(mb, 0.95)
                cross = min(sum(mb[e] for e in a95), sum(ma[e] for e in b95))
                rec = {"jaccard_95": jaccard(a95, b95), "mass_overlap": mass_overlap(ma, mb), "cross_coverage_95": cross,
                       "experts_for_95": [len(a95), len(b95)]}
                row["stability"][c] = rec
                stab[c].append(rec)
                if cross < lim - 1e-12:
                    below.append({"layer": l, "component": c, "cross_coverage_95": cross})
        res["stability"] = {
            "available": True, "halves": [halves[0].path, halves[1].path],
            "tokens": [halves[0].meta["tokens"], halves[1].meta["tokens"]],
            "components": {c: {"jaccard_95_mean": mean([r["jaccard_95"] for r in v]),
                               "mass_overlap_mean": mean([r["mass_overlap"] for r in v]),
                               "cross_coverage_95_mean": mean([r["cross_coverage_95"] for r in v]),
                               "cross_coverage_95_min": min((r["cross_coverage_95"] for r in v), default=None)}
                           for c, v in stab.items() if v},
            "coverage_limit": lim, "below": sorted(below, key=lambda r: (r["cross_coverage_95"], r["layer"], r["component"])),
            "note": "cross_coverage_95: the mass of one half that the other half's 95% set covers (the smaller of the two)"}
    else:
        res["stability"] = {"available": False,
                            "note": "needs two disjoint halves of the sample: analyze collects them itself; with existing "
                                    "statistics give --stats HALF_A --stats HALF_B"}
    # ---- the union of the components' 95% sets per layer (by mass), never below top_k + 2, and its bytes
    keep = {row["layer"]: max(len(row["union_95"]), model.top_k + 2) * model.blob_bytes[row["layer"]]
            for row in res["layers"]}
    res["projected_expert_bytes_95"] = sum(keep.values())
    res["expert_bytes"] = sum(model.blob_bytes[l] * model.experts_per_layer[l] for l in moe)
    # ---- the hardware budget, per mode
    budget = None
    if hw:
        try:
            res["target_hardware"] = hardware_budget(st, hw, context, p)
            budget = res["target_hardware"]["expert_budget"]
            res["fits_95"] = res["projected_expert_bytes_95"] <= budget
        except GenError as e:
            res["target_hardware_error"] = str(e)
    if budget is None and p.get("budget"):
        budget = parse_size(p["budget"])
    res["projection"] = project_modes(st, scores, comps, budget, p) if budget is not None else {
        "available": False, "note": "give --target-hardware or --budget to project the modes"}
    # ---- the nearly flat layers (by the profile's mass) and their share of the budget
    lim = p.get("flat_entropy", 0.9)
    flat = [row["layer"] for row in res["layers"] if row.get("profile") and row["profile"]["entropy"] > lim]
    resident = res["projection"].get("strict", {}).get("resident")
    if resident:
        per = {int(l): len(v) * model.blob_bytes[int(l)] for l, v in resident.items()}
        how = "the strict projection's resident experts"
    else:
        per, how = keep, "the union of the components' 95% sets"
    total = sum(per.values())
    res["flat_layers"] = {"entropy_limit": lim, "layers": flat,
                          "bytes": sum(per[l] for l in flat), "budget_share": (sum(per[l] for l in flat) / total) if total else None,
                          "share_of": how}
    res["note"] = ("numbers from a small sample: a feasibility check, not the whitelist; bytes and the projection are "
                   "calculations, not measurements")
    return res


def analyze(model: str | None, preps: list[str], out: str, sample: int = 4000, stats=None,
            profiles: list[str] | None = None, target_hardware: str | None = None, context: int = 32768, **p) -> dict:
    outp = pathlib.Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    stats = [stats] if isinstance(stats, str) else list(stats or [])
    if len(stats) > 2:
        raise GenError("analyze takes one --stats, or two (two disjoint halves of a sample for the stability numbers)")
    if not stats:
        if not model or not preps:
            raise GenError("analyze needs --model and --prep (or --stats with statistics that exist)")
        keys, rows, seed = subsample(preps, sample)
        # two disjoint halves by the hash of the id, collected apart: the stability numbers compare them
        halves = ([r for r in rows if hash_fraction(f"half:{seed}", r["id"]) < 0.5],
                  [r for r in rows if hash_fraction(f"half:{seed}", r["id"]) >= 0.5])
        parts = [("a", halves[0]), ("b", halves[1])] if halves[0] and halves[1] else [("", rows)]
        ckeys = ("vram_budget", "ram_budget", "block_tokens", "scratch", "scratch_max", "spill", "prefetch",
                 "checkpoint_every", "similarity", "similarity_seed")
        for name, part in parts:
            sub = outp / ("stats-" + name if name else "stats")
            if not (sub / "meta.json").exists():
                prep = write_sub_prep(outp / ("prep-" + name if name else "prep"), preps, keys, part)
                collect(model, prep, str(sub), mode=p.get("mode") or "auto", **{k: p.get(k) for k in ckeys})
            stats.append(str(sub))
    halves = None
    if len(stats) == 2:
        merged = outp / "stats"
        if str(merged.resolve()) in {str(pathlib.Path(x).resolve()) for x in stats}:
            raise GenError(f"{merged} is where analyze puts the two halves' sum: give the halves from elsewhere")
        merge(stats, str(merged))
        halves = (read_stats(stats[0]), read_stats(stats[1]))
        st = read_stats(str(merged))
    else:
        st = read_stats(stats[0])
    hw = parse_hardware(target_hardware) if target_hardware else None
    res = analyze_stats(st, profiles or [], hw, context, p, halves)
    res.update(kind="strata-whitelist-analysis", generator=GENERATOR, stats=st.path, halves=stats if halves else None,
               sample_tokens=sample, preps=preps, model_fingerprint=st.fingerprint, stats_run_id=st.meta["run_id"],
               profiles=[{"path": str(x), "sha256": sha256_file(x)} for x in profiles or []])
    write_json(outp / "analysis.json", res)
    atomic_write(outp / "analysis.md", analysis_markdown(res))
    return res


def analysis_markdown(r: dict) -> str:
    f = lambda v, d=3: "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))  # noqa: E731
    out = ["# Specialisation analysis", "", r["note"] + ".", "",
           "## Experts per component (mean over the layers; the most in one layer in brackets)", "",
           "| component | tokens | 80% | 90% | 95% | 99% | entropy | effective experts | reference mass on its 95% set |",
           "|---|---|---|---|---|---|---|---|---|"]
    for c, v in r["components"].items():
        if "experts_for_mean" not in v:
            out.append(f"| {c} | 0 | no data | | | | | | |")
            continue
        m, mx = v["experts_for_mean"], v["experts_for_max"]
        cells = " | ".join(f"{m[str(lv)]:.1f} ({mx[str(lv)]})" for lv in r["levels"])
        out.append(f"| {c} | {v['tokens']} | {cells} | {v['entropy_mean']:.3f} | {v['effective_experts_mean']:.1f} | "
                   f"{f(v['reference_mass_on_95_mean'])} |")
    if r["overlap"]:
        out += ["", "## Overlap (mean over the layers)", "",
                "| a | b | Jaccard of the 95% sets | mass overlap | a's 95% set covers of b | b's 95% set covers of a |",
                "|---|---|---|---|---|---|"]
        for k, v in r["overlap"].items():
            a, b = k.split("|")
            out.append(f"| {a} | {b} | {f(v['jaccard_95'])} | {f(v['mass_overlap'])} | {f(v['a95_covers_b'])} | "
                       f"{f(v['b95_covers_a'])} |")
    s = r["stability"]
    out += ["", "## Stability (two disjoint halves of the sample)", ""]
    if s["available"]:
        out += [f"Tokens per half: {s['tokens'][0]}, {s['tokens'][1]}. {s['note']}.", "",
                "| component | Jaccard of the 95% sets | mass overlap | cross coverage (mean) | cross coverage (lowest) |",
                "|---|---|---|---|---|"]
        for c, v in s["components"].items():
            out.append(f"| {c} | {f(v['jaccard_95_mean'])} | {f(v['mass_overlap_mean'])} | {f(v['cross_coverage_95_mean'])} | "
                       f"{f(v['cross_coverage_95_min'])} |")
        out += ["", f"Layer and component pairs with a cross coverage below {s['coverage_limit']}: {len(s['below'])}" +
                (": " + ", ".join(f"{b['layer']}/{b['component']} ({b['cross_coverage_95']:.3f})" for b in s["below"][:12])
                 if s["below"] else ".")]
    else:
        out.append(s["note"] + ".")
    pr = r["projection"]
    out += ["", "## Projection against the budget (a calculation, not a measurement)", ""]
    if "available" in pr:
        out.append(pr["note"] + ".")
    elif "problem" in pr:
        out.append(pr["problem"] + ".")
    else:
        out += [f"Experts' budget {fmt_bytes(pr['budget'])}, at least {pr['floor']} experts per layer.", "",
                "| mode | resident experts | resident bytes | component | coverage (mean, lowest) | mass on blocked | "
                "mass on on_demand |", "|---|---|---|---|---|---|---|"]
        st, sb, ti = pr["strict"], pr["substitute"], pr.get("tiered", {})
        for c, v in st["coverage"].items():
            out.append(f"| strict | {st['resident_experts']} | {fmt_bytes(st['resident_bytes'])} | {c} | "
                       f"{f(v['mean'])}, {f(v['min'])} | {f(st['blocked_mass'][c])} | - |")
        for c, v in sb["coverage"].items():
            extra = (f" (with a replacement >= {sb['threshold']}: {f(sb['blocked_mass_with_substitute'][c])} of it)"
                     if "blocked_mass_with_substitute" in sb else "")
            out.append(f"| substitute | {sb['resident_experts']} | {fmt_bytes(sb['resident_bytes'])} | {c} | "
                       f"{f(v['mean'])}, {f(v['min'])} | {f(sb['blocked_mass'][c])}{extra} | - |")
        if "problem" in ti:
            out.append(f"| tiered | | | | {ti['problem']} | | |")
        elif ti:
            for c, v in ti["coverage_with_on_demand"].items():
                out.append(f"| tiered | {ti['resident_experts']} + {ti['on_demand_experts']} on_demand | "
                           f"{fmt_bytes(ti['resident_bytes'])} | {c} | {f(ti['coverage_resident'][c]['mean'])} resident, "
                           f"{f(v['mean'])} with on_demand | {f(ti['blocked_mass'][c])} | {f(ti['on_demand_mass'][c])} |")
            out.append("")
            out.append(f"Tiered: an on_demand cache of {fmt_bytes(ti['cache'])}; at most "
                       f"{fmt_bytes(ti['on_demand_bytes_per_token_max'])} read per token ({ti['bytes_note']}).")
        if "note" in sb:
            out += ["", "Substitute: " + sb["note"] + "."]
    fl = r["flat_layers"]
    out += ["", "## Nearly flat layers", "",
            f"Layers whose profile routing has a normalised entropy above {fl['entropy_limit']}: "
            f"{', '.join(map(str, fl['layers'])) or 'none'}; they take {fmt_bytes(fl['bytes'])} "
            f"({f(fl['budget_share'])} of {fl['share_of']})."]
    out += ["", f"Expert weights for the union of the 95% sets: {fmt_bytes(r['projected_expert_bytes_95'])} of "
                f"{fmt_bytes(r['expert_bytes'])} (a calculation)."]
    if "target_hardware" in r:
        out.append(f"The target hardware leaves {fmt_bytes(r['target_hardware']['expert_budget'])} for experts.")
    return "\n".join(out) + "\n"


# ================================ EVAL ================================

def eval_whitelist(model: str, whitelist: str, prep: str, out: str, holdout: bool = True, max_samples: int = 0,
                   env: list[str] | None = None, gpu_experts=False, min_free_gb: float = 16, wrap: str | None = None,
                   mode: str | None = None) -> dict:
    """Teacher-forced on the holdout, with and without the whitelist: top-1 agreement, KL and perplexity per language,
    task and region (tools/quant_quality.py's engine run and its comparison)."""
    import quant_quality as Q  # noqa: E402
    cfg = load_model_config(model)
    wl = json.loads(pathlib.Path(whitelist).read_text(encoding="utf-8"))
    cats, samples = read_samples(pathlib.Path(prep) / ("holdout.bin" if holdout else "samples.bin"))
    ids = set(wl.get("generation", {}).get("holdout_ids") or [])
    if holdout and ids:
        samples = [s for s in samples if s["id"] in ids]
    if max_samples:
        samples = samples[:max_samples]
    if not samples:
        raise GenError("no samples to evaluate (an empty holdout? prepare --holdout 0.1)")
    outp = pathlib.Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    seg = max(len(s["tokens"]) for s in samples)
    write_json(outp / "eval.ids.json", {"version": "whitelist-eval", "seg_tokens": seg,
                                        "segments": [{"name": s["id"], "ids": s["tokens"]} for s in samples]})
    wl_cfg = dict(cfg, args=list(cfg.get("args", [])) + ["--expert-whitelist", str(pathlib.Path(whitelist).resolve())] +
                  (["--expert-whitelist-mode", mode] if mode else []))
    write_json(outp / "whitelist.config.json", wl_cfg)
    runs = {}
    for name, c in (("baseline", model), ("whitelist", str(outp / "whitelist.config.json"))):
        tsv = outp / f"{name}.tsv"
        ns = argparse.Namespace(config=c, out=str(tsv), ids=str(outp / "eval.ids.json"), env=env, gpu_experts=gpu_experts,
                                min_free_gb=min_free_gb, wrap=wrap)
        Q.cmd_run(ns)
        runs[name] = Q.read_run(str(tsv))
    res = eval_compare(runs["baseline"], runs["whitelist"], samples, cats, Q)
    res.update(kind="strata-whitelist-eval", generator=GENERATOR, whitelist=str(whitelist),
               whitelist_sha256=sha256_file(whitelist), model_fingerprint=wl.get("model_fingerprint"), prep=str(prep),
               prep_sha256=sha256_file(pathlib.Path(prep) / "prep.json"), samples=len(samples),
               set="holdout" if holdout else "train", engine={"exe": cfg["exe"], "args": cfg.get("args", [])})
    write_json(outp / "eval.json", res)
    return res


def eval_compare(ref: dict, run: dict, samples: list[dict], cats: list[str], Q) -> dict:
    """Groups the positions by the category of the token they predict and compares each group."""
    by_id = {s["id"]: s for s in samples}
    groups: dict[str, tuple[dict, dict]] = {}
    for name, rows in ref.items():
        s = by_id.get(name)
        if s is None:
            continue
        for pos, row in rows.items():
            if pos + 1 >= len(s["cats"]) or pos not in run.get(name, {}):
                continue
            c = split_key(cats[s["cats"][pos + 1]])
            for g in (f"language={c['language']}", f"language={c['language']},region={c['region']}",
                      f"task={c['task']}", "all"):
                a, b = groups.setdefault(g, ({}, {}))
                a.setdefault(name, {})[pos] = row
                b.setdefault(name, {})[pos] = run[name][pos]
    return {"groups": {g: Q.compare(a, b)["all"] for g, (a, b) in sorted(groups.items())},
            "note": "teacher-forced on held-out samples: top-1 agreement and KL against the full model, perplexity "
                    "of the text under each (measured by the engine runs in this folder)"}


# ================================ CLI ================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare", help="tokenized, deduplicated samples and a fixed holdout split")
    s.add_argument("--profile", action="append", default=[])
    s.add_argument("--out", required=True)
    s.add_argument("--tokenizer", help="'bytes' (tests), a pack's tokenizer/ folder or a GGUF; default: --model's")
    s.add_argument("--model", help="engine config (its pack's tokenizer and chat template)")
    s.add_argument("--holdout", type=float, default=0.10)
    s.add_argument("--reference", action="append", help="a reference corpus file (JSON lines), besides the profile's")
    s.add_argument("--only", action="append", choices=DATASETS, help="only these data sets (e.g. --only reference)")
    s.add_argument("--chat-template", help="model, chatml, none or a .jinja file (default: the profile's)")
    s.add_argument("--max-tokens", type=int)
    s.add_argument("--max-samples", type=int)
    s.add_argument("--seed", type=int)

    s = sub.add_parser("estimate", help="memory, scratch space, reads and the mode a collection needs (nothing runs)")
    s.add_argument("--model")
    s.add_argument("--pack")
    s.add_argument("--gguf")
    s.add_argument("--prep", required=True)
    s.add_argument("--vram-budget")
    s.add_argument("--ram-budget")
    s.add_argument("--scratch")
    s.add_argument("--scratch-max")
    s.add_argument("--block-tokens", type=int, default=0)
    s.add_argument("--spill", choices=("f32", "f16", "bf16"), default="f32")
    s.add_argument("--prefetch", action="store_true")
    s.add_argument("--nvme-gbps", type=float, default=2.0, help="the drive's read speed to assume (default 2.0)")
    s.add_argument("--prefill-tps", type=float, help="a measured prompt speed (tokens/s) for a compute-time estimate")
    s.add_argument("--json", action="store_true")

    def collect_args(s):
        s.add_argument("--mode", choices=("auto", "token", "layerwise"), default="auto")
        s.add_argument("--vram-budget")
        s.add_argument("--ram-budget")
        s.add_argument("--block-tokens", type=int)
        s.add_argument("--scratch")
        s.add_argument("--scratch-max")
        s.add_argument("--spill", choices=("f32", "f16", "bf16"))
        s.add_argument("--prefetch", action="store_true")
        s.add_argument("--checkpoint-every", type=int)
        s.add_argument("--similarity", type=float,
                       help="also the experts' output similarity (for the substitute mode) on this share of the tokens "
                            "(0.01: 1%%); every expert is computed on the CPU for each")
        s.add_argument("--similarity-seed", type=int, help="which tokens the similarity share samples (default 1)")

    def mode_args(s):
        s.add_argument("--on-demand-coverage", type=float, default=0.995,
                       help="tiered: the second coverage the on_demand tier reaches (default 0.995)")
        s.add_argument("--on-demand-cache", type=parse_size, default=DEFAULT_ON_DEMAND_CACHE,
                       help="tiered: the engine's on_demand cache, taken from the budget (default 1G)")
        s.add_argument("--substitute-threshold", type=float, default=0.5,
                       help="substitute: the similarity a replacement needs to be used (default 0.5)")
        s.add_argument("--substitute-min-tokens", type=int, default=20,
                       help="substitute: sampled picks a blocked expert needs for a replacement (default 20)")

    s = sub.add_parser("analyze", help="how specialised the experts are, on a small sample")
    s.add_argument("--model")
    s.add_argument("--prep", action="append", default=[])
    s.add_argument("--stats", action="append", default=[],
                   help="analyze statistics that exist instead of collecting a sample; twice: two disjoint halves")
    s.add_argument("--profile", action="append", default=[])
    s.add_argument("--sample", type=int, default=4000, help="tokens per --prep (default 4000)")
    s.add_argument("--target-hardware")
    s.add_argument("--budget", help="the experts' budget in bytes (16G) instead of --target-hardware")
    s.add_argument("--context", type=int, default=32768)
    s.add_argument("--dense-bytes", type=parse_size)
    s.add_argument("--kv-bytes-per-token", type=int)
    s.add_argument("--gguf")
    s.add_argument("--margin", type=int, default=2)
    s.add_argument("--flat-entropy", type=float, default=0.9, help="a layer above this normalised entropy is listed as flat")
    s.add_argument("--stability-coverage", type=float, default=0.90,
                   help="list layers whose halves cover each other's mass less than this (default 0.90)")
    s.add_argument("--out", required=True)
    mode_args(s)
    collect_args(s)

    s = sub.add_parser("collect", help="the routing statistics (runs the engine)")
    s.add_argument("--model", required=True)
    s.add_argument("--prep", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--resume", action="store_true")
    s.add_argument("--print-command", action="store_true", help="print the engine command and stop")
    collect_args(s)

    s = sub.add_parser("merge", help="add statistics of independent runs (exact)")
    s.add_argument("--in", dest="inputs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--allow-overlap", action="store_true")

    s = sub.add_parser("build", help="the whitelist manifest, its report and coverage curve")
    s.add_argument("--stats", required=True)
    s.add_argument("--profile", action="append", default=[], required=True)
    s.add_argument("--mode", choices=("coverage", "count", "fraction", "budget"), default="coverage")
    s.add_argument("--coverage", type=float)
    s.add_argument("--count", type=int)
    s.add_argument("--fraction", type=float)
    s.add_argument("--budget", help="experts (a number) or bytes (16G) over all layers, for --mode budget")
    s.add_argument("--target-hardware", help="vram=16G,ram=64G,reserve=8G[,buffer=2G]")
    s.add_argument("--context", type=int, default=32768)
    s.add_argument("--lambda", dest="lam", type=float, default=0.5)
    s.add_argument("--clip", type=float, default=3.0)
    s.add_argument("--alpha", type=float)
    s.add_argument("--protect-mass", type=float, default=0.8)
    s.add_argument("--n-min", type=int, default=20)
    s.add_argument("--include-uncertain", action="store_true")
    s.add_argument("--margin", type=int, default=2)
    s.add_argument("--region-weights", help="response=1,prompt=0.5,system=0.5,tool=0.5")
    s.add_argument("--base-weight", type=float, default=1.0)
    s.add_argument("--use-contrib", action="store_true")
    s.add_argument("--allow-under-coverage", action="store_true")
    s.add_argument("--layer-weights", help="a JSON file {layer: weight} (an ablation's result)")
    s.add_argument("--expert-groups", type=int, default=0)
    s.add_argument("--group-topk", type=int, default=0)
    s.add_argument("--runtime-buffer", type=parse_size, default=2 << 30)
    s.add_argument("--dense-bytes", type=parse_size)
    s.add_argument("--kv-bytes-per-token", type=int)
    s.add_argument("--state-bytes", type=parse_size)
    s.add_argument("--gguf", help="the model's first GGUF shard (the KV cache's size from its metadata)")
    s.add_argument("--min-tokens", type=int, default=2000, help="warn below this many tokens per component")
    s.add_argument("--min-task-tokens", type=int, default=500, help="warn below this many tokens per component and task")
    s.add_argument("--sweep", help="coverage=0.99,0.97,0.95: one manifest per value ({value} in --out)")
    s.add_argument("--whitelist-mode", choices=W.MODES, default="strict",
                   help="the manifest's mode: strict (default), substitute (needs similarity data) or tiered")
    mode_args(s)
    s.add_argument("--out", required=True)

    s = sub.add_parser("inspect", help="a readable summary of a manifest")
    s.add_argument("manifest")

    s = sub.add_parser("eval", help="a short quality check on the holdout, with the runtime")
    s.add_argument("--model", required=True)
    s.add_argument("--whitelist", required=True)
    s.add_argument("--prep", required=True)
    s.add_argument("--holdout", action="store_true", default=True)
    s.add_argument("--train", dest="holdout", action="store_false", help="the training samples instead (a check only)")
    s.add_argument("--max-samples", type=int, default=0)
    s.add_argument("--out")
    s.add_argument("--env", action="append")
    s.add_argument("--whitelist-mode", choices=W.MODES)
    s.add_argument("--min-free-gb", type=float, default=16)

    a = ap.parse_args(argv)
    try:
        if a.cmd == "prepare":
            doc = prepare(a.profile, a.out, a.tokenizer, a.model, a.holdout, a.reference, a.only, a.chat_template,
                          a.max_tokens, a.max_samples, a.seed)
            log(f"prepare: {doc['samples']['count']} samples ({doc['samples']['tokens']} tokens), holdout "
                f"{doc['holdout']['count']} ({doc['holdout']['tokens']} tokens), {len(doc['categories'])} categories, "
                f"{len(doc['dropped']['duplicates'])} duplicates dropped -> {a.out}")
            for w in doc["warnings"]:
                log(f"prepare: warning: {w}")
        elif a.cmd == "estimate":
            e = estimate(a.prep, a.model, a.pack, a.gguf, a.vram_budget, a.ram_budget, a.scratch, a.scratch_max,
                         a.block_tokens, a.spill, a.prefetch, a.nvme_gbps, a.prefill_tps)
            print(json.dumps(e, indent=1) if a.json else estimate_text(e), end="" if not a.json else "\n")
            return 0 if e["ok"] else 1
        elif a.cmd == "analyze":
            r = analyze(a.model, a.prep, a.out, a.sample, a.stats, a.profile, a.target_hardware, a.context,
                        dense_bytes=a.dense_bytes, kv_bytes_per_token=a.kv_bytes_per_token, gguf=a.gguf,
                        runtime_buffer=2 << 30, mode=a.mode, vram_budget=a.vram_budget, ram_budget=a.ram_budget,
                        block_tokens=a.block_tokens, scratch=a.scratch, scratch_max=a.scratch_max, spill=a.spill,
                        prefetch=a.prefetch, checkpoint_every=a.checkpoint_every, similarity=a.similarity,
                        similarity_seed=a.similarity_seed, budget=a.budget, margin=a.margin,
                        flat_entropy=a.flat_entropy, stability_coverage=a.stability_coverage,
                        on_demand_coverage=a.on_demand_coverage, on_demand_cache=a.on_demand_cache,
                        substitute_threshold=a.substitute_threshold, substitute_min_tokens=a.substitute_min_tokens)
            print(analysis_markdown(r), end="")
        elif a.cmd == "collect":
            return collect(a.model, a.prep, a.out, a.mode, a.resume, a.print_command, vram_budget=a.vram_budget,
                           ram_budget=a.ram_budget, block_tokens=a.block_tokens, scratch=a.scratch,
                           scratch_max=a.scratch_max, spill=a.spill, prefetch=a.prefetch,
                           checkpoint_every=a.checkpoint_every, similarity=a.similarity,
                           similarity_seed=a.similarity_seed)
        elif a.cmd == "merge":
            m = merge(a.inputs, a.out, a.allow_overlap)
            log(f"merge: {len(m['runs'])} runs, {m['samples']} samples, {m['tokens']} tokens -> {a.out}")
        elif a.cmd == "build":
            rw = {}
            for part in (a.region_weights or "").split(","):
                if part.strip():
                    k, _, v = part.partition("=")
                    rw[k.strip()] = float(v)
            lw = json.loads(pathlib.Path(a.layer_weights).read_text()) if a.layer_weights else None
            common = dict(target_hardware=a.target_hardware, context=a.context, lam=a.lam, clip=a.clip, alpha=a.alpha,
                          protect_mass=a.protect_mass, n_min=a.n_min, include_uncertain=a.include_uncertain,
                          margin=a.margin, region_weights=rw, base_weight=a.base_weight, use_contrib=a.use_contrib,
                          allow_under_coverage=a.allow_under_coverage, layer_weights=lw,
                          expert_groups=a.expert_groups, group_topk=a.group_topk, runtime_buffer=a.runtime_buffer,
                          dense_bytes=a.dense_bytes, kv_bytes_per_token=a.kv_bytes_per_token,
                          state_bytes=a.state_bytes, gguf=a.gguf, min_tokens=a.min_tokens,
                          min_task_tokens=a.min_task_tokens, whitelist_mode=a.whitelist_mode,
                          on_demand_coverage=a.on_demand_coverage, on_demand_cache=a.on_demand_cache,
                          substitute_threshold=a.substitute_threshold, substitute_min_tokens=a.substitute_min_tokens)
            jobs = []
            if a.sweep:
                key, _, vals = a.sweep.partition("=")
                if key.strip() != "coverage" or "{value}" not in a.out:
                    raise GenError("--sweep takes coverage=V1,V2,... and an --out with {value} in it")
                jobs = [(a.out.replace("{value}", v.strip()), "coverage", float(v)) for v in vals.split(",") if v.strip()]
            else:
                jobs = [(a.out, a.mode, a.coverage)]
            for out, mode, cov in jobs:
                r = build(a.stats, a.profile, out, mode, cov, a.count, a.fraction, a.budget, **common)
                t = r["totals"]
                log(f"build: {out}: {t['allowed']} of {t['experts']} experts, {fmt_bytes(t['expert_bytes_after'])} of "
                    f"{fmt_bytes(t['expert_bytes_before'])}" + (f"; fits the target hardware: {r['fits']}" if "fits" in r
                                                                 else ""))
                for w in r["warnings"]:
                    log(f"build: warning: {w}")
        elif a.cmd == "inspect":
            print(inspect(a.manifest), end="")
        elif a.cmd == "eval":
            out = a.out or str(pathlib.Path(a.whitelist).with_suffix("")) + ".eval"
            r = eval_whitelist(a.model, a.whitelist, a.prep, out, a.holdout, a.max_samples, a.env,
                               min_free_gb=a.min_free_gb, mode=a.whitelist_mode)
            for g, v in r["groups"].items():
                log(f"eval: {g:40s} positions {v['positions']:6d}  top-1 {100 * v['top1']:6.2f}%  KL {v['kl']:.5f}  "
                    f"PPL {v['ppl']:.3f} (full model {v['ppl_ref']:.3f})")
    except (GenError, W.WhitelistError, OSError) as e:
        print(f"whitelist_gen: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
