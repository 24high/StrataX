"""tools/expert_whitelist.py - expert whitelists: profiles, manifests, the model fingerprint, the pack index sidecar.

An expert whitelist (docs/EXPERT_WHITELIST.md) says which routed experts a run may use.  The engine reads it with
`--expert-whitelist M`; this tool is the Python side of the same formats, with the engine's own rules:

    python tools/expert_whitelist.py fingerprint --pack PACK [--gguf SHARD1]   the model fingerprint a manifest needs
    python tools/expert_whitelist.py fingerprint --mtp                        ... of the MTP draft layer
    python tools/expert_whitelist.py index PACK [--check]                     write (or check) PACK/experts.index.json
    python tools/expert_whitelist.py validate MANIFEST --pack PACK [--mode M]  the engine's checks, without the engine
    python tools/expert_whitelist.py check-profile P [P ...]                   check one or more profiles (and combine)

`tools/iq_pack.py index PACK` is the same `index` command.  Only the standard library is used (no numpy).  Nothing
here changes a pack's existing files: `index` writes one new file beside them.

The fingerprint is FNV-1a 64 over a canonical text of the model's geometry and the pack's expert index (where each
layer's experts start and the bytes of one expert); src/core/expert_whitelist.cpp builds the same text.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pathlib
import sys
from dataclasses import dataclass, field

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent

FORMAT_VERSION = 1
MANIFEST_KIND = "strata-expert-whitelist"
INDEX_KIND = "strata-expert-index"
PROFILE_KIND = "strata-whitelist-profile"
MODES = ("strict", "substitute", "tiered")
CANONICAL_BLOB = 1382400          # a canonical (Q2_0) pack's expert blob, tools/pack_layer.py
CANONICAL = {"n_layers": 48, "n_expert": 512, "top_k": 10, "hidden": 2560}   # include/strata/core/geometry.hpp


class WhitelistError(ValueError):
    """A manifest, profile or pack that does not pass the checks; the message names what is wrong."""


# ================================ THE MODEL AND ITS FINGERPRINT ================================

@dataclass
class ModelInfo:
    n_layers: int
    experts_per_layer: list[int]
    top_k: int
    hidden: int
    layer_offset: list[int]
    blob_bytes: list[int]
    source: str = ""                # where the index came from (native_experts.txt, manifest.json, ...)

    def fingerprint_text(self) -> str:
        t = "strata-expert-whitelist-fingerprint-v1\n"
        t += f"num_layers={self.n_layers}\ntop_k={self.top_k}\nhidden={self.hidden}\n"
        t += "experts_per_layer=" + ",".join(str(n) for n in self.experts_per_layer) + "\n"
        t += "pack_index=" + ";".join(f"{l}:{o}:{b}" for l, (o, b) in enumerate(zip(self.layer_offset, self.blob_bytes)))
        return t + "\n"

    def fingerprint(self) -> str:
        return "fnv1a64:%016x" % fnv1a64(self.fingerprint_text().encode())


def fnv1a64(data: bytes, h: int = 0xCBF29CE484222325) -> int:
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _gguf_geometry(gguf: str) -> tuple[int, int]:
    """(top_k, hidden) from a GGUF's metadata (its <arch>.expert_used_count and <arch>.embedding_length)."""
    sys.path.insert(0, str(HERE))
    import gguf_reader  # noqa: E402  (stdlib only)
    meta = gguf_reader.GGUFFile(pathlib.Path(gguf)).metadata
    arch = meta.get("general.architecture", "")
    try:
        return int(meta[f"{arch}.expert_used_count"]), int(meta[f"{arch}.embedding_length"])
    except KeyError as e:
        raise WhitelistError(f"{gguf}: the GGUF has no {e.args[0]}") from None


def pack_model_info(pack: str, top_k: int | None = None, hidden: int | None = None, gguf: str | None = None) -> ModelInfo:
    """The model as the engine sees it for this pack: the expert layout the engine computes (ExpertLayout), the
    top-k and width from the GGUF when one is given, else the canonical geometry's (or the arguments')."""
    p = pathlib.Path(pack)
    if gguf:
        gk, gh = _gguf_geometry(gguf)
        top_k = top_k or gk
        hidden = hidden or gh
    top_k = top_k or CANONICAL["top_k"]
    hidden = hidden or CANONICAL["hidden"]
    native = p / "native_experts.txt"
    if native.exists():
        n_expert = CANONICAL["n_expert"]
        rows: dict[int, tuple[int, int]] = {}
        for line in native.read_text().splitlines():
            if not line.strip():
                continue
            if line.startswith("#"):
                at = line.find("(n_expert ")
                if at >= 0:
                    n_expert = int(line[at + 10:].split(",")[0].split(")")[0].split()[0])
                continue
            f = line.split()
            layer, offset, blob = int(f[0]), int(f[3]), int(f[4])
            if layer in rows:
                raise WhitelistError(f"{native}: layer {layer} is listed twice")
            rows[layer] = (offset, blob)
        n_layers = len(rows)
        at = 0
        for l in range(n_layers):
            if l not in rows or rows[l][0] != at:
                raise WhitelistError(f"{native}: layer {l} is missing or not contiguous")
            at += rows[l][1] * n_expert
        return ModelInfo(n_layers, [n_expert] * n_layers, top_k, hidden, [rows[l][0] for l in range(n_layers)],
                         [rows[l][1] for l in range(n_layers)], "native_experts.txt")
    manifest = p / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        n_layers = int(m.get("n_layers", CANONICAL["n_layers"]))
        n_expert = int(m.get("n_experts_per_layer", CANONICAL["n_expert"]))
    else:
        raise WhitelistError(f"{pack}: neither native_experts.txt nor manifest.json: not a Strata pack")
    return ModelInfo(n_layers, [n_expert] * n_layers, top_k, hidden,
                     [l * n_expert * CANONICAL_BLOB for l in range(n_layers)], [CANONICAL_BLOB] * n_layers, "manifest.json")


def mtp_model_info() -> ModelInfo:
    """The MTP draft layer (tools/mtp_rt.py's experts.bin): one layer of the canonical geometry's experts."""
    return ModelInfo(1, [CANONICAL["n_expert"]], CANONICAL["top_k"], CANONICAL["hidden"], [0], [CANONICAL_BLOB], "mtp")


# ================================ THE PACK INDEX SIDECAR ================================

INDEX_NAME = "experts.index.json"


def index_document(pack: str, info: ModelInfo) -> dict:
    p = pathlib.Path(pack)
    src = p / info.source
    experts_bin = p / "experts.bin"
    total = sum(b * n for b, n in zip(info.blob_bytes, info.experts_per_layer))
    return {
        "format_version": FORMAT_VERSION,
        "kind": INDEX_KIND,
        "num_layers": info.n_layers,
        "expert_offset": "offset + expert * blob_bytes (bytes into experts.bin, or the same layout assembled from the GGUF)",
        "layers": [{"layer": l, "n_expert": info.experts_per_layer[l], "offset": info.layer_offset[l],
                    "blob_bytes": info.blob_bytes[l]} for l in range(info.n_layers)],
        "total_bytes": total,
        "experts_bin": {"present": experts_bin.exists(), "bytes": experts_bin.stat().st_size if experts_bin.exists() else 0},
        "source": info.source,
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest() if src.exists() else "",
        "pack_index_fnv1a64": "fnv1a64:%016x" % fnv1a64(info.fingerprint_text().split("pack_index=")[1].encode()),
    }


def write_index(pack: str, check: bool = False) -> tuple[pathlib.Path, bool]:
    """Writes PACK/experts.index.json from the pack's own layout files (nothing else in the pack is touched), or with
    `check` compares the one there.  Returns (path, ok)."""
    info = pack_model_info(pack)
    doc = index_document(pack, info)
    if doc["experts_bin"]["present"] and doc["experts_bin"]["bytes"] != doc["total_bytes"]:
        raise WhitelistError(f"{pack}/experts.bin is {doc['experts_bin']['bytes']} B, the layout says {doc['total_bytes']}")
    path = pathlib.Path(pack) / INDEX_NAME
    text = json.dumps(doc, indent=1) + "\n"
    if check:
        return path, path.exists() and json.loads(path.read_text())["layers"] == doc["layers"]
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return path, True


# ================================ MANIFESTS ================================

@dataclass
class Whitelist:
    """A validated manifest with the effective mode: per layer the tier of each original expert id."""
    mode: str
    manifest_mode: str
    n_layers: int
    top_k: int
    tier: list[list[int]]            # 0 blocked, 1 allowed (resident), 2 on_demand
    substitute: list[list[int]] = field(default_factory=list)          # in use: at or above the threshold
    profile: str = ""
    substitute_listed: list[list[int]] = field(default_factory=list)   # every listed replacement
    substitute_similarity: list[list[float | None]] = field(default_factory=list)
    substitute_threshold: float | None = None

    def routable(self, l: int, e: int) -> bool:
        t = self.tier[l][e] if 0 <= l < self.n_layers and 0 <= e < len(self.tier[l]) else 0
        return t == 1 or (t == 2 and self.mode == "tiered")


def _no_dup_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise WhitelistError(f'duplicate key "{k}"')
        out[k] = v
    return out


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate_manifest(text: str, model: ModelInfo, mode_override: str | None = None) -> Whitelist:
    """The engine's checks (src/core/expert_whitelist.cpp parse_expert_whitelist), with the same messages."""
    try:
        root = json.loads(text, object_pairs_hook=_no_dup_keys)
    except json.JSONDecodeError as e:
        raise WhitelistError(f"JSON line {e.lineno}: {e.msg}") from None
    if not isinstance(root, dict):
        raise WhitelistError("the manifest is not a JSON object")
    if "kind" in root and root["kind"] != MANIFEST_KIND:
        raise WhitelistError(f'"kind" is not "{MANIFEST_KIND}": this is not an expert whitelist manifest')
    v = root.get("format_version")
    if not _int(v):
        raise WhitelistError('"format_version" is missing or not an integer')
    if v != FORMAT_VERSION:
        raise WhitelistError(f"unknown format_version {v} (this engine reads format_version {FORMAT_VERSION})")
    fp = root.get("model_fingerprint")
    if not isinstance(fp, str) or not fp:
        raise WhitelistError('"model_fingerprint" is missing')
    want = model.fingerprint()
    if fp != want:
        raise WhitelistError(f"the whitelist was made for another model or pack (its model_fingerprint is {fp}, "
                             f"this model's is {want})")
    nl, k = root.get("num_layers"), root.get("top_k")
    if not _int(nl):
        raise WhitelistError('"num_layers" is missing or not an integer')
    if nl != model.n_layers:
        raise WhitelistError(f'"num_layers" is {nl}, the model has {model.n_layers}')
    if not _int(k):
        raise WhitelistError('"top_k" is missing or not an integer')
    if k != model.top_k:
        raise WhitelistError(f'"top_k" is {k}, the model routes {model.top_k}')
    epl = root.get("experts_per_layer")
    if epl is None:
        raise WhitelistError('"experts_per_layer" is missing')
    if _int(epl):
        for l, n in enumerate(model.experts_per_layer):
            if n > 0 and n != epl:
                raise WhitelistError(f'"experts_per_layer" is {epl}, layer {l} of the model has {n}')
    elif isinstance(epl, list):
        if len(epl) != nl:
            raise WhitelistError(f'"experts_per_layer" lists {len(epl)} layers, the model has {nl}')
        for l, n in enumerate(epl):
            if n != model.experts_per_layer[l]:
                raise WhitelistError(f'"experts_per_layer"[{l}] does not match the model ({model.experts_per_layer[l]})')
    else:
        raise WhitelistError('"experts_per_layer" must be an integer or a list')
    mm = root.get("mode", "strict")
    if mm not in MODES:
        raise WhitelistError('unknown "mode" (expected strict, substitute or tiered)')
    mode = mode_override or mm
    if mode not in MODES:
        raise WhitelistError(f"unknown mode {mode!r}")
    tier = [[0] * n for n in model.experts_per_layer]
    sub: list[list[int]] = [[] for _ in range(nl)]
    sim: list[list[float | None]] = [[] for _ in range(nl)]
    thr = root.get("substitute_threshold")
    if thr is not None and (not isinstance(thr, (int, float)) or isinstance(thr, bool) or not -1.0 <= thr <= 1.0):
        raise WhitelistError('"substitute_threshold" must be a number from -1 to 1 (the similarity a replacement needs)')
    seen = [False] * nl
    layers = root.get("layers")
    if not isinstance(layers, list):
        raise WhitelistError('"layers" is missing or not a list')

    def read_ids(lst, name, l, ne, t):
        if lst is None:
            return
        if not isinstance(lst, list):
            raise WhitelistError(f'layer {l}: "{name}" must be a list of expert ids')
        for e in lst:
            if not _int(e):
                raise WhitelistError(f'layer {l}: "{name}" holds a value that is not an integer expert id')
            if e < 0 or e >= ne:
                raise WhitelistError(f'layer {l}: expert id {e} in "{name}" is out of range (the layer has experts '
                                     f'0..{ne - 1})')
            if tier[l][e] == t:
                raise WhitelistError(f'layer {l}: expert {e} is listed twice in "{name}"')
            if tier[l][e] != 0:
                raise WhitelistError(f'layer {l}: expert {e} is both "allowed" and "on_demand"')
            tier[l][e] = t

    for L in layers:
        if not isinstance(L, dict):
            raise WhitelistError('an entry of "layers" is not an object')
        l = L.get("layer")
        if not _int(l):
            raise WhitelistError('an entry of "layers" has no integer "layer"')
        if l < 0 or l >= nl:
            raise WhitelistError(f"layer id {l} is out of range (the model has layers 0..{nl - 1})")
        if seen[l]:
            raise WhitelistError(f"layer {l} is listed twice")
        seen[l] = True
        ne = model.experts_per_layer[l]
        if ne <= 0:
            raise WhitelistError(f"layer {l} has no routed experts and cannot be filtered")
        if "allowed" not in L:
            raise WhitelistError(f'layer {l} has no "allowed" list')
        read_ids(L.get("allowed"), "allowed", l, ne, 1)
        read_ids(L.get("on_demand"), "on_demand", l, ne, 2)
        if "substitutes" in L:
            S = L["substitutes"]
            if not isinstance(S, list):
                raise WhitelistError(f'layer {l}: "substitutes" must be a list')
            row = [-1] * ne
            srow: list[float | None] = [None] * ne
            for s in S:
                if not isinstance(s, dict) or not _int(s.get("expert")) or not _int(s.get("with")):
                    raise WhitelistError(f'layer {l}: a substitute needs integer "expert" and "with"')
                e, w = s["expert"], s["with"]
                sv = s.get("similarity")
                if sv is not None and (not isinstance(sv, (int, float)) or isinstance(sv, bool) or not -1.0 <= sv <= 1.0):
                    raise WhitelistError(f'layer {l}: the substitute of expert {e} has a "similarity" that is not a number '
                                         "from -1 to 1")
                if not (0 <= e < ne and 0 <= w < ne):
                    raise WhitelistError(f"layer {l}: substitute {e} -> {w} is out of range (the layer has experts "
                                         f"0..{ne - 1})")
                if row[e] >= 0:
                    raise WhitelistError(f"layer {l}: expert {e} has two substitutes")
                if tier[l][e] == 1:
                    raise WhitelistError(f"layer {l}: expert {e} is allowed and cannot have a substitute")
                if tier[l][w] != 1:
                    raise WhitelistError(f'layer {l}: the substitute of expert {e} is expert {w}, which is not in "allowed"')
                row[e] = w
                srow[e] = float(sv) if sv is not None else None
            sub[l] = row
            sim[l] = srow
        if "scores" in L and not isinstance(L["scores"], (dict, list)):
            raise WhitelistError(f'layer {l}: "scores" must be an object or a list')
    for l in range(nl):
        if model.experts_per_layer[l] > 0 and not seen[l]:
            raise WhitelistError(f'layer {l} is an MoE layer and is missing from "layers"')
    gen = root.get("generation") if isinstance(root.get("generation"), dict) else {}
    listed = sub if any(sub) else []
    if listed:
        sub = [r if r else [-1] * model.experts_per_layer[l] for l, r in enumerate(sub)]
        sim = [r if r else [None] * model.experts_per_layer[l] for l, r in enumerate(sim)]
        listed = sub
    wl = Whitelist(mode, mm, nl, k, tier, [], str(gen.get("profile", "")), listed, sim if listed else [],
                   float(thr) if thr is not None else None)
    wl.substitute = substitute_table(wl)
    for l in range(nl):
        if model.experts_per_layer[l] <= 0:
            continue
        n = sum(wl.routable(l, e) for e in range(model.experts_per_layer[l]))
        if n < k:
            raise WhitelistError(f"layer {l} allows {n} experts in {mode} mode, fewer than the model's top_k {k} "
                                 "(top_k is not lowered)")
    return wl


def substitute_table(wl: Whitelist, threshold: float | None = None, use_manifest: bool = True) -> list[list[int]]:
    """The replacements in use: the listed ones at or above the threshold (the manifest's, or `threshold` when
    use_manifest is False); one without a similarity is always used (src/core/expert_whitelist.cpp)."""
    t = wl.substitute_threshold if use_manifest else threshold
    out = []
    for l, row in enumerate(wl.substitute_listed):
        sims = wl.substitute_similarity[l]
        out.append([w if w >= 0 and (t is None or sims[e] is None or sims[e] >= t) else -1 for e, w in enumerate(row)])
    return out


def manifest_document(model: ModelInfo, allowed: dict[int, list[int]], mode: str = "strict",
                      on_demand: dict[int, list[int]] | None = None,
                      substitutes: dict[int, list[tuple]] | None = None,
                      scores: dict[int, dict[int, float]] | None = None, generation: dict | None = None,
                      substitute_threshold: float | None = None) -> dict:
    """A manifest in the engine's format (the generator writes these; validate_manifest reads them back).
    `substitutes`: per layer (expert, with, similarity) or (expert, with, similarity, rel_l2)."""
    layers = []
    for l in range(model.n_layers):
        if model.experts_per_layer[l] <= 0:
            continue
        L: dict = {"layer": l, "allowed": sorted(allowed.get(l, []))}
        if on_demand and on_demand.get(l):
            L["on_demand"] = sorted(on_demand[l])
        if substitutes and substitutes.get(l):
            L["substitutes"] = []
            for t in substitutes[l]:
                d = {"expert": t[0], "with": t[1], "similarity": round(t[2], 6)}
                if len(t) > 3 and t[3] is not None:
                    d["rel_l2"] = round(t[3], 6)
                L["substitutes"].append(d)
        if scores and scores.get(l):
            L["scores"] = {str(e): s for e, s in sorted(scores[l].items())}
        layers.append(L)
    epl = model.experts_per_layer
    return {
        "format_version": FORMAT_VERSION,
        "kind": MANIFEST_KIND,
        "model_fingerprint": model.fingerprint(),
        "num_layers": model.n_layers,
        "experts_per_layer": epl[0] if len(set(epl)) == 1 else epl,
        "top_k": model.top_k,
        "mode": mode,
        **({"substitute_threshold": substitute_threshold} if substitute_threshold is not None else {}),
        "layers": layers,
        "generation": generation or {},
    }


# ================================ PROFILES ================================

def _paths(base: pathlib.Path, value, where: str, must_exist: bool = True) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise WhitelistError(f"{where}: must be a path or a list of paths")
    out = []
    for v in value:
        q = pathlib.Path(os.path.expanduser(v))
        if not q.is_absolute():
            q = (base / q) if (base / q).exists() or not (REPO / q).exists() else REPO / q
        if must_exist and not q.exists():
            raise WhitelistError(f"{where}: {v} does not exist")
        out.append(str(q))
    return out


def _globs(base: pathlib.Path, value, where: str) -> list[str]:
    """Source globs, relative like calibration paths: to the profile's folder, or to the repository root when only
    that matches. A glob that matches nothing stays (prepare warns about it)."""
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise WhitelistError(f"{where}: must be a glob or a list of globs")
    out = []
    for v in value:
        g = os.path.expanduser(v)
        if not os.path.isabs(g):
            here, root = str(base / g), str(REPO / g)
            g = root if not glob.glob(here, recursive=True) and glob.glob(root, recursive=True) else here
        out.append(g)
    return out


def _coverage(v, where: str) -> float:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0.0 < float(v) <= 1.0:
        raise WhitelistError(f"{where}: min_coverage must be a number in (0, 1]")
    return float(v)


def _weight(v, where: str) -> float:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not float(v) > 0.0:
        raise WhitelistError(f"{where}: weight must be a number above 0")
    return float(v)


def load_profile(path: str) -> dict:
    """Reads and checks one profile (docs/EXPERT_WHITELIST.md section 2.1); calibration paths become absolute."""
    p = pathlib.Path(path)
    try:
        doc = json.loads(p.read_text(), object_pairs_hook=_no_dup_keys)
    except json.JSONDecodeError as e:
        raise WhitelistError(f"{path}: JSON line {e.lineno}: {e.msg}") from None
    except WhitelistError as e:
        raise WhitelistError(f"{path}: {e}") from None
    base = p.parent
    w = str(path)
    if not isinstance(doc, dict) or doc.get("kind") != PROFILE_KIND:
        raise WhitelistError(f'{w}: not a profile ("kind" must be "{PROFILE_KIND}")')
    if doc.get("format_version") != FORMAT_VERSION:
        raise WhitelistError(f"{w}: unknown format_version {doc.get('format_version')!r} (this tool reads {FORMAT_VERSION})")
    name = doc.get("name")
    if not isinstance(name, str) or not name:
        raise WhitelistError(f'{w}: "name" is missing')
    langs = doc.get("target_languages")
    if not isinstance(langs, dict) or not langs:
        raise WhitelistError(f'{w}: "target_languages" must name at least one language')
    out_langs = {}
    for lang, spec in langs.items():
        where = f"{w}: target_languages.{lang}"
        if not isinstance(spec, dict):
            raise WhitelistError(f"{where}: must be an object with weight and min_coverage")
        cal = _paths(base, spec.get("calibration"), where + ".calibration")
        src = spec.get("sources", [])
        if isinstance(src, str):
            src = [src]
        if not cal and not src:
            raise WhitelistError(f"{where}: needs calibration files or source globs")
        out_langs[lang] = {"weight": _weight(spec.get("weight", 1.0), where),
                           "min_coverage": _coverage(spec.get("min_coverage", 0.95), where),
                           "calibration": cal, "sources": _globs(base, src, where + ".sources")}
    tasks = doc.get("tasks", {})
    if not isinstance(tasks, dict):
        raise WhitelistError(f'{w}: "tasks" must map task names to weights')
    tasks = {t: _weight(v, f"{w}: tasks.{t}") for t, v in tasks.items()}
    base_req = doc.get("base_requirements", {})
    if not isinstance(base_req, dict):
        raise WhitelistError(f'{w}: "base_requirements" must be an object')
    enabled = base_req.get("enabled", True)
    if not isinstance(enabled, bool):
        raise WhitelistError(f"{w}: base_requirements.enabled must be true or false")
    items = {}
    for item, spec in (base_req.get("items") or {}).items():
        where = f"{w}: base_requirements.items.{item}"
        if not isinstance(spec, dict):
            raise WhitelistError(f"{where}: must be an object")
        items[item] = {"calibration": _paths(base, spec.get("calibration"), where + ".calibration"),
                       "min_coverage": _coverage(spec.get("min_coverage", 0.9), where)}
    if enabled and not items:
        raise WhitelistError(f"{w}: base_requirements are on but list no items (set \"enabled\": false to drop them)")
    nat = base_req.get("natural_languages", ["en"])
    if not isinstance(nat, list) or not all(isinstance(x, str) for x in nat):
        raise WhitelistError(f"{w}: base_requirements.natural_languages must be a list of language codes")
    excluded = {}
    for ex, spec in (doc.get("excluded") or {}).items():
        where = f"{w}: excluded.{ex}"
        if ex in out_langs:
            raise WhitelistError(f"{where}: is also a target language")
        if not isinstance(spec, dict):
            raise WhitelistError(f"{where}: must be an object")
        excluded[ex] = {"calibration": _paths(base, spec.get("calibration"), where + ".calibration")}
    hw = doc.get("target_hardware")
    if hw is not None:
        if not isinstance(hw, dict):
            raise WhitelistError(f'{w}: "target_hardware" must be an object')
        for key in ("vram_gib", "ram_gib", "reserve_gib"):
            v = hw.get(key, 0)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                raise WhitelistError(f"{w}: target_hardware.{key} must be a number >= 0")
    limits = doc.get("limits", {})
    if not isinstance(limits, dict):
        raise WhitelistError(f'{w}: "limits" must be an object')
    for key, lo in (("max_samples", 1), ("max_tokens_per_sample", 1), ("seed", 0)):
        if key in limits and (not _int(limits[key]) or limits[key] < lo):
            raise WhitelistError(f"{w}: limits.{key} must be an integer >= {lo}")
    ref = _paths(base, doc.get("reference_corpus"), f"{w}: reference_corpus")
    tmpl = doc.get("chat_template", "model")
    if not isinstance(tmpl, str):
        raise WhitelistError(f'{w}: "chat_template" must be "model", "none" or a path')
    return {"name": name, "description": doc.get("description", ""), "target_languages": out_langs, "tasks": tasks,
            "base_requirements": {"enabled": enabled, "natural_languages": nat, "items": items if enabled else {}},
            "excluded": excluded, "target_hardware": hw, "reference_corpus": ref,
            "limits": {"max_samples": limits.get("max_samples", 2000),
                       "max_tokens_per_sample": limits.get("max_tokens_per_sample", 2048), "seed": limits.get("seed", 0)},
            "chat_template": tmpl, "files": [str(p)]}


def combine_profiles(profiles: list[dict]) -> dict:
    """`--profile rust.json --profile typescript.json`: one profile.  A language or task in several takes the
    largest weight and coverage and all their calibration files; the hardware is the smallest VRAM and RAM and the
    largest reserve; an excluded domain that another profile targets is dropped from `excluded`."""
    if not profiles:
        raise WhitelistError("no profile given")
    out = json.loads(json.dumps(profiles[0]))
    out["name"] = "+".join(p["name"] for p in profiles)
    for p in profiles[1:]:
        for lang, spec in p["target_languages"].items():
            cur = out["target_languages"].setdefault(lang, {"weight": 0.0, "min_coverage": 0.0, "calibration": [], "sources": []})
            cur["weight"] = max(cur["weight"], spec["weight"])
            cur["min_coverage"] = max(cur["min_coverage"], spec["min_coverage"])
            cur["calibration"] += [c for c in spec["calibration"] if c not in cur["calibration"]]
            cur["sources"] += [s for s in spec["sources"] if s not in cur["sources"]]
        for t, v in p["tasks"].items():
            out["tasks"][t] = max(out["tasks"].get(t, 0.0), v)
        b, ob = p["base_requirements"], out["base_requirements"]
        ob["enabled"] = ob["enabled"] or b["enabled"]
        ob["natural_languages"] += [n for n in b["natural_languages"] if n not in ob["natural_languages"]]
        for item, spec in b["items"].items():
            cur = ob["items"].setdefault(item, {"calibration": [], "min_coverage": 0.0})
            cur["min_coverage"] = max(cur["min_coverage"], spec["min_coverage"])
            cur["calibration"] += [c for c in spec["calibration"] if c not in cur["calibration"]]
        for ex, spec in p["excluded"].items():
            out["excluded"].setdefault(ex, {"calibration": []})["calibration"] += spec["calibration"]
        if p["target_hardware"]:
            if not out["target_hardware"]:
                out["target_hardware"] = dict(p["target_hardware"])
            else:
                h, oh = p["target_hardware"], out["target_hardware"]
                for key in ("vram_gib", "ram_gib"):
                    if key in h:
                        oh[key] = min(oh.get(key, h[key]), h[key])
                oh["reserve_gib"] = max(oh.get("reserve_gib", 0), h.get("reserve_gib", 0))
        for key in ("max_samples", "max_tokens_per_sample"):
            out["limits"][key] = max(out["limits"][key], p["limits"][key])
        out["reference_corpus"] += [r for r in p["reference_corpus"] if r not in out["reference_corpus"]]
        out["files"] += p["files"]
    for lang in out["target_languages"]:
        out["excluded"].pop(lang, None)
    return out


def read_calibration(path: str) -> list[dict]:
    """A calibration file: JSON lines with "prompt" (or a chat "messages" list) and optionally "response", "id",
    "task" and "language"."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise WhitelistError(f"{path}:{n}: {e.msg}") from None
            if not isinstance(row, dict):
                raise WhitelistError(f'{path}:{n}: each line needs a non-empty "prompt" or a "messages" list')
            msgs = row.get("messages")
            if msgs is not None:
                if not isinstance(msgs, list) or not msgs or not all(
                        isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str)
                        for m in msgs):
                    raise WhitelistError(f'{path}:{n}: "messages" must be a non-empty list of {{"role", "content"}}')
            elif not isinstance(row.get("prompt"), str) or not row["prompt"]:
                raise WhitelistError(f'{path}:{n}: each line needs a non-empty "prompt" or a "messages" list')
            for key in ("response", "task", "language", "id"):
                if key in row and not isinstance(row[key], str):
                    raise WhitelistError(f'{path}:{n}: "{key}" must be a string')
            rows.append(row)
    return rows


def profile_files(profile: dict) -> list[str]:
    files = []
    for spec in profile["target_languages"].values():
        files += spec["calibration"]
    for spec in profile["base_requirements"]["items"].values():
        files += spec["calibration"]
    for spec in profile["excluded"].values():
        files += spec["calibration"]
    return files + profile["reference_corpus"]


# ================================ CLI ================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fingerprint", help="the model fingerprint a manifest for this pack needs")
    f.add_argument("--pack")
    f.add_argument("--gguf", help="the model's first GGUF shard (top-k and width from its metadata)")
    f.add_argument("--top-k", type=int)
    f.add_argument("--hidden", type=int)
    f.add_argument("--mtp", action="store_true", help="the MTP draft layer's fingerprint")
    f.add_argument("--text", action="store_true", help="also print the canonical text that is hashed")
    i = sub.add_parser("index", help="write PACK/experts.index.json (changes nothing else in the pack)")
    i.add_argument("pack")
    i.add_argument("--check", action="store_true", help="only compare the index that is there")
    v = sub.add_parser("validate", help="check a manifest against a pack, as the engine does")
    v.add_argument("manifest")
    v.add_argument("--pack")
    v.add_argument("--gguf")
    v.add_argument("--top-k", type=int)
    v.add_argument("--hidden", type=int)
    v.add_argument("--mtp", action="store_true", help="the manifest is the MTP draft layer's")
    v.add_argument("--mode", choices=MODES)
    c = sub.add_parser("check-profile", help="check profiles and their calibration files; several are combined")
    c.add_argument("profiles", nargs="+")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "fingerprint":
            info = mtp_model_info() if a.mtp else pack_model_info(a.pack or "", a.top_k, a.hidden, a.gguf)
            if a.text:
                sys.stdout.write(info.fingerprint_text())
            print(info.fingerprint())
        elif a.cmd == "index":
            path, ok = write_index(a.pack, a.check)
            if a.check:
                print(f"{path}: {'matches the pack' if ok else 'missing or stale'}")
                return 0 if ok else 1
            print(f"wrote {path}")
        elif a.cmd == "validate":
            info = mtp_model_info() if a.mtp else pack_model_info(a.pack or "", a.top_k, a.hidden, a.gguf)
            wl = validate_manifest(pathlib.Path(a.manifest).read_text(), info, a.mode)
            allowed = sum(t == 1 for row in wl.tier for t in row)
            total = sum(len(row) for row in wl.tier)
            print(f"{a.manifest}: valid ({wl.mode} mode, {allowed} of {total} experts allowed)")
        elif a.cmd == "check-profile":
            prof = combine_profiles([load_profile(p) for p in a.profiles])
            rows = 0
            for path in profile_files(prof):
                for g in sorted(glob.glob(path)) or [path]:
                    rows += len(read_calibration(g))
            print(f"{prof['name']}: valid ({len(prof['target_languages'])} target languages, "
                  f"{len(prof['base_requirements']['items'])} base requirements, {rows} calibration samples)")
    except (WhitelistError, OSError) as e:
        print(f"expert_whitelist: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
