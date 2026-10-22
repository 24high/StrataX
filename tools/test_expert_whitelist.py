"""Tests for tools/expert_whitelist.py and `tools/iq_pack.py index` (no model, no GPU, no numpy).

    python -m unittest tools.test_expert_whitelist

The fingerprint and the manifest rules are the engine's (src/core/expert_whitelist.cpp); the same fixture model
(3 layers of 16 experts, top-2, width 8, 384-byte blobs) and its fingerprint are in tests/core/expert_whitelist_test.cpp,
which also reads the manifests in tests/data/expert_whitelist that this test keeps current.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import expert_whitelist as W  # noqa: E402

MINI = W.ModelInfo(3, [16] * 3, 2, 8, [0, 6144, 12288], [384] * 3, "test")
FIXTURES = REPO / "tests" / "data" / "expert_whitelist"
PROFILES = REPO / "data" / "expert_whitelist" / "profiles"
CALIBRATION = REPO / "data" / "expert_whitelist" / "calibration"
GEN = {"profile": "rust-coding", "scoring": {"method": "fixture"}, "calibration_tokens": 0, "created": "2026-10-21"}


def tree_hashes(root: pathlib.Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def fixture_docs() -> dict[str, dict]:
    return {
        "mini-strict.json": W.manifest_document(MINI, {0: [1, 5, 6, 14], 1: [0, 3, 9, 12, 13], 2: [2, 4, 7]}, generation=GEN),
        "mini-tiered.json": W.manifest_document(MINI, {0: [1, 5], 1: [0, 3], 2: [2, 4]}, mode="tiered",
                                                on_demand={0: [6, 14], 1: [9]},
                                                substitutes={0: [(2, 1, 0.81), (3, 5, 0.77)]},
                                                scores={0: {1: 0.5, 5: 0.25}}, generation=GEN),
        # the substitute mode's fields as the generator writes them: a threshold, similarities, a relative L2 distance
        "mini-substitute.json": W.manifest_document(MINI, {0: [1, 5, 6, 14], 1: [0, 3, 9, 12, 13], 2: [2, 4, 7]},
                                                    mode="substitute", substitute_threshold=0.6,
                                                    substitutes={0: [(2, 1, 0.91, 0.42), (3, 5, 0.3, 1.1), (4, 6, 0.6)],
                                                                 2: [(0, 7, 0.75)]}, generation=GEN),
    }


class Fingerprint(unittest.TestCase):
    def test_matches_the_engine(self):
        # the same text and value as tests/core/expert_whitelist_test.cpp
        self.assertEqual(MINI.fingerprint_text(),
                         "strata-expert-whitelist-fingerprint-v1\nnum_layers=3\ntop_k=2\nhidden=8\nexperts_per_layer=16,16,16\n"
                         "pack_index=0:0:384;1:6144:384;2:12288:384\n")
        self.assertEqual(MINI.fingerprint(), "fnv1a64:25d1f5de79ecddd4")

    def test_mtp(self):
        self.assertEqual(W.mtp_model_info().fingerprint(), "fnv1a64:efe972152b97798f")

    def test_changes_with_the_pack(self):
        other = W.ModelInfo(3, [16] * 3, 2, 8, [0, 6144, 12288], [384, 400, 384])
        self.assertNotEqual(other.fingerprint(), MINI.fingerprint())
        self.assertNotEqual(W.ModelInfo(3, [16] * 3, 3, 8, MINI.layer_offset, MINI.blob_bytes).fingerprint(),
                            MINI.fingerprint())


class Manifest(unittest.TestCase):
    def doc(self, **kw):
        return W.manifest_document(MINI, {0: [1, 5, 6, 14], 1: [0, 3, 9, 12, 13], 2: [2, 4, 7]}, generation=GEN, **kw)

    def fails(self, doc_or_text, needle, mode=None):
        text = doc_or_text if isinstance(doc_or_text, str) else json.dumps(doc_or_text)
        with self.assertRaises(W.WhitelistError) as cm:
            W.validate_manifest(text, MINI, mode)
        self.assertIn(needle, str(cm.exception))

    def test_fixtures_are_current(self):
        for name, doc in fixture_docs().items():
            self.assertEqual(json.loads((FIXTURES / name).read_text()), doc, f"{name}: regenerate it from fixture_docs()")
            W.validate_manifest((FIXTURES / name).read_text(), MINI)

    def test_valid(self):
        wl = W.validate_manifest(json.dumps(self.doc()), MINI)
        self.assertEqual(wl.mode, "strict")
        self.assertTrue(wl.routable(0, 14) and not wl.routable(0, 2) and not wl.routable(0, 16))
        self.assertEqual(wl.profile, "rust-coding")

    def test_errors(self):
        d = self.doc()
        d["model_fingerprint"] = "fnv1a64:0000000000000000"
        self.fails(d, "made for another model")
        d = self.doc(); d["format_version"] = 2
        self.fails(d, "unknown format_version 2")
        d = self.doc(); d["layers"].append({"layer": 3, "allowed": [1, 2]})
        self.fails(d, "layer id 3 is out of range")
        d = self.doc(); d["layers"][0]["allowed"] = [1, 16]
        self.fails(d, 'expert id 16 in "allowed" is out of range')
        d = self.doc(); d["layers"][0]["allowed"] = [1, 5, 5]
        self.fails(d, "expert 5 is listed twice")
        d = self.doc(); del d["layers"][1]
        self.fails(d, "layer 1 is an MoE layer and is missing")
        d = self.doc(); d["layers"][0]["allowed"] = [1]
        self.fails(d, "layer 0 allows 1 experts in strict mode, fewer than the model's top_k 2")
        d = self.doc(); d["layers"].append({"layer": 2, "allowed": [3, 4]})
        self.fails(d, "layer 2 is listed twice")
        self.fails('{"format_version": 1, "format_version": 1}', "duplicate key")
        self.fails('{"format_version": 1,', "JSON line 1")
        self.fails("[1, 2]", "not a JSON object")
        d = self.doc(); d["top_k"] = 3
        self.fails(d, '"top_k" is 3, the model routes 2')
        d = self.doc(); d["num_layers"] = 4
        self.fails(d, '"num_layers" is 4')
        d = self.doc(); d["experts_per_layer"] = 8
        self.fails(d, '"experts_per_layer" is 8')
        d = self.doc(); d["kind"] = "something-else"
        self.fails(d, "not an expert whitelist")
        d = self.doc(); d["mode"] = "loose"
        self.fails(d, 'unknown "mode"')
        d = self.doc(); d["layers"][0]["on_demand"] = [5]
        self.fails(d, 'both "allowed" and "on_demand"')
        d = self.doc(); d["layers"][0]["substitutes"] = [{"expert": 2, "with": 3}]
        self.fails(d, 'which is not in "allowed"')
        d = self.doc(); d["layers"][0]["substitutes"] = [{"expert": 5, "with": 1}]
        self.fails(d, "is allowed and cannot have a substitute")

    def test_substitute_threshold(self):
        wl = W.validate_manifest((FIXTURES / "mini-substitute.json").read_text(), MINI)
        self.assertEqual(wl.substitute_threshold, 0.6)
        self.assertEqual(wl.substitute_listed[0][:5], [-1, -1, 1, 5, 6])
        self.assertEqual(wl.substitute[0][:5], [-1, -1, 1, -1, 6])     # 0.3 is below 0.6; 0.6 itself is used
        self.assertEqual(wl.substitute[2][0], 7)
        self.assertEqual(W.substitute_table(wl, 0.8, use_manifest=False)[0][:5], [-1, -1, 1, -1, -1])
        self.assertEqual(W.substitute_table(wl, None, use_manifest=False)[0][:5], [-1, -1, 1, 5, 6])
        d = self.doc(substitute_threshold=1.5)
        self.fails(d, '"substitute_threshold" must be a number')
        d = self.doc(); d["substitute_threshold"] = True
        self.fails(d, '"substitute_threshold" must be a number')
        d = self.doc(); d["layers"][0]["substitutes"] = [{"expert": 2, "with": 1, "similarity": -2}]
        self.fails(d, 'has a "similarity" that is not a number')
        # a replacement without a similarity is always used
        d = self.doc(substitute_threshold=0.9); d["layers"][0]["substitutes"] = [{"expert": 2, "with": 1}]
        self.assertEqual(W.validate_manifest(json.dumps(d), MINI).substitute[0][2], 1)

    def test_tiered_and_override(self):
        doc = W.manifest_document(MINI, {0: [1], 1: [1, 2], 2: [1, 2]}, mode="tiered", on_demand={0: [9]})
        wl = W.validate_manifest(json.dumps(doc), MINI)
        self.assertTrue(wl.routable(0, 9))
        self.fails(doc, "layer 0 allows 1 experts in strict mode", mode="strict")


class PackIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="strata_wl_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def native_pack(self, fixture="iq3_s.txt"):
        pack = self.tmp / "pack"
        pack.mkdir()
        shutil.copy(REPO / "tests" / "data" / "native_experts" / fixture, pack / "native_experts.txt")
        (pack / "index.txt").write_text("strata pack index v3\n")
        (pack / "dense.bin").write_bytes(b"\x01\x02\x03\x04" * 64)
        return pack

    def test_native_layout(self):
        info = W.pack_model_info(str(self.native_pack()))
        self.assertEqual(info.n_layers, 48)
        self.assertEqual(info.experts_per_layer, [512] * 48)
        self.assertEqual(info.layer_offset[1], 1114112000)    # the fixture's second line
        self.assertEqual(info.blob_bytes[0], 2176000)
        coder = self.tmp / "coder"
        coder.mkdir()
        shutil.copy(REPO / "tests" / "data" / "native_experts" / "coder-iq1_m.txt", coder / "native_experts.txt")
        self.assertEqual(W.pack_model_info(str(coder)).experts_per_layer[0], 256)   # the header's n_expert wins

    def test_canonical_layout(self):
        pack = self.tmp / "canon"
        pack.mkdir()
        (pack / "manifest.json").write_text(json.dumps({"format_version": 1, "n_layers": 48, "n_experts_per_layer": 512}))
        info = W.pack_model_info(str(pack))
        self.assertEqual(info.layer_offset[2], 2 * 512 * 1382400)
        self.assertEqual(info.fingerprint(), W.pack_model_info(str(pack), 10, 2560).fingerprint())

    def test_index_leaves_the_pack_unchanged(self):
        pack = self.native_pack()
        before = tree_hashes(pack)
        path, ok = W.write_index(str(pack))
        self.assertTrue(ok and path.exists())
        after = tree_hashes(pack)
        after.pop(W.INDEX_NAME)
        self.assertEqual(before, after)   # every file the pack had is byte for byte what it was
        doc = json.loads(path.read_text())
        self.assertEqual(doc["num_layers"], 48)
        self.assertEqual(doc["layers"][1], {"layer": 1, "n_expert": 512, "offset": 1114112000, "blob_bytes": 1510400})
        self.assertTrue(W.write_index(str(pack), check=True)[1])
        doc["layers"][1]["offset"] += 1
        path.write_text(json.dumps(doc))
        self.assertFalse(W.write_index(str(pack), check=True)[1])

    def test_experts_bin_size_is_checked(self):
        pack = self.tmp / "tiny"
        pack.mkdir()
        (pack / "native_experts.txt").write_text(
            "# strata native experts v3: test (n_expert 4, total 128; test)\n0 18 20 0 16 0 0 0\n1 18 20 64 16 0 0 0\n")
        (pack / "experts.bin").write_bytes(b"\0" * 128)
        W.write_index(str(pack))
        (pack / "experts.bin").write_bytes(b"\0" * 100)
        with self.assertRaises(W.WhitelistError):
            W.write_index(str(pack))

    def test_iq_pack_index_command(self):
        # `tools/iq_pack.py index` runs before the packer's numpy import, and changes nothing but the new file
        pack = self.native_pack("swift-iq2_xs.txt")
        before = tree_hashes(pack)
        r = subprocess.run([sys.executable, str(HERE / "iq_pack.py"), "index", str(pack)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        after = tree_hashes(pack)
        self.assertIn(W.INDEX_NAME, after)
        after.pop(W.INDEX_NAME)
        self.assertEqual(before, after)
        r = subprocess.run([sys.executable, str(HERE / "iq_pack.py"), "index", "--check", str(pack)], capture_output=True,
                           text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_validate_against_a_pack(self):
        pack = self.native_pack()
        info = W.pack_model_info(str(pack))
        doc = W.manifest_document(info, {l: list(range(0, 512, 4)) for l in range(48)})
        (self.tmp / "wl.json").write_text(json.dumps(doc))
        r = subprocess.run([sys.executable, str(HERE / "expert_whitelist.py"), "validate", str(self.tmp / "wl.json"),
                            "--pack", str(pack)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("6144 of 24576 experts allowed", r.stdout)
        r = subprocess.run([sys.executable, str(HERE / "expert_whitelist.py"), "validate", str(self.tmp / "wl.json"),
                            "--pack", str(pack), "--top-k", "8"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("made for another model", r.stderr)


class Profiles(unittest.TestCase):
    def test_examples(self):
        rust = W.load_profile(str(PROFILES / "rust-coding.json"))
        py = W.load_profile(str(PROFILES / "python-coding.json"))
        self.assertEqual(rust["target_languages"]["rust"]["min_coverage"], 0.95)
        self.assertEqual(rust["base_requirements"]["natural_languages"], ["en", "de"])
        self.assertIn("python", rust["excluded"])
        for p in (rust, py):
            for f in W.profile_files(p):
                self.assertTrue(W.read_calibration(f))

    def test_combine(self):
        rust = W.load_profile(str(PROFILES / "rust-coding.json"))
        ts = W.load_profile(str(PROFILES / "typescript-coding.json"))
        py = W.load_profile(str(PROFILES / "python-coding.json"))
        c = W.combine_profiles([rust, ts])
        self.assertEqual(c["name"], "rust-coding+typescript-coding")
        self.assertEqual(set(c["target_languages"]), {"rust", "typescript"})
        self.assertEqual(c["target_languages"]["typescript"]["weight"], 0.5)
        self.assertTrue(c["base_requirements"]["enabled"])   # rust's are kept: one profile turning them off does not drop them
        c2 = W.combine_profiles([rust, py])
        self.assertNotIn("python", c2["excluded"])           # a target of one profile is not excluded by the other
        self.assertNotIn("rust", c2["excluded"])
        self.assertIn("prose", c2["excluded"])

    def test_errors(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="strata_wl_prof_"))
        try:
            base = json.loads((PROFILES / "rust-coding.json").read_text())

            def absolute(spec):   # the copy lives elsewhere: its calibration paths are made absolute first
                if isinstance(spec, dict):
                    for k, v in spec.items():
                        if k == "calibration":
                            spec[k] = [str((PROFILES / c).resolve()) for c in v]
                        else:
                            absolute(v)

            absolute(base)

            def fails(mutate, needle):
                d = json.loads(json.dumps(base))
                mutate(d)
                p = tmp / "p.json"
                p.write_text(json.dumps(d))
                with self.assertRaises(W.WhitelistError) as cm:
                    W.load_profile(str(p))
                self.assertIn(needle, str(cm.exception))

            (tmp / "ok.json").write_text(json.dumps(base))
            self.assertEqual(W.load_profile(str(tmp / "ok.json"))["name"], "rust-coding")
            fails(lambda d: d.pop("name"), '"name" is missing')
            fails(lambda d: d.update(kind="other"), "not a profile")
            fails(lambda d: d["target_languages"]["rust"].update(min_coverage=1.5), "min_coverage must be a number in (0, 1]")
            fails(lambda d: d["target_languages"]["rust"].update(weight=0), "weight must be a number above 0")
            fails(lambda d: d["excluded"].update(rust={"calibration": []}), "is also a target language")
            fails(lambda d: d["target_languages"]["rust"].update(calibration=["/no/such/file.jsonl"]), "does not exist")
            fails(lambda d: d["base_requirements"].update(items={}), "list no items")
            fails(lambda d: d.update(target_languages={}), "at least one language")
            fails(lambda d: d["limits"].update(seed=-1), "limits.seed")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_calibration_set(self):
        n = 0
        for f in sorted(CALIBRATION.glob("*.jsonl")):
            n += len(W.read_calibration(str(f)))
        self.assertGreater(n, 30)
        bad = pathlib.Path(tempfile.mkstemp(suffix=".jsonl")[1])
        try:
            bad.write_text('{"response": "no prompt"}\n')
            with self.assertRaises(W.WhitelistError):
                W.read_calibration(str(bad))
        finally:
            os.unlink(bad)


if __name__ == "__main__":
    unittest.main()
