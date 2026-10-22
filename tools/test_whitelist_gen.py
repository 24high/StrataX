"""Tests for tools/whitelist_gen.py (no model, no GPU, no numpy).

    python -m unittest tools.test_whitelist_gen

The statistics fixture tests/data/expert_whitelist/stats-mini is written by tests/core/routing_stats_test.cpp from a
synthetic mini-MoE (4 layers, layer 2 dense, 16 experts, top-2) whose router sends rust tokens to experts 0-3, python to
4-7 and prose (the base requirement "instructions" and the reference corpus) to 8-11; 12-15 are general.  So a correct
generator must find exactly those experts.  The C++ test checks that the fixture is what its collector writes; this
test reads it with the Python reader, so the two sides of the format stay in step.
"""
from __future__ import annotations

import array
import contextlib
import io
import json
import math
import os
import pathlib
import shutil
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import expert_whitelist as W  # noqa: E402
import whitelist_gen as G  # noqa: E402

FIXTURE = REPO / "tests" / "data" / "expert_whitelist" / "stats-mini"
# the same samples on a model whose expert 4 is a copy of expert 0 and expert 5 is expert 1 with its output halved, with
# the experts' similarity data (a quarter of the tokens): written by routing_stats_test too
FIXTURE_SIM = REPO / "tests" / "data" / "expert_whitelist" / "stats-mini-sim"
NATIVE = REPO / "tests" / "data" / "native_experts"


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="strata_wlgen_"))
        (self.tmp / "cal.jsonl").write_text('{"prompt": "x"}\n')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def profile(self, name, langs, base=None, tasks=None, hw=None):
        """A profile for the fixture's categories (its calibration file is a placeholder: build reads only weights)."""
        doc = {"format_version": 1, "kind": W.PROFILE_KIND, "name": name,
               "target_languages": {k: {"weight": w, "min_coverage": c, "calibration": ["cal.jsonl"]}
                                    for k, (w, c) in langs.items()},
               "tasks": tasks if tasks is not None else {"codegen": 1.0, "compiler_errors": 0.9},
               "base_requirements": {"enabled": bool(base),
                                     "items": {k: {"calibration": ["cal.jsonl"], "min_coverage": c}
                                               for k, c in (base or {}).items()}}}
        if hw:
            doc["target_hardware"] = hw
        p = self.tmp / f"{name}.json"
        p.write_text(json.dumps(doc))
        return str(p)

    def build(self, profiles, out="w.json", **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            r = G.build(str(FIXTURE), profiles, str(self.tmp / out), **kw)
        doc = json.loads((self.tmp / out).read_text())
        return r, {L["layer"]: L["allowed"] for L in doc["layers"]}


# ================================ THE STATISTICS FORMAT ================================

class StatsFormat(Tmp):
    def test_reads_the_cpp_fixture(self):
        st = G.read_stats(str(FIXTURE))
        self.assertEqual(sorted(st.layers), [0, 1, 3])
        self.assertEqual(st.model.experts_per_layer, [16, 16, 0, 16])
        self.assertEqual(st.model.fingerprint(), st.fingerprint)
        self.assertTrue(st.has_contrib)
        # every token was counted once per MoE layer, every pick once
        for l, ld in st.layers.items():
            self.assertEqual(sum(ld.tokens), st.meta["tokens"])
            picks = sum(sum(ld.col(c, 0)) for c in range(ld.C))
            self.assertEqual(picks, 2 * st.meta["tokens"])
            ranks = sum(sum(ld.col(c, 4)) + sum(ld.col(c, 5)) for c in range(ld.C))
            self.assertEqual(ranks, picks)

    def test_damaged_files_are_refused(self):
        bad = self.tmp / "bad"
        shutil.copytree(FIXTURE, bad)
        data = bytearray((bad / "layer_001.bin").read_bytes())
        data[100] ^= 1
        (bad / "layer_001.bin").write_bytes(bytes(data))
        with self.assertRaises(G.GenError) as cm:
            G.read_stats(str(bad))
        self.assertIn("CRC-32", str(cm.exception))
        shutil.rmtree(bad)
        shutil.copytree(FIXTURE, bad)
        meta = json.loads((bad / "meta.json").read_text())
        meta["model"]["blob_bytes"][0] += 1   # another pack: the recorded fingerprint no longer matches
        (bad / "meta.json").write_text(json.dumps(meta))
        with self.assertRaises(G.GenError) as cm:
            G.read_stats(str(bad))
        self.assertIn("fingerprint", str(cm.exception))

    def test_python_writer_matches_the_cpp_files(self):
        st = G.read_stats(str(FIXTURE))
        out = self.tmp / "copy"
        G.write_stats(str(out), dict(st.meta), st.layers)
        for l in st.layers:
            name = "layer_%03d.bin" % l
            self.assertEqual((out / name).read_bytes(), (FIXTURE / name).read_bytes())

    def test_samples_file_matches_the_cpp_files(self):
        cats, samples = G.read_samples(FIXTURE / "samples.bin")
        self.assertEqual(len(cats), 9)
        self.assertEqual(samples[0]["id"], "rust-0001")
        G.write_samples(self.tmp / "s.bin", cats, samples)
        self.assertEqual((self.tmp / "s.bin").read_bytes(), (FIXTURE / "samples.bin").read_bytes())


# ================================ MERGE ================================

class Merge(Tmp):
    def variant(self, name, scale, extra_category=False, run_suffix=None):
        """A second independent run: the fixture's numbers times `scale`, other sample ids and run id, optionally one
        more category."""
        st = G.read_stats(str(FIXTURE))
        meta = json.loads(json.dumps(st.meta))
        rid = G.fnv_hex(name)
        meta["run_id"] = rid
        meta["runs"] = [dict(meta["runs"][0], run_id=rid)]
        meta["samples_done"] = [f"{s}-{run_suffix or name}" for s in meta["samples_done"]]
        meta["collect"] = {"variant": name}
        meta["tokens"] *= scale
        layers = {}
        keys = list(st.categories)
        if extra_category:
            keys = keys + ["excluded|french|excluded|prompt|prefill"]
            meta["categories"] = meta["categories"] + [dict(G.split_key(keys[-1]), key=keys[-1])]
        for l, ld in st.layers.items():
            n = G.LayerData(l, ld.E, ld.k, len(keys))
            for c in range(ld.C):
                n.tokens[c] = ld.tokens[c] * scale
            vals = array.array("Q", (v * scale for v in ld.values))
            if extra_category:
                vals.extend(v + 1 for v in ld.values[:ld.E * ld.W])
                n.tokens[ld.C] = 7
            n.values = vals
            layers[l] = n
        out = self.tmp / name
        G.write_stats(str(out), meta, layers)
        return str(out)

    def files(self, d):
        d = pathlib.Path(d)
        return {p.name: p.read_bytes() for p in sorted(d.iterdir())}

    def test_commutative_and_associative(self):
        a, b, c = str(FIXTURE), self.variant("b", 2), self.variant("c", 3, extra_category=True)
        ab, ba = self.tmp / "ab", self.tmp / "ba"
        G.merge([a, b], str(ab))
        G.merge([b, a], str(ba))
        self.assertEqual(self.files(ab), self.files(ba))
        ab_c, bc, a_bc = self.tmp / "ab_c", self.tmp / "bc", self.tmp / "a_bc"
        G.merge([str(ab), c], str(ab_c))
        G.merge([b, c], str(bc))
        G.merge([a, str(bc)], str(a_bc))
        self.assertEqual(self.files(ab_c), self.files(a_bc))
        # and the numbers are the sums
        m = G.read_stats(str(ab_c))
        f = G.read_stats(a)
        self.assertEqual(m.meta["tokens"], f.meta["tokens"] * 6)
        ci = m.categories.index(f.categories[0])
        self.assertEqual(m.layers[0].col(ci, 1)[5], f.layers[0].col(0, 1)[5] * 6)
        self.assertIn("excluded|french|excluded|prompt|prefill", m.categories)
        self.assertEqual(len(m.meta["runs"]), 3)

    def test_refusals(self):
        with self.assertRaises(G.GenError) as cm:
            G.merge([str(FIXTURE), str(FIXTURE)], str(self.tmp / "twice"))
        self.assertIn("more than one input", str(cm.exception))
        b = self.variant("b", 1, run_suffix="x")
        same_ids = self.variant("same", 1, run_suffix="x")
        with self.assertRaises(G.GenError) as cm:
            G.merge([b, same_ids], str(self.tmp / "overlap"))
        self.assertIn("count twice", str(cm.exception))
        G.merge([b, same_ids], str(self.tmp / "overlap"), allow_overlap=True)
        other = self.tmp / "other"
        shutil.copytree(FIXTURE, other)
        meta = json.loads((other / "meta.json").read_text())
        meta["model"]["blob_bytes"] = [400, 400, 0, 400]
        meta["model_fingerprint"] = G.model_from_meta(meta["model"]).fingerprint()
        meta["run_id"] = "fnv1a64:0000000000000001"
        meta["runs"][0]["run_id"] = meta["run_id"]
        st = G.read_stats(str(FIXTURE))
        G.write_stats(str(other), meta, st.layers)
        with self.assertRaises(G.GenError) as cm:
            G.merge([str(FIXTURE), str(other)], str(self.tmp / "models"))
        self.assertIn("another model", str(cm.exception))


# ================================ BUILD ================================

class Build(Tmp):
    def test_finds_the_language_experts(self):
        rust = self.profile("rust", {"rust": (1.0, 0.95)})
        _, allowed = self.build([rust])
        self.assertEqual(allowed, {0: [0, 1, 2, 3], 1: [0, 1, 2, 3], 3: [0, 1, 2, 3]})

    def test_two_languages_and_a_base_requirement(self):
        rust = self.profile("rust", {"rust": (1.0, 0.95)})
        py = self.profile("python", {"python": (0.5, 0.9)}, tasks={"codegen": 1.0})
        _, allowed = self.build([rust, py], "w2.json")
        self.assertEqual(allowed, {l: list(range(8)) for l in (0, 1, 3)})
        base = self.profile("rust-base", {"rust": (1.0, 0.95)}, base={"instructions": 0.9})
        r, allowed = self.build([base], "w3.json")
        self.assertEqual(allowed, {l: [0, 1, 2, 3, 8, 9, 10, 11] for l in (0, 1, 3)})
        self.assertGreaterEqual(min(r["components"]["base:instructions"]["coverage_per_layer"].values()), 0.9)
        # a language with little data is not swamped by another: python's own coverage is met with its weight 0.5
        r, _ = self.build([rust, py], "w4.json")
        self.assertGreaterEqual(min(r["components"]["lang:python"]["coverage_per_layer"].values()), 0.9)

    def test_deterministic(self):
        rust = self.profile("rust", {"rust": (1.0, 0.95)})
        self.build([rust], "a.json")
        self.build([rust], "b.json")
        self.assertEqual((self.tmp / "a.json").read_bytes(), (self.tmp / "b.json").read_bytes())
        self.assertEqual((self.tmp / "a.coverage.csv").read_bytes(), (self.tmp / "b.coverage.csv").read_bytes())

    def test_the_manifest_is_valid_for_the_engine(self):
        rust = self.profile("rust", {"rust": (1.0, 0.95)})
        self.build([rust])
        st = G.read_stats(str(FIXTURE))
        wl = W.validate_manifest((self.tmp / "w.json").read_text(), st.model)
        self.assertTrue(wl.routable(0, 3) and not wl.routable(0, 4))
        doc = json.loads((self.tmp / "w.json").read_text())
        gen = doc["generation"]
        self.assertEqual(gen["scoring"]["method"], G.SCORING_METHOD)
        for key in ("lambda", "clip", "protect_mass", "n_min", "margin", "region_weights", "targets", "floor"):
            self.assertIn(key, gen["scoring"]["params"])
        self.assertEqual(gen["quantization"], "f32 (synthetic)")
        self.assertEqual(gen["stats"][0]["run_id"], st.meta["run_id"])

    def test_count_fraction_floor(self):
        rust = self.profile("rust", {"rust": (1.0, 0.95)})
        _, allowed = self.build([rust], "c.json", mode="count", count=6)
        self.assertTrue(all(len(v) == 6 and v[:4] == [0, 1, 2, 3] for v in allowed.values()))
        _, allowed = self.build([rust], "f.json", mode="fraction", fraction=0.5)
        self.assertTrue(all(len(v) == 8 for v in allowed.values()))
        # top_k + margin is the floor whatever the coverage asks
        _, allowed = self.build([rust], "low.json", coverage=0.1, protect_mass=0.0, margin=3)
        self.assertTrue(all(len(v) == 5 for v in allowed.values()))

    def test_uncertain_experts(self):
        rust = self.profile("rust", {"rust": (1.0, 0.95)})
        r, _ = self.build([rust], "u.json", n_min=10 ** 6, protect_mass=0.5, margin=0)
        self.assertTrue(all(sum(L["uncertain"].values()) > 0 for L in r["layers"]))
        self.assertTrue(r["violations"])   # the 95% sits on experts that are all "uncertain" now: reported per layer
        r2, allowed = self.build([rust], "u2.json", n_min=10 ** 6, include_uncertain=True)
        self.assertFalse(r2["violations"])

    def test_budget_and_target_hardware(self):
        rust = self.profile("rust", {"rust": (1.0, 0.95)})
        py = self.profile("python", {"python": (0.5, 0.95)}, tasks={"codegen": 1.0})
        with self.assertRaises(G.GenError) as cm:
            self.build([rust], "b1.json", mode="budget", budget="10")
        self.assertIn("floor", str(cm.exception))
        _, allowed = self.build([rust], "b2.json", mode="budget", budget="14")
        self.assertEqual(sum(len(v) for v in allowed.values()), 12)   # no other expert carries rust: nothing to add
        # two languages, low minimum coverage: the budget's rest goes by marginal utility, to the languages' experts
        lo_r = self.profile("rust-lo", {"rust": (1.0, 0.5)})
        lo_p = self.profile("python-lo", {"python": (0.5, 0.5)}, tasks={"codegen": 1.0})
        _, allowed = self.build([lo_r, lo_p], "b5.json", mode="budget", budget="20", protect_mass=0.0)
        self.assertEqual(sum(len(v) for v in allowed.values()), 20)
        self.assertTrue(all(e < 8 for v in allowed.values() for e in v))
        # the protected experts of both languages (80% of each one's mass) fit 22, their 95% does not
        with self.assertRaises(G.GenError) as cm:
            self.build([rust, py], "b3.json", mode="budget", budget="22")
        self.assertIn("min_coverage", str(cm.exception))
        r, allowed = self.build([rust, py], "b4.json", mode="budget", budget="22", allow_under_coverage=True)
        self.assertEqual(sum(len(v) for v in allowed.values()), 22)
        self.assertTrue(r["violations"] and r["warnings"])
        # the budget from the hardware: VRAM + RAM - reserve - non-expert weights - KV cache - buffer, shown in the report
        r, allowed = self.build([rust], "hw.json", mode="budget", target_hardware="vram=8K,ram=8K,reserve=0,buffer=1K",
                                context=64, kv_bytes_per_token=16)
        hw = r["target_hardware"]
        self.assertEqual(hw["expert_budget"], 16384 - 4224 - 64 * 16 - 1024)
        self.assertTrue(r["fits"])
        self.assertLessEqual(r["totals"]["expert_bytes_after"], hw["expert_budget"])
        self.assertIn("a calculation", (self.tmp / "hw.report.md").read_text())
        with self.assertRaises(G.GenError) as cm:
            self.build([rust], "hw2.json", mode="budget", target_hardware="vram=4K,ram=2K,reserve=0,buffer=0",
                       context=64, kv_bytes_per_token=16)
        self.assertIn("does not hold even the floor", str(cm.exception))

    def test_report_and_sweep(self):
        rust = self.profile("rust", {"rust": (1.0, 0.95)})
        out = str(self.tmp / "rust-{value}.json")
        with contextlib.redirect_stdout(io.StringIO()):
            rc = G.main(["build", "--stats", str(FIXTURE), "--profile", rust, "--sweep", "coverage=0.99,0.5",
                         "--out", out])
        self.assertEqual(rc, 0)
        for v in ("0.99", "0.5"):
            self.assertTrue((self.tmp / f"rust-{v}.json").exists())
            self.assertTrue((self.tmp / f"rust-{v}.report.md").exists())
        csv = (self.tmp / "rust-0.99.coverage.csv").read_text().splitlines()
        self.assertEqual(csv[0], "layer,experts,profile_mass,lang:rust")
        self.assertEqual(len(csv), 1 + 3 * 16)
        rep = json.loads((self.tmp / "rust-0.5.report.json").read_text())
        self.assertTrue(any("below the profile's min_coverage" in w for w in rep["warnings"]))
        text = G.inspect(str(self.tmp / "rust-0.99.json"))
        self.assertIn("allowed 12 of 48 experts", text)
        self.assertIn("lang:rust: mean coverage", text)

    def test_group_limited_routing(self):
        chosen = {0, 1, 4, 8, 9, 10, 11}
        blocked = G.apply_groups(chosen, 16, 4, 2, 4)   # 4 groups of 4, the router picks 2 groups: each needs 2 experts
        self.assertEqual(blocked, [1])                    # group 1 (experts 4-7) has one allowed expert: blocked
        self.assertEqual(chosen, {0, 1, 8, 9, 10, 11})


# ================================ PREPARE ================================

# ================================ THE MODES IN BUILD ================================

class BuildModes(Tmp):
    def test_tiered_second_coverage(self):
        # a resident tier for 60% of rust (and no margin), so the second coverage has something to add
        rust = self.profile("rust", {"rust": (1.0, 0.6)})
        kw = dict(coverage=0.6, margin=0, protect_mass=0.5)
        r, allowed = self.build([rust], "t.json", whitelist_mode="tiered", on_demand_coverage=0.995, **kw)
        _, strict = self.build([rust], "strict.json", **kw)
        doc = json.loads((self.tmp / "t.json").read_text())
        st = G.read_stats(str(FIXTURE))
        wl = W.validate_manifest(json.dumps(doc), st.model)
        self.assertEqual(doc["mode"], "tiered")
        self.assertEqual(allowed, strict)   # the resident tier is the strict selection
        od = {L["layer"]: L.get("on_demand", []) for L in doc["layers"]}
        self.assertTrue(any(od.values()))
        t = r["tiered"]
        for L in t["layers"]:
            self.assertFalse(set(od[L["layer"]]) & set(allowed[L["layer"]]))
            c = L["coverage"]["lang:rust"]
            self.assertGreaterEqual(c["with_on_demand"], 0.995 - 1e-9)   # the second coverage is reached
            self.assertLess(c["resident"], 0.995)
            self.assertTrue(all(wl.routable(L["layer"], e) and not wl.tier[L["layer"]][e] == 1 for e in od[L["layer"]]))
        self.assertEqual(t["on_demand_experts"], sum(len(v) for v in od.values()))
        self.assertGreater(t["on_demand_bytes_per_token_max"], 0)
        # deterministic, and the strict override still has its top_k
        self.build([rust], "t2.json", whitelist_mode="tiered", on_demand_coverage=0.995, **kw)
        self.assertEqual((self.tmp / "t.json").read_bytes(), (self.tmp / "t2.json").read_bytes())
        self.assertGreaterEqual(min(len(v) for v in allowed.values()), st.model.top_k)   # the floor: top_k + margin 0
        W.validate_manifest(json.dumps(doc), st.model, "strict")
        # budget mode: the on_demand cache comes out of the budget
        rb, _ = self.build([rust], "tb.json", mode="budget", budget="8K", whitelist_mode="tiered", on_demand_cache=672)
        self.assertEqual(rb["budget"]["limit"], 8192 - 672)

    def test_substitute_table_from_the_similarity_data(self):
        rust = self.profile("rust", {"rust": (1.0, 0.95)})
        with contextlib.redirect_stdout(io.StringIO()):
            r = G.build(str(FIXTURE_SIM), [rust], str(self.tmp / "s.json"), whitelist_mode="substitute",
                        substitute_threshold=0.9, substitute_min_tokens=1)
        doc = json.loads((self.tmp / "s.json").read_text())
        st = G.read_stats(str(FIXTURE_SIM))
        wl = W.validate_manifest(json.dumps(doc), st.model)
        self.assertEqual(doc["mode"], "substitute")
        self.assertEqual(doc["substitute_threshold"], 0.9)
        for L in doc["layers"]:
            l = L["layer"]
            self.assertEqual(L["allowed"], [0, 1, 2, 3])
            subs = {x["expert"]: x for x in L["substitutes"]}
            # the twins: expert 4's output is expert 0's, expert 5's is half of expert 1's
            self.assertEqual(subs[4]["with"], 0)
            self.assertAlmostEqual(subs[4]["similarity"], 1.0, places=5)
            self.assertAlmostEqual(subs[4]["rel_l2"], 0.0, places=5)
            self.assertEqual(subs[5]["with"], 1)
            self.assertAlmostEqual(subs[5]["similarity"], 1.0, places=5)
            self.assertAlmostEqual(subs[5]["rel_l2"], 1.0, places=5)
            # every listed one is the most similar allowed expert; the engine uses those at or above the threshold
            sd = st.similarity[l]
            for e, x in subs.items():
                best = max(sd.mean_cos(e, j) for j in (0, 1, 2, 3))
                self.assertAlmostEqual(x["similarity"], max(-1.0, min(1.0, best)), places=5)
                self.assertEqual(wl.substitute[l][e], x["with"] if x["similarity"] >= 0.9 else -1)
            self.assertTrue(any(x["similarity"] < 0.9 for x in subs.values()))   # some below: the fallback
        self.assertGreaterEqual(r["substitute"]["at_threshold"], 6)
        self.assertEqual(r["substitute"]["listed"], sum(len(L["substitutes"]) for L in doc["layers"]))
        # too few sampled picks: nothing listed
        with contextlib.redirect_stdout(io.StringIO()):
            r2 = G.build(str(FIXTURE_SIM), [rust], str(self.tmp / "s2.json"), whitelist_mode="substitute",
                         substitute_min_tokens=10 ** 6)
        self.assertEqual(r2["substitute"]["listed"], 0)
        # statistics without similarity data are refused for the substitute mode
        with self.assertRaises(G.GenError) as cm:
            self.build([rust], "s3.json", whitelist_mode="substitute")
        self.assertIn("similarity data", str(cm.exception))
        # inspect names the table
        self.assertIn("substitutes:", G.inspect(str(self.tmp / "s.json")))


# ================================ THE SIMILARITY DATA ================================

class Similarity(Tmp):
    def test_reads_the_cpp_fixture(self):
        st = G.read_stats(str(FIXTURE_SIM))
        self.assertEqual(sorted(st.similarity), [0, 1, 3])
        self.assertEqual(st.meta["similarity"]["fraction"], 0.25)
        for l, sd in st.similarity.items():
            self.assertGreater(sd.tokens, 0)
            self.assertEqual(sum(sd.row_tokens(e) for e in range(sd.E)), 2 * sd.tokens)   # top-2: two rows per token
            self.assertAlmostEqual(sd.mean_cos(4, 0), 1.0, places=6)
            self.assertAlmostEqual(sd.mean_rel(4, 0), 0.0, places=6)
            self.assertAlmostEqual(sd.mean_rel(5, 1), 1.0, places=6)
            self.assertAlmostEqual(sd.mean_rel(1, 5), 0.5, places=6)
        # the statistics are those of the twins model, the same format as without similarity data
        self.assertEqual(G.read_stats(str(FIXTURE)).categories, st.categories)

    def test_sampling_matches_the_collector(self):
        # the fixture's samples with a share of 0.25 and seed 7: the C++ collector counted this many sampled tokens
        _, samples = G.read_samples(FIXTURE / "samples.bin")
        n = sum(G.similarity_sampled(0.25, 7, s["id"], i) for s in samples for i in range(len(s["tokens"])))
        self.assertEqual(n, G.read_stats(str(FIXTURE_SIM)).meta["similarity"]["tokens"])
        self.assertTrue(G.similarity_sampled(1.0, 1, "x", 0) and not G.similarity_sampled(0.0, 1, "x", 0))

    def test_python_writer_matches_the_cpp_files(self):
        st = G.read_stats(str(FIXTURE_SIM))
        out = self.tmp / "copy"
        G.write_stats(str(out), json.loads(json.dumps(st.meta)), st.layers, st.similarity)
        for f in st.meta["similarity"]["files"]:
            self.assertEqual((out / f["file"]).read_bytes(), (FIXTURE_SIM / f["file"]).read_bytes())
        self.assertEqual(G.read_stats(str(out)).similarity[0].values, st.similarity[0].values)

    def test_damaged_files_are_refused(self):
        bad = self.tmp / "bad"
        shutil.copytree(FIXTURE_SIM, bad)
        data = bytearray((bad / "similarity_001.bin").read_bytes())
        data[100] ^= 1
        (bad / "similarity_001.bin").write_bytes(bytes(data))
        with self.assertRaises(G.GenError) as cm:
            G.read_stats(str(bad))
        self.assertIn("damaged", str(cm.exception))
        meta = json.loads((FIXTURE_SIM / "meta.json").read_text())
        meta["similarity"]["config_fnv1a64"] = "fnv1a64:0000000000000002"   # another sampling than the files say
        shutil.copy(FIXTURE_SIM / "similarity_001.bin", bad / "similarity_001.bin")
        (bad / "meta.json").write_text(json.dumps(meta))
        with self.assertRaises(G.GenError) as cm:
            G.read_stats(str(bad))
        self.assertIn("sampling", str(cm.exception))

    def variant(self, name, scale, with_sim=True):
        st = G.read_stats(str(FIXTURE_SIM))
        meta = json.loads(json.dumps(st.meta))
        rid = G.fnv_hex(name)
        meta.update(run_id=rid, runs=[dict(meta["runs"][0], run_id=rid)], collect={"variant": name},
                    samples_done=[f"{x}-{name}" for x in meta["samples_done"]])
        layers = {l: G.LayerData(l, ld.E, ld.k, ld.C, array.array("Q", (v * scale for v in ld.tokens)),
                                 array.array("Q", (v * scale for v in ld.values))) for l, ld in st.layers.items()}
        sim = {l: G.SimData(l, sd.E, sd.tokens * scale, array.array("Q", (v * scale for v in sd.values)))
               for l, sd in st.similarity.items()} if with_sim else None
        if not with_sim:
            meta.pop("similarity")
        out = self.tmp / name
        G.write_stats(str(out), meta, layers, sim)
        return str(out)

    def test_merge_adds_the_similarity_data(self):
        a, b, c = self.variant("a", 1), self.variant("b", 2), self.variant("c", 3, with_sim=False)
        G.merge([a, b, c], str(self.tmp / "abc"))
        G.merge([c, b, a], str(self.tmp / "cba"))
        files = lambda d: {p.name: p.read_bytes() for p in sorted(pathlib.Path(d).iterdir())}  # noqa: E731
        self.assertEqual(files(self.tmp / "abc"), files(self.tmp / "cba"))
        m, f = G.read_stats(str(self.tmp / "abc")), G.read_stats(str(FIXTURE_SIM))
        self.assertEqual(m.similarity[1].values[3], f.similarity[1].values[3] * 3)   # a and b: 1 + 2
        self.assertEqual(m.similarity[1].tokens, f.similarity[1].tokens * 3)
        self.assertAlmostEqual(m.similarity[1].mean_cos(4, 0), f.similarity[1].mean_cos(4, 0), places=9)
        self.assertEqual(len(m.meta["similarity"]["runs"]), 2)
        self.assertEqual(len(m.meta["similarity"]["runs_without"]), 1)
        self.assertEqual(m.meta["similarity"]["fraction"], 0.25)
        # without any similarity data the merge has none
        G.merge([c, self.variant("d", 1, with_sim=False)], str(self.tmp / "cd"))
        self.assertIsNone(G.read_stats(str(self.tmp / "cd")).similarity)


# ================================ ANALYZE (spec 4.4a) ================================

def synthetic_stats(path, name: str, layers: dict, tokens: int = 80, sim: dict | None = None) -> str:
    """Statistics with known numbers: 2 MoE layers of 16 experts, top-2, blobs of 100 B; `layers` gives per layer and
    category key {expert: picks}, each pick a router weight of 1 (so the masses are exact ratios)."""
    model = W.ModelInfo(2, [16, 16], 2, 8, [0, 1600], [100, 100], "synthetic")
    keys = sorted({k for spec in layers.values() for k in spec})
    rid = G.fnv_hex(name)
    meta = {"format_version": 1, "kind": G.STATS_KIND, "model_fingerprint": model.fingerprint(),
            "model": {"num_layers": 2, "experts_per_layer": [16, 16], "top_k": 2, "hidden": 8, "layer_offset": [0, 1600],
                      "blob_bytes": [100, 100], "moe_layers": [0, 1]},
            "dense_bytes": 1000, "quantization": "synthetic",
            "fixed_point": {"gate_frac_bits": G.GATE_FRAC_BITS, "contrib_frac_bits": G.CONTRIB_FRAC_BITS},
            "has_contrib": False, "categories": [dict(G.split_key(k), key=k) for k in keys],
            "tokens": tokens * len(keys), "samples": len(keys), "run_id": rid,
            "runs": [{"run_id": rid, "mode": "token", "created": "2026-10-21", "samples": len(keys), "tokens": tokens}],
            "mode": "token", "created": "2026-10-21", "generator": "test", "run": {}, "collect": {"name": name},
            "samples_done": [f"{name}-{k}" for k in keys]}
    lds = {}
    for l, spec in layers.items():
        ld = G.LayerData(l, 16, 2, len(keys))
        for ci, k in enumerate(keys):
            ld.tokens[ci] = tokens
            for e, n in spec.get(k, {}).items():
                at = (ci * 16 + e) * ld.W
                ld.values[at] = n                          # count
                ld.values[at + 1] = n << G.GATE_FRAC_BITS  # weight sum: n picks of weight 1
                ld.values[at + 4] = n                      # rank 0
        lds[l] = ld
    if sim:
        meta["similarity"] = {"fraction": 0.01, "seed": 1, "config_fnv1a64": "fnv1a64:0000000000000001",
                              "frac_bits": 32, "rel_cap": 4, "weighting": G.SIM_WEIGHTING, "tokens": 30}
    G.write_stats(str(path), meta, lds, sim)
    return str(path)


RUST, PY, REF = "target|rust|codegen|response|prefill", "target|python|codegen|response|prefill", \
    "reference|prose|reference|response|prefill"
STRONG = {RUST: {0: 64, 1: 48, 2: 32, 3: 16}, PY: {4: 64, 5: 48, 6: 32, 7: 16}, REF: {e: 10 for e in range(16)}}
FLAT = {RUST: {e: 10 for e in range(16)}, PY: {e: 10 for e in range(16)}, REF: {e: 10 for e in range(16)}}


class Analyze(Tmp):
    def setUp(self):
        super().setUp()
        self.prof = self.profile("rp", {"rust": (1.0, 0.95), "python": (1.0, 0.95)}, tasks={"codegen": 1.0})
        # layer 0: rust on experts 0-3 (0.4, 0.3, 0.2, 0.1), python on 4-7; layer 1: every expert alike.  The
        # reference corpus is uniform in both.
        E = 16
        vals = [0] * (2 * E + 2 * E * E)
        w = 100 << 32
        for e in (3, 7):
            vals[e], vals[E + e] = w, 30
        for j in range(E):   # blocked 3: like 0 (cosine 0.9), unlike the rest (0); blocked 7: cosine 0.2 to all
            vals[2 * E + 3 * E + j] = int(w * (1.9 if j == 0 else 1.0))
            vals[2 * E + E * E + 3 * E + j] = int(w * (0.3 if j == 0 else 1.4))
            vals[2 * E + 7 * E + j] = int(w * 1.2)
            vals[2 * E + E * E + 7 * E + j] = int(w * 1.0)
        sim = {0: G.SimData(0, E, 30, array.array("Q", vals)), 1: G.SimData(1, E, 30)}
        self.a = synthetic_stats(self.tmp / "half-a", "a", {0: STRONG, 1: FLAT}, sim=sim)
        self.b = synthetic_stats(self.tmp / "half-b", "b", {0: STRONG, 1: FLAT}, sim=sim)

    def run_analyze(self, stats, **kw):
        kw.setdefault("budget", "1200")
        with contextlib.redirect_stdout(io.StringIO()):
            return G.analyze(None, [], str(self.tmp / "an"), stats=stats, profiles=[self.prof], margin=2,
                             on_demand_cache=200, **kw)

    def test_known_specialisation(self):
        r = self.run_analyze([self.a, self.b])
        L0, L1 = r["layers"]
        # experts for 80 / 90 / 95 / 99 % and the entropy: strong in layer 0, none in layer 1
        rust0, rust1 = L0["components"]["lang:rust"], L1["components"]["lang:rust"]
        self.assertEqual(rust0["experts_for"], {"0.8": 3, "0.9": 3, "0.95": 4, "0.99": 4})
        self.assertEqual(rust1["experts_for"], {"0.8": 13, "0.9": 15, "0.95": 16, "0.99": 16})
        h = -sum(x * math.log(x) for x in (0.4, 0.3, 0.2, 0.1))
        self.assertAlmostEqual(rust0["entropy"], h / math.log(16), places=9)
        self.assertAlmostEqual(rust0["effective_experts"], math.exp(h), places=9)
        self.assertAlmostEqual(rust1["entropy"], 1.0, places=9)
        self.assertAlmostEqual(rust1["effective_experts"], 16.0, places=9)
        c = r["components"]["lang:rust"]
        self.assertEqual(c["experts_for_mean"], {"0.8": 8.0, "0.9": 9.0, "0.95": 10.0, "0.99": 10.0})
        self.assertEqual(c["experts_for_max"]["0.8"], 13)
        self.assertEqual(r["components"]["reference"]["experts_for_mean"]["0.95"], 16.0)
        # overlap: the two languages share nothing in layer 0 and everything in layer 1; rust and the reference a quarter
        ov0, ov1 = L0["overlap"]["lang:python|lang:rust"], L1["overlap"]["lang:python|lang:rust"]
        self.assertEqual((ov0["jaccard_95"], ov0["mass_overlap"], ov0["a95_covers_b"]), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(ov1["jaccard_95"], 1.0)
        self.assertAlmostEqual(ov1["mass_overlap"], 1.0, places=9)
        rr = L0["overlap"]["lang:rust|reference"]
        self.assertAlmostEqual(rr["jaccard_95"], 0.25)
        self.assertAlmostEqual(rr["mass_overlap"], 0.25, places=9)
        self.assertAlmostEqual(rr["a95_covers_b"], 0.25, places=9)   # rust's set keeps a quarter of the reference
        self.assertAlmostEqual(rr["b95_covers_a"], 1.0, places=9)
        self.assertAlmostEqual(r["overlap"]["lang:python|lang:rust"]["jaccard_95"], 0.5)
        # stability: two identical halves cover each other fully
        s = r["stability"]
        self.assertTrue(s["available"])
        self.assertEqual(s["below"], [])
        self.assertAlmostEqual(s["components"]["lang:rust"]["cross_coverage_95_min"], 1.0, places=9)
        self.assertAlmostEqual(s["components"]["lang:rust"]["jaccard_95_mean"], 1.0)
        # the flat layer, and its share of the strict projection's budget
        self.assertEqual(r["flat_layers"]["layers"], [1])
        self.assertAlmostEqual(r["flat_layers"]["budget_share"], 0.5)
        self.assertTrue((self.tmp / "an" / "analysis.md").exists() and (self.tmp / "an" / "analysis.json").exists())

    def test_projection_per_mode(self):
        pr = self.run_analyze([self.a, self.b])["projection"]
        # strict: 12 experts of 100 B: the floor of 4 per layer, then the most mass per byte
        st = pr["strict"]
        self.assertEqual(st["resident"], {"0": [0, 1, 2, 4, 5, 6], "1": [0, 1, 2, 3, 4, 5]})
        self.assertEqual(st["resident_bytes"], 1200)
        self.assertAlmostEqual(st["coverage"]["lang:rust"]["per_layer"]["0"], 0.9, places=9)
        self.assertAlmostEqual(st["coverage"]["lang:rust"]["per_layer"]["1"], 0.375, places=9)
        self.assertAlmostEqual(st["blocked_mass"]["lang:rust"], 1 - (0.9 + 0.375) / 2, places=9)
        # substitute: rust's blocked expert 3 of layer 0 has a replacement (cosine 0.9); python's 7 has none above 0.5
        sb = pr["substitute"]
        self.assertEqual((sb["blocked_experts"], sb["with_substitute"]), (20, 1))
        self.assertAlmostEqual(sb["blocked_mass_with_substitute"]["lang:rust"], 0.5, places=9)   # 1.0 and 0.0
        self.assertAlmostEqual(sb["blocked_mass_with_substitute"]["lang:python"], 0.0, places=9)
        # tiered: 1000 B resident (the cache takes 200), the rest of a 0.995 coverage on_demand
        t = pr["tiered"]
        self.assertEqual((t["resident_experts"], t["on_demand_experts"]), (10, 14))
        self.assertAlmostEqual(t["coverage_resident"]["lang:rust"]["mean"], (0.9 + 0.25) / 2, places=9)
        self.assertAlmostEqual(t["coverage_with_on_demand"]["lang:rust"]["mean"], 1.0, places=9)
        self.assertAlmostEqual(t["on_demand_mass"]["lang:rust"], 1.0 - (0.9 + 0.25) / 2, places=9)
        self.assertAlmostEqual(t["blocked_mass"]["lang:rust"], 0.0, places=9)
        self.assertAlmostEqual(t["on_demand_picks_per_token"]["0"], 0.2, places=9)   # experts 3 and 7: 32 of 160
        self.assertAlmostEqual(t["on_demand_picks_per_token"]["1"], 1.5, places=9)
        self.assertAlmostEqual(t["on_demand_bytes_per_token_max"], 170.0, places=6)
        # a budget below the floor says so
        self.assertIn("problem", self.run_analyze([self.a, self.b], budget="700")["projection"])

    def test_unstable_halves(self):
        moved = dict(STRONG, **{RUST: {8: 64, 9: 48, 10: 32, 11: 16}})   # half b: rust on other experts in layer 0
        b2 = synthetic_stats(self.tmp / "half-b2", "b2", {0: moved, 1: FLAT})
        r = self.run_analyze([self.a, b2])
        rec = r["layers"][0]["stability"]["lang:rust"]
        self.assertEqual((rec["jaccard_95"], rec["mass_overlap"], rec["cross_coverage_95"]), (0.0, 0.0, 0.0))
        self.assertEqual(r["stability"]["below"][0], {"layer": 0, "component": "lang:rust", "cross_coverage_95": 0.0})
        self.assertEqual(len(r["stability"]["below"]), 1)   # layer 1 and python agree
        with self.assertRaises(G.GenError):
            self.run_analyze([self.a, self.b, b2])


class Prepare(Tmp):
    def setUp(self):
        super().setUp()
        rows = [
            {"id": "r1", "prompt": "fn main() {}", "response": "It does nothing.", "task": "codegen", "language": "rust"},
            {"id": "r2", "prompt": "let x = 1;", "task": "compiler_errors"},
            {"id": "r3", "prompt": "fn main() {}", "response": "It does nothing.", "task": "codegen"},   # a duplicate
            {"id": "r4", "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "loop"},
                                      {"role": "assistant", "content": "loop {}"}], "task": "codegen"},
            {"id": "r5", "prompt": "toml?", "language": "toml", "task": "mystery"},
        ] + [{"id": f"rr{i}", "prompt": f"struct S{i};", "response": "ok", "task": "codegen"} for i in range(40)]
        (self.tmp / "rust.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (self.tmp / "py.jsonl").write_text('{"prompt": "def f(): pass"}\n')
        (self.tmp / "base.jsonl").write_text('{"prompt": "Please answer in German."}\n')
        doc = {"format_version": 1, "kind": W.PROFILE_KIND, "name": "rust-test",
               "target_languages": {"rust": {"weight": 1.0, "min_coverage": 0.95, "calibration": ["rust.jsonl"]}},
               "tasks": {"codegen": 1.0, "compiler_errors": 0.9},
               "base_requirements": {"enabled": True, "items": {"instructions": {"calibration": ["base.jsonl"],
                                                                                 "min_coverage": 0.9}}},
               "excluded": {"python": {"calibration": ["py.jsonl"]}},
               "limits": {"max_samples": 100, "max_tokens_per_sample": 64, "seed": 3},
               "chat_template": "chatml"}
        (self.tmp / "p.json").write_text(json.dumps(doc))

    def prep(self, out="prep", **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return G.prepare([str(self.tmp / "p.json")], str(self.tmp / out), tokenizer="bytes", **kw)

    def test_samples_regions_and_split(self):
        doc = self.prep(holdout=0.25)
        cats, train = G.read_samples(self.tmp / "prep" / "samples.bin")
        _, held = G.read_samples(self.tmp / "prep" / "holdout.bin")
        ids = {s["id"] for s in train}
        self.assertFalse(ids & {s["id"] for s in held})          # a holdout sample never goes into collect
        self.assertEqual(sorted(s["id"] for s in held), doc["holdout"]["ids"])
        self.assertNotIn("r3", ids | {s["id"] for s in held})     # the duplicate of r1
        self.assertEqual(doc["dropped"]["duplicates"], [{"id": "r3", "same_as": "r1"}])
        self.assertTrue(10 <= len(held) <= 30)
        # ChatML through the byte tokenizer: the user turn is prompt, the assistant turn response, system its own
        both = {s["id"]: s for s in train + held}
        r4 = both["r4"]
        regions = [G.split_key(cats[c])["region"] for c in r4["cats"]]
        self.assertEqual(r4["tokens"][0], 256)                    # <|im_start|>
        self.assertEqual(regions[0], "system")
        self.assertIn("prompt", regions)
        self.assertEqual(regions[-1], "response")
        text = bytes(t for t in r4["tokens"] if t < 256).decode()
        self.assertIn("assistant\nloop {}", text)
        # datasets and languages: the rows' own language, base items, excluded domains
        keys = {tuple(G.split_key(k)[f] for f in ("dataset", "language")) for k in cats}
        self.assertEqual(keys, {("target", "rust"), ("target", "toml"), ("base", "instructions"), ("excluded", "python")})
        self.assertTrue(any("toml" in w for w in doc["warnings"]))
        self.assertTrue(any("mystery" in w for w in doc["warnings"]))
        # the summary counts every token once
        self.assertEqual(sum(r["tokens"] for r in doc["summary"]["train"]), doc["samples"]["tokens"])

    def test_deterministic_and_recorded(self):
        a = self.prep("a")
        b = self.prep("b")
        self.assertEqual((self.tmp / "a" / "samples.bin").read_bytes(), (self.tmp / "b" / "samples.bin").read_bytes())
        self.assertEqual(a["holdout"]["ids"], b["holdout"]["ids"])
        self.assertEqual(a["inputs"][0]["sha256"], G.sha256_file(self.tmp / "rust.jsonl"))
        self.assertEqual(a["tokenizer"]["kind"], "bytes")
        self.assertEqual(G.read_prep(str(self.tmp / "a"))["samples"]["count"], a["samples"]["count"])
        (self.tmp / "a" / "samples.bin").write_bytes(b"changed")
        with self.assertRaises(G.GenError):
            G.read_prep(str(self.tmp / "a"))

    def test_limits(self):
        doc = self.prep(max_tokens=20, max_samples=10, holdout=0.0)
        _, train = G.read_samples(self.tmp / "prep" / "samples.bin")
        self.assertTrue(all(len(s["tokens"]) <= 20 for s in train))
        self.assertGreater(doc["dropped"]["truncated"], 0)
        per = {}
        for s in train:
            per[s["id"].rstrip("0123456789")] = per.get(s["id"].rstrip("0123456789"), 0) + 1
        self.assertLessEqual(len(train), 10 + 3)   # rust capped at 10; base, excluded, toml one each
        self.assertGreater(doc["dropped"]["over_max_samples"], 0)

    def test_source_globs_relative_to_the_profile(self):
        (self.tmp / "code").mkdir()
        (self.tmp / "code" / "a.rs").write_text("fn a() {}\n")
        (self.tmp / "code" / "b.rs").write_text("fn b() {}\n")
        doc = json.loads((self.tmp / "p.json").read_text())
        doc["target_languages"]["rust"]["sources"] = ["code/*.rs", "nothing/*.rs"]
        (self.tmp / "p.json").write_text(json.dumps(doc))
        prof = W.load_profile(str(self.tmp / "p.json"))
        self.assertEqual(prof["target_languages"]["rust"]["sources"][0], str(self.tmp / "code" / "*.rs"))
        cwd = os.getcwd()
        os.chdir(self.tmp.parent)                  # not the profile's folder: the globs must not depend on it
        try:
            out = self.prep(holdout=0.0)
        finally:
            os.chdir(cwd)
        _, train = G.read_samples(self.tmp / "prep" / "samples.bin")
        self.assertEqual(sum(s["id"].startswith("rust-source-") for s in train), 2)
        self.assertTrue(any("nothing/*.rs" in w and "matches no file" in w for w in out["warnings"]))

    def test_template_that_renders_earlier_messages_differently(self):
        class Qwenish:
            """Consecutive tool results share one user turn and one <|im_end|>; an assistant turn before the last
            user message has no thinking block (as the Qwen templates render them)."""
            def render(self, msgs, add_generation_prompt=False):
                last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
                out = ""
                for i, m in enumerate(msgs):
                    if m["role"] == "tool":
                        out += ("" if i and msgs[i - 1]["role"] == "tool" else "<|im_start|>user")
                        out += "\n<tool_response>\n" + m["content"] + "\n</tool_response>"
                        if i + 1 == len(msgs) or msgs[i + 1]["role"] != "tool":
                            out += "<|im_end|>\n"
                    elif m["role"] == "assistant":
                        think = "<think>\n\n</think>\n\n" if i > last_user else ""
                        out += "<|im_start|>assistant\n" + think + m["content"] + "<|im_end|>\n"
                    else:
                        out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                return out

        msgs = [{"role": "user", "content": "weather in Bonn and Köln?"},
                {"role": "assistant", "content": "<tool_call>a</tool_call>\n<tool_call>b</tool_call>"},
                {"role": "tool", "content": "Bonn 12C"}, {"role": "tool", "content": "Köln 13C"},
                {"role": "assistant", "content": "Bonn 12C, Köln 13C."},
                {"role": "user", "content": "thanks"}, {"role": "assistant", "content": "You're welcome."}]
        t = G.Template("chatml", None)
        t.mode, t.jinja = "model", Qwenish()
        spans = t.spans(msgs)
        self.assertEqual("".join(s for _, s in spans), Qwenish().render(msgs))     # every character once, in order
        self.assertEqual([r for r, _ in spans], ["prompt", "response", "tool", "tool", "response", "prompt", "response"])
        self.assertEqual(spans[2][1], "<|im_start|>user\n<tool_response>\nBonn 12C\n</tool_response>")
        self.assertEqual(spans[3][1], "\n<tool_response>\nKöln 13C\n</tool_response><|im_end|>\n")
        self.assertEqual(spans[4][1], "<|im_start|>assistant\nBonn 12C, Köln 13C.<|im_end|>\n")   # no thinking block now
        self.assertEqual(spans[6][1], "<|im_start|>assistant\n<think>\n\n</think>\n\nYou're welcome.<|im_end|>\n")
        # a template that keeps every prefix: the pieces are the prefixes' differences, as before
        plain = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        self.assertEqual([s for _, s in t.spans(plain[:1])], ["<|im_start|>user\nhi<|im_end|>\n"])
        self.assertEqual([s for _, s in t.spans(plain)][1], "<|im_start|>assistant\n<think>\n\n</think>\n\nhello<|im_end|>\n")

    def test_only_reference(self):
        (self.tmp / "ref.jsonl").write_text('{"prompt": "Bonjour"}\n{"prompt": "Hello"}\n')
        with contextlib.redirect_stdout(io.StringIO()):
            doc = G.prepare([], str(self.tmp / "ref"), tokenizer="bytes", reference=[str(self.tmp / "ref.jsonl")],
                            holdout=0.0)
        self.assertEqual(doc["categories"], ["reference|reference|reference|prompt|prefill"])
        self.assertEqual(doc["samples"]["count"], 2)


# ================================ ESTIMATE, COLLECT, ANALYZE, EVAL ================================

class Planning(Tmp):
    def test_python_plan_matches_the_scheduler(self):
        # the numbers of tests/core/routing_stats_test.cpp's small_budgets(): its run printed 4 blocks of up to 367
        # tokens; the Python plan (the estimate) must agree with the C++ scheduler
        _, samples = G.read_samples(FIXTURE / "samples.bin")
        n = {"row_floats": 8, "hidden": 8, "top_k": 2, "experts_per_layer": [16, 16, 0, 16],
             "dense_per_layer": [1056] * 4, "blob_bytes": [336, 336, 0, 336]}
        plan = G.plan_layerwise(n, [len(s["tokens"]) for s in samples], 9, 1056 + 336 + 40 * 3 * 8 * 4, 56 * 1024, 0, 0,
                                4, False)
        self.assertEqual((plan["blocks"], plan["block_tokens"]), (4, 367))
        self.assertFalse(plan["problems"])
        # nothing limits the block: 16384 tokens, or the longest sample when that is longer (never a refusal)
        free = G.plan_layerwise(n, [20000, 300, 300], 9, 0, 0, 0, 0, 4, False)
        self.assertEqual((free["block_tokens"], free["blocks"], free["problems"]), (20000, 2, []))
        small = G.plan_layerwise(n, [len(s["tokens"]) for s in samples], 9, 1500, 8 * 1024, 0, 0, 4, False)
        self.assertTrue(any("RAM budget" in p for p in small["problems"]))
        self.assertTrue(any("VRAM budget" in p for p in small["problems"]))

    def pack_and_prep(self):
        pack = self.tmp / "pack"
        pack.mkdir()
        shutil.copy(NATIVE / "iq3_s.txt", pack / "native_experts.txt")
        (pack / "dense.bin").write_bytes(b"\0" * (48 * 1000))
        prep = self.tmp / "prep"
        rows = [{"id": f"s{i}", "prompt": "fn x() {} " * (1 + i % 7)} for i in range(30)]
        (self.tmp / "rust.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        doc = {"format_version": 1, "kind": W.PROFILE_KIND, "name": "r",
               "target_languages": {"rust": {"weight": 1.0, "min_coverage": 0.95, "calibration": ["rust.jsonl"]}},
               "base_requirements": {"enabled": False}, "chat_template": "none"}
        (self.tmp / "p.json").write_text(json.dumps(doc))
        with contextlib.redirect_stdout(io.StringIO()):
            G.prepare([str(self.tmp / "p.json")], str(prep), tokenizer="bytes", holdout=0.0)
        cfg = {"exe": "/opt/strata/strata-engine", "args": ["--pack", str(pack), "--native", "/m/x.gguf", "--serve",
                                                             "--expert-cache", "auto"]}
        (self.tmp / "cfg.json").write_text(json.dumps(cfg))
        return pack, prep

    def test_estimate(self):
        pack, prep = self.pack_and_prep()
        e = G.estimate(str(prep), pack=str(pack), vram_budget="16G", ram_budget="64G")
        self.assertTrue(e["token_mode"]["fits"])   # a 34 GiB IQ3_S pack in 16 + 64 GiB
        self.assertEqual(e["recommended_mode"], "token")
        e = G.estimate(str(prep), pack=str(pack), vram_budget="8G", ram_budget="16G")
        self.assertFalse(e["token_mode"]["fits"])
        self.assertEqual(e["recommended_mode"], "layerwise")
        self.assertGreater(e["token_mode"]["drive_reads"], 0)
        self.assertIn("recommended: --mode layerwise", G.estimate_text(e))
        e = G.estimate(str(prep), pack=str(pack), vram_budget="8G", ram_budget="1M")
        self.assertFalse(e["ok"])
        self.assertTrue(e["layerwise"]["problems"])

    def test_collect_command(self):
        pack, prep = self.pack_and_prep()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = G.main(["collect", "--model", str(self.tmp / "cfg.json"), "--prep", str(prep), "--out",
                         str(self.tmp / "stats"), "--mode", "layerwise", "--ram-budget", "48G", "--block-tokens", "4096",
                         "--similarity", "0.01", "--similarity-seed", "3", "--print-command"])
        self.assertEqual(rc, 0)
        cmd = out.getvalue().split()
        self.assertNotIn("--serve", cmd)
        for flag, val in (("--routing-stats-mode", "layerwise"), ("--routing-stats-ram-budget", "48G"),
                          ("--routing-stats-block-tokens", "4096"), ("--routing-stats-samples", str(prep / "samples.bin")),
                          ("--routing-stats-similarity", "0.01"), ("--routing-stats-similarity-seed", "3")):
            self.assertEqual(cmd[cmd.index(flag) + 1], val)
        meta = json.loads((self.tmp / "stats" / "collect.json").read_text())
        self.assertEqual(meta["prep"]["samples_sha256"], G.sha256_file(prep / "samples.bin"))
        self.assertEqual(meta["mode"], "layerwise")

    def test_analyze_existing_stats(self):
        rust = self.tmp / "rust.json"
        (self.tmp / "cal.jsonl").write_text('{"prompt": "x"}\n')
        rust.write_text(json.dumps({"format_version": 1, "kind": W.PROFILE_KIND, "name": "rust",
                                    "target_languages": {"rust": {"weight": 1.0, "min_coverage": 0.95,
                                                                  "calibration": ["cal.jsonl"]}},
                                    "base_requirements": {"enabled": False}}))
        r = G.analyze(None, [], str(self.tmp / "an"), stats=str(FIXTURE), profiles=[str(rust)],
                      target_hardware="vram=8K,ram=8K,reserve=0", kv_bytes_per_token=16, runtime_buffer=0, context=64)
        self.assertNotIn("verdict", r)   # numbers only
        self.assertLessEqual(r["components"]["lang:rust"]["experts_for_mean"]["0.95"], 4)
        self.assertTrue(r["fits_95"])
        self.assertFalse(r["stability"]["available"])   # one statistics folder: no halves to compare
        self.assertIn("strict", r["projection"])
        r2 = G.analyze(None, [], str(self.tmp / "an2"), stats=str(FIXTURE))   # no profile: every target and base item
        self.assertEqual(set(r2["components"]), {"lang:rust", "lang:python", "base:instructions", "reference"})
        self.assertEqual(r2["layers"][0]["union_95"], list(range(12)))   # three languages, four experts each
        self.assertEqual(r2["projected_expert_bytes_95"], 3 * 12 * 336)

    def write_gguf(self, path, meta: dict, tensors: list[tuple[str, list[int], int]]):
        """A GGUF v3 header (metadata and tensor table, no data): what gguf_reader.py and these stages read."""
        import struct
        out = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(meta)))
        for k, v in meta.items():
            kb = k.encode()
            out += struct.pack("<Q", len(kb)) + kb
            if isinstance(v, str):
                vb = v.encode()
                out += struct.pack("<IQ", 8, len(vb)) + vb
            else:
                out += struct.pack("<II", 4, v)
        for name, shape, ty in tensors:
            nb = name.encode()
            out += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(shape))
            out += struct.pack(f"<{len(shape)}Q", *shape) + struct.pack("<IQ", ty, 0)
        pathlib.Path(path).write_bytes(bytes(out))

    def test_kv_and_dense_bytes_from_the_gguf(self):
        arch = "qwen4exp"
        meta = {"general.architecture": arch, f"{arch}.block_count": 8, f"{arch}.full_attention_interval": 4,
                f"{arch}.attention.head_count_kv": 2, f"{arch}.attention.key_length": 256, f"{arch}.ssm.state_size": 128,
                f"{arch}.ssm.time_step_rank": 48, f"{arch}.ssm.conv_kernel": 4, f"{arch}.ssm.group_count": 16,
                f"{arch}.expert_used_count": 10, f"{arch}.embedding_length": 2560}
        g1 = self.tmp / "m-00001-of-00002.gguf"
        self.write_gguf(g1, meta, [("token_embd.weight", [2560, 1000], 1), ("blk.0.attn_q.weight", [2560, 512], 0),
                                   ("blk.0.ffn_gate_exps.weight", [2560, 640, 512], 1)])
        self.write_gguf(self.tmp / "m-00002-of-00002.gguf", {"general.architecture": arch},
                        [("blk.7.ffn_down_shexp.weight", [640, 2560], 1),
                         ("per_layer_token_embd.weight", [2048, 1000], 1)])   # read from the drive: not counted
        per_tok, state, how = G.kv_from_gguf(str(g1))
        self.assertEqual(per_tok, 2 * 2 * 256 * 2 * 2)   # 2 full-attention layers of 8, K and V, FP16
        self.assertEqual(state, 6 * (128 * 48 * 128 + (2 * 128 * 16 + 128 * 48) * 3) * 4)
        self.assertIn("full-attention layers", how)
        # every tensor that is not a routed expert's and not the PLE table, over both shards
        self.assertEqual(G.gguf_dense_bytes(str(g1)), 2560 * 1000 * 2 + 2560 * 512 * 4 + 640 * 2560 * 2)

    def test_eval_grouping(self):
        import quant_quality as Q
        cats = ["target|rust|codegen|prompt|prefill", "target|rust|codegen|response|prefill"]
        samples = [{"id": "a", "tokens": [1, 2, 3, 4], "cats": [0, 0, 1, 1]}]
        top = [(5, -0.1), (6, -2.5)] + [(100 + i, -9.0) for i in range(Q.TOPK - 2)]
        ref = {"a": {0: (2, -0.1, 5, top), 1: (3, -0.2, 5, top), 2: (4, -0.3, 5, top)}}
        run = {"a": {0: (2, -0.1, 5, top), 1: (3, -0.4, 6, top), 2: (4, -0.3, 5, top)}}
        r = G.eval_compare(ref, run, samples, cats, Q)
        self.assertEqual(r["groups"]["all"]["positions"], 3)
        self.assertEqual(r["groups"]["language=rust,region=response"]["positions"], 2)   # positions 1, 2 predict 3, 4
        self.assertAlmostEqual(r["groups"]["language=rust,region=response"]["top1"], 0.5)
        self.assertEqual(r["groups"]["language=rust,region=prompt"]["top1"], 1.0)


if __name__ == "__main__":
    unittest.main()
