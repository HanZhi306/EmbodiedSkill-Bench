"""Regression checks for running the public code without a skill library."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from embodied_skill_bench import cli
from embodied_skill_bench.catalog import SkillCatalog
from embodied_skill_bench.experiments.common import create_backend
from embodied_skill_bench.experiments.native_typed import run_native_typed_condition
from embodied_skill_bench.io import project_root, sha256_file
from embodied_skill_bench.runtime.native_typed_contract import P25_PROTOCOL, P3_PROTOCOL, contract_sha256


class PublicNativeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "configs").mkdir()
        shutil.copy2(project_root() / "configs/clean100.json", self.root / "configs/clean100.json")

    def starter(self, **changes):
        values = dict(runner="codex", model="test-model", cli_version="test-cli-version",
                      reasoning_effort="medium", mode="native", native_protocol="p3",
                      ik_identity=None, ik_endpoint=None, output=str(self.root / "run.json"))
        values.update(changes)
        with patch.object(cli, "project_root", return_value=self.root):
            return cli.init_config(argparse.Namespace(**values))

    def test_doctor_reports_absent_assets_without_claiming_full_validation(self):
        with patch.object(cli, "project_root", return_value=self.root):
            report = cli.doctor()
        self.assertTrue(report["ok"])
        self.assertEqual(report["clean100_tasks"], 100)
        self.assertEqual(report["skill_assets"], "not_included")
        self.assertEqual(report["verified_skill_files"], 0)
        self.assertFalse(report["simulator_checked"])
        self.assertFalse(report["agent_api_checked"])

    def test_partial_skill_restore_is_an_error(self):
        (self.root / "skill-pool").mkdir()
        with patch.object(cli, "project_root", return_value=self.root):
            with self.assertRaisesRegex(ValueError, "both skill-pool"):
                cli.doctor()

    def test_local_skill_hash_tampering_is_rejected(self):
        directory = self.root / "skill-pool/action/example-skill"
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text("# Local test skill\n", encoding="utf-8")
        metadata = {"id": "example-skill", "kind": "action", "description": "Test fixture",
                    "interaction": {"class": "simulator-action", "changes_simulator_state": True}}
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        index = {"total": 1, "kind_counts": {"action": 1}, "skills": [{
            "id": "example-skill", "path": "skill-pool/action/example-skill",
            "files_sha256": {p.name: sha256_file(p) for p in directory.iterdir()}}]}
        (self.root / "skills-index.json").write_text(json.dumps(index), encoding="utf-8")
        with patch.object(cli, "project_root", return_value=self.root), \
             patch("embodied_skill_bench.catalog.project_root", return_value=self.root):
            self.assertEqual(cli.doctor()["skill_assets"], "verified")
            (directory / "SKILL.md").write_text("Changed after indexing", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                cli.doctor()

    def test_native_p3_requires_audited_service_instead_of_substituting_osc(self):
        with self.assertRaisesRegex(ValueError, "requires --ik-identity"):
            self.starter()
        self.assertFalse((self.root / "run.json").exists())

    def test_p3_starter_freezes_identity_and_native_contract_without_skills(self):
        identity = self.root / "identity.json"
        identity.write_text(json.dumps({"service": "local-test-service", "revision": "test"}), encoding="utf-8")
        self.starter(ik_identity=str(identity), ik_endpoint="http://127.0.0.1:8766")
        config = json.loads((self.root / "run.json").read_text())
        self.assertEqual(config["experiment"], "native-typed-no-skill")
        self.assertNotIn("skill_set", config)
        self.assertEqual(config["backend"]["controller_kind"], "JOINT_POSITION")
        self.assertEqual(config["native_typed"]["protocol"], P3_PROTOCOL)
        self.assertEqual(config["native_typed"]["contract_sha256"], contract_sha256(P3_PROTOCOL))
        self.assertEqual(config["native_typed"]["ik_provider"]["expected_identity"]["revision"], "test")
        cli.validate_config(config)
        config["native_typed"]["contract_sha256"] = "invalid"
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            cli.validate_config(config)

    def test_p25_is_an_explicit_distinct_development_mode(self):
        report = self.starter(native_protocol="p25")
        config = json.loads((self.root / "run.json").read_text())
        self.assertEqual(config["native_typed"]["protocol"], P25_PROTOCOL)
        self.assertEqual(config["native_typed"]["ik_provider"]["kind"], "unavailable")
        self.assertIn("not the paper", report["note"])
        cli.validate_config(config)

    def test_empty_catalog_does_not_trigger_discovery_during_backend_creation(self):
        config = {"kind": "rpent-libero", "suite": "libero_10_lan", "task": 0, "seed": 26}
        backend = Mock()
        module = SimpleNamespace(RPentLiberoBackend=backend)
        with patch.object(SkillCatalog, "discover", side_effect=AssertionError("library access")), \
             patch.dict(sys.modules, {"embodied_skill_bench.runtime.backends.rpent_libero": module}):
            create_backend(config, self.root, [], catalog=SkillCatalog([]), primitive_code=True,
                           primitive_code_perception_mode="calibrated-geometry-p2",
                           controller_kind="JOINT_POSITION")
        arguments = backend.call_args.kwargs
        self.assertFalse(arguments["enable_vla"])
        self.assertFalse(arguments["enable_sam3"])
        self.assertEqual(arguments["controller_kind"], "JOINT_POSITION")

    def test_native_preflight_can_fail_without_accessing_private_library(self):
        with patch.object(SkillCatalog, "discover", side_effect=AssertionError("library access")), \
             patch("embodied_skill_bench.experiments.native_typed.load_agent_specs",
                   return_value={"agent": {"runner": "codex", "model": "test-model"}}), \
             patch("embodied_skill_bench.experiments.native_typed.AgentRunner") as runner:
            runner.return_value.version.side_effect = RuntimeError("test CLI unavailable")
            result = run_native_typed_condition(
                name="test-native", agent_id="agent", backend_config={}, limits={},
                output_dir=self.root / "episode", strict_versions=True, parent_config={},
                protocol=P3_PROTOCOL, native_typed_config={})
        self.assertEqual(result["classification"]["failure_class"], "agent_protocol_failure")
        self.assertIsNone(result["agent_outcome"])
        self.assertEqual(result["classification"]["incidents"][0]["source"], "agent_cli_preflight")
        self.assertEqual(result["skills"], [])

    def test_runtime_manifest_does_not_require_private_index(self):
        config = {"name": "test-native", "experiment": "native-typed-no-skill", "backend": {}}
        with patch.object(cli, "validate_config"), \
             patch.object(cli, "run_one", return_value={"run_dir": "test", "classification": {"eligible": True}}):
            ok, output = cli.run_config(config, self.root / "batch")
        self.assertTrue(ok)
        manifest = json.loads((output / "runtime-manifest.json").read_text())
        self.assertNotIn("skills-index.json", manifest["files_sha256"])
        self.assertIn("configs/clean100.json", manifest["files_sha256"])


if __name__ == "__main__":
    unittest.main()
