"""Ensure force-added snapshots and staged credential bytes are rejected."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_release_guard", ROOT / "scripts/check_public_release.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class ReleaseGuardTests(unittest.TestCase):
    def test_private_assets_and_snapshots_are_rejected_at_any_depth(self):
        for path in ["skill-pool/action/example/SKILL.md", "archive/data/objects/source.py",
                     "backup/skills-index.json", "backup/01_inventory.json", "models/weights.safetensors"]:
            with self.subTest(path=path):
                self.assertIsNotNone(guard.path_violation(path))
        self.assertIsNone(guard.path_violation("docs/images/benchmark-overview.png"))
        self.assertIsNone(guard.path_violation("src/embodied_skill_bench/runtime/session.py"))

    def test_staged_bytes_are_checked_even_when_worktree_is_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            path = root / "config.txt"
            # Synthetic detector fixture; not a usable credential.
            path.write_text("gh" + "p_" + "a" * 36, encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "config.txt"], check=True)
            path.write_text("clean working tree contents", encoding="utf-8")
            problems = guard.check_staged(root)
            self.assertEqual(len(problems), 1)
            self.assertIn("possible credential", problems[0])
            self.assertNotIn("a" * 36, problems[0])


if __name__ == "__main__":
    unittest.main()
