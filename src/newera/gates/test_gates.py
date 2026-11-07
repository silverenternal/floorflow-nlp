from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from newera.gates.checks import (
    check_directories,
    check_single_git_worktree,
    check_sources,
    check_versioned_paths,
)
from newera.gates.evidence import check_evidence


class GateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "src/newera/nlp_flow/components/prototype").mkdir(parents=True)
        (self.root / "project").mkdir()

    def test_exploratory_overflow_warns_but_formal_overflow_fails(self) -> None:
        for directory in (self.root / "src/newera/nlp_flow/components/prototype", self.root / "project"):
            for i in range(6):
                (directory / str(i)).touch()
        findings = check_directories(self.root)
        self.assertEqual({item["level"] for item in findings}, {"error", "warning"})
        self.assertTrue(any("prototype" in f["message"] and f["level"] == "warning" for f in findings))

    def test_ignored_local_data_does_not_count(self) -> None:
        for name in (".git", ".venv", ".data", "outputs", "checkpoints", ".pytest_cache"):
            (self.root / name).mkdir()
        self.assertEqual(check_directories(self.root), [])

    def test_syntax_and_duplicate_config_keys_fail(self) -> None:
        (self.root / "project/invalid.py").write_text("def broken(:\n")
        (self.root / "project/invalid.json").write_text('{"a": 1, "a": 2}')
        self.assertEqual(len(check_sources(self.root)), 2)

    def test_version_iteration_paths_fail(self) -> None:
        (self.root / "project/configs").mkdir(parents=True)
        (self.root / "project/configs/model_v2.json").touch()
        (self.root / "src/newera/baseline.py").touch()
        findings = check_versioned_paths(self.root)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["check"], "git_is_the_version_history")

    def test_non_git_fixture_skips_local_worktree_count(self) -> None:
        self.assertEqual(check_single_git_worktree(self.root), [])

    def test_missing_or_invalid_evidence_warns_without_failing(self) -> None:
        self.assertEqual(check_evidence(self.root, "explore", None), [])
        findings = check_evidence(self.root, "release", None)
        self.assertTrue(any("model_card" in f["message"] for f in findings))
        self.assertTrue(all(f["level"] == "warning" for f in findings))

    def test_non_object_evidence_manifest_warns_without_failing(self) -> None:
        manifest = self.root / "manifest.json"
        manifest.write_text("[]")
        findings = check_evidence(self.root, "research", manifest)
        self.assertTrue(any("manifest must be a JSON object" in f["message"] for f in findings))
        self.assertTrue(all(f["level"] == "warning" for f in findings))

    def test_cli_warnings_exit_zero_but_formal_overflow_fails(self) -> None:
        command = [sys.executable, "-m", "newera.gates", "--root", str(self.root), "--stage", "research"]
        warning = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy(), check=False)
        self.assertEqual(warning.returncode, 0)
        self.assertFalse(json.loads(warning.stdout)["qualified"])
        for index in range(6):
            (self.root / "project" / str(index)).touch()
        failure = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy(), check=False)
        self.assertEqual(failure.returncode, 1)
        self.assertEqual(json.loads(failure.stdout)["status"], "failed")

    def test_wrapped_command_runs_after_warnings_and_forwards_exit(self) -> None:
        marker = self.root / "ran"
        command = [sys.executable, "-m", "newera.gates", "--root", str(self.root), "--stage", "research",
                   "--execute", sys.executable, "-c", "from pathlib import Path; Path('ran').touch(); raise SystemExit(7)"]
        result = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy(), check=False)
        self.assertEqual(result.returncode, 7)
        self.assertTrue(marker.is_file())
        self.assertIn('"qualified": false', result.stderr)

    def test_wrapped_command_does_not_run_after_formal_error(self) -> None:
        for index in range(6):
            (self.root / "project" / str(index)).touch()
        command = [sys.executable, "-m", "newera.gates", "--root", str(self.root),
                   "--execute", sys.executable, "-c", "from pathlib import Path; Path('ran').touch()"]
        result = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy(), check=False)
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.root / "ran").exists())

    def test_modified_artifact_is_reported(self) -> None:
        (self.root / "config.json").write_text("changed")
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"config": "config.json", "config_sha256": "0" * 64}))
        findings = check_evidence(self.root, "research", manifest)
        self.assertTrue(any("SHA-256 mismatch" in f["message"] for f in findings))
        self.assertTrue(all(f["level"] == "warning" for f in findings))

    def test_existing_files_are_not_treated_as_verified_evidence(self) -> None:
        (self.root / "config.json").write_text("{}")
        (self.root / "eval.json").write_text("{}")
        manifest = self.root / "manifest.json"
        digest = hashlib.sha256(b"{}").hexdigest()
        manifest.write_text(json.dumps({"run_id": "r1", "config": "config.json", "config_sha256": digest, "data_version": "v1", "data_fingerprint": "sha256:test", "code_revision": "test", "environment": "cpu", "started_at": "2026-01-01T00:00:00Z", "seed": 0, "budget": "1 cpu second", "stop_conditions": "end", "evaluation_protocol": "eval.json", "evaluation_protocol_sha256": digest}))
        findings = check_evidence(self.root, "research", manifest)
        self.assertEqual(len(findings), 1)
        self.assertIn("do not verify", findings[0]["message"])


if __name__ == "__main__":
    unittest.main()
