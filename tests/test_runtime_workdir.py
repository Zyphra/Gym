# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from nemo_gym import runtime_workdir as module


class RuntimeWorkdirTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.source = self.base / "source component"
        self.source.mkdir()
        (self.source / "asset.txt").write_text("retained asset\n")
        self.root = self.base / "jobcache"
        self.root.mkdir(mode=0o700)
        self.bin = self.base / "bin"
        self.bin.mkdir()
        (self.bin / "python").symlink_to(sys.executable)
        self.prior_path = os.environ["PATH"]
        os.environ["PATH"] = str(self.bin) + ":" + self.prior_path

    def tearDown(self):
        os.environ["PATH"] = self.prior_path
        self.tmp.cleanup()

    def test_private_directories_preserve_inputs_and_new_outputs(self):
        one = module.component_runtime_directory(self.source, self.root, "one")
        two = module.component_runtime_directory(self.source, self.root, "two")
        self.assertNotEqual(one, two)
        self.assertEqual(one.stat().st_mode & 0o777, 0o700)
        self.assertTrue((one / "asset.txt").is_symlink())
        self.assertEqual((one / "asset.txt").read_text(), "retained asset\n")
        (one / "output.txt").write_text("private")
        self.assertFalse((self.source / "output.txt").exists())
        self.assertFalse((two / "output.txt").exists())

    def test_rejects_symlink_root(self):
        link = self.base / "link"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            module.component_runtime_directory(self.source, link, "bad")

    def test_rejects_writable_root(self):
        self.root.chmod(0o770)
        with self.assertRaises(ValueError):
            module.component_runtime_directory(self.source, self.root, "bad")

    def test_rejects_absolute_entrypoint(self):
        with self.assertRaises(ValueError):
            module.component_launch_command("true", self.source, Path("/etc/passwd"), runtime_root=self.root)

    def test_default_native_relative_launch(self):
        (self.source / "app.py").write_text("import os; print(os.getcwd())")
        command = module.component_launch_command(f"cd {shlex.quote(str(self.source))}", self.source, Path("app.py"))
        result = subprocess.run(["/bin/bash", "-c", command], text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), str(self.source))

    def test_explicit_interpreter_survives_unrelated_path(self):
        (self.source / "app.py").write_text("import sys; print(sys.executable)")
        command = module.component_launch_command(
            "true",
            self.source,
            Path("app.py"),
            runtime_root=self.root,
            python_executable=sys.executable,
        )
        environment = os.environ.copy()
        environment["PATH"] = "/does-not-exist"
        result = subprocess.run(
            ["/bin/bash", "-c", command],
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(Path(result.stdout.strip()), Path(sys.executable))

    def test_owned_invalidated_source_cwd(self):
        script = """import json,os,time\nfrom pathlib import Path\nPath(os.environ['READY']).write_text('ready')\nwhile not Path(os.environ['GO']).exists(): time.sleep(.01)\ntry:\n cwd=os.getcwd()\nexcept FileNotFoundError:\n print(json.dumps({'deleted_cwd':True}),flush=True);raise SystemExit(41)\nasset=Path('asset.txt').read_text()\nPath('output.txt').write_text('private output')\nprint(json.dumps({'cwd':cwd,'asset':asset,'file':__file__,'config':os.environ['CONFIG_TEST']}),flush=True)\n"""
        (self.source / "app.py").write_text(script)
        processes = []
        for candidate in (False, True):
            ready = self.base / f"ready-{candidate}"
            go = self.base / f"go-{candidate}"
            command = module.component_launch_command(
                f"cd {shlex.quote(str(self.source))}",
                self.source,
                Path("app.py"),
                runtime_root=self.root if candidate else None,
                environment={"CONFIG_TEST": "exact value"},
            )
            env = os.environ.copy()
            env.update(READY=str(ready), GO=str(go))
            p = subprocess.Popen(
                ["/bin/bash", "-c", command], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            processes.append((candidate, p, ready, go))
        import time

        deadline = time.monotonic() + 10
        while not all(x[2].exists() for x in processes):
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        old = self.base / "old"
        self.source.rename(old)
        shutil.copytree(old, self.source)
        shutil.rmtree(old)
        for _, _, _, go in processes:
            go.write_text("go")
        for candidate, p, _, _ in processes:
            out, err = p.communicate(timeout=10)
            if not candidate:
                self.assertEqual(p.returncode, 41, err)
                self.assertTrue(json.loads(out)["deleted_cwd"])
            else:
                self.assertEqual(p.returncode, 0, err)
                data = json.loads(out)
                self.assertEqual(data["asset"], "retained asset\n")
                self.assertEqual(data["config"], "exact value")
                self.assertTrue(Path(data["cwd"]).is_relative_to(self.root))
                self.assertEqual(data["file"], str(self.source / "app.py"))
        self.assertFalse((self.source / "output.txt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
