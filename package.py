"""Create the source backup zip and verify it by running the tests from an unzipped copy.

    python package.py            -> dist/Group84_Q_learning_SARSA_SetA_source.zip (+ copy of the PDF in dist/)

Only the PDF is submitted (assignment rule). The zip is the team's backup and the
bundle to copy into the BITS virtual lab.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

ZIP_NAME = "Group84_Q_learning_SARSA_SetA_source.zip"
PDF_NAME = "Group 84 - Q_learning_SARSA_Set A.pdf"
EXCLUDE_DIRS = {".venv", "venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "dist", ".git",
                ".claude", "pdf_pages"}
EXCLUDE_FILES = {"drl.pdf", ".DS_Store"}


def collect(root: str):
    """Yield (absolute path, archive name) of every file that belongs in the zip."""
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in EXCLUDE_DIRS]          # prune excluded folders
        for f in files:
            if f in EXCLUDE_FILES or f.endswith((".pyc", ".zip")):
                continue
            full = os.path.join(d, f)
            yield full, os.path.relpath(full, root)


def build_zip(root: str = ".") -> str:
    """Write dist/<ZIP_NAME> and return its path."""
    os.makedirs(os.path.join(root, "dist"), exist_ok=True)
    out = os.path.join(root, "dist", ZIP_NAME)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for full, arc in collect(root):
            z.write(full, os.path.join("Group84_Q_learning_SARSA_SetA", arc))
    if os.path.exists(os.path.join(root, PDF_NAME)):                   # submission copy
        shutil.copy(os.path.join(root, PDF_NAME), os.path.join(root, "dist", PDF_NAME))
    return out


def test_unzipped(zip_path: str) -> int:
    """Unzip into a temporary folder and run pytest there with the current interpreter."""
    tmp = tempfile.mkdtemp(prefix="g84_")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    proj = os.path.join(tmp, "Group84_Q_learning_SARSA_SetA")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=proj)
    shutil.rmtree(tmp, ignore_errors=True)
    return r.returncode


if __name__ == "__main__":
    path = build_zip()
    size = os.path.getsize(path) / 1e6
    print(f"zip -> {path} ({size:.1f} MB)")
    sys.exit(test_unzipped(path))
