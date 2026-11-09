"""Compliance tests: banned libraries, documentation rule, required files, naming."""

import ast
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY_FILES = [p for p in list((ROOT / "drl_sticky").glob("*.py")) +
            [ROOT / f for f in ("run_experiments.py", "pilot_sweep.py", "build_notebook.py", "make_pdf.py", "package.py")]
            if p.exists()]
BANNED = {"stable_baselines3", "sb3_contrib", "ray", "rllib", "tianshou", "torch", "tensorflow",
          "keras", "jax", "d3rlpy", "pfrl", "acme", "dopamine", "garage", "rl_coach"}


def _imports(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                yield n.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module.split(".")[0]


def test_no_rl_libraries():
    for p in PY_FILES:
        bad = set(_imports(p)) & BANNED
        assert not bad, f"{p.name} imports {bad}"


def test_every_function_and_class_documented():
    """Assignment rule: comments/descriptions on every function (else -1 mark)."""
    missing = []
    for p in PY_FILES:
        tree = ast.parse(p.read_text())
        if ast.get_docstring(tree) is None:
            missing.append(f"{p.name} (module)")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is None:
                    missing.append(f"{p.name}:{node.name}")
    assert not missing, missing


def test_required_files_present():
    for f in ["README.md", "PLAN.md", "requirements.txt", "requirements-dev.txt", "run_experiments.py",
              "drl_sticky/env.py", "drl_sticky/agents.py", "build_notebook.py", "make_pdf.py",
              "CONTRIBUTION.md", "docs/traceability.md"]:
        assert (ROOT / f).exists(), f


def test_pdf_name_constant():
    import make_pdf
    assert make_pdf.PDF_NAME == "Group 84 - Q_learning_SARSA_Set A.pdf"


def test_wrapper_subclasses_gym_wrapper():
    import gymnasium as gym
    from drl_sticky.env import StickyActuatorMountainCar
    assert issubclass(StickyActuatorMountainCar, gym.Wrapper)


def test_contribution_declaration_lists_all_members():
    text = (ROOT / "CONTRIBUTION.md").read_text()
    for name in ["SHREYANSH SHARMA", "LAKSHYA KANCHAN", "VINAY PRATAP SINGH BHADAURIA"]:
        assert name in text
    assert text.count("100%") >= 3


def test_every_visible_notebook_code_cell_commented():
    """Notebook cells shown in the PDF carry a comment; functions defined there have docstrings."""
    import build_notebook
    for cell in build_notebook.cells():
        if cell.cell_type != "code" or "remove_input" in cell.metadata.get("tags", []):
            continue
        src = cell.source
        if src.startswith("%%writefile"):
            continue                                   # module files are covered by the docstring test
        assert any(l.strip().startswith("#") for l in src.splitlines()), src[:80]
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert ast.get_docstring(node), node.name
