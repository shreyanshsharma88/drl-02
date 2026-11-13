"""Consistency tests between saved artefacts: results.json, Q-tables, notebook and PDF."""

import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "results.json"
NB = ROOT / "Group84_Q_learning_SARSA_SetA.ipynb"
PDF = ROOT / "Group 84 - Q_learning_SARSA_Set A.pdf"
need_results = pytest.mark.skipif(not RES.exists(), reason="run the notebook / run_experiments.py first")


def test_pdf_export_without_browser(tmp_path, monkeypatch):
    """Export tables, embedded plots and long logs without external executables."""
    import base64
    import io
    from PIL import Image
    from pypdf import PdfReader
    from make_pdf import html_to_pdf

    monkeypatch.setenv("PATH", "")
    image = io.BytesIO()
    Image.new("RGB", (40, 20), "red").save(image, format="PNG")
    encoded = base64.b64encode(image.getvalue()).decode("ascii")
    records = "\n".join(f"episode {episode:04d} reward -100" for episode in range(600))
    source = tmp_path / "report.html"
    target = tmp_path / "report.pdf"
    source.write_text(
        '<html><body><h1>Export check</h1><p>Reward <strong>summary</strong>.</p>'
        '<table><tr><th>Agent</th><th>Reward</th></tr>'
        '<tr><td>SARSA</td><td>-100</td></tr></table>'
        f'<img src="data:image/png;base64,{encoded}"><pre>{records}</pre>'
        '<p>End of report</p></body></html>', encoding="utf-8",
    )
    assert html_to_pdf(str(source), str(target))
    monkeypatch.undo()   # restore PATH before reading back (pypdf 7 checks PATH-dependent defaults)
    reader = PdfReader(target)
    text = "\n".join(page.extract_text() for page in reader.pages)
    for expected in ["Export check", "SARSA", "episode 0000", "episode 0599", "End of report"]:
        assert expected in text
    assert len(reader.pages) > 1
    assert any(list(page.images) for page in reader.pages)


@need_results
def test_results_consistent_with_saved_qtables():
    """Re-evaluating the saved Q-tables reproduces the greedy-test table in results.json."""
    from drl_sticky.agents import QLearningAgent
    from drl_sticky.config import Config
    from drl_sticky.evaluate import evaluate_greedy
    R = json.loads(RES.read_text())
    cfg = Config(**{k: v for k, v in R["config"].items() if k in Config.__dataclass_fields__})
    for key, ev in R["evaluation"].items():
        agent = QLearningAgent((cfg.n_pos_bins, cfg.n_vel_bins))
        agent.Q = np.load(ROOT / "results" / "qtables" / f"{key}.npy")
        again = evaluate_greedy(agent, key.endswith("modified"), cfg)
        assert again["mean_reward"] == pytest.approx(ev["mean_reward"])
        assert again["success_rate"] == pytest.approx(ev["success_rate"])


@need_results
def test_config_meets_assignment_minimums():
    R = json.loads(RES.read_text())
    c = R["config"]
    assert c["n_pos_bins"] >= 20 and c["n_vel_bins"] >= 20 and c["n_episodes"] >= 5000
    assert set(R["evaluation"]) == {"qlearning_original", "sarsa_original", "qlearning_modified", "sarsa_modified"}


@pytest.mark.skipif(not NB.exists(), reason="notebook not built yet")
def test_notebook_executed_without_errors_and_ordered():
    import nbformat
    nb = nbformat.read(str(NB), as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert all(o.get("output_type") != "error" for o in cell.get("outputs", [])), cell.source[:80]
    text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    order = ["Group Contribution Declaration", "(a) Modified Environment", "(b) Q-Learning", "(c) SARSA",
             "(d) Performance Evaluation", "(e) Discussion", "(f) Virtual-lab", "Appendix C"]
    pos = [text.find(h) for h in order]
    assert all(p >= 0 for p in pos) and pos == sorted(pos), pos


@pytest.mark.skipif(not PDF.exists(), reason="PDF not built yet")
def test_pdf_first_page_and_contents():
    from pypdf import PdfReader
    reader = PdfReader(str(PDF))
    first = reader.pages[0].extract_text()
    assert "Group Contribution Declaration" in first
    for name in ["SHREYANSH SHARMA", "LAKSHYA KANCHAN", "VINAY PRATAP SINGH BHADAURIA"]:
        assert name in first
    full = "".join(p.extract_text() for p in reader.pages[:60])
    for module in ["env.py", "agents.py", "discretizer.py", "train.py", "evaluate.py", "verify.py", "plotting.py"]:
        assert f"drl_sticky/{module}" in full, module
