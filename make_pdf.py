"""Build the single submission PDF: notebook -> execute -> HTML -> PDF.

    python make_pdf.py                # rebuild notebook, execute it, export the PDF
    python make_pdf.py --no-execute   # export an already-executed notebook (e.g. run in the virtual lab)
    python make_pdf.py --html-only    # stop after HTML (print it to PDF from any browser)

Produces:  "Group 84 - Q_learning_SARSA_Set A.pdf"  (exact name required by the assignment)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

import nbformat

PDF_NAME = "Group 84 - Q_learning_SARSA_Set A.pdf"
NB_NAME = "Group84_Q_learning_SARSA_SetA.ipynb"
HTML_NAME = "Group84_Q_learning_SARSA_SetA.html"

# Print stylesheet injected into the HTML export (A4, wrapped code, page numbers).
PRINT_CSS = """
<style>
@page { size: A4; margin: 14mm 12mm 16mm 12mm;
        @bottom-center { content: "Group 84 - Q_learning_SARSA_Set A  |  page " counter(page) " of " counter(pages);
                         font-size: 8pt; color: #666; } }
html, body { background: #fff !important; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 10pt; }
.jp-Notebook, #notebook-container { padding: 0 !important; width: 100% !important; }
.jp-InputPrompt, .jp-OutputPrompt, .prompt, .input_prompt, .output_prompt { display: none !important; min-width: 0 !important; width: 0 !important; }
.jp-Cell { padding: 2px 0 !important; break-inside: auto; }
pre, code, .jp-RenderedText pre, .highlight pre, .CodeMirror pre {
  white-space: pre-wrap !important; word-break: break-word !important; overflow-wrap: anywhere !important;
  font-size: 7.6pt !important; line-height: 1.28 !important; }
.jp-InputArea-editor, .highlight { background: #f6f6f4 !important; border: 1px solid #e4e3df !important; }
.jp-RenderedHTMLCommon h1 { font-size: 18pt; }
.jp-RenderedHTMLCommon h2 { font-size: 14pt; break-before: page; border-bottom: 2px solid #2a78d6; padding-bottom: 3px; }
.jp-RenderedHTMLCommon h2.nobreak, .jp-RenderedHTMLCommon h2:first-of-type { break-before: auto; }
.jp-RenderedHTMLCommon h3 { font-size: 11.5pt; break-after: avoid; }
.jp-RenderedHTMLCommon p, .jp-RenderedHTMLCommon li { font-size: 10pt; line-height: 1.45; }
.jp-RenderedHTMLCommon table { font-size: 8pt; border-collapse: collapse; break-inside: avoid; margin: 4px 0; }
.jp-RenderedHTMLCommon td, .jp-RenderedHTMLCommon th { padding: 2px 4px !important; border: 1px solid #ddd; }
.jp-OutputArea-output table { font-size: 7pt !important; max-width: 100% !important; }
.jp-OutputArea-output th { white-space: normal !important; word-break: break-word !important; overflow-wrap: anywhere !important; }
.jp-OutputArea-output td, .jp-OutputArea-output tbody th { white-space: nowrap !important; word-break: normal !important; overflow-wrap: normal !important; }
.jp-OutputArea-output, .jp-OutputArea-child { overflow: visible !important; max-width: 100% !important; }
.jp-RenderedImage img, .jp-OutputArea-output img { max-width: 100% !important; break-inside: avoid; }
.dense-log pre { font-size: 5.9pt !important; line-height: 1.18 !important; }
.jp-OutputArea-output pre { font-size: 7.4pt !important; }
</style>
"""


def execute_notebook(path: str, timeout: int = 3600) -> None:
    """Run every cell of the notebook in place (working directory = project root)."""
    from nbclient import NotebookClient
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": os.path.dirname(os.path.abspath(path))}})
    t0 = time.time()
    client.execute()                                         # raises on any cell error
    nbformat.write(nb, path)
    print(f"executed notebook in {time.time() - t0:.0f}s")


def export_html(nb_path: str, html_path: str) -> None:
    """Convert the executed notebook to standalone HTML and inject the print CSS."""
    from nbconvert import HTMLExporter
    from traitlets.config import Config as TConfig
    nb = nbformat.read(nb_path, as_version=4)
    c = TConfig()
    c.TagRemovePreprocessor.enabled = True                     # hide code of text-rendering cells
    c.TagRemovePreprocessor.remove_input_tags = {"remove_input"}
    exporter = HTMLExporter(template_name="lab", config=c)
    exporter.embed_images = True                              # self-contained file
    body, _ = exporter.from_notebook_node(nb)
    body = body.replace("</head>", PRINT_CSS + "</head>", 1)  # add print styles
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(body)


def find_chrome() -> str | None:
    """Locate a Chrome/Chromium/Edge binary for headless printing."""
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ]
    for name in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "msedge", "chrome"]:
        p = shutil.which(name)
        if p:
            candidates.append(p)
    return next((c for c in candidates if os.path.exists(c)), None)


def html_to_pdf(html_path: str, pdf_path: str) -> bool:
    """Print the HTML to PDF with headless Chrome; returns False if no browser is available."""
    chrome = find_chrome()
    if chrome is None:
        return False
    cmd = [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--no-sandbox",
           f"--print-to-pdf={os.path.abspath(pdf_path)}", "file://" + os.path.abspath(html_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    return os.path.exists(pdf_path)


def main() -> None:
    """Build notebook -> (execute) -> HTML -> PDF."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-execute", action="store_true", help="use the already executed notebook")
    ap.add_argument("--html-only", action="store_true")
    args = ap.parse_args()
    if not args.no_execute:
        import build_notebook
        build_notebook.build(NB_NAME)                          # regenerate cells from the package
        execute_notebook(NB_NAME)
    export_html(NB_NAME, HTML_NAME)
    print("HTML ->", HTML_NAME)
    if os.path.exists("results/results.json") and os.path.isdir("docs"):
        import write_results_md
        write_results_md.main()                                 # keep docs/results.md in sync
    if args.html_only:
        return
    if html_to_pdf(HTML_NAME, PDF_NAME):
        print("PDF  ->", PDF_NAME)
    else:
        print("No Chrome/Chromium found: open", HTML_NAME, "in a browser and Print -> Save as PDF as", PDF_NAME)
        sys.exit(1)


if __name__ == "__main__":
    main()
