"""Build the single submission PDF: notebook -> execute -> HTML -> PDF.

    python make_pdf.py                # rebuild notebook, execute it, export the PDF
    python make_pdf.py --no-execute   # export an already-executed notebook (e.g. run in the virtual lab)
    python make_pdf.py --html-only    # export only standalone HTML

Produces:  "Group 84 - Q_learning_SARSA_Set A.pdf"  (exact name required by the assignment)
PDF export uses Python packages only; no browser, LaTeX or system renderer is needed.
"""

from __future__ import annotations

import argparse
import os
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


def html_to_pdf(html_path: str, pdf_path: str) -> bool:
    """Render notebook HTML as paginated PDF without a browser or external commands."""
    import base64
    import io
    import tempfile
    from html import escape
    from pathlib import Path

    from bs4 import BeautifulSoup, NavigableString
    from matplotlib import get_data_path
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, LongTable, Paragraph, Preformatted, SimpleDocTemplate, Spacer

    fonts = Path(get_data_path()) / "fonts" / "ttf"
    for name, filename in [("Report", "DejaVuSans.ttf"), ("ReportBold", "DejaVuSans-Bold.ttf"),
                           ("ReportItalic", "DejaVuSans-Oblique.ttf"),
                           ("ReportBoldItalic", "DejaVuSans-BoldOblique.ttf"),
                           ("ReportMono", "DejaVuSansMono.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold",
                                  italic="ReportItalic", boldItalic="ReportBoldItalic")
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "Report"
    styles["Normal"].fontSize = 9
    styles["Normal"].leading = 12
    styles["Normal"].spaceAfter = 5
    table_style = ParagraphStyle("ReportTable", parent=styles["Normal"], fontSize=6, leading=8)
    code_style = ParagraphStyle("ReportCode", fontName="ReportMono", fontSize=6.4, leading=8,
                                spaceAfter=6)
    log_style = ParagraphStyle("ReportLog", parent=code_style, fontSize=5.5, leading=6.5)
    width = A4[0] - 68
    height = A4[1] - 80
    source = Path(html_path).resolve()
    soup = BeautifulSoup(source.read_text(encoding="utf-8"), "html.parser")

    def inline(node):
        """Keep supported inline formatting while escaping notebook text."""
        if isinstance(node, NavigableString):
            return escape(str(node))
        content = "".join(inline(child) for child in node.children)
        tag = {"strong": "b", "em": "i"}.get(node.name, node.name)
        if tag in {"b", "i", "u", "sub", "sup"}:
            return f"<{tag}>{content}</{tag}>"
        if tag == "code":
            return f'<font name="ReportMono">{content}</font>'
        if tag == "br":
            return "<br/>"
        return content

    def blocks(node):
        """Translate notebook content blocks into splittable ReportLab flowables."""
        if isinstance(node, NavigableString):
            if str(node).strip():
                yield Paragraph(escape(str(node)), styles["Normal"])
            return
        tag = node.name
        if tag in {"script", "style", "head", "button", "svg"}:
            return
        if "jp-InputPrompt" in node.get("class", []) or "jp-OutputPrompt" in node.get("class", []):
            return
        if tag == "pre":
            dense = node.find_parent(class_="dense-log") is not None
            style = log_style if dense else code_style
            line_length = int(width / pdfmetrics.stringWidth("M", style.fontName, style.fontSize))
            yield Preformatted(node.get_text(), style, maxLineLength=line_length)
        elif tag == "table":
            rows = [row.find_all(["th", "td"], recursive=False) for row in node.find_all("tr")]
            rows = [row for row in rows if row]
            if not rows:
                return
            columns = max(len(row) for row in rows)
            weights = [max(6, min(32, max((len(row[index].get_text()) if index < len(row) else 0)
                                       for row in rows))) for index in range(columns)]
            data = [[Paragraph(inline(cell), table_style) for cell in row]
                    + [""] * (columns - len(row)) for row in rows]
            header = int(any(cell.name == "th" for cell in rows[0]))
            table = LongTable(data, colWidths=[width * weight / sum(weights) for weight in weights],
                              repeatRows=header, hAlign="LEFT")
            table.setStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 3),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 3)])
            yield table
            yield Spacer(1, 6)
        elif tag == "img":
            location = node.get("src", "")
            if location.startswith("data:image/") and ";base64," in location:
                image_source = io.BytesIO(base64.b64decode(location.split(",", 1)[1]))
            elif "://" not in location and location:
                image_source = str(source.parent / location)
            else:
                raise ValueError(f"PDF images must be embedded or local: {location[:80]}")
            image = Image(image_source)
            scale = min(width / image.imageWidth, (height - 12) / image.imageHeight, 1)
            image.drawWidth = image.imageWidth * scale
            image.drawHeight = image.imageHeight * scale
            yield image
            yield Spacer(1, 6)
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li"}:
            style = styles[f"Heading{tag[1]}"] if tag.startswith("h") else styles["Normal"]
            prefix = "- " if tag == "li" else ""
            yield Paragraph(prefix + inline(node), style)
            for image in node.find_all("img"):
                yield from blocks(image)
        else:
            for child in node.children:
                yield from blocks(child)

    def footer(canvas, document):
        """Add the submission title and page number to every page."""
        canvas.setFont("Report", 7)
        canvas.drawString(34, 24, "Group 84 - Q_learning_SARSA_Set A")
        canvas.drawRightString(A4[0] - 34, 24, f"Page {document.page}")

    target = Path(pdf_path).resolve()
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".pdf", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        document = SimpleDocTemplate(str(temporary_path), pagesize=A4, leftMargin=34, rightMargin=34,
                                     topMargin=36, bottomMargin=44, title=Path(PDF_NAME).stem)
        document.build(list(blocks(soup.body or soup)), onFirstPage=footer, onLaterPages=footer)
        temporary_path.replace(target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True


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
    html_to_pdf(HTML_NAME, PDF_NAME)
    print("PDF  ->", PDF_NAME)


if __name__ == "__main__":
    main()
