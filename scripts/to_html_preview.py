"""Render assignment4_finetune_comparison.ipynb as a standalone HTML page
for the Hermes preview pane (no jupyter/nbconvert dependency)."""
import html
import json
import pathlib
import re

REPO = pathlib.Path(r"C:/Users/maran/OneDrive/Desktop/assignment-4")
NB = REPO / "assignment4_finetune_comparison.ipynb"
OUT = pathlib.Path(__import__("os").environ.get("LOCALAPPDATA", "/tmp")) / "a4_notebook_preview.html"


def md_inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\1">\2</a>', text)
    text = re.sub(r"^> ?(.*)$", r"<blockquote>\1</blockquote>", text, flags=re.M)
    return text


def md_cell(src: str) -> str:
    lines = src.splitlines()
    out, para, in_list = [], [], False

    def flush_para():
        nonlocal para
        if para:
            out.append(f"<p>{md_inline(' '.join(para))}</p>")
            para = []

    for raw in lines:
        line = raw.rstrip()
        if not line:
            flush_para()
            in_list = False
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            flush_para()
            lvl = len(m.group(1)) + 2
            out.append(f"<h{lvl}>{md_inline(m.group(2))}</h{lvl}>")
            in_list = False
            continue
        if re.match(r"^\s*[-*]\s+", line):
            flush_para()
            item = re.sub(r"^\s*[-*]\s+", "", line)
            out.append("<li>" + md_inline(item) + "</li>")
            in_list = True
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            flush_para()
            item = re.sub(r"^\s*\d+\.\s+", "", line)
            out.append("<li>" + md_inline(item) + "</li>")
            in_list = True
            continue
        if line.startswith("|") and line.endswith("|"):
            flush_para()
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not out or not out[-1].startswith("<table>"):
                out.append("<table>" + "".join(f"<td>{md_inline(c)}</td>" for c in cells) + "</table>")
            else:
                out.append("<table>" + "".join(f"<td>{md_inline(c)}</td>" for c in cells) + "</table>")
            in_list = False
            continue
        if in_list:
            out.append(f"<li>{md_inline(line)}</li>")
        else:
            para.append(line)
    flush_para()
    return "".join(out)


def outputs_html(cell) -> str:
    parts = []
    for o in cell.get("outputs", []):
        ot = o.get("output_type")
        if ot == "stream":
            parts.append(f"<pre class='out'>{html.escape(''.join(o.get('text', [])))}</pre>")
        elif ot in ("execute_result", "display_data"):
            d = o.get("data", {})
            if "text/plain" in d:
                parts.append(f"<pre class='out'>{html.escape(''.join(d['text/plain']))}</pre>")
            for key in ("image/png", "image/jpeg"):
                if key in d:
                    src = d[key]
                    src = src if not isinstance(src, list) else "".join(src)
                    parts.append(f"<img src='data:{key};base64,{src}' style='max-width:100%'>")
    return "".join(parts)


nb = json.loads(NB.read_text(encoding="utf-8"))
body = []
for i, cell in enumerate(nb["cells"], 1):
    src = "".join(cell.get("source", []))
    if cell["cell_type"] == "markdown":
        body.append(f"<div class='md'><div class='tag'>MD {i}</div>{md_cell(src)}</div>")
    elif cell["cell_type"] == "code":
        body.append(
            f"<div class='code'><div class='tag'>IN [{i}]</div>"
            f"<pre class='src'>{html.escape(src)}</pre>{outputs_html(cell)}</div>"
        )

css = """
body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 24px auto; max-width: 940px; padding: 0 16px; background: #fbfbfa; color: #1d1d1f; }
h1 { font-size: 22px; border-bottom: 2px solid #e5e5e5; padding-bottom: 8px; }
.md { background: #fff; border: 1px solid #e8e6e3; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px; }
.code { background: #fff; border: 1px solid #e8e6e3; border-radius: 8px; margin-bottom: 14px; overflow: hidden; }
.tag { font-size: 11px; color: #8a8884; letter-spacing: .05em; padding: 6px 12px; background: #f4f2ef; border-bottom: 1px solid #e8e6e3; font-weight: 600; }
pre.src { margin: 0; padding: 12px 16px; background: #faf9f7; overflow-x: auto; font: 12.5px/1.55 'Cascadia Code', Consolas, monospace; white-space: pre; }
pre.out { margin: 0; padding: 10px 16px; background: #f0f7f0; border-top: 1px solid #e8e6e3; color: #1f3d1f; font: 12.5px/1.5 'Cascadia Code', Consolas, monospace; white-space: pre-wrap; }
.md p { line-height: 1.55; }
.md h3 { font-size: 16px; margin: 14px 0 6px; }
.md h4 { font-size: 14px; margin: 12px 0 4px; }
.md h5 { font-size: 13px; margin: 10px 0 4px; }
.md code { background: #f1efe9; padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }
.md li { margin: 2px 0; }
.md table { border-collapse: collapse; margin: 8px 0; display: block; overflow-x: auto; }
.md td { border: 1px solid #e0ddd8; padding: 4px 10px; font-size: 13px; }
blockquote { border-left: 3px solid #c9c4bc; margin: 8px 0; padding-left: 10px; color: #5a5753; }
"""

title = nb.get("metadata", {}).get("title", "Assignment 4 — Fine-Tuning Comparison")
page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{css}</style></head><body>
<h1>📓 {html.escape(title)}</h1>
<p style="color:#8a8884;font-size:13px">{len(nb['cells'])} cells &nbsp;·&nbsp; {NB.name} &nbsp;·&nbsp; rendered preview only — empty outputs mean the notebook hasn't run in Colab yet</p>
{''.join(body)}
</body></html>"""
OUT.write_text(page, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
