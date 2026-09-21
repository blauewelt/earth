#!/usr/bin/env python3
"""Build the paper's editing page — `paper.tex` as blocks Chris can edit and
comment on, with the last compiled PDF beside it.

WHY. The paper is LaTeX in a repository; Chris reads on a phone and wants to
edit and comment without a LaTeX installation or a git client. The page this
script writes is published as a claude.ai Artifact. It shows the source split
into blocks (a paragraph, an environment, a heading, one bibliography entry),
each carrying its first line number in `paper.tex`. A block's edit is saved in
the artifact's own database (collection `edits`, one document per block, keyed
by the block's id); comments go to the artifact's shared comment store through
the claude.ai comment composer. Nothing here compiles LaTeX: the "PDF" tab is
the last build, rendered to page images, and an edit reaches the PDF when
`apply_edits.py` folds it into `paper.tex` and `build.sh` rebuilds.

BLOCK IDS are the first ten hex digits of the SHA-1 of the block's text, so an
edit keeps pointing at the text it was made against: after `paper.tex` changes,
a block whose text changed gets a new id, and an edit made against its old
text shows up on the page as an edit of an earlier version rather than being
laid silently over new text.

Usage:
    python3 ml/paper/editor/build_editor.py [--out PATH]
The round trip (blocks joined back == paper.tex, byte for byte) is asserted
before anything is written.
"""
import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PAPER))

HEAD_RE = re.compile(r"^\\(section|subsection)\*?\{")
BEGIN_RE = re.compile(r"\\begin\{([A-Za-z*]+)\}")
END_RE = re.compile(r"\\end\{([A-Za-z*]+)\}")
FLAT_ENVS = {"document"}                  # blank lines inside still split


def brace_arg(s, i):
    """The balanced {...} argument starting at s[i] == '{' -> (text, end)."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{" and (j == 0 or s[j - 1] != "\\"):
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def plain_title(tex):
    """A heading's LaTeX, readable: math and commands reduced to their text."""
    t = re.sub(r"\\label\{[^}]*\}", "", tex)
    t = t.replace("---", "\u2014").replace("--", "\u2013").replace("~", "\u00a0")
    t = re.sub(r"\$\^\\circ\$", "\u00b0", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", "", t)
    return re.sub(r"[{}$]", "", t).strip()


def split_blocks(text):
    """paper.tex -> spans [(start, end, kind)], separators are what lies between."""
    lines = text.split("\n")
    offs, o = [], 0
    for ln in lines:
        offs.append(o)
        o += len(ln) + 1
    spans = []
    cur = None                     # [start_line, end_line_inclusive, kind]
    depth = 0
    in_bib = False
    in_preamble = True

    def close():
        nonlocal cur
        if cur is not None:
            s, e, k = cur
            spans.append((offs[s], offs[e] + len(lines[e]), k, s + 1))
            cur = None

    for i, ln in enumerate(lines):
        st = ln.strip()
        if in_preamble:
            if st.startswith("\\begin{document}"):
                close()
                in_preamble = False
                cur = [i, i, "text"]
                close()
                continue
            if cur is None:
                if st == "":
                    continue
                cur = [i, i, "preamble"]
            else:
                cur[1] = i
            continue
        if in_bib:
            if st.startswith("\\end{thebibliography}"):
                close()
                cur = [i, i, "bib"]
                close()
                in_bib = False
                continue
            if st.startswith("\\bibitem"):
                close()
                cur = [i, i, "bib"]
                continue
            if cur is None:
                cur = [i, i, "bib"]
            else:
                cur[1] = i
            continue
        if st.startswith("\\begin{thebibliography}"):
            close()
            cur = [i, i, "bib"]
            close()
            in_bib = True
            continue
        if st == "" and depth == 0:
            close()
            continue
        if HEAD_RE.match(st) and depth == 0:
            close()
            kind = "section" if st.startswith("\\section") else "subsection"
            cur = [i, i, kind]
        elif cur is None:
            cur = [i, i, "text"]
        else:
            cur[1] = i
        for m in BEGIN_RE.finditer(ln):
            if m.group(1) not in FLAT_ENVS:
                depth += 1
        for m in END_RE.finditer(ln):
            if m.group(1) not in FLAT_ENVS:
                depth = max(0, depth - 1)
    close()
    return spans


def build_blocks(text):
    spans = split_blocks(text)
    blocks, seen = [], {}
    sec_n, sub_n, appendix = 0, 0, False
    section_label = ""
    for idx, (a, b, kind, line) in enumerate(spans):
        body = text[a:b]
        nxt = spans[idx + 1][0] if idx + 1 < len(spans) else len(text)
        sep = text[b:nxt]
        h = hashlib.sha1(body.encode()).hexdigest()[:10]
        seen[h] = seen.get(h, 0) + 1
        bid = h if seen[h] == 1 else f"{h}-{seen[h]}"
        if "\\appendix" in body:
            appendix, sec_n = True, 0
        title = number = None
        if kind in ("section", "subsection"):
            st = body.lstrip()
            arg, _ = brace_arg(st, st.index("{"))
            title = plain_title(arg)
            if kind == "section":
                sec_n += 1
                sub_n = 0
                number = chr(64 + sec_n) if appendix else str(sec_n)
            else:
                sub_n += 1
                number = f"{chr(64 + sec_n) if appendix else sec_n}.{sub_n}"
            section_label = f"{number} {title}"
        elif kind == "bib":
            section_label = "References"
        blocks.append({"id": bid, "kind": kind, "line": line, "text": body,
                       "sep": sep, "title": title, "number": number,
                       "where": section_label or "Front matter"})
    lead = text[:spans[0][0]] if spans else text
    rebuilt = lead + "".join(bl["text"] + bl["sep"] for bl in blocks)
    assert rebuilt == text, "block split does not round-trip paper.tex"
    return lead, blocks


def page_images(pdf, dpi=96, quality=78):
    tmp = tempfile.mkdtemp(prefix="paper_pages_")
    subprocess.run(["pdftoppm", "-r", str(dpi), "-jpeg", "-jpegopt",
                    f"quality={quality}", pdf, os.path.join(tmp, "p")],
                   check=True)
    out = []
    for n in sorted(os.listdir(tmp)):
        with open(os.path.join(tmp, n), "rb") as f:
            out.append("data:image/jpeg;base64," +
                       base64.b64encode(f.read()).decode())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "paper_editor.html"))
    a = ap.parse_args()
    tex_path = os.path.join(PAPER, "paper.tex")
    text = open(tex_path, encoding="utf-8").read()
    lead, blocks = build_blocks(text)
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short=8",
                             "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    figs = {}
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                           text):
        p = os.path.join(PAPER, "figs", name)
        with open(p, "rb") as f:
            figs["figs/" + name] = base64.b64encode(f.read()).decode()
    data = {
        "commit": commit,
        "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "lines": text.count("\n") + 1,
        "lead": lead,
        "blocks": blocks,
        "pages": page_images(os.path.join(PAPER, "paper.pdf")),
        "figs": figs,
    }
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    tpl = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
    html = tpl.replace("__PAPER_DATA__", blob)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    kinds = {}
    for bl in blocks:
        kinds[bl["kind"]] = kinds.get(bl["kind"], 0) + 1
    print(f"{a.out}: {len(blocks)} blocks {kinds}, {len(data['pages'])} "
          f"pages, {len(html) / 1e6:.2f} MB, commit {commit}")


if __name__ == "__main__":
    sys.exit(main())
