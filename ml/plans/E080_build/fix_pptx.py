#!/usr/bin/env python3
"""pptxgenjs emits one <a:pPr> per RUN inside a paragraph; renderers take the
last one, which wipes the paragraph's bullet whenever an item has inline runs.
Keep only the first <a:pPr> in each <a:p>."""
import re, shutil, sys, zipfile

PATH = sys.argv[1]
PPR = re.compile(r"<a:pPr\b[^>]*/>|<a:pPr\b[^>]*>.*?</a:pPr>", re.S)
PARA = re.compile(r"<a:p>.*?</a:p>", re.S)


def fix_para(m):
    p = m.group(0)
    seen = {"n": 0}

    def repl(mm):
        seen["n"] += 1
        return mm.group(0) if seen["n"] == 1 else ""

    return PPR.sub(repl, p)


src = PATH + ".orig"
shutil.copy(PATH, src)
zin = zipfile.ZipFile(src)
n_fixed = 0
with zipfile.ZipFile(PATH, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if re.match(r"ppt/(slides|notesSlides)/[^/]+\.xml$", item.filename):
            txt = data.decode("utf-8")
            new = PARA.sub(fix_para, txt)
            if new != txt:
                n_fixed += 1
            data = new.encode("utf-8")
        zout.writestr(item, data)
zin.close()
print(f"stripped duplicate pPr in {n_fixed} parts of {PATH}")
