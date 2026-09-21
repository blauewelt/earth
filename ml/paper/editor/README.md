# The paper's editing page

A claude.ai Artifact that shows `ml/paper/paper.tex` as blocks Chris can edit
and comment on, with the last compiled PDF beside it:
[Earth Paper Editor](https://claude.ai/artifact/SjEDUHmfHuo3SvHQH2d9BJ)
(private to Chris; the link opens only for people he shares it with).

- **Build**: `python3 ml/paper/editor/build_editor.py --out <path>` after every
  rebuild of the PDF, then republish that file to the SAME artifact (pass the
  URL above as `url` from a new conversation). The page embeds `paper.tex`,
  the PDF's pages as images and the figures (for the Overleaf zip).
- **Edits** live in the artifact's database, collection `edits`, one document
  per block, keyed by the SHA-1 prefix of the block's text. Read them with the
  ArtifactData tool (`list`, collection `edits`), save as JSON, run
  `python3 ml/paper/editor/apply_edits.py edits.json`, rebuild the paper,
  commit, then delete the applied documents and republish the page. An edit
  whose block changed since it was made is shown on the page as "made against
  an earlier version" and is never applied automatically.
- **Comments** are the artifact's own comment threads (the claude.ai comment
  composer); read and answer them with the ArtifactComments tool. A remote
  session is not notified of new ones — they are read when Chris says so.
- **Download project** gives an Overleaf-ready zip (paper.tex with the page's
  edits, the figures, a README).
