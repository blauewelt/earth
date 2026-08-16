# Where the artifacts live — off-GitHub archive plan

**Status: decided, awaiting two accounts.** Chris, 2026-08-16: *"We also need
a backup strategy that is not github. You proposed some research-centric
services (archiv related?). Let's get that started."*

The service is **Zenodo** (CERN's research data repository — the arXiv-adjacent
one), plus **Hugging Face** for the working copies. They do different jobs and
we need both; the reasoning is below, and every quota was read from the
providers' own documentation on 2026-08-16 rather than remembered.

---

## 1 · Why GitHub has to stop being the archive

Measured today: `model-checkpoints-v1` alone holds **290 assets / 135.7 GB**,
after August's 85 GB prune. Against that:

- **2 GiB per asset**, hard. Our 205M heads with optimiser state are ~2.5 GB,
  so every one of them ships as a weights-only file *plus* split
  `.full.partNN` pieces *plus* a `.sha256` manifest — machinery that exists
  only to work around the cap.
- **1000 assets per release.** At 290 and climbing three per experiment.
- **No published size or bandwidth guarantee.** GitHub's terms describe
  releases as intended for distributing software, not as a data repository.
  Chris's own words in August: *"I am wary about how many GBs we push through
  github without paying for it."*
- It is not citable. A paper cannot reference a mutable release asset.

---

## 2 · The split, and why it is two services rather than one

| | **Hugging Face Hub** | **Zenodo** |
|---|---|---|
| **Role here** | the working artifact store — checkpoints, tensors, embedding caches | the frozen, citable snapshot at paper time |
| **Mutability** | versioned, overwrite freely | immutable once published; new version = new DOI |
| **Identifier** | repo path | **a real DOI** |
| **Per-file limit** | 500 GB hard, <200 GB recommended | part of the record quota |
| **Capacity** | free public storage is *best-effort* for a few GB; PRO ($9/mo) covers up to 10 TB public; add-ons $12/TB/month | 50 GB per record by default; a one-time increase to **200 GB / <100 files** can be requested |
| **Good at** | large, changing, frequently pulled | permanent, small-ish, cited |
| **Bad at** | citation, permanence guarantees | iteration, big mutable working sets |

The two are complementary rather than redundant: **Hugging Face is where a
box pulls a 33 GB tensor at 3 a.m.; Zenodo is what the paper's Data
Availability section points at.** Using only Zenodo would mean re-requesting
quota every time an experiment produces a checkpoint, and its <100-file limit
alone rules it out as a working store. Using only Hugging Face would leave
the paper citing a URL that can be force-pushed.

Note for Hugging Face specifically: they grant **free public storage for
impactful research** given a dataset card, a real format (Parquet/WebDataset
preferred) and evidence of reuse. Worth applying for once the pentad tensor
exists — but the plan should not depend on a grant, so PRO at $9/month is
the assumption.

---

## 3 · What goes where

| artifact | size | destination | why |
|---|---|---|---|
| stage-2 heads, all tiers | ~1–2.5 GB each, 136 GB total | **HF**, `blauewelt/earth-checkpoints` | too many and too mutable for Zenodo; the 2 GiB split machinery can be retired entirely |
| codec checkpoints | ~0.5 GB each | **HF** | same |
| the monthly family-3 tensor | 10.9 GB | **HF** + **Zenodo at paper time** | it is the input every published number was computed from — the single most important thing to make citable |
| GLORYS12 daily pull | ~110 GB | **HF** only | re-derivable from CMEMS; archiving it is a convenience, not a duty |
| pentad / daily tensors | 33 / 165 GB | **HF** | working data |
| embedding caches `Z` | 5.2 GB and up | **nowhere** | re-derivable from codec + tensor in ~80 min; keep the box-local and GitHub copies, archive neither |
| the paper + figures + experiment log | small | git, and **Zenodo** at submission | git history is already the timestamp; Zenodo adds the DOI |

**The rule that decides the ambiguous cases:** archive what cannot be
recomputed. A checkpoint is a training run nobody will pay for twice. An
embedding cache is eighty minutes of GPU.

---

## 4 · What I need from Chris (this is the whole blocker)

1. **A Hugging Face account**, and a token with write scope, pasted into a
   project doc the way the CMEMS and GitHub credentials already are (a new
   `claude/huggingface-access.md`, same footing). PRO ($9/month) if we are not
   waiting on a storage grant.
2. **A Zenodo account** — free, no tier needed — and a personal access token
   with `deposit:write` and `deposit:actions`. Zenodo also has a **sandbox**
   (`sandbox.zenodo.org`) with separate accounts, which is where the upload
   path should be exercised first so a broken script never creates a junk DOI.

Neither can be created from this sandbox, and I would rather ask than
improvise around it.

**Optional third**, worth considering separately: an S3-compatible bucket
(Cloudflare R2, ~$0.015/GB/month, **zero egress**) as the box-facing pull
target. Hugging Face works fine for that, so this is only if pull bandwidth
or rate limits become the constraint — E-033 §5 has the arithmetic.

---

## 5 · What happens once the tokens exist

1. Exercise the whole path against **Zenodo sandbox** and a throwaway HF repo.
2. Mirror the crown jewels first: the xl89/xl144 200k heads, the xl55 tier,
   the 41M codec, and the family-3 tensor.
3. Verify by **downloading back and checking sha256 against the manifest** —
   an upload that reports success is not evidence the bytes are retrievable
   (ml/CLAUDE.md §0.2), and this is a backup, so the restore path is the
   only part that matters.
4. Only then prune GitHub further, and retire the 2 GiB split machinery.
5. At paper submission: one Zenodo record with the tensor, the final
   checkpoints and the code snapshot; put the DOI in the paper.

---

*Quotas above are from Hugging Face's and Zenodo's own documentation, read
2026-08-16. Both change; re-read before relying on a number.*

Sources:
- [Hugging Face — storage limits](https://huggingface.co/docs/hub/storage-limits)
- [Zenodo — quota increase](https://help.zenodo.org/docs/deposit/manage-files/quota-increase)
