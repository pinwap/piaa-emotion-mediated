# PIAA project handoff

Updated: 2026-09-09. This is a continuity note, not a transcript or a fresh audit certificate.

## Read first

- `paperskill.md`: detailed research concepts and lessons.
- `docs/METHODOLOGY.md` and `docs/DATA.md`: implementation and data protocol.
- `jaist_thai_week_15min_th.md`: initial presentation outline. The user rejected its density and pace. Revise toward a slower, easier talk with concrete examples before translating into English.
- `paper/supplementary_release/`: supplementary sources.
- `src/modeling/pipeline.py`, `src/experiments/faithfulness.py`, `src/utils/metrics.py`: verify execution and statistics here before asserting facts.

## User preferences

Explain in Thai, step by step. Paper prose should be concise, clear professional English. Explain symbols consistently. Do not overwhelm with terminology or rushed detail. Do not claim an audit passed without checking the actual files. Never push to a remote or add yourself as coauthor. The user controls Git publication. Preserve existing edits and frozen results.

## Project status and decisions

The user reports that the WACV paper and supplementary were submitted. Next work includes a 15-minute JAIST Thai Week talk and later arXiv preparation. Hayashi requested significance annotations directly on figures. PARA is on hold following concerns about the emotion annotation distribution and category confounding. The user's university QCNN project takes priority over nonurgent future research.

Formula Swap is frozen to K=5 sampled peers. Do not silently substitute a cyclic peer protocol. Image Swap uses the 49 nonzero cyclic shifts of 50 evaluation images. These are different experiments.

Anchor C executes y_pop(x) + z(x)^T delta_u + b_u. Its population prediction comes from raw visual features. Seven emotion coefficients describe the feature-dependent personal correction. The derived w_total = w_pop + delta is an emotion-space interpretation, not a decomposition of the raw-feature population branch. Confirm exact diagnostic scope in code.

Average statistics must preserve the three domains of each participant as a cluster: 387 user-domain rows, 129 participant clusters. Average seeds within unit first where multiple seeds exist. Holm families and one-/two-sided alternatives must match the actual statistics generator.

Earlier files and assistant statements conflict in some rounded numbers and SEMs. Treat historical messages as context, not numerical authority. Read frozen CSVs and recompute the precise contrast before updating claims. In the latest inspected raw database, primary Hybrid and Direct SROCC were approximately .459236 and .477579, giving a full-precision gap of approximately .018343. Printed means .459 and .478 subtract to .019, which is a rounding artifact.

## Transfer checklist

Move the working repository including uncommitted/untracked documents, plus required ignored output/data/feature files. Git clone alone does not copy these. Recreate the environment from the lockfile rather than copying a virtual environment between operating systems. Also copy the submitted PDF/ZIP and original attachments stored outside the repository if historical comparison is needed.

This note preserves decisions and current work. It does not replace the full conversation history. Prefer the app's supported remote connection or chat handoff for continuing the actual thread when available.

## First prompt on the new computer

Read PROJECT_HANDOFF.md, paperskill.md, docs/METHODOLOGY.md and jaist_thai_week_15min_th.md. We are continuing this project. First simplify the JAIST Thai Week presentation: the previous 11-slide draft is too dense and fast. Use a concrete example to explain personalization and the model, then select the essential results for 15 minutes. Draft Thai slide content and speaker script before English. Verify numerical and mathematical claims against the current files. Do not push or change frozen experimental results.
