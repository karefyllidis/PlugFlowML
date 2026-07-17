# HydrAI → Open Tutorial Course — Transformation Plan

Agreed 2026-07-17 (interview with repo owner). This document is the blueprint for
repositioning open_HydrAI as a 10-lesson open-source course:
**"ML surrogates for physical simulation"** — taught for a generic *ML-for-science*
graduate audience, with the plug-flow reactor as the worked example rather than
the subject.

---

## 1. Decisions (locked)

| Dimension | Decision |
|---|---|
| Audience | Generic "ML for science" grads. The PFR is *an example system*; the transferable pipeline (simulate → explore → baseline → NN → PINN → symbolic regression → Bayesian optimisation) is the curriculum. Chemistry explained only as far as needed. |
| Scope | All 10 notebooks, one linear course: Lesson 1 → Lesson 10. |
| Format | Enrich `Main_1–10` **in place** — no parallel Tutorial_N set. Single source of truth. |
| Pedagogy | Per notebook: learning objectives + prerequisites header, "concept boxes" (short theory markdown at point of use), inline exercises with solutions. No checkpoint quizzes or capstones (for now). |
| Data access | Ship a small **sample dataset as a GitHub Release asset**; notebooks download it in a bootstrap cell. Full 23 GB dataset stays local/unpublished. |
| Mechanism | The 153-species kinetic mechanism **cannot be shared**. `Main_1` and `Main_2` become **read-along lessons** with committed, pre-executed outputs. Hands-on work starts at `Main_3`. |
| Sensitivity line | Only the mechanism YAML is sensitive. Simulated (derived) data, heat-flux/reactant configs, parameter ranges, and HPC docs are all publishable. |
| Runtime | **Colab-first**: every notebook gets an "Open in Colab" badge and is tested on free-tier Colab. Config defaults sized so each hands-on lesson finishes in minutes, not hours. |
| Positioning | Same repo (`open_HydrAI`), **course-first README**: lead with the 10-lesson course; research framing and authorship retained below. |
| Citation | `CITATION.cff` + Zenodo-archived release (DOI). No JOSE paper for now. |
| Community | Light infrastructure: issue templates ("bug in lesson N" / "question"), GitHub Discussions for Q&A, short `CONTRIBUTING.md` welcoming typo/exercise fixes. |

---

## 2. Course narrative

Each lesson keeps its `Main_N` identity but gains a course title stressing the
*method*, e.g.:

| Lesson | Notebook | Method taught | Mode |
|---|---|---|---|
| 1 | Main_1 | Physics simulation as data source (stiff ODE ground truth) | read-along |
| 2 | Main_2 | Design of experiments: LHS sampling, batch data generation | read-along |
| 3 | Main_3 | EDA + feature engineering for simulation data; dimensionality lumping | hands-on |
| 4 | Main_4 | Tree-ensemble baselines; honest metrics | hands-on |
| 5 | Main_5 | Hyperparameter tuning; predicting full spatial profiles | hands-on |
| 6 | Main_6 | Neural surrogates; leakage-safe splits; physics-aware evaluation | hands-on |
| 7 | Main_7 | Physics-informed NNs: residual losses, curriculum, collocation | hands-on |
| 8 | Main_8 | Symbolic regression: distilling NNs into equations | hands-on |
| 9 | Main_9 | Surrogate validation against ground truth | hands-on |
| 10 | Main_10 | Bayesian optimisation over a surrogate + physical validation | hands-on |

Lessons 1–2 teach *how the data was made* (and how to make your own if you have
a mechanism); Lesson 3's bootstrap cell downloads the sample release so the
hands-on track is self-sufficient from there.

---

## 3. Technical workstream (Phase A — foundations)

1. **Sample dataset**
   - Script (`scripts/dev/make_sample_dataset.py`): subsample N runs (target ≈ 100–200
     runs ≈ tens of MB compressed) from the full training pickle; export.
   - Format: prefer **Parquet** for the public asset (pickle from the internet is a
     security/version liability); add a small loader shim so Main_3 accepts either.
     If effort must be minimal, a subset `.pkl` matching the current schema is the fallback.
   - Attach to a GitHub Release (`sample-data-vX`); notebooks reference the release URL.
   - Write a dataset card (columns, units, provenance, sampling, license) — the
     `dataset-card` skill exists for this.
2. **Packaging for Colab**
   - Add minimal `pyproject.toml` so `pip install git+https://github.com/karefyllidis/open_HydrAI`
     provides `src/` imports; bootstrap cell = pip install + data download + config load.
3. **nbstripout exemption**
   - `.gitattributes`: unset the filter for `Main_1`/`Main_2` so their executed
     outputs (figures, profiles) are committed for read-along use. All other
     notebooks stay stripped.
4. **Colab-scale configs**
   - Re-tune per-notebook `configs/ml/*.json` defaults to free-Colab budgets
     (subsample rows, epochs, Optuna trials, PySR iterations ≈ 5–15 min/lesson).
     Keep research-scale values documented in `docs/ML_CONFIG_GUIDE.md` (or a
     `*_research.json` copy) so the pipeline remains research-usable.

## 4. Course workstream (Phase B — enrichment)

Per notebook (order: 3 → 4 → 6 → 8 → 10 → 5 → 7 → 9 → 1 → 2, most-taught first):

- Header cell: lesson number/title, learning objectives (3–5), prerequisites,
  estimated runtime, Colab badge.
- Concept boxes at point of use (e.g. Lesson 6: "why run-level splits", Lesson 7:
  "what a residual loss is"), written for the ML-for-science reader — chemistry
  minimally, methods thoroughly, with a recurring "in your domain, this maps to…" line.
- 2–4 inline exercises per notebook (`# 💪 EXERCISE` cells with TODOs) + solutions
  (collapsed cell or `notebooks/solutions/` — decide once, apply everywhere).
- Main_1–2 additionally: convert to read-along voice ("what you're seeing and why"),
  execute fully, commit outputs.

## 5. Publishing workstream (Phase C — positioning)

- README rewrite: course-first pitch, lesson table, "start here" path
  (Colab Lesson 3 in one click), then research framing/architecture as today.
- `CITATION.cff`; tag a release; archive on Zenodo for a DOI; add DOI badge.
- `.github/`: issue templates (lesson bug / question), `CONTRIBUTING.md`,
  enable GitHub Discussions.
- Verify licence coverage: MIT for code; state data licence (e.g. CC-BY-4.0) in the
  dataset card.

---

## 6. Implementation notes (2026-07-17 build)

Decisions that changed from §3–5 during implementation:

- **Fresh sample generation instead of subsampling.** The 23 GB research pickle
  cannot be loaded on a 32 GB workstation, so the sample was **regenerated**:
  150 LHS runs (seed 7, `configs/ml/main2_sample_dataset_config.json`,
  committed), 138 successful, 27 661 rows — independent draw, clean provenance.
  See `docs/SAMPLE_DATASET_CARD.md`.
- **Clone-based Colab bootstrap instead of pyproject packaging.** Notebooks
  already `sys.path.insert(project_root)`; on Colab the bootstrap cell clones
  the repo and `%cd`s into `notebooks/`, keeping one code path. No
  `pyproject.toml` was added.
- **Config-pinned data resolution.** Main_3 gained `run_stamp_sample`
  (`use_run_stamp: "sample"`), Main_4–10 gained `processed_stem: "sample150"`
  (glob-latest fallback) so the sample never silently shadows a research
  export and vice versa.
- **Pretrained-models release asset.** Colab sessions don't persist across
  lessons, so Lessons 8–10 download `models_pretrained_sample.zip` (SimpleNN,
  PINN, SR artifacts trained on the sample at course budgets) and run
  standalone.
- **Read-along guard in Main_10.** The Cantera validation section checks for a
  local mechanism and degrades to read-along when absent.
- Style templates for all lesson cells: `docs/COURSE_STYLE_GUIDE.md`.

## 7. Known risks / open items

- **PySR on Colab (Lesson 8)**: PySR needs a Julia backend; first-run setup on Colab
  takes several minutes and occasionally breaks. Mitigation: pinned install cell,
  tiny default budget, and committed fallback outputs so the lesson is completable read-along.
- **Colab session limits (Lessons 6–7)**: tutorial-scale epochs must fit CPU/T4
  free tier; verify end-to-end on a fresh Colab account before release.
- **Pickle → Parquet shim**: touching Main_3's ingest needs care not to break the
  research path with the full pickle. Loader must accept both.
- **Sample size**: pick run count after checking per-run row size; hard cap: single
  release asset ≤ 2 GB (aim ≪ 100 MB).
- **Sync burden**: exercises live inside the pipeline notebooks; any pipeline change
  must keep exercise cells runnable. Add a CI smoke-run (papermill on the sample
  data) if maintenance becomes real.
