# HydrAI Course Style Guide

Authoring rules for the tutorial layer inside `notebooks/Main_1–10`. Every
lesson must follow these templates so the course reads as one voice.
Companion blueprint: `docs/TUTORIAL_COURSE_PLAN.md`.

## 1. Framing and voice

- Audience: **graduate students learning "ML for science"** — comfortable with
  Python and basic ML vocabulary, *not* assumed to know chemical engineering.
- The plug-flow reactor is the **worked example, not the subject**. Every
  method lesson explains the transferable idea first, the PFR instance second.
- Chemistry is explained only as far as the method requires (one or two
  sentences, plain language). Methods are explained properly.
- Where natural (roughly once or twice per lesson, not every box), close a
  concept box with an *"In your domain:"* line mapping the idea to other
  fields (climate models, CFD, battery simulation, epidemiology…).
- Tone: direct, concrete, no hype. Sentences over bullet-walls. British or
  American spelling both fine; be consistent within a lesson.

## 2. Lesson map

| Lesson | Notebook | Course title | Mode | Colab extras (`%pip install -q …`) | Release assets to download |
|---|---|---|---|---|---|
| 1 | Main_1 | Simulation as a data source | read-along | — | — |
| 2 | Main_2 | Design of experiments & batch data generation | read-along | — | — |
| 3 | Main_3 | Exploring simulation data & feature engineering | hands-on | — | raw sample → `data/training/` |
| 4 | Main_4 | Tree-ensemble baselines | hands-on | `xgboost scikit-optimize` | processed sample → `data/processed/` |
| 5 | Main_5 | Hyperparameter tuning & full-profile prediction | hands-on | `xgboost scikit-optimize` | processed sample |
| 6 | Main_6 | Neural surrogates done honestly | hands-on | `optuna torchinfo` | processed sample |
| 7 | Main_7 | Physics-informed neural networks | hands-on | `optuna torchinfo` | processed sample |
| 8 | Main_8 | Distilling networks into equations (SR) | hands-on | `pysr` | processed sample + pretrained models zip |
| 9 | Main_9 | Validating surrogates against ground truth | hands-on | — | processed sample + pretrained models zip |
| 10 | Main_10 | Bayesian optimisation over a surrogate | hands-on | `optuna cantera` | processed sample + pretrained models zip |

- Read-along lessons (1–2) require the proprietary kinetic mechanism, which is
  **not redistributable**. They ship with executed outputs committed
  (`.gitattributes` exempts them from nbstripout) and open with the read-along
  banner (§3.2) instead of a bootstrap cell.
- Any section that needs the mechanism at runtime (e.g. Main_10's Cantera
  validation) must be guarded: check the mechanism file exists, else print a
  clear "read-along only — mechanism not available" message and skip without
  raising errors.

## 3. Cell templates

### 3.1 Lesson header (first cell of every notebook, markdown)

Replace the existing title cell with:

```markdown
# Lesson N — <Course title>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karefyllidis/open_HydrAI/blob/main/notebooks/<FILENAME>.ipynb)

*Part of the [HydrAI course](../README.md): machine-learning surrogates for physical simulation, taught on a chemical reactor.*

**Mode:** hands-on (Colab-ready) · **Runtime:** ~X min on free Colab · **Builds on:** Lesson M

**What you'll learn**

1. <objective — a skill, phrased as something the student can do afterwards>
2. …(3–5 total)

**The example system.** <2–4 sentences: what the PFR data represents, only
what this lesson needs. Link to Lesson 1 for the full story.>
```

For read-along lessons the **Mode** line is
`**Mode:** read-along (requires a kinetic mechanism — outputs shown below)` and
the Colab badge is omitted, with this banner appended:

```markdown
> ⚠️ **Read-along lesson.** This notebook needs a detailed kinetic mechanism
> that we cannot redistribute. All outputs are committed, so you can follow
> every step without running it. If you have your own Cantera mechanism, point
> `configs/simulation/main1_reactant_database.json` at it and the notebook runs
> end-to-end. Hands-on work starts in [Lesson 3](Main_3_data_exploration_feature_engineering.ipynb).
```

### 3.2 Colab bootstrap (second cell of hands-on lessons, code)

```python
# ══ 0. COLAB BOOTSTRAP ══
# Running locally? This cell is a no-op — just run it and move on.
import sys, subprocess
from pathlib import Path

if "google.colab" in sys.modules:
    if Path.cwd().name != "notebooks":            # fresh Colab VM
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/karefyllidis/open_HydrAI.git"], check=True)
        %cd open_HydrAI/notebooks
    %pip install -q <LESSON EXTRAS>               # omit line if none
    _REL = "https://github.com/karefyllidis/open_HydrAI/releases/download/sample-data-v1"
    import urllib.request, zipfile
    for _f, _d in [<ASSET LIST>]:                 # e.g. ("features_targets_training_data_complete_sample150.pkl", "../data/processed")
        _p = Path(_d); _p.mkdir(parents=True, exist_ok=True)
        if not (_p / _f).exists():
            print(f"Downloading {_f} …")
            urllib.request.urlretrieve(f"{_REL}/{_f}", _p / _f)
            if _f.endswith(".zip"):
                with zipfile.ZipFile(_p / _f) as _z:
                    _z.extractall(_p)
```

Asset lists per lesson are in the §2 table; the models zip is
`("models_pretrained_sample.zip", "../models")`.

### 3.3 Concept box (markdown, at point of use)

```markdown
> 🧠 **Concept — <short name>.** <3–8 sentences explaining the transferable
> idea: what it is, why it matters here, what goes wrong without it.>
>
> *In your domain:* <one sentence mapping it elsewhere — optional, ≤2 per lesson.>
```

Good concept-box subjects are method ideas (run-level splits, residual losses,
acquisition functions), not code walkthroughs.

### 3.4 Exercise + solution

Markdown cell:

```markdown
#### 💪 Exercise N.M — <imperative title>

<What to do and what to look for, 2–5 sentences. State the expected
observation ("you should see R² drop noticeably") so students can self-check.>
```

Code cell immediately after — **must execute cleanly as shipped** (scaffold
runs and prints something sensible even before the student edits it):

```python
# 💪 Exercise N.M — your turn (edit and re-run)
# TODO: <the one or two lines the student should change>
<runnable scaffold using copies of pipeline variables>
```

Markdown solution cell immediately after:

```markdown
<details><summary><b>💡 Solution & discussion — Exercise N.M</b></summary>

​```python
<solution code>
​```

<2–5 sentences of discussion: what happens and why.>
</details>
```

Rules: 2–4 exercises per lesson, numbered `N.1, N.2, …` (N = lesson number).
Exercises must operate on **copies** (`df.copy()`, re-instantiated models) —
never mutate state that downstream pipeline cells rely on.

## 4. Hard rules

1. **Never break the pipeline.** All original computation cells keep working,
   top-to-bottom, on the sample dataset. Enrichment adds cells; it does not
   reorder sections or rename exports.
2. Keep the `# ══ N. TITLE ══` section banners and numbering. The bootstrap is
   section 0.
3. Notebook flag cells keep the repo convention: `IF_*` booleans and `Path`s
   only; numbers/strings live in `configs/ml/*.json`. Tutorial defaults favour
   the fast path (`IF_HYPERPARAM_TUNING = False` etc.).
4. Exercise/solution/concept cells must not require packages outside the
   lesson's §2 extras column.
5. Respect project data rules (mass fractions only; run-level splits) and
   plotting rules (`docs/HYDRAI_PROJECT_CONVENTIONS.md`, CLAUDE.md) in any code
   you add — including exercise solutions.
6. Notebooks are edited cell-by-cell (NotebookEdit); nbstripout keeps outputs
   out of git for hands-on lessons — do not fight it.
