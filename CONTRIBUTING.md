# Contributing to HydrAI

Thanks for helping make this course better. Contributions of all sizes are
welcome — fixing a typo in a concept box counts.

## Easy wins

- **Typos, unclear explanations, broken links** — open a PR directly.
- **A lesson fails on Colab** — open an issue with the "Lesson bug" template;
  include the lesson number, the cell that failed, and the full error.
- **A better exercise or concept box** — yes please. Follow
  [docs/COURSE_STYLE_GUIDE.md](docs/COURSE_STYLE_GUIDE.md) (templates for
  headers, concept boxes, exercises + collapsed solutions).

## Ground rules

1. **Don't break the pipeline.** Every hands-on lesson must run top-to-bottom
   on the sample dataset with the shipped configs, on free Colab.
2. **Style is documented, not debated.** Course cells follow
   `docs/COURSE_STYLE_GUIDE.md`; code follows
   `docs/HYDRAI_PROJECT_CONVENTIONS.md` (mass fractions only, run-level
   splits, plotting rules, config-driven numbers).
3. **Notebook hygiene.** `nbstripout` is registered via `.gitattributes`
   (`nbstripout --install` after cloning if you commit notebooks). Lessons 1–2
   are the exception — their executed outputs are committed on purpose.
4. **Data and mechanisms.** Never commit datasets, model artifacts, or kinetic
   mechanism files. The sample dataset lives in GitHub Releases; mechanisms
   are bring-your-own.

## Dev setup

```bash
git clone https://github.com/karefyllidis/open_HydrAI.git
cd open_HydrAI
pip install -r requirements.txt
nbstripout --install   # respect the .gitattributes filter
```

To test a lesson end-to-end headlessly:

```bash
jupyter nbconvert --to notebook --execute notebooks/Main_4_train_and_evaluate_tree_models_IO.ipynb \
  --output /tmp/Main_4_test.ipynb
```

## Questions

Use [GitHub Discussions](https://github.com/karefyllidis/open_HydrAI/discussions)
for anything that isn't a bug: conceptual questions about the methods,
adapting the pipeline to your own simulator, course feedback.
