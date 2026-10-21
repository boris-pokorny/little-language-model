# little-language-model

A little language model from scratch in plain PyTorch: BPE tokenizer -> transformer -> sampling.

Code lives in the `llm` package; the pipeline stages are run as modules via `pdm run`.

## Setup

Put one or more PDFs into `data/corpus/` (versioned by DVC, not git). One coherent text works best for a model this small.

```sh
pdm install
```

## Pipeline

```sh
pdm run ingest                  # 1. data ingestion:         data/corpus/*.pdf -> data/feature/corpus.txt + vocab.json
pdm run extract                 # 2. feature extraction:     build BPE merge table -> data/feature/merges.txt
pdm run tokenize                # 3. feature transformation: tokenize corpus.txt -> data/feature/tokens.txt  (or: pdm run tokenize "some text", or pipe into pdm run tokenize -)
pdm run train                   # 4. model training:         train TinyGPT -> data/model/model.pt (+ model.json card), print samples
pdm run evaluate                # 5. model evaluation:       score frozen benchmarks data/bench/*.txt -> data/metrics/eval.json, data/metrics/samples.md
pdm run chat                    # 5b. inference:             same numbers as evaluate, then prompt -> continuation loop (real words only, --help for decoding knobs)
pdm run agent "some goal"       # 6. model serving:          simplistic agent, propose K continuations -> critique -> keep best / backtrack, with corpus lookup
```

`agent` is a generate -> critique -> select loop around the model: each round it samples `--candidates` continuations of the draft, scores them with the model's own perplexity plus a repetition penalty, keeps the best (or backtracks when the round is clearly worse than the last) and prints a Thought/Action/Observation trace quoting the corpus passage the kept text echoes. The LM only continues text; the loop does the deciding.

Or run the whole thing with `pdm run repro` (= `dvc repro`), which only re-runs the stages whose inputs changed.

Parameters (merge count, model size, training schedule, benchmark settings) live in `params.yaml`; CLI flags override them for one-off runs.
`train` auto-picks CUDA / MPS / CPU; override with `--device cpu|cuda|mps`, shorten with `--steps N`. Training keeps the best-validation weights and stops early after `--patience` evals without improvement.
The CPU-only torch wheel is pinned in `pyproject.toml`; remove the `[[tool.pdm.source]]` block on a CUDA machine.

## Reproducing the experiment

The PDFs are not in git (DVC tracks them, and no remote is configured), so fetch them yourself. The committed
`params.yaml`, `data/bench/` and `data/metrics/` describe a two-book corpus:

| file in `data/corpus/` | source |
|---|---|
| `think-and-grow-rich-napoleon-hill.pdf` | Napoleon Hill, *Think and Grow Rich* (1937), the free "Digital Download PDF eBook Edition" from www.think-and-grow-rich-ebook.com (253 pages) |
| `pg79832-images-3.pdf` | G. B. Stern, *A Deputy Was King* (1926), [Project Gutenberg #79832](https://www.gutenberg.org/ebooks/79832), the EPUB converted to PDF (339 pages) |

Put them under those names into `data/corpus/` and run `pdm run dvc repro`. With byte-identical files DVC
reports everything up to date (the hashes in `data/corpus.dvc` and `dvc.lock` match); with a different edition
or page layout the stages re-run, the page numbers in `ingest.exclude_pages` may be off, and the scores will
differ from the committed ones. The benchmark pages are the last chapters of *Think and Grow Rich* (pages
228-253 of that edition); the front and back matter of both books and the running header/footer the ebook
edition stamps on every page are excluded via `ingest.exclude_pages` / `ingest.strip_phrases`.

## Using your own corpus

With different PDFs the committed benchmark and metrics are stale, so start a new experiment:

1. Put your PDFs into `data/corpus/` and run `pdm run dvc add data/corpus`.
2. Edit `ingest.exclude_pages` in `params.yaml`: delete the old entries (ingest warns about keys that match no
   PDF) and list front/back matter plus the pages you want kept out of training. Put recurring page headers,
   footers and ads into `ingest.strip_phrases` so they are not learned (and not scored).
3. Build a benchmark from the held-out pages: write their text to a new `data/bench/holdout-v1.txt` (same
   normalisation as `data/feature/corpus.txt`: one page per line, lowercase words, phrases stripped), put a few
   prompts in `data/bench/prompts.txt`, and run `pdm run dvc add data/bench`.
4. `pdm run dvc repro` regenerates `dvc.lock`, `data/metrics/eval.json` and `data/metrics/samples.md` for your
   corpus; commit them together with the `.dvc` files.

## Versioning

Everything that can change is versioned along four axes by one mechanism, [DVC](https://dvc.org) on top of git.
A git commit is the version id: it pins the code, `params.yaml`, the data hashes in `data/*.dvc`, the artifact
hashes in `dvc.lock` and the resulting `data/metrics/eval.json`. `git checkout <rev> && pdm run dvc checkout` restores the exact
corpus, tokenizer and model of that commit.

| Axis | Version identity | Human alias |
|---|---|---|
| raw data (`data/corpus/`) | hash in `data/corpus.dvc` | git tag `data-vN` |
| benchmarks (`data/bench/`) | hash in `data/bench.dvc`; files are append-only, never edited | git tag `bench-vN` |
| tokenizer + model | hashes of every stage's deps and outs in `dvc.lock` | git tag `model-vN`, `artifacts: tinygpt` in `dvc.yaml` |
| pipeline software | `version` in `pyproject.toml`, lock file, each stage's source files as DVC deps | git tag `vX.Y.Z` |

Each checkpoint is also self-describing: `data/model/model.json` (and the `meta` key inside `model.pt`) records the
pipeline version, git commit, sha256 of the token/merge files it was trained on, all params, library versions,
steps run and best validation loss. `data/metrics/eval_meta.json` records the same for an evaluation run.

**Adding data and training a new model**

```sh
cp new-book.pdf data/corpus/
pdm run dvc add data/corpus      # new data version (updates data/corpus.dvc)
pdm run dvc repro                # re-runs ingest -> ... -> evaluate; dvc.lock + data/metrics/ update
pdm run dvc metrics diff HEAD    # new model vs the committed one, on the same frozen benchmarks
git diff data/metrics/samples.md      # qualitative diff on the fixed prompts
git commit -am "Train on corpus v2" && git tag data-v2 && git tag model-v2
```

**Comparing models and experiments**

```sh
pdm run dvc metrics diff model-v1 model-v2   # any two revisions (tags, commits, branches)
pdm run dvc params diff model-v1 model-v2    # what hyperparameters differed
git diff model-v1 model-v2 -- data/corpus.dvc params.yaml   # what data / config differed
pdm run dvc exp run -S train.lr=1e-3         # try a parameter without committing
pdm run dvc exp show                         # table of experiments: params + metrics side by side
pdm run dvc exp apply <name> / pdm run dvc exp remove   # keep or drop
```

Benchmarks are scored with `word_ppl` and `bpc` (bits per character), which do not depend on the tokenizer and so
stay comparable when a bigger corpus changes the BPE merges. `token_ppl` is reported too but only comparable
between models that share `merges.txt`. `val_loss` is the in-distribution number on the training split and moves
with the corpus. New benchmarks are new files (`data/bench/holdout-v2.txt` + `pdm run dvc add data/bench`); existing ones
are never edited, so old models keep their scores.

**Restoring a past model**

```sh
git checkout model-v1 && pdm run dvc checkout   # data/, model.pt and metrics as they were
pdm run chat --no-eval
pdm run dvc artifacts get . tinygpt --rev model-v1 -o /tmp/model-v1.pt   # or fetch just the weights
```

**Versioning the software.** Bump `version` in `pyproject.toml`, commit, tag `vX.Y.Z`. Because every stage lists
its source files as deps, a code change invalidates exactly the stages that read it on the next `pdm run dvc repro`,
and the pipeline version is stamped into each model card. Adding a stage means adding a block to `dvc.yaml`
and, if it has knobs, a section to `params.yaml` read via `llm.params.load_params`.

**Storage.** Artifacts live in the local `.dvc/cache`. To share them with another machine or CI, enable a remote
(see the comments in `.dvc/config`) and use `pdm run dvc push` / `pdm run dvc pull`.

What the repo does not show:

- LLM application work: no model APIs, RAG, embeddings, tool calling or prompt/evaluation harness.
- Scale: two books (about 350k training tokens), a 64-token context and a 4-layer, ~1M-parameter model
  (4 MB) that trains in minutes on a CPU. The text it produces is locally plausible, not coherent.
- Deployment and collaboration: no service or container, no DVC remote (artifacts stay in the local cache),
  no tests or CI.
- A real agent: `llm/agent.py` is a sample-and-rerank loop, as its own docstring says.
