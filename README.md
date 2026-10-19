# little-language-model

A little language model from scratch in plain PyTorch: BPE tokenizer -> transformer -> sampling.

Code lives in the `llm` package; the pipeline stages are run as modules via `pdm run`.

## Setup

Put one or more PDFs into `data/corpus/` (ignored by git). One coherent text works best for a model this small.

```sh
pdm install
```

## Pipeline

```sh
pdm run ingest                  # 1. data ingestion:         data/corpus/*.pdf -> data/feature/corpus.txt + vocab.json
pdm run extract                 # 2. feature extraction:     build BPE merge table -> data/feature/merges.txt
pdm run tokenize                # 3. feature transformation: tokenize corpus.txt -> data/feature/tokens.txt  (or: pdm run tokenize "some text", or pipe into pdm run tokenize -)
pdm run train                   # 4. model training:         train TinyGPT -> data/model/model.pt, print samples
pdm run chat                    # 5. evaluation + inference: loss/perplexity on held-out text, then prompt -> continuation loop (real words only, --help for decoding knobs)
pdm run agent "some goal"       # 6. model serving:          simplistic agent, propose K continuations -> critique -> keep best / backtrack, with corpus lookup
```

`agent` is a generate -> critique -> select loop around the model: each round it samples `--candidates` continuations of the draft, scores them with the model's own perplexity plus a repetition penalty, keeps the best (or backtracks when the round is clearly worse than the last) and prints a Thought/Action/Observation trace quoting the corpus passage the kept text echoes. The LM only continues text; the loop does the deciding.

`train` auto-picks CUDA / MPS / CPU; override with `--device cpu|cuda|mps`, shorten with `--steps N`. Training keeps the best-validation weights and stops early after `--patience` evals without improvement.
The CPU-only torch wheel is pinned in `pyproject.toml`; remove the `[[tool.pdm.source]]` block on a CUDA machine.

What the repo does not show:

- LLM application work: no model APIs, RAG, embeddings, tool calling or prompt/evaluation harness.
- Scale and deployment: one book, a 64-token context, a model of roughly 4 MB, no service or container around it.
- A real agent: `llm/agent.py` is a sample-and-rerank loop, as its own docstring says.
