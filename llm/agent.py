"""Model serving: a simplistic agent driven by the tiny language model.

The model is a plain next-token LM trained on a small corpus: it cannot follow
instructions or call tools. So the agent loop supplies the "reasoning" and
uses the model as a proposer and a critic:

    Thought      how far the draft has come, what to try next
    Action       propose  - sample K continuations of the draft
                 critique - score each with the model's own perplexity plus a
                            penalty for repeating words already in the draft
    Observation  keep the best candidate and quote the corpus passage it echoes
                 most closely (word-overlap search over the tokenized corpus); if
                 every candidate scores --backtrack times worse than the last
                 accepted one, drop that chunk instead and try again

After --steps rounds the accumulated draft is printed as the final answer.

Usage:
    pdm run agent ["goal text"] [--steps N] [--candidates K] [--chunk T] [--seed S]
    plus the decoding knobs of chat.py (--temperature, --top-k, ...)
"""

import argparse
import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from llm.backend import Backend, select_backend
from llm.chat import add_decoding_args, build_logits_filter, load_model, trim_partial_word
from llm.log import setup_logging
from llm.model import LanguageModel
from llm.tokenize_text import TOKENS_PATH, tokenize
from llm.train_model import detokenize

log = logging.getLogger(__name__)

WINDOW = 20  # words per corpus passage
STRIDE = 10


class Corpus:
    """Word-overlap search over the training text, sliced into short passages."""

    def __init__(self) -> None:
        words = detokenize(TOKENS_PATH.read_text().split()).split()
        self.passages = [
            (set(words[i : i + WINDOW]), " ".join(words[i : i + WINDOW]))
            for i in range(0, max(1, len(words) - WINDOW + 1), STRIDE)
        ]

    def recall(self, text: str) -> tuple[float, str]:
        query = set(text.split())
        if not query:
            return 0.0, ""
        return max(
            (len(query & ws) / len(query | ws), passage) for ws, passage in self.passages
        )


@dataclass
class Candidate:
    ids: list[int]
    text: str
    ppl: float
    repetition: float

    @property
    def score(self) -> float:
        return self.ppl * (1 + self.repetition)


class Agent:
    def __init__(
        self,
        model: LanguageModel,
        itos: list[str],
        backend: Backend,
        logits_filter,
        temperature: float,
    ) -> None:
        self.model = model
        self.itos = itos
        self.backend = backend
        self.logits_filter = logits_filter
        self.temperature = temperature
        self.corpus = Corpus()

    def text(self, ids: list[int]) -> str:
        return detokenize([self.itos[i] for i in ids])

    # -- tools -------------------------------------------------------------

    def propose(self, draft: list[int], k: int, length: int) -> list[list[int]]:
        ctx = draft[-self.model.cfg.block_size :]
        idx = self.backend.to(torch.tensor([ctx] * k, dtype=torch.long))
        out = self.model.generate(
            idx, length, temperature=self.temperature, logits_filter=self.logits_filter
        )
        cands = []
        for row in out[:, len(ctx) :].tolist():
            toks = trim_partial_word([self.itos[i] for i in row])
            cands.append(row[: len(toks)])
        return cands

    def critique(self, draft: list[int], cand: list[int]) -> Candidate:
        """Mean cross-entropy of the candidate tokens given the draft, as perplexity."""
        seq = (draft + cand)[-self.model.cfg.block_size :]
        n = min(len(cand), len(seq) - 1)
        x = self.backend.to(torch.tensor([seq[:-1]], dtype=torch.long))
        with torch.no_grad(), self.backend.autocast():
            logits, _ = self.model(x)
        targets = self.backend.to(torch.tensor(seq[-n:], dtype=torch.long))
        loss = F.cross_entropy(logits[0, -n:].float(), targets).item()
        words = self.text(cand).split()
        seen = set(self.text(draft).split())
        repetition = sum(w in seen for w in words) / len(words)
        return Candidate(cand, " ".join(words), math.exp(loss), repetition)

    # -- loop --------------------------------------------------------------

    def run(self, goal: str, steps: int, k: int, chunk: int, backtrack: float) -> str:
        stoi = {tok: i for i, tok in enumerate(self.itos)}
        draft = [stoi[t] for t in tokenize(goal) if t in stoi]
        if not draft:
            _, seed = self.corpus.recall(goal)
            seed = seed or "the"
            log.info(f"Thought: no known tokens in goal, seeding from corpus: {seed!r}")
            draft = [stoi[t] for t in tokenize(seed) if t in stoi]
        accepted: list[Candidate] = []
        retry = False  # accept the round after a backtrack unconditionally

        for step in range(1, steps + 1):
            log.info(f"--- step {step}/{steps} ---")
            log.info(f"Thought: draft is {len(self.text(draft).split())} words; propose {k} continuations")
            log.info(f"Action: propose(chunk={chunk})")
            cands = [c for c in self.propose(draft, k, chunk) if c]
            if not cands:
                log.info("Observation: nothing usable proposed, retrying")
                continue
            log.info("Action: critique each candidate")
            scored = sorted((self.critique(draft, c) for c in cands), key=lambda c: c.score)
            for c in scored:
                log.info(f"  ppl {c.ppl:7.1f}  rep {c.repetition:.2f}  score {c.score:7.1f}  {c.text!r}")
            best = scored[0]

            if accepted and backtrack and not retry and best.score > backtrack * accepted[-1].score:
                dropped = accepted.pop()
                del draft[-len(dropped.ids) :]
                retry = True
                log.info(f"Observation: all candidates weak, backtrack: dropped {dropped.text!r}")
                continue

            retry = False
            accepted.append(best)
            draft += best.ids
            overlap, passage = self.corpus.recall(best.text)
            log.info(f"Observation: kept {best.text!r}")
            log.info(f"             echoes ({overlap:.2f}): {passage!r}")

        print("\nFinal answer:")
        return self.text(draft)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_decoding_args(ap)
    ap.add_argument("goal", nargs="?", default="the secret of success is")
    ap.add_argument("--steps", type=int, default=5, help="propose/critique rounds")
    ap.add_argument("--candidates", type=int, default=4, help="continuations per round")
    ap.add_argument("--chunk", type=int, default=20, help="tokens per continuation")
    ap.add_argument(
        "--backtrack", type=float, default=1.3, help="drop last chunk if best score exceeds this x its score, 0 = off"
    )
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    backend = select_backend(args.device)
    model, itos = load_model(backend)
    agent = Agent(model, itos, backend, build_logits_filter(itos, args), args.temperature)
    log.info(f"Goal: {args.goal!r}")
    print(agent.run(args.goal, args.steps, args.candidates, args.chunk, args.backtrack))


if __name__ == "__main__":
    main()
