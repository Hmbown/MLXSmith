from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


_DEFAULT_SUMMARY_PROMPT = (
    "You are a concise summarizer. Preserve key facts, constraints, and requirements.\n\n"
    "Text:\n{chunk}\n\nSummary:"
)


@dataclass
class RecursiveStats:
    depth: int
    chunks: int
    tokens_in: int
    tokens_out: int
    truncated: bool


def _chunk_ids(ids: list[int], *, chunk_tokens: int, overlap_tokens: int) -> Iterable[list[int]]:
    if chunk_tokens <= 0:
        return []
    overlap = max(0, min(overlap_tokens, max(0, chunk_tokens - 1)))
    start = 0
    total = len(ids)
    while start < total:
        end = min(total, start + chunk_tokens)
        yield ids[start:end]
        if end >= total:
            break
        start = max(0, end - overlap)


def _strip_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    return text.strip()


def recursive_compact(
    llm,
    prompt: str,
    *,
    max_seq_len: int,
    chunk_tokens: int = 1024,
    overlap_tokens: int = 128,
    keep_last_tokens: int = 384,
    summary_tokens: int = 256,
    max_depth: int = 3,
    temperature: float = 0.2,
    seed: Optional[int] = None,
    summary_prompt: Optional[str] = None,
) -> Tuple[str, RecursiveStats]:
    """Recursively compress long prompts into a shorter representation."""
    if max_seq_len <= 0:
        return prompt, RecursiveStats(depth=0, chunks=0, tokens_in=0, tokens_out=0, truncated=False)

    ids = list(llm.encode(prompt))
    tokens_in = len(ids)
    if tokens_in <= max_seq_len:
        return prompt, RecursiveStats(depth=0, chunks=0, tokens_in=tokens_in, tokens_out=tokens_in, truncated=False)

    if max_depth <= 0:
        tail_ids = ids[-max_seq_len:]
        tail_text = llm.decode(tail_ids)
        return tail_text, RecursiveStats(depth=0, chunks=0, tokens_in=tokens_in, tokens_out=len(tail_ids), truncated=True)

    keep_last = max(0, min(int(keep_last_tokens), tokens_in))
    tail_ids = ids[-keep_last:] if keep_last > 0 else []
    context_ids = ids[:-keep_last] if keep_last > 0 else ids

    if not context_ids:
        tail_ids = ids[-max_seq_len:]
        tail_text = llm.decode(tail_ids)
        return tail_text, RecursiveStats(depth=0, chunks=0, tokens_in=tokens_in, tokens_out=len(tail_ids), truncated=True)

    summaries = []
    prompt_tpl = summary_prompt or _DEFAULT_SUMMARY_PROMPT
    chunks = list(_chunk_ids(context_ids, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens))
    for idx, chunk in enumerate(chunks):
        chunk_text = llm.decode(chunk)
        prompt_text = prompt_tpl.format(chunk=chunk_text, index=idx + 1, total=len(chunks))
        gen = llm.generate(
            prompt_text,
            max_new_tokens=summary_tokens,
            temperature=temperature,
            seed=None if seed is None else int(seed) + idx,
        )
        summary = _strip_prefix(gen.text, prompt_text)
        if summary:
            summaries.append(summary)

    if not summaries:
        tail_ids = ids[-max_seq_len:]
        tail_text = llm.decode(tail_ids)
        return tail_text, RecursiveStats(depth=0, chunks=0, tokens_in=tokens_in, tokens_out=len(tail_ids), truncated=True)

    summary_block = "\n".join(f"- {s}" for s in summaries)
    tail_text = llm.decode(tail_ids)
    compacted = f"Context summary:\n{summary_block}\n\nTask:\n{tail_text}".strip()

    compacted_ids = list(llm.encode(compacted))
    if len(compacted_ids) >= tokens_in and max_depth <= 1:
        tail_ids = ids[-max_seq_len:]
        tail_text = llm.decode(tail_ids)
        return tail_text, RecursiveStats(depth=1, chunks=len(chunks), tokens_in=tokens_in, tokens_out=len(tail_ids), truncated=True)

    next_prompt, stats = recursive_compact(
        llm,
        compacted,
        max_seq_len=max_seq_len,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        keep_last_tokens=keep_last_tokens,
        summary_tokens=summary_tokens,
        max_depth=max_depth - 1,
        temperature=temperature,
        seed=seed,
        summary_prompt=summary_prompt,
    )

    return next_prompt, RecursiveStats(
        depth=stats.depth + 1,
        chunks=len(chunks),
        tokens_in=tokens_in,
        tokens_out=len(llm.encode(next_prompt)),
        truncated=stats.truncated,
    )
