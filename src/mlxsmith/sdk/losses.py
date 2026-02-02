from __future__ import annotations

from typing import Any, Callable, Optional, Sequence


LOSS_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_loss(name: str):
    def decorator(fn: Callable[..., Any]):
        LOSS_REGISTRY[name] = fn
        return fn

    return decorator


def get_loss(name: str) -> Callable[..., Any]:
    if name not in LOSS_REGISTRY:
        raise KeyError(f"Unknown loss: {name}")
    return LOSS_REGISTRY[name]


def _require_mx(backend) -> Any:
    mx = getattr(backend, "mx", None)
    if mx is None:
        raise RuntimeError("Backend does not expose mx; cannot compute preference losses.")
    return mx


def _to_mx_scalar(mx: Any, value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        return mx.array(value)
    except Exception:
        return value


def preference_diff(
    backend,
    chosen_ids: Sequence[int],
    rejected_ids: Sequence[int],
    *,
    prompt_len_chosen: int,
    prompt_len_rejected: int,
    reference_backend: Optional[Any] = None,
) -> Any:
    logp_c = backend.sequence_logprob(chosen_ids, prompt_len=prompt_len_chosen)
    logp_r = backend.sequence_logprob(rejected_ids, prompt_len=prompt_len_rejected)
    ref_diff = 0.0
    if reference_backend is not None:
        ref_logp_c = reference_backend.sequence_logprob(chosen_ids, prompt_len=prompt_len_chosen)
        ref_logp_r = reference_backend.sequence_logprob(rejected_ids, prompt_len=prompt_len_rejected)
        ref_diff = ref_logp_c - ref_logp_r
    return (logp_c - logp_r) - ref_diff


@register_loss("dpo")
def dpo_loss(
    backend,
    chosen_ids: Sequence[int],
    rejected_ids: Sequence[int],
    *,
    prompt_len_chosen: int,
    prompt_len_rejected: int,
    beta: float = 0.1,
    reference_backend: Optional[Any] = None,
    kl_coeff: float = 0.0,
) -> Any:
    mx = _require_mx(backend)
    diff = preference_diff(
        backend,
        chosen_ids,
        rejected_ids,
        prompt_len_chosen=prompt_len_chosen,
        prompt_len_rejected=prompt_len_rejected,
        reference_backend=reference_backend,
    )
    scaled = _to_mx_scalar(mx, beta) * diff
    loss = mx.log1p(mx.exp(-scaled))

    if reference_backend is not None and kl_coeff > 0:
        logp_c = backend.sequence_logprob(chosen_ids, prompt_len=prompt_len_chosen)
        ref_logp_c = reference_backend.sequence_logprob(chosen_ids, prompt_len=prompt_len_chosen)
        loss = loss + _to_mx_scalar(mx, kl_coeff) * (logp_c - ref_logp_c)

    return loss


@register_loss("orpo")
def orpo_loss(
    backend,
    chosen_ids: Sequence[int],
    rejected_ids: Sequence[int],
    *,
    prompt_len_chosen: int,
    prompt_len_rejected: int,
    beta: float = 0.1,
    reference_backend: Optional[Any] = None,
    kl_coeff: float = 0.0,
    train_on_prompt: bool = False,
) -> Any:
    mx = _require_mx(backend)
    diff = preference_diff(
        backend,
        chosen_ids,
        rejected_ids,
        prompt_len_chosen=prompt_len_chosen,
        prompt_len_rejected=prompt_len_rejected,
        reference_backend=reference_backend,
    )
    nll = backend.sft_loss(chosen_ids, train_on_prompt=train_on_prompt, prompt_len=prompt_len_chosen)
    or_term = -_to_mx_scalar(mx, beta) * mx.log(mx.sigmoid(diff))
    loss = _to_mx_scalar(mx, nll) + or_term

    if reference_backend is not None and kl_coeff > 0:
        logp_c = backend.sequence_logprob(chosen_ids, prompt_len=prompt_len_chosen)
        ref_logp_c = reference_backend.sequence_logprob(chosen_ids, prompt_len=prompt_len_chosen)
        loss = loss + _to_mx_scalar(mx, kl_coeff) * (logp_c - ref_logp_c)

    return loss


@register_loss("preference")
def preference_loss(
    backend,
    chosen_ids: Sequence[int],
    rejected_ids: Sequence[int],
    *,
    prompt_len_chosen: int,
    prompt_len_rejected: int,
    algo: str = "dpo",
    beta: float = 0.1,
    reference_backend: Optional[Any] = None,
    kl_coeff: float = 0.0,
    train_on_prompt: bool = False,
) -> Any:
    if algo.lower() == "orpo":
        return orpo_loss(
            backend,
            chosen_ids,
            rejected_ids,
            prompt_len_chosen=prompt_len_chosen,
            prompt_len_rejected=prompt_len_rejected,
            beta=beta,
            reference_backend=reference_backend,
            kl_coeff=kl_coeff,
            train_on_prompt=train_on_prompt,
        )
    return dpo_loss(
        backend,
        chosen_ids,
        rejected_ids,
        prompt_len_chosen=prompt_len_chosen,
        prompt_len_rejected=prompt_len_rejected,
        beta=beta,
        reference_backend=reference_backend,
        kl_coeff=kl_coeff,
    )


@register_loss("cross_entropy")
def cross_entropy_loss(
    backend,
    token_ids: Sequence[int],
    *,
    prompt_len: int,
    train_on_prompt: bool = False,
) -> Any:
    return backend.sft_loss(token_ids, train_on_prompt=train_on_prompt, prompt_len=prompt_len)


def _mx_clip(mx: Any, x: Any, lo: float, hi: float) -> Any:
    if hasattr(mx, "minimum") and hasattr(mx, "maximum"):
        return mx.minimum(mx.maximum(x, _to_mx_scalar(mx, lo)), _to_mx_scalar(mx, hi))
    return min(max(x, lo), hi)


@register_loss("importance_sampling")
def importance_sampling_loss(
    backend,
    token_ids: Sequence[int],
    *,
    prompt_len: int,
    advantage: float,
    behavior_logprob: Optional[Any] = None,
) -> Any:
    mx = _require_mx(backend)
    logp = backend.sequence_logprob(token_ids, prompt_len=prompt_len)
    if behavior_logprob is None:
        behavior_logprob = logp
    ratio = mx.exp(logp - behavior_logprob)
    return -ratio * _to_mx_scalar(mx, advantage)


@register_loss("ppo")
def ppo_loss(
    backend,
    token_ids: Sequence[int],
    *,
    prompt_len: int,
    advantage: float,
    behavior_logprob: Any,
    clip: float = 0.2,
) -> Any:
    mx = _require_mx(backend)
    logp = backend.sequence_logprob(token_ids, prompt_len=prompt_len)
    ratio = mx.exp(logp - behavior_logprob)
    adv = _to_mx_scalar(mx, advantage)
    clipped = _mx_clip(mx, ratio, 1.0 - clip, 1.0 + clip)
    return -mx.minimum(ratio * adv, clipped * adv)


@register_loss("cispo")
def cispo_loss(
    backend,
    token_ids: Sequence[int],
    *,
    prompt_len: int,
    advantage: float,
    behavior_logprob: Any,
    clip: float = 0.2,
    penalty: float = 0.1,
) -> Any:
    mx = _require_mx(backend)
    logp = backend.sequence_logprob(token_ids, prompt_len=prompt_len)
    ratio = mx.exp(logp - behavior_logprob)
    adv = _to_mx_scalar(mx, advantage)
    clipped = _mx_clip(mx, ratio, 1.0 - clip, 1.0 + clip)
    penalty_term = _to_mx_scalar(mx, penalty) * (ratio - clipped) ** 2
    return -(clipped * adv) + penalty_term


@register_loss("dro")
def dro_loss(
    backend,
    token_ids: Sequence[int],
    *,
    prompt_len: int,
    advantage: float,
    temperature: float = 1.0,
) -> Any:
    mx = _require_mx(backend)
    logp = backend.sequence_logprob(token_ids, prompt_len=prompt_len)
    weight = mx.exp(_to_mx_scalar(mx, advantage) / _to_mx_scalar(mx, temperature))
    return -weight * logp
