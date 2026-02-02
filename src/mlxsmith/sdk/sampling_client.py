"""SamplingClient SDK for MLXSmith.

Client for text sampling with logprobs support, including top-k logprobs
per token. Designed for distillation workflows where teacher model logprobs
are needed for student training.

Example:
    >>> from mlxsmith.sdk import SamplingClient
    >>> 
    >>> # Local backend
    >>> client = SamplingClient(backend=loaded_model.backend)
    >>> 
    >>> # Single sample with logprobs
    >>> result = client.sample("What is 2+2?", logprobs_k=5)
    >>> print(result.text)
    >>> for token_lp in result.top_k_logprobs:
    ...     print(token_lp)  # {"token": logprob, ...}
    >>> 
    >>> # Batch sampling
    >>> results = client.sample_batch(
    ...     ["Q1", "Q2", "Q3"],
    ...     max_tokens=100,
    ...     logprobs_k=5
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .future import APIFuture, SdkFuturePool


@dataclass
class SampleResult:
    """Result from a sampling operation.
    
    Attributes:
        text: Generated text (completion only, prompt excluded)
        token_ids: Full sequence of token IDs (prompt + completion)
        prompt_len: Length of prompt in tokens
        logprobs: Log probability of each generated token
        top_k_logprobs: Top-k logprobs per token (list of {token: logprob} dicts)
        finish_reason: Why generation stopped ("stop", "length", etc.)
        metrics: Additional metrics (perplexity, etc.)
    """
    text: str
    token_ids: List[int]
    prompt_len: int
    logprobs: List[float] = field(default_factory=list)
    top_k_logprobs: Optional[List[Dict[str, float]]] = None
    prompt_logprobs: Optional[List[float]] = None
    prompt_top_k_logprobs: Optional[List[Dict[str, float]]] = None
    finish_reason: str = "stop"
    metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def completion_token_ids(self) -> List[int]:
        """Get token IDs for the completion only."""
        return self.token_ids[self.prompt_len:]
    
    @property
    def avg_logprob(self) -> float:
        """Average log probability of generated tokens."""
        if not self.logprobs:
            return 0.0
        return sum(self.logprobs) / len(self.logprobs)
    
    @property
    def perplexity(self) -> float:
        """Perplexity of the completion."""
        import math
        avg_lp = self.avg_logprob
        return math.exp(-avg_lp) if avg_lp != 0 else 1.0


@dataclass
class SampleBatchResult:
    """Result from a batch sampling operation."""
    results: List[SampleResult]
    total_tokens: int = 0
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __getitem__(self, idx: int) -> SampleResult:
        return self.results[idx]
    
    @property
    def texts(self) -> List[str]:
        """Get all completion texts."""
        return [r.text for r in self.results]
    
    @property
    def all_token_ids(self) -> List[List[int]]:
        """Get all token ID sequences."""
        return [r.token_ids for r in self.results]
    
    @property
    def all_top_k_logprobs(self) -> List[List[Dict[str, float]]]:
        """Get all top-k logprobs."""
        return [r.top_k_logprobs for r in self.results if r.top_k_logprobs is not None]


class SamplingClient:
    """Client for sampling with logprobs support.
    
    Can be used with a local backend or an API endpoint for distributed
    sampling (e.g., teacher model running on a different machine).
    
    Example:
        >>> # Local sampling
        >>> client = SamplingClient(backend=backend)
        >>> result = client.sample("Hello", max_tokens=10, logprobs_k=5)
        >>> 
        >>> # API-based sampling (for remote teacher model)
        >>> api_client = SamplingClient(
        ...     api_endpoint="http://teacher:8000",
        ...     api_key="secret"
        ... )
        >>> result = api_client.sample("Hello", max_tokens=10, logprobs_k=5)
    """
    
    def __init__(
        self,
        backend: Any = None,
        pool: Optional[SdkFuturePool] = None,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """Initialize SamplingClient.
        
        Args:
            backend: Local LLM backend instance
            pool: Optional SdkFuturePool for async execution
            api_endpoint: Optional API endpoint for remote sampling
            api_key: API key for authentication
            timeout: Default timeout for operations
            
        Raises:
            ValueError: If neither backend nor api_endpoint is provided
        """
        if backend is None and api_endpoint is None:
            raise ValueError("Either backend or api_endpoint must be provided")
        
        self.backend = backend
        self.pool = pool or SdkFuturePool(max_workers=4)
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.timeout = timeout
        self._session = None
    
    def sample(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        logprobs_k: int = 0,
        include_prompt_logprobs: bool = False,
        prompt_logprobs_k: int = 0,
    ) -> SampleResult:
        """Sample a single completion.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            seed: Random seed for reproducibility
            stop: Stop sequences
            logprobs_k: Number of top logprobs to return per token (0 = none)
            
        Returns:
            SampleResult with text, token IDs, and logprobs
            
        Example:
            >>> result = client.sample(
            ...     "What is the capital of France?",
            ...     max_tokens=50,
            ...     temperature=0.7,
            ...     logprobs_k=5
            ... )
            >>> print(result.text)
            >>> print(f"Perplexity: {result.perplexity}")
        """
        if self.api_endpoint:
            return self._sample_api(
                prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                stop,
                logprobs_k,
                include_prompt_logprobs,
                prompt_logprobs_k,
            )
        else:
            return self._sample_local(
                prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                stop,
                logprobs_k,
                include_prompt_logprobs,
                prompt_logprobs_k,
            )
    
    def sample_async(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        logprobs_k: int = 0,
        include_prompt_logprobs: bool = False,
        prompt_logprobs_k: int = 0,
    ) -> APIFuture[SampleResult]:
        """Async version of sample().
        
        Returns:
            APIFuture that resolves to SampleResult
            
        Example:
            >>> future = client.sample_async("Hello", max_tokens=10)
            >>> future.then(lambda r: print(r.text))
            >>> result = future.result()
        """
        def _do_sample():
            return self.sample(
                prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                stop,
                logprobs_k,
                include_prompt_logprobs,
                prompt_logprobs_k,
            )
        
        return self.pool.submit(_do_sample)
    
    def sample_batch(
        self,
        prompts: Sequence[str],
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        logprobs_k: int = 0,
        include_prompt_logprobs: bool = False,
        prompt_logprobs_k: int = 0,
        max_workers: Optional[int] = None,
    ) -> SampleBatchResult:
        """Sample completions for multiple prompts.
        
        Args:
            prompts: List of prompt strings
            max_tokens: Maximum tokens per completion
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            seed: Random seed (different for each prompt if provided)
            stop: Stop sequences
            logprobs_k: Number of top logprobs per token
            max_workers: Number of parallel workers (None = use pool default)
            
        Returns:
            SampleBatchResult with all results
            
        Example:
            >>> prompts = ["Q: 2+2=", "Q: 3*4=", "Q: 10/2="]
            >>> results = client.sample_batch(prompts, max_tokens=10, logprobs_k=5)
            >>> for prompt, result in zip(prompts, results):
            ...     print(f"{prompt} {result.text}")
        """
        # Use different seeds for each prompt if seed provided
        seeds = [seed + i if seed is not None else None for i in range(len(prompts))]
        
        # Submit all samples to thread pool
        futures = [
            self.sample_async(
                prompt=p,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=s,
                stop=stop,
                logprobs_k=logprobs_k,
                include_prompt_logprobs=include_prompt_logprobs,
                prompt_logprobs_k=prompt_logprobs_k,
            )
            for p, s in zip(prompts, seeds)
        ]
        
        # Collect results
        results = [f.result(timeout=self.timeout) for f in futures]
        total_tokens = sum(len(r.token_ids) for r in results)
        
        return SampleBatchResult(results=results, total_tokens=total_tokens)
    
    def sample_batch_async(
        self,
        prompts: Sequence[str],
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        logprobs_k: int = 0,
        include_prompt_logprobs: bool = False,
        prompt_logprobs_k: int = 0,
    ) -> APIFuture[SampleBatchResult]:
        """Async version of sample_batch().
        
        Returns:
            APIFuture that resolves to SampleBatchResult
        """
        def _do_batch():
            return self.sample_batch(
                prompts,
                max_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                stop,
                logprobs_k,
                include_prompt_logprobs,
                prompt_logprobs_k,
            )
        
        return self.pool.submit(_do_batch)
    
    def get_logprobs_for_texts(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        top_k: int = 0,
    ) -> List[List[Dict[str, float]]]:
        """Get logprobs for given prompt-completion pairs.
        
        This is useful for distillation where you have student-generated
        texts and want teacher logprobs for them.
        
        Args:
            prompts: List of prompts
            completions: List of completions (parallel to prompts)
            top_k: Number of top logprobs per token
            
        Returns:
            List of top-k logprobs for each completion
            
        Example:
            >>> prompts = ["Q: What is 2+2?"]
            >>> completions = ["The answer is 4."]
            >>> logprobs = client.get_logprobs_for_texts(prompts, completions, top_k=5)
            >>> # logprobs[0] is list of {token: logprob} for each token
        """
        results = []
        for prompt, completion in zip(prompts, completions):
            # Encode and get logprobs
            prompt_ids = self.backend.encode(prompt)
            full_ids = self.backend.encode(prompt + completion)
            
            # Try to get logprobs from the backend
            if hasattr(self.backend, 'generate_with_logprobs'):
                gen = self.backend.generate_with_logprobs(
                    prompt,
                    max_new_tokens=len(full_ids) - len(prompt_ids),
                    logprobs=top_k,
                )
                results.append(gen.top_k_logprobs or [])
            else:
                results.append([])
        
        return results

    def get_prompt_token_logprobs(
        self,
        prompts: Sequence[str],
    ) -> List[List[float]]:
        """Get per-token logprobs for prompt tokens.

        Returns a list of logprob lists (one per prompt). The first token is
        omitted because it has no previous context.
        """
        results: List[List[float]] = []
        for prompt in prompts:
            if not hasattr(self.backend, "token_logprobs"):
                results.append([])
                continue
            prompt_ids = self.backend.encode(prompt)
            logprobs, _ = self.backend.token_logprobs(
                prompt_ids,
                prompt_len=len(prompt_ids),
                top_k=0,
                include_prompt=True,
            )
            results.append(logprobs)
        return results

    def get_prompt_top_k_logprobs(
        self,
        prompts: Sequence[str],
        top_k: int = 5,
    ) -> List[List[Dict[str, float]]]:
        """Get top-k logprobs for each prompt token.

        Returns a list of per-token top-k dicts for each prompt.
        """
        results: List[List[Dict[str, float]]] = []
        for prompt in prompts:
            if not hasattr(self.backend, "token_logprobs"):
                results.append([])
                continue
            prompt_ids = self.backend.encode(prompt)
            _, top_k_logprobs = self.backend.token_logprobs(
                prompt_ids,
                prompt_len=len(prompt_ids),
                top_k=top_k,
                include_prompt=True,
            )
            results.append(top_k_logprobs or [])
        return results
    
    def compute_sequence_logprobs(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
    ) -> List[float]:
        """Compute total logprob for each prompt-completion pair.
        
        Args:
            prompts: List of prompts
            completions: List of completions
            
        Returns:
            List of total logprobs
        """
        results = []
        for prompt, completion in zip(prompts, completions):
            prompt_ids = self.backend.encode(prompt)
            full_ids = self.backend.encode(prompt + completion)
            
            if hasattr(self.backend, 'sequence_logprob'):
                logprob = self.backend.sequence_logprob(full_ids, prompt_len=len(prompt_ids))
                results.append(float(logprob))
            else:
                results.append(0.0)
        
        return results
    
    # ========================================================================
    # Internal methods
    # ========================================================================
    
    def _sample_local(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        seed: Optional[int],
        stop: Optional[Sequence[str]],
        logprobs_k: int,
        include_prompt_logprobs: bool,
        prompt_logprobs_k: int,
    ) -> SampleResult:
        """Sample using local backend."""
        if logprobs_k > 0:
            gen = self.backend.generate_with_logprobs(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k_sampling=top_k,
                seed=seed,
                logprobs=logprobs_k,
            )
        else:
            gen = self.backend.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
            )
        
        # Extract completion text (remove prompt prefix if present)
        completion_text = gen.text
        if completion_text.startswith(prompt):
            completion_text = completion_text[len(prompt):]
        
        # Apply stop sequences
        finish_reason = "stop"
        if stop:
            for stop_seq in stop:
                if stop_seq in completion_text:
                    completion_text = completion_text[:completion_text.index(stop_seq)]
                    finish_reason = "stop"
                    break
        
        # Check if we hit length limit
        if len(gen.token_ids) - gen.prompt_len >= max_tokens:
            finish_reason = "length"
        
        prompt_logprobs = None
        prompt_top_k_logprobs = None
        if (include_prompt_logprobs or prompt_logprobs_k > 0) and hasattr(self.backend, "token_logprobs"):
            prompt_ids = self.backend.encode(prompt)
            try:
                plogps, ptopk = self.backend.token_logprobs(
                    prompt_ids,
                    prompt_len=len(prompt_ids),
                    top_k=prompt_logprobs_k if prompt_logprobs_k > 0 else 0,
                    include_prompt=True,
                )
                if include_prompt_logprobs:
                    prompt_logprobs = plogps
                if prompt_logprobs_k > 0:
                    prompt_top_k_logprobs = ptopk or []
            except Exception:
                prompt_logprobs = None
                prompt_top_k_logprobs = None

        return SampleResult(
            text=completion_text,
            token_ids=gen.token_ids,
            prompt_len=gen.prompt_len,
            logprobs=gen.logprobs or [],
            top_k_logprobs=gen.top_k_logprobs,
            prompt_logprobs=prompt_logprobs,
            prompt_top_k_logprobs=prompt_top_k_logprobs,
            finish_reason=finish_reason,
        )
    
    def _sample_api(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        seed: Optional[int],
        stop: Optional[Sequence[str]],
        logprobs_k: int,
        include_prompt_logprobs: bool,
        prompt_logprobs_k: int,
    ) -> SampleResult:
        """Sample using remote API endpoint."""
        import urllib.request
        import urllib.error
        import json
        
        url = f"{self.api_endpoint}/internal/rollout"
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "include_tokens": True,
            "include_logprobs": True,
            "include_top_k_logprobs": logprobs_k if logprobs_k > 0 else None,
            "include_prompt_logprobs": bool(include_prompt_logprobs),
            "include_prompt_top_k_logprobs": prompt_logprobs_k if prompt_logprobs_k > 0 else None,
            "include_text": True,
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode())
                
                return SampleResult(
                    text=data.get("completion", ""),
                    token_ids=data.get("token_ids", []),
                    prompt_len=data.get("prompt_len", 0),
                    logprobs=data.get("logprobs", []),
                    top_k_logprobs=data.get("top_k_logprobs"),
                    prompt_logprobs=data.get("prompt_logprobs"),
                    prompt_top_k_logprobs=data.get("prompt_top_k_logprobs"),
                    finish_reason="stop",
                )
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"API error: {e.code} - {e.read().decode()}")
        except Exception as e:
            raise RuntimeError(f"Failed to sample from API: {e}")
    
    def shutdown(self) -> None:
        """Shutdown the client and its thread pool."""
        self.pool.shutdown(wait=True)


class DistillationSampler:
    """Helper for knowledge distillation workflows.
    
    Combines a student training client with a teacher sampling client
    to provide teacher logprobs for student-generated samples.
    
    Example:
        >>> sampler = DistillationSampler(
        ...     teacher_client=teacher_sampling_client,
        ...     student_client=student_training_client,
        ... )
        >>> 
        >>> # Generate samples from student and get teacher logprobs
        >>> prompts = ["Q: What is 2+2?"]
        >>> samples, teacher_logprobs = sampler.sample_with_teacher_logprobs(
        ...     prompts,
        ...     max_tokens=50,
        ...     logprobs_k=5
        ... )
    """
    
    def __init__(
        self,
        teacher_client: SamplingClient,
        student_client: Optional['SamplingClient'] = None,  # type: ignore
    ):
        """Initialize DistillationSampler.
        
        Args:
            teacher_client: SamplingClient for the teacher model
            student_client: Optional SamplingClient for the student model
                           (if None, uses teacher for sampling too)
        """
        self.teacher = teacher_client
        self.student = student_client or teacher_client
    
    def sample_with_teacher_logprobs(
        self,
        prompts: Sequence[str],
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 1.0,
        logprobs_k: int = 5,
    ) -> tuple[SampleBatchResult, List[List[Dict[str, float]]]]:
        """Sample from student and get teacher logprobs for those samples.
        
        This is the key operation for distillation: the student generates
        samples, then the teacher evaluates the log probability of each
        token in those samples.
        
        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens per sample
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            logprobs_k: Number of top logprobs to get from teacher
            
        Returns:
            Tuple of (student samples, teacher top-k logprobs)
            
        Example:
            >>> prompts = ["Question: What is Python?"]
            >>> samples, teacher_lps = sampler.sample_with_teacher_logprobs(
            ...     prompts, max_tokens=100, logprobs_k=5
            ... )
            >>> # Use samples and teacher_lps for distillation training
        """
        # Sample from student
        student_samples = self.student.sample_batch(
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs_k=0,  # Don't need student logprobs
        )
        
        # Get teacher logprobs for the student-generated completions
        completions = student_samples.texts
        teacher_logprobs = self.teacher.get_logprobs_for_texts(
            prompts, completions, top_k=logprobs_k
        )
        
        return student_samples, teacher_logprobs
    
    def sample_with_teacher_logprobs_async(
        self,
        prompts: Sequence[str],
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 1.0,
        logprobs_k: int = 5,
    ) -> APIFuture[tuple[SampleBatchResult, List[List[Dict[str, float]]]]]:
        """Async version of sample_with_teacher_logprobs()."""
        def _do_sample():
            return self.sample_with_teacher_logprobs(
                prompts, max_tokens, temperature, top_p, logprobs_k
            )
        
        return self.student.pool.submit(_do_sample)
