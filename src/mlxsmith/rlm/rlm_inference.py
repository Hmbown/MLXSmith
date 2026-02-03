"""RLM Inference - Multi-turn REPL-based generation.

This module implements the canonical RLM inference paradigm where the model
interacts with a Python REPL environment, executing code, observing output,
making recursive sub-calls, and signaling completion via FINAL().

Based on Zhang et al. (arXiv:2512.24601).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .docker_repl import DockerRLMEnvironment, DockerREPLConfig
from .repl import (
    RLMEnvironment,
    REPLConfig,
    REPLResult,
    extract_code_blocks,
    format_repl_output,
)


@dataclass
class RLMInferenceConfig:
    """Configuration for RLM inference."""
    max_turns: int = 20
    max_new_tokens_per_turn: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: Optional[int] = None

    # REPL config
    max_output_chars: int = 4000
    max_exec_iterations: int = 50
    timeout_per_exec_s: float = 30.0

    # Sub-call config
    sub_call_max_tokens: int = 512
    sub_call_temperature: float = 0.3

    # Sandbox
    sandbox: str = "local"  # "local", "docker"
    docker_image: str = "python:3.11-slim"
    docker_memory_mb: int = 512
    docker_cpus: float = 1.0
    docker_pids: int = 128
    docker_network: bool = False


@dataclass
class RLMTurn:
    """A single turn in RLM inference."""
    turn_number: int
    model_output: str
    code_blocks: List[str]
    repl_results: List[REPLResult]
    timestamp: float = field(default_factory=time.time)


@dataclass
class RLMTrajectory:
    """Complete trajectory of an RLM inference run."""
    context: str
    turns: List[RLMTurn]
    final_answer: Optional[str]
    success: bool
    total_tokens: int = 0
    sub_calls: int = 0
    duration_ms: float = 0.0


RLM_SYSTEM_PROMPT = """You are an AI assistant with access to a Python REPL environment.

You have access to:
- `context`: A string variable containing the input/task
- `llm_query(prompt)`: Make a sub-call to a language model (handles ~500K chars)
- `llm_batch(prompts)`: Make parallel sub-calls (list of prompts -> list of results)
- `FINAL(answer)`: Call this when you have your final answer
- `FINAL_VAR(varname)`: Call this if your answer is stored in a variable

To execute code, wrap it in ```repl or ```python blocks.
IMPORTANT: Tool calls like llm_query(...) and FINAL(...) must be inside a code block to execute.

Strategy for long inputs:
1. First, probe the context to understand its structure (print first lines, count items, etc.)
2. Chunk the data strategically (by section, paragraph, or semantic unit)
3. Use llm_query() to process chunks recursively
4. Accumulate results in variables
5. Call FINAL() with your answer

Example:
```repl
# First, understand the context
print(f"Context length: {len(context)} chars")
print(context[:500])
```

Then based on what you see, decompose and process the input."""


def run_rlm_inference(
    llm,
    context: str,
    *,
    config: Optional[RLMInferenceConfig] = None,
    system_prompt: Optional[str] = None,
    seed: Optional[int] = None,
) -> RLMTrajectory:
    """Run RLM inference on a context.

    The model interacts with a Python REPL, executing code blocks,
    observing output, and making recursive sub-calls until it
    calls FINAL() or reaches max_turns.

    Args:
        llm: Language model backend with generate() method
        context: The input text to process
        config: Inference configuration
        system_prompt: Override the default system prompt
        seed: Random seed for generation

    Returns:
        RLMTrajectory with the complete interaction history and final answer
    """
    config = config or RLMInferenceConfig()
    system_prompt = system_prompt or RLM_SYSTEM_PROMPT
    if config.sandbox not in ("local", "docker"):
        raise ValueError(f"Unknown sandbox: {config.sandbox}. Choose from: local, docker")

    t0 = time.time()
    sub_call_count = 0

    # Create sub-call function that uses the same LLM
    def llm_query_fn(prompt: str) -> str:
        nonlocal sub_call_count
        sub_call_count += 1
        gen = llm.generate(
            prompt,
            max_new_tokens=config.sub_call_max_tokens,
            temperature=config.sub_call_temperature,
            top_p=config.top_p,
        )
        # Strip prompt if echoed
        text = gen.text
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()

    # Initialize REPL environment
    if config.sandbox == "docker":
        docker_config = DockerREPLConfig(
            image=config.docker_image,
            memory_mb=config.docker_memory_mb,
            cpus=config.docker_cpus,
            pids=config.docker_pids,
            max_iterations=config.max_exec_iterations,
            timeout_s=config.timeout_per_exec_s,
            network=config.docker_network,
            max_output_chars=config.max_output_chars,
        )
        env = DockerRLMEnvironment(context, llm_query_fn, docker_config)
    else:
        repl_config = REPLConfig(
            max_output_chars=config.max_output_chars,
            max_iterations=config.max_exec_iterations,
            timeout_per_exec_s=config.timeout_per_exec_s,
        )
        env = RLMEnvironment(context, llm_query_fn, repl_config)

    # Build initial prompt
    conversation = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Process the input stored in the `context` variable.\n\n"
                "Preview (truncated):\n\n"
                f"{context[:2000]}{'...(truncated)' if len(context) > 2000 else ''}"
            ),
        },
    ]

    turns: List[RLMTurn] = []
    total_tokens = 0

    for turn_num in range(config.max_turns):
        # Format conversation for generation
        prompt = _format_conversation(conversation)

        # Generate model response
        gen = llm.generate(
            prompt,
            max_new_tokens=config.max_new_tokens_per_turn,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            seed=seed,
        )

        model_output = gen.text
        if model_output.startswith(prompt):
            model_output = model_output[len(prompt):]
        model_output = model_output.strip()

        total_tokens += len(gen.token_ids) if gen.token_ids else 0

        # Extract and execute code blocks
        code_blocks = extract_code_blocks(model_output)
        repl_results: List[REPLResult] = []

        for _, code in code_blocks:
            result = env.execute(code)
            repl_results.append(result)

            # Check if FINAL was called
            if result.final_answer is not None:
                turns.append(RLMTurn(
                    turn_number=turn_num,
                    model_output=model_output,
                    code_blocks=[code for _, code in code_blocks],
                    repl_results=repl_results,
                ))

                return RLMTrajectory(
                    context=context,
                    turns=turns,
                    final_answer=result.final_answer,
                    success=True,
                    total_tokens=total_tokens,
                    sub_calls=sub_call_count,
                    duration_ms=(time.time() - t0) * 1000,
                )

        # Record turn
        turns.append(RLMTurn(
            turn_number=turn_num,
            model_output=model_output,
            code_blocks=[code for _, code in code_blocks],
            repl_results=repl_results,
        ))

        # Build REPL output for next turn
        if repl_results:
            repl_feedback = "\n\n".join(
                f"[Execution {i+1}]\n{format_repl_output(r)}"
                for i, r in enumerate(repl_results)
            )
            conversation.append({"role": "assistant", "content": model_output})
            conversation.append({"role": "user", "content": f"REPL Output:\n{repl_feedback}\n\nContinue processing or call FINAL() when done."})
        else:
            # No code blocks - prompt model to use the REPL
            conversation.append({"role": "assistant", "content": model_output})
            conversation.append({"role": "user", "content": "Please use the REPL to process the context. Write code in ```repl blocks."})

        # Check if we should stop (no progress)
        if turn_num > 3 and not code_blocks:
            # Model isn't using REPL, try to extract answer from output
            break

    # Max turns reached without FINAL()
    # Try to extract answer from last turn or namespace
    final_answer = None

    # Check if 'answer' variable exists in namespace
    if env.get_variable("answer") is not None:
        final_answer = str(env.get_variable("answer"))
    elif env.get_variable("result") is not None:
        final_answer = str(env.get_variable("result"))
    elif turns:
        # Use last model output as answer
        final_answer = turns[-1].model_output

    return RLMTrajectory(
        context=context,
        turns=turns,
        final_answer=final_answer,
        success=False,
        total_tokens=total_tokens,
        sub_calls=sub_call_count,
        duration_ms=(time.time() - t0) * 1000,
    )


def _format_conversation(messages: List[Dict[str, str]]) -> str:
    """Format conversation for generation."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def save_trajectory(trajectory: RLMTrajectory, path: Path) -> None:
    """Save trajectory to JSON file for training."""
    data = {
        "context": trajectory.context,
        "final_answer": trajectory.final_answer,
        "success": trajectory.success,
        "total_tokens": trajectory.total_tokens,
        "sub_calls": trajectory.sub_calls,
        "duration_ms": trajectory.duration_ms,
        "turns": [
            {
                "turn_number": t.turn_number,
                "model_output": t.model_output,
                "code_blocks": t.code_blocks,
                "repl_results": [
                    {
                        "stdout": r.stdout,
                        "stderr": r.stderr,
                        "success": r.success,
                        "exception": r.exception,
                        "final_answer": r.final_answer,
                    }
                    for r in t.repl_results
                ],
            }
            for t in trajectory.turns
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_trajectory(path: Path) -> RLMTrajectory:
    """Load trajectory from JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))

    turns = []
    for t_data in data.get("turns", []):
        repl_results = [
            REPLResult(
                stdout=r.get("stdout", ""),
                stderr=r.get("stderr", ""),
                success=r.get("success", False),
                exception=r.get("exception"),
                final_answer=r.get("final_answer"),
            )
            for r in t_data.get("repl_results", [])
        ]
        turns.append(RLMTurn(
            turn_number=t_data.get("turn_number", 0),
            model_output=t_data.get("model_output", ""),
            code_blocks=t_data.get("code_blocks", []),
            repl_results=repl_results,
        ))

    return RLMTrajectory(
        context=data.get("context", ""),
        turns=turns,
        final_answer=data.get("final_answer"),
        success=data.get("success", False),
        total_tokens=data.get("total_tokens", 0),
        sub_calls=data.get("sub_calls", 0),
        duration_ms=data.get("duration_ms", 0.0),
    )


def trajectory_to_training_pairs(
    trajectory: RLMTrajectory,
    include_failed: bool = False,
) -> List[Dict[str, str]]:
    """Convert trajectory to training pairs for SFT.

    Returns list of {prompt, response} dicts suitable for SFT training.
    Each turn becomes a training example.
    """
    if not trajectory.success and not include_failed:
        return []

    pairs = []
    conversation_so_far = [
        {"role": "system", "content": RLM_SYSTEM_PROMPT},
        {"role": "user", "content": f"Process this input:\n\n{trajectory.context[:2000]}"},
    ]

    for turn in trajectory.turns:
        # Prompt is conversation so far
        prompt = _format_conversation(conversation_so_far)

        # Response is model output for this turn
        response = turn.model_output

        pairs.append({"prompt": prompt, "response": response})

        # Update conversation for next turn
        conversation_so_far.append({"role": "assistant", "content": turn.model_output})
        if turn.repl_results:
            repl_feedback = "\n\n".join(
                f"[Execution {i+1}]\n{format_repl_output(r)}"
                for i, r in enumerate(turn.repl_results)
            )
            conversation_so_far.append({"role": "user", "content": f"REPL Output:\n{repl_feedback}"})

    return pairs
