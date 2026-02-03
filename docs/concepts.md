# Concepts

MLXSmith supports several training modes. Each addresses a different stage of the model improvement pipeline. This page explains what each mode does and when to use it.

## SFT (Supervised Fine-Tuning)

SFT trains a model to follow instructions by learning from prompt-response pairs. This is the foundation — most fine-tuning starts here.

**Data format:** JSONL with `{prompt, response}` fields.

**When to use:** You have examples of correct behavior and want the model to replicate them. This is the simplest and most common training mode.

**Method:** LoRA or QLoRA adapters are trained on top of the base model weights. The base model is not modified.

See [SFT guide](cli/sft.md).

## Preference Training (DPO/ORPO)

Preference training teaches a model which responses are better by learning from comparison pairs. Instead of showing the model a single correct answer, you show it a preferred answer and a rejected answer.

**Data format:** JSONL with `{prompt, chosen, rejected}` fields.

**When to use:** You have preference data — pairs where one response is better than another — and want to align the model's behavior accordingly. Typically run after SFT.

**Algorithms:**

| Algorithm | Description |
|-----------|-------------|
| DPO | Direct Preference Optimization — standard offline preference learning |
| ORPO | Odds Ratio Preference Optimization — combines SFT and preference in one loss |
| IPO | Identity Preference Optimization |
| CPO | Contrastive Preference Optimization — no reference model needed |
| SimPO | Length-normalized preference optimization — reference-free |
| Hinge | Margin-based preference loss |
| TDPO | Token-level DPO |

See [Preference training guide](cli/preference-training.md).

## KTO (Kahneman-Tversky Optimization)

KTO trains from binary feedback — each response is labeled as good or bad — rather than requiring paired comparisons. This is useful when you have thumbs-up/thumbs-down data but not side-by-side preference pairs.

**Data format:** JSONL with `{prompt, response, label}` where label is `true`/`false` or `1`/`0`.

**When to use:** You have binary quality labels on individual responses. KTO is grounded in behavioral economics (loss aversion, prospect theory) and can work when paired preference data is unavailable.

See [KTO guide](cli/kto.md).

## Reinforcement Fine-Tuning (GRPO)

GRPO (Generalized Reward Policy Optimization) trains a model using reward signals from a verifier. For each task, the model generates multiple candidate solutions (rollouts). A verifier scores each one, and the model is updated using policy gradients.

**Data format:** A YAML environment file defining tasks, paired with a verifier script.

**When to use:** You have a way to programmatically verify whether a model's output is correct — test cases, regex patterns, schema validation, or an LLM judge. This is the primary mode for improving reasoning and code generation.

**Loss variants:** `grpo`, `dr_grpo` (doubly robust), `dapo` (decaying advantage).

See [Reinforcement training guide](cli/reinforcement-training.md).

## Online DPO

Online DPO generates preference data on-the-fly. At each step, the model generates multiple candidate responses to a prompt. An LLM judge scores them, and the best and worst become the chosen/rejected pair for a DPO update.

**Data format:** JSONL with `{prompt}` fields.

**When to use:** You want preference-based training but don't have pre-collected preference data. The judge model creates the preference signal in real time.

See [Online DPO guide](cli/online-dpo.md).

## Self-Verification Training

Self-verification training is similar to online DPO but uses the model (or a separate verifier model) to assess its own outputs. The verification scores are used as reward signals for policy gradient updates.

**Data format:** JSONL with `{prompt}` fields.

**When to use:** You want the model to learn from its own self-assessment, using process-level or outcome-level verification.

See [Self-verify guide](cli/self-verify.md).

## Knowledge Distillation

Distillation transfers knowledge from a larger teacher model to a smaller student model.

**Modes:**
- **Offline:** The teacher generates responses, and the student is trained on them via SFT.
- **OPD (Online Preference Distillation):** The student generates candidates, the teacher scores them, and an importance-sampled loss updates the student.

**Data format:** JSONL with `{prompt}` fields.

**When to use:** You have a strong large model and want to create a smaller, faster model that approaches its quality.

See [Distillation guide](cli/distillation.md).

## Judge Training

Judge training fine-tunes a model to act as a scoring model (for use in online DPO, self-verification, or evaluation). It is standard SFT on judge-format data — prompt/response pairs where the responses are scoring judgments.

**Data format:** JSONL with `{prompt, response}` in judge format.

**When to use:** You want a dedicated judge model for automated evaluation or reward scoring.

See [Judge guide](cli/judge.md).

## RLM (Recursive Language Model)

RLM is a self-improving loop that combines task generation, training, and evaluation across multiple iterations:

1. **Generate** tasks (from the model itself or from a dataset)
2. **Collect** rollouts (multiple candidate solutions per task)
3. **Verify** rollouts (using configured verifiers)
4. **Train** on graded rollouts (GRPO policy gradient update)
5. **Evaluate** on a held-out benchmark
6. **Gate** — accept or reject the updated adapter based on benchmark score

Each iteration produces a new adapter checkpoint. The gating mechanism prevents regression by only promoting adapters that improve on the best historical score.

**When to use:** Long-running improvement campaigns where the model iteratively generates its own training signal and improves over many cycles.

See [RLM guide](cli/rlm.md).

## Related guides

- [CLI Reference](cli/README.md)
- [Data Tools](cli/data.md)
- [Synthetic Data](cli/synthetic-data.md)
- [Eval and Bench](cli/eval-and-bench.md)
- [Serving](cli/serving.md)
- [Configuration](cli/configuration.md)

## Pipeline

The `pipeline` command chains training stages in sequence: SFT, then preference training, then RFT, then RLM. Each stage feeds its output adapter to the next.

```bash
mlxsmith pipeline \
  --data-sft data/sft \
  --data-pref data/prefs \
  --env envs/coding.yaml \
  --verifier verifiers/regex.py
```

See [SFT guide](cli/sft.md#pipeline).

## Verifiers

Verifiers are the reward signal for reinforcement training, evaluation, and online preference methods. They determine whether a model's output is correct.

Built-in verifiers:
- **regex** — pattern matching on output text
- **pytest** — sandboxed Python test execution
- **jsonschema** — JSON structure validation
- **docker** — containerized test execution
- **llm_judge** — LLM-based scoring (ThinkPRM-style)
- **compose** — combine multiple verifiers (AND/OR/weighted)

See [Verifiers reference](VERIFIERS.md).

## Environment Plugins

Environments package tasks, verifiers, and metadata into reusable units for RL training. They can be versioned, packaged, published to a local registry, and shared.

See [Environments reference](ENVIRONMENTS.md).
