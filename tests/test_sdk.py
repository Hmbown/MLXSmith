"""Tests for MLXSmith SDK.

Tests APIFuture, TrainingClient, SamplingClient, and logprobs functionality.
"""

import time

import pytest

from mlxsmith.config import ProjectConfig
from mlxsmith.infer import ChatMessage, run_chat
from mlxsmith.llm.backend import DecodingConfig
from mlxsmith.sdk import (
    # Core SDK functions
    create_optimizer,
    get_loss,
    importance_sampling_loss,
    SdkFuturePool,
    load_model,
    logprobs,
    optim_step,
    preference_forward_backward,
    sample,
    sft_forward_backward,
    APIFuture,
    APIFutureState,
    completed_future,
    failed_future,
    cancelled_future,
    
    # Training Client
    TrainingClient,
    TrainingBatch,
    ForwardBackwardResult,
    CheckpointResult,
    WeightsResult,
    DistillationTrainingClient,
    
    # Sampling Client
    SamplingClient,
    SampleResult,
    SampleBatchResult,
    DistillationSampler,
)


# =============================================================================
# APIFuture Tests
# =============================================================================

class TestAPIFuture:
    """Test APIFuture functionality."""
    
    def test_future_initial_state(self):
        """Test that a new future starts in pending state."""
        future = APIFuture()
        assert future.state == APIFutureState.PENDING
        assert not future.done
        assert future.progress == 0.0
    
    def test_completed_future(self):
        """Test creating a completed future."""
        future = completed_future("test_result")
        assert future.state == APIFutureState.COMPLETED
        assert future.done
        assert future.result() == "test_result"
        assert future.progress == 100.0
    
    def test_failed_future(self):
        """Test creating a failed future."""
        exc = ValueError("test error")
        future = failed_future(exc)
        assert future.state == APIFutureState.FAILED
        assert future.done
        assert future.exception() is not None
    
    def test_cancelled_future(self):
        """Test creating a cancelled future."""
        future = cancelled_future()
        assert future.state == APIFutureState.CANCELLED
        assert future.done
        assert future.cancelled
    
    def test_then_callback(self):
        """Test then() callback."""
        results = []
        future = completed_future(42)
        future.then(lambda x: results.append(x))
        assert results == [42]
    
    def test_catch_callback(self):
        """Test catch() callback."""
        errors = []
        future = failed_future(ValueError("test"))
        future.catch(lambda e: errors.append(str(e)))
        assert "test" in errors[0]
    
    def test_finally_callback(self):
        """Test finally_() callback."""
        called = []
        future = completed_future(42)
        future.finally_(lambda: called.append(True))
        assert called == [True]
    
    def test_progress_tracking(self):
        """Test progress tracking."""
        future = APIFuture()
        progress_updates = []
        
        future.on_progress(lambda p, m: progress_updates.append((p, m)))
        future.update_progress(50.0, "halfway")
        
        assert future.progress == 50.0
        assert future.progress_message == "halfway"
        assert progress_updates == [(50.0, "halfway")]
    
    def test_chaining_callbacks(self):
        """Test chaining multiple callbacks."""
        results = []
        future = completed_future(42)
        
        (future
            .then(lambda x: results.append(f"then:{x}"))
            .finally_(lambda: results.append("finally")))
        
        assert "then:42" in results
        assert "finally" in results
    
    def test_pool_submit(self):
        """Test submitting to thread pool."""
        pool = SdkFuturePool(max_workers=1)
        
        def slow_task():
            time.sleep(0.01)
            return "done"
        
        future = pool.submit(slow_task)
        assert isinstance(future, APIFuture)
        assert future.result(timeout=1.0) == "done"
        
        pool.shutdown()


# =============================================================================
# TrainingClient Tests
# =============================================================================

class TestTrainingClient:
    """Test TrainingClient functionality."""
    
    def test_client_initialization(self):
        """Test TrainingClient initialization."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg)
        
        client = TrainingClient(loaded.backend)
        assert client.backend == loaded.backend
        assert client.step == 0
        assert client.optimizer is None
        
        client.shutdown()
    
    def test_create_optimizer(self):
        """Test creating optimizer."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg)
        
        client = TrainingClient(loaded.backend)
        future = client.create_optimizer(lr=1e-4, weight_decay=0.01)
        future.result()
        
        assert client.optimizer is not None
        
        client.shutdown()
    
    def test_forward_backward_sft(self):
        """Test forward/backward with SFT batch."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg, apply_lora_if_missing=True)
        
        client = TrainingClient(loaded.backend)
        
        batch = TrainingBatch(
            prompts=["What is 2+2?", "What is 3+3?"],
            responses=["The answer is 4.", "The answer is 6."],
            loss_type="sft",
        )
        
        future = client.forward_backward(batch)
        result = future.result()
        
        assert isinstance(result, ForwardBackwardResult)
        assert result.loss is not None
        assert result.batch_size == 2
        assert "num_samples" in result.metrics
        
        client.shutdown()
    
    def test_forward_backward_preference(self):
        """Test forward/backward with preference batch."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg, apply_lora_if_missing=True)
        
        client = TrainingClient(loaded.backend)
        
        batch = TrainingBatch(
            prompts=["What is 2+2?"],
            responses=["The answer is 4."],
            rejected_responses=["I don't know."],
            loss_type="dpo",
        )
        
        future = client.forward_backward(batch)
        result = future.result()
        
        assert isinstance(result, ForwardBackwardResult)
        
        client.shutdown()
    
    def test_save_and_load_state(self):
        """Test saving and loading state."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg, apply_lora_if_missing=True)
        
        client = TrainingClient(loaded.backend)
        client._step = 100  # Simulate some training
        
        import tempfile
        import shutil
        import os
        
        tmpdir = tempfile.mkdtemp()
        try:
            # Save state
            save_path = os.path.join(tmpdir, "checkpoint")
            save_future = client.save_state(save_path, metadata={"test": True})
            save_result = save_future.result()
            
            assert isinstance(save_result, CheckpointResult)
            assert save_result.success
            assert "checkpoint" in save_result.path
            
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        
        client.shutdown()
    
    def test_get_weights(self):
        """Test getting weights."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg)
        
        client = TrainingClient(loaded.backend)
        future = client.get_weights()
        result = future.result()
        
        assert isinstance(result, WeightsResult)
        assert result.success
        
        client.shutdown()
    
    def test_set_weights(self):
        """Test setting weights."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg)
        
        client = TrainingClient(loaded.backend)
        
        # Mock weights
        weights = {"layer1.weight": [1.0, 2.0], "layer1.bias": [0.5]}
        future = client.set_weights(weights)
        result = future.result()
        
        assert isinstance(result, WeightsResult)
        assert result.success
        assert result.num_tensors == 2
        
        client.shutdown()
    
    def test_training_state(self):
        """Test training state management."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg)
        
        client = TrainingClient(loaded.backend)
        
        client.update_training_state({"epoch": 5, "best_score": 0.95})
        state = client.training_state
        
        assert state["epoch"] == 5
        assert state["best_score"] == 0.95
        
        client.shutdown()
    
    def test_batch_types(self):
        """Test TrainingBatch properties."""
        # SFT batch
        sft_batch = TrainingBatch(
            prompts=["Q1", "Q2"],
            responses=["A1", "A2"],
            loss_type="sft",
        )
        assert len(sft_batch) == 2
        assert not sft_batch.is_preference
        assert not sft_batch.is_rl
        
        # Preference batch
        pref_batch = TrainingBatch(
            prompts=["Q1"],
            responses=["Good"],
            rejected_responses=["Bad"],
            loss_type="dpo",
        )
        assert pref_batch.is_preference
        
        # RL batch
        rl_batch = TrainingBatch(
            prompts=["Q1"],
            responses=["A1"],
            advantages=[1.0],
            loss_type="ppo",
        )
        assert rl_batch.is_rl


# =============================================================================
# SamplingClient Tests
# =============================================================================

class TestSamplingClient:
    """Test SamplingClient functionality."""
    
    def test_client_initialization(self):
        """Test SamplingClient initialization."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg)
        
        client = SamplingClient(backend=loaded.backend)
        assert client.backend == loaded.backend
        
        client.shutdown()
    
    def test_sample_basic(self):
        """Test basic sampling."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg)
        
        client = SamplingClient(backend=loaded.backend)
        result = client.sample("Hello", max_tokens=10)
        
        assert isinstance(result, SampleResult)
        assert result.text is not None
        assert len(result.token_ids) > 0
        assert result.prompt_len >= 0
        
        client.shutdown()
    
    def test_sample_with_logprobs(self):
        """Test sampling with logprobs."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg)
        
        client = SamplingClient(backend=loaded.backend)
        result = client.sample("Hello", max_tokens=5, logprobs_k=5)
        
        assert isinstance(result, SampleResult)
        assert result.logprobs is not None
        assert len(result.logprobs) == 5
        assert result.top_k_logprobs is not None
        assert len(result.top_k_logprobs) == 5
        
        # Check top-k structure
        for token_lp in result.top_k_logprobs:
            assert isinstance(token_lp, dict)
            assert len(token_lp) <= 5
        
        client.shutdown()
    
    def test_sample_result_properties(self):
        """Test SampleResult computed properties."""
        result = SampleResult(
            text="test",
            token_ids=[1, 2, 3, 4, 5],
            prompt_len=2,
            logprobs=[-0.5, -0.3, -0.2],
        )
        
        assert result.completion_token_ids == [3, 4, 5]
        assert result.avg_logprob == pytest.approx(-0.333, abs=0.01)
        assert result.perplexity > 0
    
    def test_sample_batch(self):
        """Test batch sampling."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg)
        
        client = SamplingClient(backend=loaded.backend)
        prompts = ["Q1", "Q2", "Q3"]
        results = client.sample_batch(prompts, max_tokens=5, logprobs_k=3)
        
        assert isinstance(results, SampleBatchResult)
        assert len(results) == 3
        assert len(results.texts) == 3
        assert results.total_tokens > 0
        
        client.shutdown()
    
    def test_sample_async(self):
        """Test async sampling."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg)
        
        client = SamplingClient(backend=loaded.backend)
        future = client.sample_async("Hello", max_tokens=5)
        
        assert isinstance(future, APIFuture)
        result = future.result(timeout=5.0)
        assert isinstance(result, SampleResult)
        
        client.shutdown()
    
    def test_get_logprobs_for_texts(self):
        """Test getting logprobs for existing texts."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg)
        
        client = SamplingClient(backend=loaded.backend)
        prompts = ["What is 2+2?"]
        completions = ["The answer is 4."]
        
        logprobs_list = client.get_logprobs_for_texts(prompts, completions, top_k=5)
        
        assert len(logprobs_list) == 1
        # Each element is a list of {token: logprob} dicts
        for token_lps in logprobs_list[0]:
            assert isinstance(token_lps, dict)
        
        client.shutdown()


# =============================================================================
# Distillation Tests
# =============================================================================

class TestDistillation:
    """Test distillation-related functionality."""
    
    def test_distillation_sampler(self):
        """Test DistillationSampler."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg)
        
        # Use same client for both teacher and student in test
        client = SamplingClient(backend=loaded.backend)
        sampler = DistillationSampler(teacher_client=client, student_client=client)
        
        prompts = ["What is Python?"]
        samples, teacher_logprobs = sampler.sample_with_teacher_logprobs(
            prompts, max_tokens=10, logprobs_k=5
        )
        
        assert isinstance(samples, SampleBatchResult)
        assert len(teacher_logprobs) == 1
        
        client.shutdown()
    
    def test_distillation_training_client(self):
        """Test DistillationTrainingClient."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg)
        
        sampling_client = SamplingClient(backend=loaded.backend)
        train_client = DistillationTrainingClient(
            loaded.backend,
            teacher_sampling_client=sampling_client,
        )
        
        assert train_client.teacher_client is sampling_client
        
        train_client.shutdown()
        sampling_client.shutdown()


# =============================================================================
# Top-k Logprobs Tests
# =============================================================================

class TestTopKLogprobs:
    """Test top-k logprobs extraction."""
    
    def test_mock_backend_generate_with_logprobs(self):
        """Test mock backend returns top-k logprobs."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg)
        
        gen = loaded.backend.generate_with_logprobs(
            "Test prompt",
            max_new_tokens=5,
            logprobs=3,
        )
        
        assert gen.logprobs is not None
        assert len(gen.logprobs) == 5
        assert gen.top_k_logprobs is not None
        assert len(gen.top_k_logprobs) == 5
        
        # Each position should have up to 3 tokens
        for token_dict in gen.top_k_logprobs:
            assert isinstance(token_dict, dict)
            assert len(token_dict) <= 3
            for token, logprob in token_dict.items():
                assert isinstance(token, str)
                assert isinstance(logprob, float)
                assert logprob < 0  # Logprobs should be negative
    
    def test_generate_without_logprobs(self):
        """Test that generate_without_logprobs works."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg)
        
        gen = loaded.backend.generate_with_logprobs(
            "Test prompt",
            max_new_tokens=5,
            logprobs=0,  # No top-k logprobs
        )
        
        assert gen.logprobs is not None
        assert gen.top_k_logprobs is None or len(gen.top_k_logprobs) == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestSDKIntegration:
    """Integration tests for the SDK."""
    
    def test_end_to_end_training_loop(self):
        """Test a complete training loop."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg, apply_lora_if_missing=True)
        
        client = TrainingClient(loaded.backend)
        
        # Create optimizer
        client.create_optimizer(lr=1e-4).result()
        
        # Training loop
        for i in range(3):
            batch = TrainingBatch(
                prompts=[f"Q{i}"],
                responses=[f"A{i}"],
                loss_type="sft",
            )
            
            # Forward/backward
            fb_result = client.forward_backward(batch).result()
            assert fb_result.loss is not None
            
            # In real usage: optimizer step
            # client.optim_step(fb_result.grads).result()
        
        assert client.step == 0  # No optim_step called
        
        client.shutdown()
    
    def test_sample_and_train_interaction(self):
        """Test sampling then training on the samples."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg, apply_lora_if_missing=True)
        
        # Sample
        sampler = SamplingClient(backend=loaded.backend)
        samples = sampler.sample_batch(
            ["Prompt 1", "Prompt 2"],
            max_tokens=10,
        )
        
        # Train on samples (would be typical for RL)
        trainer = TrainingClient(loaded.backend)
        batch = TrainingBatch(
            prompts=["Prompt 1", "Prompt 2"],
            responses=samples.texts,
            loss_type="sft",
        )
        result = trainer.forward_backward(batch).result()
        
        assert result.loss is not None
        
        sampler.shutdown()
        trainer.shutdown()
    
    def test_future_pool_operations(self):
        """Test SdkFuturePool with various operations."""
        cfg = ProjectConfig()
        cfg.model.backend = "mock"
        loaded = load_model("dummy/model", cfg)
        
        pool = SdkFuturePool(max_workers=2)
        
        # Submit sample
        fut1 = pool.submit_sample(
            loaded.backend,
            ["hi", "hello"],
            DecodingConfig(max_new_tokens=4, temperature=0.0)
        )
        
        # Submit generic function
        def compute(x):
            return x * x
        
        fut2 = pool.submit(compute, 5)
        
        # Wait for results
        results1 = fut1.result(timeout=5)
        result2 = fut2.result(timeout=5)
        
        assert len(results1) == 2
        assert result2 == 25
        
        pool.shutdown()


# =============================================================================
# Legacy Tests (keep for backwards compatibility)
# =============================================================================

def test_sdk_sample_and_logprobs_mock():
    cfg = ProjectConfig()
    cfg.model.backend = "mock"
    cfg.model.use_chat_template = False

    loaded = load_model("dummy/model", cfg)
    gens = sample(loaded.backend, ["hi"], DecodingConfig(max_new_tokens=4, temperature=0.0))
    assert len(gens) == 1
    assert gens[0].prompt_len == 2

    lps = logprobs(loaded.backend, ["hi"], [" there"])
    assert len(lps) == 1


def test_sdk_train_steps_mock():
    cfg = ProjectConfig()
    cfg.model.backend = "mock"

    loaded = load_model("dummy/model", cfg, apply_lora_if_missing=True)
    loss, grads = sft_forward_backward(loaded.backend, "hi", " there", train_on_prompt=False, max_seq_len=64)
    assert loss is not None
    assert grads is None

    loss, grads = preference_forward_backward(
        loaded.backend,
        "hi",
        " there",
        " bye",
        algo="dpo",
        beta=0.1,
        max_seq_len=64,
    )
    assert loss is not None
    assert grads is None

    loss, grads = preference_forward_backward(
        loaded.backend,
        "hi",
        " there",
        " bye",
        algo="orpo",
        beta=0.1,
        max_seq_len=64,
    )
    assert loss is not None
    assert grads is None

    opt, _params = create_optimizer(loaded.backend, lr=1e-4, weight_decay=0.0)
    optim_step(loaded.backend, opt, grads)


def test_run_chat_strips_prompt_mock():
    cfg = ProjectConfig()
    cfg.model.backend = "mock"
    cfg.model.use_chat_template = False

    out = run_chat(
        cfg,
        "dummy/model",
        [ChatMessage(role="user", content="Hi")],
        max_new_tokens=4,
        temperature=0.0,
    )
    assert out == "****"


def test_loss_registry_mock():
    cfg = ProjectConfig()
    cfg.model.backend = "mock"
    loaded = load_model("dummy/model", cfg)
    ids = loaded.backend.encode("hello")
    loss_fn = get_loss("cross_entropy")
    loss = loss_fn(loaded.backend, ids, prompt_len=2, train_on_prompt=False)
    assert loss is not None

    is_loss = importance_sampling_loss(loaded.backend, ids, prompt_len=2, advantage=1.0, behavior_logprob=0.1)
    assert is_loss is not None
    is_loss_list = importance_sampling_loss(
        loaded.backend,
        ids,
        prompt_len=2,
        advantage=1.0,
        behavior_logprob=[0.05, 0.05],
    )
    assert is_loss_list == pytest.approx(-1.0)


def test_sdk_future_pool_mock():
    cfg = ProjectConfig()
    cfg.model.backend = "mock"
    loaded = load_model("dummy/model", cfg)
    pool = SdkFuturePool(max_workers=1)
    fut = pool.submit_sample(loaded.backend, ["hi"], DecodingConfig(max_new_tokens=2, temperature=0.0))
    out = fut.result(timeout=5)
    assert len(out) == 1
    pool.shutdown()
