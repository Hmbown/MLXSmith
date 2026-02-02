from mlxsmith.config import ProjectConfig
from mlxsmith.rlm.generate import generate_tasks
from mlxsmith.rlm.mutate import mutate_tasks
from mlxsmith.llm.registry import get_llm_backend


def test_rlm_mutation_fallback():
    cfg = ProjectConfig()
    cfg.model.backend = "mock"

    llm = get_llm_backend(cfg.model.backend)
    llm.load("dummy/model")

    tasks = generate_tasks(
        llm,
        tasks_per_iter=1,
        temperature=0.7,
        max_new_tokens=32,
        top_p=1.0,
        top_k=None,
        require_recursion=False,
        task_domains=["strings"],
    )

    mutated = mutate_tasks(
        llm,
        tasks,
        mutations_per_task=1,
        max_total=1,
        temperature=0.7,
        max_new_tokens=32,
        top_p=1.0,
        top_k=None,
        require_recursion=False,
    )

    assert len(mutated) == 1
    assert mutated[0].prompt
    assert mutated[0].tests
