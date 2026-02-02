from __future__ import annotations

import json
from typing import Iterable, List, Optional

from ..util import sha1_text
from .generate import GeneratedTask, extract_json_objects, task_to_prompt, task_to_tests


def mutate_tasks(
    llm,
    tasks: Iterable[GeneratedTask],
    *,
    mutations_per_task: int,
    max_total: Optional[int] = None,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    require_recursion: bool = False,
) -> List[GeneratedTask]:
    tasks_list = list(tasks)
    if mutations_per_task <= 0 or not tasks_list:
        return tasks_list

    mutated: List[GeneratedTask] = list(tasks_list)

    for task in tasks_list:
        for idx in range(mutations_per_task):
            prompt = (
                "Mutate the following coding task to increase diversity. "
                "Return ONE JSON object with fields: id, description, signature, tests.\n\n"
                f"ORIGINAL_ID: {task.id}\n"
                f"ORIGINAL_PROMPT:\n{task.prompt}\n\n"
                f"ORIGINAL_TESTS:\n{task.tests}\n"
            )
            gen = llm.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            items = extract_json_objects(gen.text)
            if not items:
                continue

            item = items[0]
            tid = item.get("id") or f"{task.id}_m{idx}" or sha1_text(json.dumps(item, sort_keys=True))[:12]
            task_prompt = task_to_prompt(item, require_recursion=require_recursion)
            tests = task_to_tests(item)

            if len(task_prompt) < 10 or not tests:
                continue

            mutated.append(
                GeneratedTask(
                    id=str(tid),
                    prompt=task_prompt,
                    tests=tests,
                    description=item.get("description"),
                )
            )

            if max_total is not None and len(mutated) >= max_total:
                break
        if max_total is not None and len(mutated) >= max_total:
            break

    # Deduplicate by id/prompt hash
    seen = set()
    deduped: List[GeneratedTask] = []
    for t in mutated:
        key = t.id or sha1_text(t.prompt)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)
        if max_total is not None and len(deduped) >= max_total:
            break

    return deduped
