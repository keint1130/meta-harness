"""Deterministic random few-shot baseline over external solved examples."""

import hashlib
import random


class Candidate:
    def __init__(self, memory, task_context=None, k=3, max_solution_chars=2500):
        self.memory = memory
        self.k = k
        self.max_solution_chars = max_solution_chars

    def build_prompt(self, problem: str) -> str:
        if not self.memory or self.k <= 0:
            selected = []
        else:
            seed = int(hashlib.sha256(problem.encode()).hexdigest()[:16], 16)
            rng = random.Random(seed)
            selected = rng.sample(range(len(self.memory)), min(self.k, len(self.memory)))
        examples = []
        for rank, index in enumerate(selected, 1):
            row = self.memory[index]
            if not row.get("problem") or not row.get("solution"):
                continue
            examples.append(
                f"Example {rank}:\nProblem: {row['problem']}\n"
                f"Solution: {str(row['solution'])[:self.max_solution_chars]}"
            )
        context = "\n\n".join(examples)
        return (
            "Solve the following math problem step by step. The examples are "
            "generic references, not an answer key.\n\n"
            f"{problem}\n\n{context}\n\nPut the final answer within \\boxed{{}}."
        )
