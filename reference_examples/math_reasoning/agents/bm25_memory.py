"""Task-label-free BM25 retrieval baseline over external solved examples."""

from math_retriever import MathBM25


class Candidate:
    def __init__(self, memory, task_context=None, k=3, max_solution_chars=2500):
        self.memory = [row for row in memory if row.get("problem") and row.get("solution")]
        self.index = MathBM25([row["problem"] for row in self.memory])
        self.k = k
        self.max_solution_chars = max_solution_chars

    def build_prompt(self, problem: str) -> str:
        examples = []
        for rank, (_, index) in enumerate(self.index.query(problem, k=self.k), 1):
            row = self.memory[index]
            solution = str(row["solution"])[: self.max_solution_chars]
            examples.append(
                f"Example {rank}:\nProblem: {row['problem']}\nSolution: {solution}"
            )
        context = "\n\n".join(examples)
        return (
            "Solve the problem rigorously. Use the retrieved examples only as "
            "methodological references and ignore any irrelevant details.\n\n"
            f"Problem:\n{problem}\n\nRetrieved solved examples:\n{context}\n\n"
            "Put the final answer within \\boxed{}."
        )
