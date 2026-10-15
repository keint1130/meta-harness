"""No-retrieval baseline."""


class Candidate:
    def __init__(self, memory=None, task_context=None):
        # Deliberately ignore both external memory and task identity.
        pass

    def build_prompt(self, problem: str) -> str:
        return (
            f"{problem}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

