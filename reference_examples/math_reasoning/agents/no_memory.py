"""No-retrieval baseline."""


class Candidate:
    def __init__(self, test_problems=None):
        self.test_problems = test_problems or []

    def build_prompt(self, problem: str) -> str:
        return (
            f"{problem}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

