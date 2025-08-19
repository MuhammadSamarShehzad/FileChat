from typing import TypedDict, List


class QAState(TypedDict, total=False):
    question: str
    retrieved: List[str]
    answer: str