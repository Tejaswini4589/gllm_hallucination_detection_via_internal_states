"""
Dataset Loader Module
Loads and caches the TruthfulQA dataset from HuggingFace for ground truth comparison.
"""

from datasets import load_dataset


def load_truthfulqa(split: str = "validation"):
    """
    Load the TruthfulQA dataset (generation config) from HuggingFace.

    Args:
        split: Dataset split to load ('validation' is the only available split)

    Returns:
        HuggingFace Dataset object with fields:
            - question     (str)
            - best_answer  (str)
            - correct_answers (List[str])
            - incorrect_answers (List[str])
            - source (str)
    """
    dataset = load_dataset("truthful_qa", "generation")
    return dataset[split]


def build_qa_lookup(split: str = "validation") -> dict:
    """
    Return a dict mapping each question (lowercased) -> best_answer.
    Useful for fast exact-lookup.

    Args:
        split: Dataset split to use

    Returns:
        dict {question_lower: best_answer}
    """
    data = load_truthfulqa(split)
    return {entry["question"].strip().lower(): entry["best_answer"] for entry in data}