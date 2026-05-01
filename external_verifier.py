"""
External Verifier Module
Uses TruthfulQA, CoQA, SQuAD, NQ, and TriviaQA as ground truth datasets.
Semantic similarity is used to both find matching questions and score responses.
"""

import re
import numpy as np
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from dataset_loader import get_all_qa_pairs


class ExternalVerifier:
    """
    Handles external factual verification.

    Ground-truth priority
    ---------------------
    1. Exact match across aggregated datasets.
    2. Semantic match across aggregated datasets.
    3. Default neutral value (0.5 risk) when no source is found above threshold.

    Question matching uses pre-computed sentence embeddings so it works even
    when the user's prompt differs slightly from the dataset question.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        split: str = "validation",
        semantic_threshold: float = 0.80,
    ):
        """
        Args:
            model_name:        Sentence-Transformers model for embedding.
            split:             Dataset split to load.
            semantic_threshold: Minimum cosine similarity to accept a dataset
                               question as a semantic match (0–1).
        """
        print(f"Loading sentence transformer model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        self.semantic_threshold = semantic_threshold

        # ── Load aggregated datasets ──────────────────────────────────────────
        print(f"Loading datasets (split='{split}'). This might take a few moments...")
        self.qa_pairs = get_all_qa_pairs(split)
        print(f"Loaded {len(self.qa_pairs)} total QA entries.")

        # Pre-compute question embeddings for fast semantic search
        print("Pre-computing question embeddings...")
        questions = [entry["question"] for entry in self.qa_pairs]
        self._question_embeddings = self.embedding_model.encode(
            questions, batch_size=64, show_progress_bar=False
        )
        print("External verifier ready.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_relation_query(self, query: str) -> Optional[str]:
        """
        Build a relation-preserving search query for factoid prompts.

        Examples
        --------
        "What is the capital of Australia?" -> "capital of Australia"
        "Who is the president of France?"   -> "president of France"
        """
        q = query.strip().rstrip("?")

        patterns = [
            r"(?i)^what\s+is\s+the\s+(.+?)\s+of\s+(.+)$",
            r"(?i)^who\s+is\s+the\s+(.+?)\s+of\s+(.+)$",
            r"(?i)^what\s+was\s+the\s+(.+?)\s+of\s+(.+)$",
            r"(?i)^who\s+was\s+the\s+(.+?)\s+of\s+(.+)$",
        ]

        for pattern in patterns:
            match = re.match(pattern, q)
            if match:
                relation = match.group(1).strip()
                entity = match.group(2).strip()
                return f"{relation} of {entity}"

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def find_ground_truth(self, question: str) -> Optional[Dict[str, str]]:
        """
        Find the ground-truth for *question*.

        Priority: Exact match -> Semantic match -> None.

        Returns:
            Dict with keys ``text`` (ground truth string) and ``source``
            (dataset name), or ``None``.
        """
        q_lower = question.strip().lower()

        # ── 1. Exact match ─────────────────────────────────────────
        for i, entry in enumerate(self.qa_pairs):
            if entry["question"].strip().lower() == q_lower:
                source = entry["source"]
                print(f"  [ExternalVerifier] Exact {source} match (entry #{i}).")
                return {"text": entry["best_answer"], "source": source}

        # ── 2. Semantic match ──────────────────────────────────────
        query_emb = self.embedding_model.encode([question])
        sims = cosine_similarity(query_emb, self._question_embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_entry = self.qa_pairs[best_idx]

        print(
            f"  [ExternalVerifier] Best {best_entry['source']} match: "
            f"'{best_entry['question'][:80]}' "
            f"(sim={best_sim:.4f})"
        )

        if best_sim >= self.semantic_threshold:
            return {"text": best_entry["best_answer"], "source": best_entry["source"]}

        print(
            f"  [ExternalVerifier] Similarity {best_sim:.4f} < threshold "
            f"{self.semantic_threshold:.2f} -> No reliable match found."
        )

        return None

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity between two texts via sentence embeddings."""
        embeddings = self.embedding_model.encode([text1, text2])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(sim)

    def verify_responses(
        self,
        responses: List[str],
        ground_truth: str,
    ) -> Dict[str, object]:
        """
        Score each generated response against the ground-truth answer.

        Returns:
            {
                "similarities":          List[float],
                "external_consistency":  float,   # mean similarity
                "external_risk":         float,   # 1 - consistency
                "ground_truth":          str,
            }
        """
        similarities = []
        for i, response in enumerate(responses):
            sim = self.compute_similarity(response, ground_truth)
            similarities.append(sim)
            print(f"  Response {i + 1} similarity to ground truth: {sim:.4f}")

        external_consistency = float(np.mean(similarities))
        external_risk = 1.0 - external_consistency

        return {
            "similarities": similarities,
            "external_consistency": external_consistency,
            "external_risk": external_risk,
            "ground_truth": ground_truth,
        }

    def compute_external_metrics(
        self,
        prompt: str,
        responses: List[str],
    ) -> Optional[Dict[str, object]]:
        """
        Full external verification pipeline for a single prompt.

        Returns:
            Metrics dict (see ``verify_responses``) augmented with
            ``ground_truth_source`` key, or ``None`` if no ground truth found.
        """
        result = self.find_ground_truth(prompt)

        if result is None:
            print(
                f"  Warning: No ground truth found "
                f"for: '{prompt[:80]}'"
            )
            return None

        ground_truth = result["text"]
        source = result["source"]
        print(f"  Ground truth ({source}): {str(ground_truth)[:120]}...")

        metrics = self.verify_responses(responses, ground_truth)
        metrics["ground_truth_source"] = source
        return metrics

