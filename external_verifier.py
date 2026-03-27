"""
External Verifier Module
Uses TruthfulQA dataset (HuggingFace) as ground truth for external verification.
Semantic similarity is used to both find matching questions and score responses.
"""

import numpy as np
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from dataset_loader import load_truthfulqa


class ExternalVerifier:
    """
    Handles external factual verification using the TruthfulQA dataset.

    Ground truth is fetched directly from HuggingFace (`truthful_qa`, generation config)
    so no local JSON file is required.  Question matching is done first by exact string
    comparison and then by semantic (embedding) similarity so it works even when the
    user's prompt is slightly different from the dataset question.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        split: str = "validation",
        semantic_threshold: float = 0.80,
    ):
        """
        Initialize the external verifier.

        Args:
            model_name: Sentence-Transformers model used for both question matching
                        and response–ground-truth similarity scoring.
            split: TruthfulQA split to load ('validation' is the only available one).
            semantic_threshold: Minimum cosine similarity required to accept a
                                question as a semantic match (0–1).
        """
        print(f"Loading sentence transformer model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        self.semantic_threshold = semantic_threshold

        # ── Load TruthfulQA from HuggingFace ──────────────────────────────────
        print(f"Loading TruthfulQA dataset (split='{split}') from HuggingFace...")
        raw = load_truthfulqa(split)

        # Materialise into a plain list so we can iterate multiple times cheaply
        self.qa_pairs: List[Dict] = list(raw)
        print(f"Loaded {len(self.qa_pairs)} TruthfulQA entries.")

        # Pre-compute question embeddings for fast semantic search
        print("Pre-computing question embeddings...")
        questions = [entry["question"] for entry in self.qa_pairs]
        self._question_embeddings = self.embedding_model.encode(
            questions, batch_size=64, show_progress_bar=False
        )
        print("TruthfulQA verifier ready.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _best_answer(self, idx: int) -> str:
        """Return the canonical best answer for a dataset entry."""
        entry = self.qa_pairs[idx]
        # 'best_answer' is always present in the generation config
        return entry.get("best_answer", "")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def find_ground_truth(self, question: str) -> Optional[str]:
        """
        Find the ground-truth answer for *question* from TruthfulQA.

        Strategy
        --------
        1. Exact string match (case-insensitive, stripped).
        2. Semantic similarity: embed the query and compare against all pre-computed
           question embeddings; return the best_answer of the closest match if its
           similarity exceeds ``self.semantic_threshold``.

        Args:
            question: The prompt / question to look up.

        Returns:
            best_answer string if a match is found, else ``None``.
        """
        q_lower = question.strip().lower()

        # ── 1. Exact match ────────────────────────────────────────────────────
        for i, entry in enumerate(self.qa_pairs):
            if entry["question"].strip().lower() == q_lower:
                print(f"  [ExternalVerifier] Exact match found (entry #{i}).")
                return self._best_answer(i)

        # ── 2. Semantic match ─────────────────────────────────────────────────
        query_emb = self.embedding_model.encode([question])
        sims = cosine_similarity(query_emb, self._question_embeddings)[0]  # (N,)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        print(
            f"  [ExternalVerifier] Best semantic match: "
            f"'{self.qa_pairs[best_idx]['question'][:80]}' "
            f"(sim={best_sim:.4f})"
        )

        if best_sim >= self.semantic_threshold:
            return self._best_answer(best_idx)

        print(
            f"  [ExternalVerifier] Similarity {best_sim:.4f} < threshold "
            f"{self.semantic_threshold:.2f} → no ground truth found."
        )
        return None

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts using sentence embeddings.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Cosine similarity in [0, 1].
        """
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

        Args:
            responses: List of model-generated responses.
            ground_truth: The canonical TruthfulQA answer.

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

        Args:
            prompt: Original question / prompt.
            responses: List of generated responses to evaluate.

        Returns:
            Metrics dict (see ``verify_responses``) or ``None`` if no TruthfulQA
            entry matches the prompt with sufficient confidence.
        """
        ground_truth = self.find_ground_truth(prompt)

        if ground_truth is None:
            print(
                f"  Warning: No TruthfulQA ground truth found for: '{prompt[:80]}'"
            )
            return None

        print(f"  Ground truth: {ground_truth}")
        return self.verify_responses(responses, ground_truth)
