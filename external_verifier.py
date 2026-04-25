"""
External Verifier Module
Uses TruthfulQA dataset (HuggingFace) as primary ground truth.
Falls back to a Wikipedia article summary when no TruthfulQA match is found.
Semantic similarity is used to both find matching questions and score responses.
"""

import re
import numpy as np
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from dataset_loader import load_truthfulqa


# ── Optional Wikipedia import (graceful degradation) ─────────────────────────

try:
    import wikipedia as _wiki_module
    _WIKI_AVAILABLE = True
except ImportError:
    _wiki_module = None
    _WIKI_AVAILABLE = False


class ExternalVerifier:
    """
    Handles external factual verification.

    Ground-truth priority
    ---------------------
    1. TruthfulQA (HuggingFace) — exact then semantic match.
    2. Wikipedia summary          — searched when TruthfulQA has no match.
    3. Default neutral value (0.5 risk) when both sources fail.

    Question matching uses pre-computed sentence embeddings so it works even
    when the user's prompt differs slightly from the dataset question.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        split: str = "validation",
        semantic_threshold: float = 0.80,
        wiki_sentences: int = 3,
    ):
        """
        Args:
            model_name:        Sentence-Transformers model for embedding.
            split:             TruthfulQA split ('validation' is the only one).
            semantic_threshold: Minimum cosine similarity to accept a TruthfulQA
                               question as a semantic match (0–1).
            wiki_sentences:    Number of sentences to extract from the Wikipedia
                               article summary used as ground truth.
        """
        print(f"Loading sentence transformer model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        self.semantic_threshold = semantic_threshold
        self.wiki_sentences = wiki_sentences

        # ── Load TruthfulQA from HuggingFace ──────────────────────────────────
        print(f"Loading TruthfulQA dataset (split='{split}') from HuggingFace...")
        raw = load_truthfulqa(split)

        self.qa_pairs: List[Dict] = list(raw)
        print(f"Loaded {len(self.qa_pairs)} TruthfulQA entries.")

        # Pre-compute question embeddings for fast semantic search
        print("Pre-computing question embeddings...")
        questions = [entry["question"] for entry in self.qa_pairs]
        self._question_embeddings = self.embedding_model.encode(
            questions, batch_size=64, show_progress_bar=False
        )
        print(f"TruthfulQA verifier ready. Wikipedia fallback: {'enabled' if _WIKI_AVAILABLE else 'disabled (install wikipedia package)'}.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _best_answer(self, idx: int) -> str:
        """Return the canonical best answer for a dataset entry."""
        return self.qa_pairs[idx].get("best_answer", "")

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

    def _split_sentences(self, text: str) -> List[str]:
        """Split a paragraph into simple sentence candidates."""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]

    def _select_best_ground_truth_span(self, question: str, text: str) -> str:
        """
        Pick the sentence or short span that best answers the question.

        This avoids using an entire generic country/article summary as the
        reference answer for narrow factoid prompts such as capitals.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return text.strip()
        if len(sentences) == 1:
            return sentences[0]

        embeddings = self.embedding_model.encode([question] + sentences)
        query_embedding = [embeddings[0]]
        sentence_embeddings = embeddings[1:]
        scores = cosine_similarity(query_embedding, sentence_embeddings)[0]
        best_idx = int(np.argmax(scores))
        best_sentence = sentences[best_idx]

        # Include the following sentence when it is very short and likely
        # completes the answer span.
        if best_idx + 1 < len(sentences) and len(best_sentence.split()) < 8:
            follow_up = sentences[best_idx + 1]
            if len(follow_up.split()) <= 20:
                return f"{best_sentence} {follow_up}".strip()

        return best_sentence

    def _extract_topic(self, query: str) -> str:
        """
        Heuristically extract the main topic/entity from a question so that
        the Wikipedia search is anchored to the right subject.

        Examples
        --------
        "What is the capital of France?"  → "France"
        "Who invented the telephone?"     → "telephone invention"
        "What year did WW2 end?"          → "World War 2"
        """
        q = query.strip().rstrip("?").lower()

        # Pattern: "what is the X of Y" → topic = Y
        m = re.search(r'\bof\s+([a-z][a-z\s\-]+)$', q)
        if m:
            return m.group(1).strip().title()

        # Pattern: "who invented/discovered/created X" → topic = X
        m = re.search(r'\b(?:invented?|discovered?|created?|founded?|wrote?|built?)\s+(?:the\s+)?(.+)$', q)
        if m:
            return m.group(1).strip().title()

        # Pattern: "when did/was X …" → topic = X (the first proper noun-ish phrase)
        m = re.search(r'\bwhen\s+(?:did|was|were)\s+(?:the\s+)?(.+?)\s+(?:\w+ed|happen|occur|take place)', q)
        if m:
            return m.group(1).strip().title()

        # Fallback: return the full query (original behaviour)
        return query

    def _search_wikipedia(self, query: str) -> Optional[str]:
        """
        Search Wikipedia for a relevant article summary.

        Strategy
        --------
        1. Extract the core topic from the question (e.g. "France" from
           "What is the capital of France?").
        2. Run ``wikipedia.search()`` for the topic, then the full query.
        3. Try the candidates in the order returned by Wikipedia (which uses
           BM25/PageRank and is generally better than title embedding similarity).
           Prioritize exact matches with the topic.
        """
        if not _WIKI_AVAILABLE:
            print("  [Wikipedia] Package not installed — skipping fallback.")
            return None

        try:
            topic = self._extract_topic(query)
            relation_query = self._extract_relation_query(query)

            candidates: list[str] = []
            seen: set[str] = set()
            
            def add_candidates(search_term):
                try:
                    hits = _wiki_module.search(search_term, results=5)
                    for c in hits:
                        if c not in seen and "question" not in c.lower():
                            seen.add(c)
                            candidates.append(c)
                except Exception:
                    pass

            # First search the relation-preserving query when available, then
            # the extracted topic, then the full query.
            if relation_query:
                add_candidates(relation_query)
            if topic and topic.lower() != query.lower():
                add_candidates(topic)
            add_candidates(query)

            if not candidates:
                print("  [Wikipedia] No search results returned.")
                return None

            # Prioritize exact match with the relation query, then with topic.
            ranked = []
            for c in candidates:
                c_lower = c.lower()
                if relation_query and c_lower == relation_query.lower():
                    ranked.insert(0, c)
                elif c_lower == topic.lower():
                    ranked.insert(0, c)
                else:
                    ranked.append(c)

            print(f"  [Wikipedia] Top candidates: {ranked[:4]}")

            for title in ranked:
                try:
                    summary = _wiki_module.summary(
                        title,
                        sentences=self.wiki_sentences,
                        auto_suggest=False,
                    )
                    if summary and len(summary.strip()) > 20:
                        best_span = self._select_best_ground_truth_span(query, summary)
                        print(f"  [Wikipedia] Selected article: '{title}' ({len(summary)} chars)")
                        return best_span
                except _wiki_module.exceptions.DisambiguationError as e:
                    for opt in e.options[:3]:
                        try:
                            summary = _wiki_module.summary(
                                opt,
                                sentences=self.wiki_sentences,
                                auto_suggest=False,
                            )
                            if summary and len(summary.strip()) > 20:
                                best_span = self._select_best_ground_truth_span(query, summary)
                                print(f"  [Wikipedia] Disambiguation resolved to '{opt}'")
                                return best_span
                        except Exception:
                            continue
                except _wiki_module.exceptions.PageError:
                    continue
                except Exception as exc:
                    print(f"  [Wikipedia] Error fetching '{title}': {exc}")
                    continue

            print("  [Wikipedia] All candidates failed.")
            return None

        except Exception as exc:
            print(f"  [Wikipedia] Search failed: {exc}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def find_ground_truth(self, question: str) -> Optional[Dict[str, str]]:
        """
        Find the ground-truth for *question*.

        Priority: TruthfulQA (exact → semantic) → Wikipedia → None.

        Returns:
            Dict with keys ``text`` (ground truth string) and ``source``
            (``"TruthfulQA"`` or ``"Wikipedia"``), or ``None``.
        """
        q_lower = question.strip().lower()

        # ── 1. Exact TruthfulQA match ─────────────────────────────────────────
        for i, entry in enumerate(self.qa_pairs):
            if entry["question"].strip().lower() == q_lower:
                print(f"  [ExternalVerifier] Exact TruthfulQA match (entry #{i}).")
                return {"text": self._best_answer(i), "source": "TruthfulQA"}

        # ── 2. Semantic TruthfulQA match ──────────────────────────────────────
        query_emb = self.embedding_model.encode([question])
        sims = cosine_similarity(query_emb, self._question_embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        print(
            f"  [ExternalVerifier] Best TruthfulQA match: "
            f"'{self.qa_pairs[best_idx]['question'][:80]}' "
            f"(sim={best_sim:.4f})"
        )

        if best_sim >= self.semantic_threshold:
            return {"text": self._best_answer(best_idx), "source": "TruthfulQA"}

        print(
            f"  [ExternalVerifier] Similarity {best_sim:.4f} < threshold "
            f"{self.semantic_threshold:.2f} -> trying Wikipedia fallback."
        )

        # ── 3. Wikipedia fallback ─────────────────────────────────────────────
        wiki_text = self._search_wikipedia(question)
        if wiki_text:
            return {"text": wiki_text, "source": "Wikipedia"}

        print("  [ExternalVerifier] No ground truth found from any source.")
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
                f"  Warning: No ground truth found (TruthfulQA or Wikipedia) "
                f"for: '{prompt[:80]}'"
            )
            return None

        ground_truth = result["text"]
        source = result["source"]
        print(f"  Ground truth ({source}): {ground_truth[:120]}...")

        metrics = self.verify_responses(responses, ground_truth)
        metrics["ground_truth_source"] = source
        return metrics
