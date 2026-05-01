"""
Page 3 - Evaluation Page.
ROC curve, confusion matrix, accuracy, precision, recall, and F1.
<<<<<<< HEAD

The model is run for every sampled question so that the `final_risk` score
produced by the full hallucination-detection pipeline is used as the classifier
score for the ROC curve.  The true label is determined by comparing the model's
*own generated response* to the CSV ground-truth answer via cosine similarity:

    similarity >= label_threshold  →  label = 0  (model was correct)
    similarity <  label_threshold  →  label = 1  (model hallucinated)
=======
>>>>>>> 348eac36cba6edb8b73207e4b53b5a0fa24ab3c1
"""

import ast

<<<<<<< HEAD
import numpy as np
=======
>>>>>>> 348eac36cba6edb8b73207e4b53b5a0fa24ab3c1
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from analyzer import HallucinationAnalyzer


@st.cache_resource
def load_analyzer(model_name, semantic_threshold):
    return HallucinationAnalyzer(
        model_name=model_name,
        semantic_threshold=semantic_threshold,
    )


LIGHT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.03)",
    font=dict(color="#4b5563"),
    margin=dict(t=50, b=40, l=50, r=20),
)


def plot_roc(y_true, y_scores, auc_val):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#475569", dash="dash", width=1),
            name="Random Chance (AUC = 0.50)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(fpr) + [1, 0],
            y=list(tpr) + [0, 0],
            fill="toself",
            fillcolor="rgba(2,132,199,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines+markers",
            name=f"ROC Curve (AUC = {auc_val:.4f})",
            line=dict(color="#0284c7", width=2.5),
            marker=dict(size=4, color="#0284c7"),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=f"ROC Curve (AUC = {auc_val:.4f})", font=dict(color="#111827", size=15)),
        xaxis=dict(title="False Positive Rate", range=[0, 1]),
        yaxis=dict(title="True Positive Rate", range=[0, 1.02]),
        legend=dict(x=0.55, y=0.08, font=dict(size=12)),
        height=430,
        **LIGHT_LAYOUT,
    )
    return fig


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Correct (0)", "Hallucinated (1)"]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale=[[0, "#e0f2fe"], [0.5, "#38bdf8"], [1, "#0369a1"]],
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=20, color="#111827"),
            showscale=False,
        )
    )
    fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(color="#111827", size=15)),
        xaxis=dict(title="Predicted", side="bottom"),
        yaxis=dict(title="Actual", autorange="reversed"),
        height=350,
        **LIGHT_LAYOUT,
    )
    return fig


<<<<<<< HEAD
def plot_risk_distribution(details: list) -> go.Figure:
    """Box / strip plot showing risk score distribution per true label."""
    label_key = "True Label (label=0 → Correct, 1 → Hallucinated)"
    correct_scores = [d["Final Risk"] for d in details if d[label_key] == "Correct"]
    halluc_scores  = [d["Final Risk"] for d in details if d[label_key] == "Hallucinated"]

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=correct_scores, name="Correct (label=0)",
        marker_color="#10b981", boxmean=True,
        hovertemplate="Risk: %{y:.4f}<extra>Correct</extra>",
    ))
    fig.add_trace(go.Box(
        y=halluc_scores, name="Hallucinated (label=1)",
        marker_color="#ef4444", boxmean=True,
        hovertemplate="Risk: %{y:.4f}<extra>Hallucinated</extra>",
    ))
    fig.update_layout(
        title=dict(text="Final Risk Distribution by True Label", font=dict(color="#111827", size=13)),
        yaxis=dict(title="Final Risk Score", range=[0, 1.05]),
        height=320,
        **LIGHT_LAYOUT,
    )
    return fig


=======
>>>>>>> 348eac36cba6edb8b73207e4b53b5a0fa24ab3c1
def metric_card(label: str, value: str, sub: str, color: str):
    st.markdown(
        f"""
    <div class='card' style='text-align:center; border-top:3px solid {color};'>
        <p style='color:rgba(0,0,0,0.5); font-size:0.78rem; margin:0 0 2px; font-weight:700;'>{label}</p>
        <p style='color:#111827; font-size:1.8rem; font-weight:700; margin:0;'>{value}</p>
        <p style='color:rgba(0,0,0,0.4); font-size:0.75rem; margin:0;'>{sub}</p>
    </div>""",
        unsafe_allow_html=True,
    )


<<<<<<< HEAD
=======
def _parse_incorrect_answers(raw_value: str) -> list[str]:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return []

    try:
        parsed = ast.literal_eval(raw_value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (SyntaxError, ValueError):
        pass

    return [chunk.strip().strip("'\"") for chunk in raw_value.split(",") if chunk.strip()]


def _score_candidate_answer(analyzer: HallucinationAnalyzer, question: str, answer: str) -> float | None:
    ground_truth = analyzer.external_verifier.find_ground_truth(question)
    if ground_truth is None:
        return None
    similarity = analyzer.external_verifier.compute_similarity(answer, ground_truth["text"])
    return 1.0 - similarity


>>>>>>> 348eac36cba6edb8b73207e4b53b5a0fa24ab3c1
def render(cfg: dict):
    st.markdown(
        """
    <div class='hero'>
        <h1>Evaluation</h1>
<<<<<<< HEAD
        <p>Run the full model pipeline on sampled questions and measure ROC / classification performance.</p>
=======
        <p>Measure how well the risk score separates factual answers from incorrect ones.</p>
>>>>>>> 348eac36cba6edb8b73207e4b53b5a0fa24ab3c1
    </div>
    """,
        unsafe_allow_html=True,
    )

<<<<<<< HEAD
    with st.expander("How this works", expanded=False):
        st.markdown(
            """
**End-to-end evaluation — the model generates responses for every question.**

| Step | What happens |
|---|---|
| **1. Sample** | N questions are drawn from `generation_validation.csv` |
| **2. Analyze** | The full pipeline runs: model generates responses → EigenScore, Stability, Grounding, External similarity → `final_risk` |
| **3. Label** | The model's own generated response is compared to the CSV ground-truth answer via cosine similarity. If similarity ≥ *Label threshold* → **Correct (0)**, else → **Hallucinated (1)** |
| **4. ROC** | `final_risk` is used as the classifier score; true labels from step 3 are used as ground truth |

**Formula reference**

| Metric | Formula |
|---|---|
| **Accuracy** | (TP + TN) / Total |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1** | 2 · P · R / (P + R) |
| **AUC** | Area under ROC (0.5 = random, 1.0 = perfect) |
=======
    with st.expander("Formula Reference", expanded=False):
        st.markdown(
            """
| Metric | Formula |
|---|---|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1 Score** | 2 * Precision * Recall / (Precision + Recall) |
| **AUC** | Area under the ROC curve (0.5 = random, 1.0 = perfect) |
>>>>>>> 348eac36cba6edb8b73207e4b53b5a0fa24ab3c1
"""
        )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        roc_samples = st.slider(
<<<<<<< HEAD
            "Questions to evaluate",
            3, 20, 5, 1,
            key="roc_samples",
            help="Number of questions the model will actually run. Each takes ~10–30 s depending on model size.",
        )
    with c2:
        label_threshold = st.slider(
            "Label threshold (Gen↔GT similarity)",
            0.10, 0.90, 0.45, 0.05,
            key="label_thresh",
            help=(
                "How similar the model's generated response must be to the ground truth "
                "to be counted as Correct. Below this → True Label = Hallucinated."
            ),
        )

    c3, c4 = st.columns(2)
    with c3:
        risk_threshold = st.slider(
            "Risk threshold (confusion matrix)",
            0.10, 0.90, 0.30, 0.05,
            key="risk_thresh",
            help=(
                "GPT-2 final_risk scores cluster around 0.25–0.35. "
                "Set this near the middle of that range. "
                "Risk score above this → Predicted Hallucinated."
            ),
        )
    with c4:
        roc_score_key = st.selectbox(
            "Score used for ROC curve",
            ["Final Risk", "Ext. Similarity (inverted)", "EigenScore component"],
            index=0,
            key="roc_score_key",
            help=(
                "Which score to use as the classifier for the ROC curve. "
                "'Final Risk' uses the full hybrid score. "
                "'Ext. Similarity (inverted)' uses 1 - external_consistency, which is "
                "more reliable when GPT-2 responses are noisy. "
                "'EigenScore component' uses the normalised eigen risk alone."
            ),
=======
            "QA sample pairs",
            5,
            60,
            20,
            5,
            key="roc_samples",
            help="Total question-answer pairs evaluated across correct and incorrect answers.",
        )
    with c2:
        risk_threshold = st.slider(
            "Risk threshold",
            0.1,
            0.9,
            0.5,
            0.05,
            key="risk_thresh",
            help="Risk score above this threshold is classified as hallucination.",
>>>>>>> 348eac36cba6edb8b73207e4b53b5a0fa24ab3c1
        )
    run_btn = st.button("Run Batch Evaluation", use_container_width=True, key="run_roc")
    st.markdown("</div>", unsafe_allow_html=True)

    if not run_btn:
        st.markdown(
            """
        <div class='card' style='text-align:center; padding:2.5rem;'>
            <h3 style='color:#0369a1;'>Waiting for Evaluation</h3>
            <p style='color:rgba(0,0,0,0.6);'>
<<<<<<< HEAD
                Click <strong>Run Batch Evaluation</strong> to run the model on sampled
                questions and compute ROC / classification metrics.
            </p>
            <p style='color:rgba(0,0,0,0.45); font-size:0.85rem;'>
                ⚠️ Each question requires a full model forward pass. 5 questions ≈ 1–3 min.
=======
                Click <strong>Run Batch Evaluation</strong> to score labelled answers from the validation CSV.
>>>>>>> 348eac36cba6edb8b73207e4b53b5a0fa24ab3c1
            </p>
        </div>""",
            unsafe_allow_html=True,
        )
        return

<<<<<<< HEAD
    # ── Load CSV ────────────────────────────────────────────────────────────
=======
>>>>>>> 348eac36cba6edb8b73207e4b53b5a0fa24ab3c1
    try:
        df = pd.read_csv("generation_validation.csv")
    except FileNotFoundError:
        st.error("generation_validation.csv not found in the working directory.")
        return

<<<<<<< HEAD
    df = df.dropna(subset=["question", "best_answer"])
    df = df[df["best_answer"].astype(str).str.strip() != ""]
    sample_count = min(roc_samples, len(df))
    sampled = df.sample(n=sample_count, random_state=42).reset_index(drop=True)

    # ── Load model ──────────────────────────────────────────────────────────
    try:
        analyzer = load_analyzer(cfg["model_name"], cfg["semantic_threshold"])
    except Exception as exc:
        st.error(f"Model loading failed: {exc}")
        return

    # ── Run pipeline for every question ────────────────────────────────────
    st.info(
        f"Running the full analysis pipeline on {sample_count} questions. "
        "Progress is shown below — this may take several minutes."
    )

    y_true, y_scores, details = [], [], []
    progress = st.progress(0.0)
    status   = st.empty()

    for idx, row in sampled.iterrows():
        question  = str(row["question"]).strip()
        gt_text   = str(row["best_answer"]).strip()

        status.markdown(
            f"**[{idx + 1}/{sample_count}]** Running model on: *{question[:80]}*…"
        )

        try:
            results = analyzer.analyze(
                prompt        = question,
                num_responses = cfg["num_responses"],
                max_length    = cfg["max_length"],
                temperature   = cfg["temperature"],
                alpha         = cfg["alpha"],
                beta          = cfg["beta"],
                w1            = cfg["w1"],
                w2            = cfg["w2"],
                w3            = cfg["w3"],
            )
        except Exception as exc:
            st.warning(f"Skipped question {idx + 1} due to error: {exc}")
            progress.progress((idx + 1) / sample_count)
            continue

        final_risk     = results["final_risk"]
        primary_resp   = results["primary_response"]

        # Determine true label: compare model's own response to ground truth
        gen_sim = analyzer.external_verifier.compute_similarity(primary_resp, gt_text)
        true_label = 0 if gen_sim >= label_threshold else 1   # 0=correct, 1=hallucinated

        y_true.append(true_label)

        ext_sim_val   = results["external"]["external_consistency"]
        import math
        raw_eigen     = results["eigen"]["eigen_score"]
        eigen_norm    = float(1.0 / (1.0 + math.exp(-raw_eigen)))

        # Pick the ROC classifier score based on user's selection
        if roc_score_key == "Ext. Similarity (inverted)":
            roc_score = 1.0 - ext_sim_val
        elif roc_score_key == "EigenScore component":
            roc_score = eigen_norm
        else:
            roc_score = final_risk

        predicted = 1 if roc_score > risk_threshold else 0

        y_scores.append(roc_score)

        details.append({
            "Question":          question[:65] + ("…" if len(question) > 65 else ""),
            "Ground Truth":      gt_text[:65]  + ("…" if len(gt_text)   > 65 else ""),
            "Model Response":    primary_resp[:65] + ("…" if len(primary_resp) > 65 else ""),
            "Gen↔GT Sim":        round(gen_sim, 4),
            "True Label (label=0 → Correct, 1 → Hallucinated)": "Hallucinated" if true_label == 1 else "Correct",
            "ROC Score used":    round(roc_score, 4),
            "Final Risk":        round(final_risk, 4),
            "Ext. Similarity":   round(ext_sim_val, 4),
            "EigenScore":        round(raw_eigen, 4),
            "Stability":         round(results["stability"]["stability_score"], 4),
            "Predicted (score > threshold)": "Hallucinated" if predicted == 1 else "Correct",
            "Correct?":          "✅" if predicted == true_label else "❌",
        })

        progress.progress((idx + 1) / sample_count)

    status.empty()
    progress.empty()

    # ── Guard: need at least 2 classes ─────────────────────────────────────
    if len(y_true) < 2:
        st.error("Not enough samples completed successfully to compute metrics.")
        return

    unique_labels = set(y_true)
    if len(unique_labels) < 2:
        only = "Correct" if 0 in unique_labels else "Hallucinated"
        st.warning(
            f"All {len(y_true)} completed samples were labelled **{only}** "
            f"(label threshold = {label_threshold:.2f}). "
            "Try lowering the label threshold so some responses are labelled as hallucinated, "
            "or increase the number of questions."
        )
        # Still show the per-sample table so the user can inspect
        with st.expander("Per-sample results", expanded=True):
            st.dataframe(pd.DataFrame(details), use_container_width=True)
        return

    # ── Metrics ─────────────────────────────────────────────────────────────
    y_pred    = [1 if s > risk_threshold else 0 for s in y_scores]
    auc_val   = auc(*roc_curve(y_true, y_scores)[:2])
    acc       = accuracy_score(y_true, y_pred)
    prec      = precision_score(y_true, y_pred, zero_division=0)
    rec       = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    n_correct = y_true.count(0)
    n_halluc  = y_true.count(1)

    st.success(
        f"Evaluated **{len(y_true)}** questions — "
        f"**{n_correct}** labelled Correct, **{n_halluc}** labelled Hallucinated."
    )
=======
    df = df.dropna(subset=["question", "best_answer", "incorrect_answers"])
    df = df[df["incorrect_answers"].astype(str).str.strip() != ""]
    sample_count = min(max(1, roc_samples // 2), len(df))

    sampled = df.sample(n=sample_count, random_state=42)
    labeled_answers = []
    for _, row in sampled.iterrows():
        wrong_answers = _parse_incorrect_answers(row["incorrect_answers"])
        if not wrong_answers:
            continue
        labeled_answers.append((row["question"], row["best_answer"], 0))
        labeled_answers.append((row["question"], wrong_answers[0], 1))

    if len(labeled_answers) < 4:
        st.warning("Not enough valid labelled answers were found in the CSV.")
        return

    with st.spinner("Scoring labelled answers against the detected ground truth."):
        try:
            analyzer = load_analyzer(cfg["model_name"], cfg["semantic_threshold"])
        except Exception as exc:
            st.error(f"Model loading failed: {exc}")
            return

        y_true, y_scores, details = [], [], []
        progress = st.progress(0.0)
        status = st.empty()

        for index, (question, answer, label) in enumerate(labeled_answers):
            status.text(f"Sample {index + 1}/{len(labeled_answers)}: {question[:70]}...")
            score = _score_candidate_answer(analyzer, question, answer)
            if score is None:
                progress.progress((index + 1) / len(labeled_answers))
                continue

            y_true.append(label)
            y_scores.append(score)
            predicted = 1 if score > risk_threshold else 0
            details.append(
                {
                    "Question": question[:60] + ("..." if len(question) > 60 else ""),
                    "Candidate Answer": answer[:70] + ("..." if len(answer) > 70 else ""),
                    "Label": "Hallucinated" if label == 1 else "Correct",
                    "Risk Score": round(score, 4),
                    "Predicted": "Hallucinated" if predicted else "Correct",
                    "Correct?": "Yes" if predicted == label else "No",
                }
            )
            progress.progress((index + 1) / len(labeled_answers))

        status.empty()
        progress.empty()

    if len(set(y_true)) < 2:
        st.warning("Not enough label variety remained after filtering.")
        return

    y_pred = [1 if score > risk_threshold else 0 for score in y_scores]
    auc_val = auc(*roc_curve(y_true, y_scores)[:2])
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
>>>>>>> 348eac36cba6edb8b73207e4b53b5a0fa24ab3c1

    st.subheader("Performance Metrics")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        metric_card("AUC", f"{auc_val:.4f}", "Area under ROC", "#0284c7")
    with k2:
        metric_card("Accuracy", f"{acc:.2%}", "(TP+TN) / Total", "#60a5fa")
    with k3:
        metric_card("Precision", f"{prec:.2%}", "TP / (TP+FP)", "#34d399")
    with k4:
        metric_card("Recall", f"{rec:.2%}", "TP / (TP+FN)", "#f59e0b")
    with k5:
<<<<<<< HEAD
        metric_card("F1 Score", f"{f1:.2%}", "Harmonic mean P·R", "#ef4444")
=======
        metric_card("F1 Score", f"{f1:.2%}", "Harmonic mean of precision and recall", "#ef4444")
>>>>>>> 348eac36cba6edb8b73207e4b53b5a0fa24ab3c1

    col_roc, col_cm = st.columns([3, 2])
    with col_roc:
        st.plotly_chart(plot_roc(y_true, y_scores, auc_val), use_container_width=True)
    with col_cm:
        st.plotly_chart(plot_confusion(y_true, y_pred), use_container_width=True)

<<<<<<< HEAD
    st.plotly_chart(plot_risk_distribution(details), use_container_width=True)

    with st.expander("Per-sample results", expanded=False):
        st.dataframe(pd.DataFrame(details), use_container_width=True)

    st.caption(
        "The model's full analysis pipeline (EigenScore + Stability + Grounding + External) "
        "is run for each question. final_risk is used as the ROC classifier score. "
        "The true label is determined by comparing the model's own generated response "
        f"to the CSV ground truth (label threshold = {label_threshold:.2f})."
=======
    with st.expander("Per-sample results"):
        st.dataframe(pd.DataFrame(details), use_container_width=True)

    st.caption(
        "This evaluation now scores the labelled answers in the CSV directly against the detected ground truth, instead of ignoring the supplied answer text."
>>>>>>> 348eac36cba6edb8b73207e4b53b5a0fa24ab3c1
    )
