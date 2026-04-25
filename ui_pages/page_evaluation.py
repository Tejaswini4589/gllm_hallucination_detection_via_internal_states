"""
Page 3 - Evaluation Page.
ROC curve, confusion matrix, accuracy, precision, recall, and F1.
"""

import ast

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


def render(cfg: dict):
    st.markdown(
        """
    <div class='hero'>
        <h1>Evaluation</h1>
        <p>Measure how well the risk score separates factual answers from incorrect ones.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

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
"""
        )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        roc_samples = st.slider(
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
        )
    run_btn = st.button("Run Batch Evaluation", use_container_width=True, key="run_roc")
    st.markdown("</div>", unsafe_allow_html=True)

    if not run_btn:
        st.markdown(
            """
        <div class='card' style='text-align:center; padding:2.5rem;'>
            <h3 style='color:#0369a1;'>Waiting for Evaluation</h3>
            <p style='color:rgba(0,0,0,0.6);'>
                Click <strong>Run Batch Evaluation</strong> to score labelled answers from the validation CSV.
            </p>
        </div>""",
            unsafe_allow_html=True,
        )
        return

    try:
        df = pd.read_csv("generation_validation.csv")
    except FileNotFoundError:
        st.error("generation_validation.csv not found in the working directory.")
        return

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
        metric_card("F1 Score", f"{f1:.2%}", "Harmonic mean of precision and recall", "#ef4444")

    col_roc, col_cm = st.columns([3, 2])
    with col_roc:
        st.plotly_chart(plot_roc(y_true, y_scores, auc_val), use_container_width=True)
    with col_cm:
        st.plotly_chart(plot_confusion(y_true, y_pred), use_container_width=True)

    with st.expander("Per-sample results"):
        st.dataframe(pd.DataFrame(details), use_container_width=True)

    st.caption(
        "This evaluation now scores the labelled answers in the CSV directly against the detected ground truth, instead of ignoring the supplied answer text."
    )
