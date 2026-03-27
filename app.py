"""
Streamlit UI for Hybrid LLM Hallucination Detection System
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyzer import HallucinationAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc


# Page configuration
st.set_page_config(
    page_title="Hallucination Detection System",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_analyzer(model_name, semantic_threshold):
    """Load the analyzer (cached to avoid reloading on every re-run)"""
    return HallucinationAnalyzer(
        model_name=model_name,
        semantic_threshold=semantic_threshold
    )


def get_risk_color(risk_score):
    """Return color based on risk score"""
    if risk_score < 0.3:
        return "#28a745"  # Green
    elif risk_score < 0.6:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red


def get_risk_label(risk_score):
    """Return risk label based on score"""
    if risk_score < 0.3:
        return "LOW", "risk-low"
    elif risk_score < 0.6:
        return "MEDIUM", "risk-medium"
    else:
        return "HIGH", "risk-high"


def plot_eigenvalue_spectrum(eigenvalues):
    """Create interactive eigenvalue spectrum bar chart"""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(range(len(eigenvalues))),
        y=eigenvalues,
        name='Eigenvalue',
        marker_color='#1f77b4',
        opacity=0.85
    ))

    fig.update_layout(
        title="Hidden-State Covariance Eigenvalue Spectrum",
        xaxis_title="Eigenvalue Index",
        yaxis_title="Eigenvalue Magnitude",
        hovermode='x unified',
        height=400
    )

    return fig


def plot_metrics_comparison(results):
    """Create comparison plot for all metrics"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Internal Risk Components',
            'External Similarity Scores',
            'Layer-wise Stability',
            'Final Risk Breakdown'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'pie'}]]
    )
    
    # Internal risk components
    internal_components = [
        results['internal_risk']['eigen_score_component'],
        results['internal_risk']['stability_component'],
        results['internal_risk']['grounding_component']
    ]
    fig.add_trace(
        go.Bar(
            x=['Eigen Score', 'Stability', 'Grounding'],
            y=internal_components,
            marker_color=['#ff7f0e', '#2ca02c', '#d62728'],
            name='Components'
        ),
        row=1, col=1
    )
    
    # External similarity scores
    if results['external']['ground_truth'] != "N/A":
        similarities = results['external']['similarities']
        fig.add_trace(
            go.Bar(
                x=[f'Response {i+1}' for i in range(len(similarities))],
                y=similarities,
                marker_color='#9467bd',
                name='Similarity'
            ),
            row=1, col=2
        )
    
    # Layer-wise stability
    if 'layer_similarities' in results['stability']:
        layer_sims = results['stability']['layer_similarities']
        fig.add_trace(
            go.Bar(
                x=[f'L{i}-{i+1}' for i in range(len(layer_sims))],
                y=layer_sims,
                marker_color='#8c564b',
                name='Stability'
            ),
            row=2, col=1
        )
    
    # Final risk breakdown
    fig.add_trace(
        go.Pie(
            labels=['Internal Risk', 'External Risk'],
            values=[
                results['weights']['alpha'] * results['internal_risk']['internal_risk'],
                results['weights']['beta'] * results['external']['external_risk']
            ],
            marker_colors=['#e377c2', '#7f7f7f']
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    
    return fig


def plot_roc_curve(y_true, y_scores, auc_score):
    """Create an interactive ROC curve plot using Plotly."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    fig = go.Figure()

    # Diagonal chance line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='#888888', width=1, dash='dash'),
        name='Random Chance (AUC = 0.50)',
        hoverinfo='skip'
    ))

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines+markers',
        name=f'ROC Curve (AUC = {auc_score:.4f})',
        line=dict(color='#1f77b4', width=2.5),
        marker=dict(size=4),
        hovertemplate=(
            'FPR: %{x:.3f}<br>TPR: %{y:.3f}<br>'
            'Threshold: %{text}<extra></extra>'
        ),
        text=[f'{t:.3f}' for t in thresholds]
    ))

    # Shade area under curve
    fig.add_trace(go.Scatter(
        x=list(fpr) + [1, 0],
        y=list(tpr) + [0, 0],
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=dict(
            text=f'ROC Curve — Hallucination Detection (AUC = {auc_score:.4f})',
            font=dict(size=16)
        ),
        xaxis=dict(title='False Positive Rate', range=[0, 1]),
        yaxis=dict(title='True Positive Rate', range=[0, 1.02]),
        hovermode='x unified',
        height=500,
        legend=dict(x=0.6, y=0.1),
        plot_bgcolor='#fafafa'
    )

    return fig


def main():
    # Header
    st.markdown('<div class="main-header">🔍 Hybrid LLM Hallucination Detection System</div>', 
                unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    MODEL_LABELS = {
        "EleutherAI/gpt-neo-125M": "GPT-Neo 125M",
        "EleutherAI/gpt-neo-1.3B": "GPT-Neo 1.3B",
        "EleutherAI/gpt-neo-2.7B": "GPT-Neo 2.7B",
        "EleutherAI/pythia-2.8b": "Pythia 2.8B",
    }

    model_name = st.sidebar.selectbox(
        "Model",
        [
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "EleutherAI/gpt-neo-125M",
            "EleutherAI/gpt-neo-1.3B",
            "EleutherAI/gpt-neo-2.7B",
            "EleutherAI/pythia-2.8b",
        ],
        index=0,
        format_func=lambda x: MODEL_LABELS.get(x, x)
    )

    # TruthfulQA info banner
    st.sidebar.info(
        "🗂️ **Ground Truth Source**\n\n"
        "External verification uses the **TruthfulQA** dataset "
        "(HuggingFace, `generation` config). Questions are matched "
        "semantically, so your prompt doesn't need to match exactly."
    )

    semantic_threshold = st.sidebar.slider(
        "Semantic Match Threshold",
        min_value=0.50,
        max_value=1.00,
        value=0.80,
        step=0.05,
        help=(
            "Minimum cosine similarity required to consider a TruthfulQA entry "
            "a match for your prompt. Lower = fuzzier matching."
        )
    )

    num_responses = st.sidebar.slider(
        "Number of Responses",
        min_value=1,
        max_value=10,
        value=5
    )
    
    max_length = st.sidebar.slider(
        "Max Generation Length",
        min_value=10,
        max_value=100,
        value=50
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1
    )
    
    st.sidebar.subheader("Risk Weights")
    alpha = st.sidebar.slider("Alpha (Internal)", 0.0, 1.0, 0.6, 0.1)
    beta = st.sidebar.slider("Beta (External)", 0.0, 1.0, 0.4, 0.1)
    
    st.sidebar.subheader("Internal Metric Weights")
    w1 = st.sidebar.slider("w1 (Eigen Score)", 0.0, 1.0, 0.4, 0.1)
    w2 = st.sidebar.slider("w2 (Stability)", 0.0, 1.0, 0.3, 0.1)
    w3 = st.sidebar.slider("w3 (Grounding)", 0.0, 1.0, 0.3, 0.1)


    
    # Main content
    st.header("📝 Input Prompt")
    prompt = st.text_area(
        "Enter your question or prompt:",
        height=100,
        placeholder="e.g., What is the capital of France?"
    )
    
    analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
    if analyze_button and prompt:
        with st.spinner("Loading model and analyzing..."):
            try:
                # Load analyzer
                analyzer = load_analyzer(model_name, semantic_threshold)
                
                # Run analysis
                results = analyzer.analyze(
                    prompt=prompt,
                    num_responses=num_responses,
                    max_length=max_length,
                    temperature=temperature,
                    alpha=alpha,
                    beta=beta,
                    w1=w1,
                    w2=w2,
                    w3=w3,
                )
                
                # Display results
                st.success("Analysis complete!")
                
                # Final Risk Score (prominent display)
                st.header("🎯 Final Hallucination Risk Score")
                final_risk = results['final_risk']
                risk_label, risk_class = get_risk_label(final_risk)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.metric(
                        label="Risk Score",
                        value=f"{final_risk:.4f}",
                        delta=f"{risk_label} Risk",
                        delta_color="inverse"
                    )
                
                # Generated Responses
                st.header("💬 Generated Responses")
                for i, response in enumerate(results['responses'], 1):
                    with st.expander(f"Response {i}", expanded=(i == 1)):
                        st.write(response)
                
                # Internal Metrics
                st.header("📊 Internal Metrics")

                # Helper labels
                def _eigen_label(v):
                    return "🔴 High — uncertain" if v > 0.6 else ("🟡 Medium" if v > 0.3 else "🟢 Low — confident")
                def _stability_label(v):
                    return "🟢 High — stable" if v > 0.7 else ("🟡 Medium" if v > 0.4 else "🔴 Low — unstable")
                def _grounding_label(v):
                    return "🟢 High — grounded" if v > 0.7 else ("🟡 Medium" if v > 0.4 else "🔴 Low — ungrounded")
                def _risk_label(v):
                    return "🔴 High — risky" if v > 0.6 else ("🟡 Medium" if v > 0.3 else "🟢 Low — safe")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    eigen_val = results['eigen']['eigen_score']
                    # Normalise for label display: higher (less negative) = more uncertain
                    import math
                    norm_eig_display = float(1.0 / (1.0 + math.exp(-eigen_val / max(1.0, abs(eigen_val) + 1e-9))))
                    st.metric("Eigen Score", f"{eigen_val:.4f}",
                              delta=_eigen_label(norm_eig_display), delta_color="off")
                with col2:
                    st.metric("Responses Used", f"{results['eigen']['num_responses']}")
                with col3:
                    stab = results['stability']['stability_score']
                    st.metric("Stability Score", f"{stab:.4f}",
                              delta=_stability_label(stab), delta_color="off")
                with col4:
                    gnd = results['grounding']['grounding_score']
                    st.metric("Grounding Score", f"{gnd:.4f}",
                              delta=_grounding_label(gnd), delta_color="off")

                col5, col6 = st.columns(2)
                with col5:
                    irisk = results['internal_risk']['internal_risk']
                    st.metric("Internal Risk", f"{irisk:.4f}",
                              delta=_risk_label(irisk), delta_color="off")
                with col6:
                    st.metric(
                        "Eigenvalues",
                        f"{len(results['eigen']['eigenvalues'])} values (K×K matrix)",
                        help="Covariance matrix is (K×K) where K = number of responses"
                    )

                # Eigenvalue Spectrum
                st.subheader("📈 Eigenvalue Spectrum")
                eig_fig = plot_eigenvalue_spectrum(results['eigen']['eigenvalues'])
                st.plotly_chart(eig_fig, use_container_width=True)
                
                # External Metrics
                st.header("🌐 External Verification Metrics")
                
                if results['external']['ground_truth'] != "N/A":
                    st.info(f"**Ground Truth:** {results['external']['ground_truth']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "External Consistency",
                            f"{results['external']['external_consistency']:.4f}"
                        )
                    with col2:
                        st.metric(
                            "External Risk",
                            f"{results['external']['external_risk']:.4f}"
                        )
                    
                    st.subheader("Similarity Scores per Response")
                    for i, sim in enumerate(results['external']['similarities'], 1):
                        st.write(f"Response {i}: **{sim:.4f}**")
                else:
                    st.warning("No ground truth available for this prompt")
                
                # Comprehensive Visualizations
                st.header("📊 Comprehensive Metrics Visualization")
                metrics_fig = plot_metrics_comparison(results)
                st.plotly_chart(metrics_fig, use_container_width=True)
                
                # Risk Interpretation
                st.header("💡 Risk Interpretation")
                if final_risk < 0.3:
                    st.success("✅ **LOW RISK** - The response appears reliable and well-grounded.")
                elif final_risk < 0.6:
                    st.warning("⚠️ **MEDIUM RISK** - The response may contain some uncertainties or inconsistencies.")
                else:
                    st.error("❌ **HIGH RISK** - The response likely contains hallucinations or factual errors.")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.exception(e)
    
    elif analyze_button:
        st.warning("Please enter a prompt to analyze.")
    
    # ── ROC Curve Section ────────────────────────────────────────────────────
    st.markdown("---")
    st.header("📉 ROC Curve — Batch Evaluation")
    st.markdown(
        "Evaluates the hallucination detector on a random sample of TruthfulQA entries. "
        "**Correct answers** are treated as label 0 (no hallucination) and "
        "**incorrect answers** as label 1 (hallucination). "
        "The model generates one response per Q-A pair and the final risk score is used as the classifier score."
    )

    roc_col1, roc_col2 = st.columns(2)
    with roc_col1:
        roc_samples = st.slider(
            "Number of QA samples",
            min_value=5, max_value=60, value=20, step=5,
            help="Total Q-A pairs to evaluate (split evenly between correct and incorrect answers)."
        )
    with roc_col2:
        roc_max_length = st.slider(
            "Max generation length (ROC)",
            min_value=10, max_value=80, value=30, step=10,
            help="Shorter is faster; longer may be more accurate."
        )

    run_roc = st.button("📊 Run ROC Evaluation", use_container_width=True)

    if run_roc:
        try:
            csv_path = "generation_validation.csv"
            df = pd.read_csv(csv_path)

            # Keep only rows that have both correct and incorrect answers
            df = df.dropna(subset=["best_answer", "incorrect_answers"])
            df = df[df["incorrect_answers"].str.strip() != ""]

            half = roc_samples // 2
            correct_rows = df.sample(n=min(half, len(df)), random_state=42)
            incorrect_rows = df.sample(n=min(half, len(df)), random_state=99)

            prompts_labels = []
            for _, row in correct_rows.iterrows():
                prompts_labels.append((row["question"], row["best_answer"], 0))
            for _, row in incorrect_rows.iterrows():
                # Pick the first incorrect answer from the string-list
                raw = row["incorrect_answers"]
                first_wrong = raw.strip("[]").split("'")[1] if "'" in raw else raw.split(",")[0].strip()
                prompts_labels.append((row["question"], first_wrong, 1))

            with st.spinner(f"Running batch evaluation on {len(prompts_labels)} samples — this may take a few minutes…"):
                analyzer = load_analyzer(model_name, semantic_threshold)

                y_true = []
                y_scores = []
                progress = st.progress(0)
                status_text = st.empty()

                for i, (question, answer, label) in enumerate(prompts_labels):
                    status_text.text(f"Evaluating sample {i + 1}/{len(prompts_labels)}: {question[:60]}…")
                    try:
                        result = analyzer.analyze(
                            prompt=question,
                            num_responses=3,
                            max_length=roc_max_length,
                            temperature=0.8,
                            alpha=alpha,
                            beta=beta,
                            w1=w1, w2=w2, w3=w3
                        )
                        y_true.append(label)
                        y_scores.append(result["final_risk"])
                    except Exception:
                        pass  # skip failed samples
                    progress.progress((i + 1) / len(prompts_labels))

                status_text.empty()
                progress.empty()

            if len(set(y_true)) < 2:
                st.warning("Not enough label diversity to plot a ROC curve. Try increasing the sample count.")
            else:
                auc_score = auc(*roc_curve(y_true, y_scores)[:2])
                roc_fig = plot_roc_curve(y_true, y_scores, auc_score)
                st.plotly_chart(roc_fig, use_container_width=True)

                # Summary metrics
                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    st.metric("AUC", f"{auc_score:.4f}")
                with rc2:
                    st.metric("Samples evaluated", len(y_scores))
                with rc3:
                    mean_risk_positive = np.mean([s for s, l in zip(y_scores, y_true) if l == 1])
                    mean_risk_negative = np.mean([s for s, l in zip(y_scores, y_true) if l == 0])
                    st.metric("Avg risk: hallucinated", f"{mean_risk_positive:.4f}")

                # Detailed table
                with st.expander("🔎 Per-sample scores"):
                    detail_df = pd.DataFrame({
                        "Label": ["Hallucinated" if l == 1 else "Correct" for l in y_true],
                        "Risk Score": [f"{s:.4f}" for s in y_scores]
                    })
                    st.dataframe(detail_df, use_container_width=True)

        except Exception as e:
            st.error(f"ROC evaluation failed: {str(e)}")
            st.exception(e)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Hybrid LLM Hallucination Detection System | "
        "Powered by GPT-2, GPT-Neo &amp; TransformerLens"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
