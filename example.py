"""
Example script demonstrating the Hybrid LLM Hallucination Detection System
"""

from analyzer import HallucinationAnalyzer


def run_example():
    """
    Run a simple example analysis.
    """
    print("Initializing Hybrid LLM Hallucination Detection System...")
    print("This may take a moment on first run (downloading models)...\n")
    
    # Initialize analyzer — ground truth is loaded automatically from TruthfulQA
    analyzer = HallucinationAnalyzer(
        model_name="gpt2",
        semantic_threshold=0.80   # adjust if prompts don't match dataset questions
    )
    
    # Example prompt from ground truth
    prompt = "What is the capital of France?"
    
    # Run analysis
    results = analyzer.analyze(
        prompt=prompt,
        num_responses=5,
        max_length=30,
        temperature=0.8
    )
    
    # Print summary
    analyzer.print_summary(results)
    
    # Show entropy plot
    print("\nDisplaying entropy curve...")
    plt = analyzer.plot_entropy_curve(results['entropy']['entropy_curve'])
    plt.savefig('example_entropy_curve.png', dpi=300, bbox_inches='tight')
    print("Entropy curve saved to 'example_entropy_curve.png'")
    
    print("\n" + "="*80)
    print("Example complete!")
    print("="*80)
    print("\nTo run the Streamlit UI, use: streamlit run app.py")
    print("To run custom prompts, use: python main.py --prompt \"Your question here\"")


if __name__ == "__main__":
    run_example()
