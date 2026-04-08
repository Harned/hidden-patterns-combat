from hmmlearn import hmm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from economic_hmm_analysis import EconomicHMMAnalyzer

def main():
    print("=== Extended Validation Demo ===")
    
    # Initialize analyzer
    analyzer = EconomicHMMAnalyzer()
    
    # Load data
    print("\n1. Loading data...")
    df = analyzer.load_data("../data/synthetic_cashflow.csv")
    
    # Define feature columns
    feature_columns = ["cash_inflow", "overdue_ratio", "avg_payment_delay_days", "contract_terms_index"]
    
    # Preprocess data
    print(f"\n2. Preprocessing data...")
    X_scaled = analyzer.preprocess_data(df, feature_columns)
    
    # Extended validation: Find optimal number of states (2 to 8)
    print(f"\n3. Extended model selection (2 to 8 states)...")
    results_df, optimal_n_states = analyzer.find_optimal_states(X_scaled, min_states=2, max_states=8)
    
    if results_df is not None:
        # Plot AIC and BIC vs number of states
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(results_df["n_states"], results_df["aic"], "o-", label="AIC")
        plt.title("AIC vs Number of States")
        plt.xlabel("Number of States")
        plt.ylabel("AIC")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(results_df["n_states"], results_df["bic"], "o-", label="BIC", color="orange")
        plt.title("BIC vs Number of States")
        plt.xlabel("Number of States")
        plt.ylabel("BIC")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nOptimal number of states based on BIC: {optimal_n_states}")
        
        # Fit model with optimal number of states
        print(f"\n4. Fitting model with optimal number of states ({optimal_n_states})...")
        model = analyzer.fit_model(X_scaled, n_hidden_states=optimal_n_states)
        
        # Perform multiple stability checks with different numbers of states
        print(f"\n5. Stability analysis for different numbers of states...")
        stability_results = {}
        for n_states in range(2, 9):
            print(f"  Testing stability with {n_states} states...")
            try:
                temp_analyzer = EconomicHMMAnalyzer()
                # Fit a temporary model to get stability
                temp_model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type="full",
                    n_iter=1000,
                    tol=1e-4,
                    random_state=42
                )
                temp_model.fit(X_scaled)
                
                stability = temp_analyzer.stability_check(
                    X_scaled, 
                    n_runs=5,  # Reduced for demo speed
                    n_hidden_states=n_states
                )
                if stability:
                    stability_results[n_states] = stability
            except:
                print(f"    Failed for {n_states} states")
                continue
        
        print("\nStability summary:")
        for n_states, results in stability_results.items():
            if results:
                print(f"  {n_states} states: CV = {results['cv']:.3f}")
    
    print(f"\n=== Extended validation completed ===")

if __name__ == "__main__":
    main()

