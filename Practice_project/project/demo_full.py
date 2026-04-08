import numpy as np
import pandas as pd
from economic_hmm_analysis import EconomicHMMAnalyzer

def main():
    print("=== Economic HMM Analysis Demo ===")
    
    # Initialize analyzer
    analyzer = EconomicHMMAnalyzer()
    
    # Load data
    print("\n1. Loading data...")
    df = analyzer.load_data("../data/synthetic_cashflow.csv")
    print(f"Data shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Define feature columns
    feature_columns = ["cash_inflow", "overdue_ratio", "avg_payment_delay_days", "contract_terms_index"]
    
    # Preprocess data
    print(f"\n2. Preprocessing data...")
    X_scaled = analyzer.preprocess_data(df, feature_columns)
    print(f"Preprocessed data shape: {X_scaled.shape}")
    
    # Define known events
    events = {
        "2016-03": "контрагентский шок",
        "2020-04": "кризис/пандемия", 
        "2022-02": "изменение внешних условий"
    }
    
    # Find optimal number of states
    print(f"\n3. Finding optimal number of states...")
    results_df, optimal_n_states = analyzer.find_optimal_states(X_scaled, min_states=2, max_states=8)
    
    if optimal_n_states is None:
        print("Could not find optimal number of states, using default (4)")
        optimal_n_states = 4
    
    # Fit model with optimal number of states
    print(f"\n4. Fitting HMM model with {optimal_n_states} hidden states...")
    model = analyzer.fit_model(X_scaled, n_hidden_states=optimal_n_states, covariance_type="full")
    
    # Print log-likelihood
    log_likelihood = model.score(X_scaled)
    print(f"Log-likelihood of final model: {log_likelihood:.2f}")
    
    # Get state sequence
    print(f"\n5. Decoding state sequence...")
    state_sequence = analyzer.decode_states(X_scaled)
    
    # Print state sequence with dates
    print("\nState sequence over time:")
    for i in range(0, len(df), 12):  # Print every 12 months (yearly)
        date = df.iloc[i]["date"]
        state = state_sequence[i]
        print(f"{date}: State {state}")
    
    # Print transition matrix
    print(f"\n6. Transition matrix:")
    print(analyzer.transition_matrix)
    
    # Interpret states
    print(f"\n7. Interpreting states...")
    interpretations = analyzer.interpret_states(df, feature_columns)
    
    # Perform temporal validation
    print(f"\n8. Performing temporal validation...")
    temp_validation_results = analyzer.temporal_validation(
        X_scaled, 
        train_ratio=0.8, 
        n_hidden_states=optimal_n_states
    )
    
    # Perform stability check
    print(f"\n9. Performing stability check...")
    stability_results = analyzer.stability_check(
        X_scaled, 
        n_runs=10,  # Reduced for demo speed
        n_hidden_states=optimal_n_states
    )
    
    # Compare with known events
    print(f"\n10. Comparing with known events...")
    df_with_states = analyzer.compare_with_events(df, events)
    
    # Visualize results (uncomment to show plots)
    print(f"\n11. Visualizing results...")
    print("Note: Visualization is commented out in this demo to avoid blocking execution.")
    # analyzer.visualize_results(df, feature_columns)
    
    print(f"\n=== Demo completed successfully! ===")

if __name__ == "__main__":
    main()

