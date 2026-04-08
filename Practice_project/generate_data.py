import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data():
    """
    Generate synthetic time series data for HMM analysis of cash flow decline reasons.
    
    Returns:
        DataFrame with columns: date, cash_inflow, overdue_ratio, avg_payment_delay_days, contract_terms_index, true_state
    """
    np.random.seed(42)
    
    # Parameters
    T = 120  # 10 years of monthly data
    start_date = datetime(2015, 1, 1)
    
    # Define hidden states
    states = ["NORMAL", "LIQUIDITY_DROP", "TERMS_WORSENING", "DEBT_OVERDUE_CRISIS"]
    n_states = len(states)
    
    # Define event dates that influence state transitions
    events = {
        "2016-03": "контрагентский шок",
        "2020-04": "кризис/пандемия", 
        "2022-02": "изменение внешних условий"
    }
    
    # Generate date range
    dates = pd.date_range(start=start_date, periods=T, freq="ME").strftime("%Y-%m").tolist()
    
    # Initialize arrays
    cash_inflow = np.zeros(T)
    overdue_ratio = np.zeros(T)
    avg_payment_delay_days = np.zeros(T)
    contract_terms_index = np.zeros(T)
    true_state = np.zeros(T, dtype=int)
    
    # Define state parameters (mean values for each state)
    state_params = {
        0: {  # NORMAL
            "cash_inflow_mean": 100,
            "cash_inflow_std": 10,
            "overdue_ratio_mean": 0.05,
            "overdue_ratio_std": 0.02,
            "avg_payment_delay_mean": 15,
            "avg_payment_delay_std": 5,
            "contract_terms_mean": 20,
            "contract_terms_std": 5
        },
        1: {  # LIQUIDITY_DROP
            "cash_inflow_mean": 70,
            "cash_inflow_std": 15,
            "overdue_ratio_mean": 0.15,
            "overdue_ratio_std": 0.05,
            "avg_payment_delay_mean": 35,
            "avg_payment_delay_std": 10,
            "contract_terms_mean": 30,
            "contract_terms_std": 8
        },
        2: {  # TERMS_WORSENING
            "cash_inflow_mean": 85,
            "cash_inflow_std": 12,
            "overdue_ratio_mean": 0.08,
            "overdue_ratio_std": 0.03,
            "avg_payment_delay_mean": 20,
            "avg_payment_delay_std": 7,
            "contract_terms_mean": 60,
            "contract_terms_std": 10
        },
        3: {  # DEBT_OVERDUE_CRISIS
            "cash_inflow_mean": 50,
            "cash_inflow_std": 20,
            "overdue_ratio_mean": 0.40,
            "overdue_ratio_std": 0.10,
            "avg_payment_delay_mean": 50,
            "avg_payment_delay_std": 15,
            "contract_terms_mean": 40,
            "contract_terms_std": 12
        }
    }
    
    # Define transition probabilities (higher probability to stay in same state)
    # But with some probability to transition to other states, especially around event dates
    base_transition_matrix = np.array([
        [0.8, 0.1, 0.05, 0.05],  # From NORMAL
        [0.1, 0.7, 0.1, 0.1],    # From LIQUIDITY_DROP
        [0.1, 0.1, 0.7, 0.1],    # From TERMS_WORSENING
        [0.1, 0.1, 0.1, 0.7]     # From DEBT_OVERDUE_CRISIS
    ])
    
    # Event dates as indices
    event_indices = {}
    for event_date, description in events.items():
        # Convert event date to index (approximate)
        year, month = map(int, event_date.split("-"))
        event_dt = datetime(year, month, 1)
        idx = (event_dt.year - start_date.year) * 12 + (event_dt.month - start_date.month)
        if 0 <= idx < T:
            event_indices[idx] = description
    
    # Generate state sequence with potential transitions around event dates
    current_state = 0  # Start in NORMAL state
    true_state[0] = current_state
    
    for t in range(1, T):
        # Check if this time point is near an event
        is_near_event = any(abs(t - event_idx) <= 2 for event_idx in event_indices.keys())
        
        if is_near_event:
            # Higher probability of state change near events
            # Increase probability of moving to crisis states
            event_transition = base_transition_matrix.copy()
            # Increase probability of moving to crisis states
            event_transition[current_state, 1] = 0.3  # LIQUIDITY_DROP
            event_transition[current_state, 2] = 0.2  # TERMS_WORSENING
            event_transition[current_state, 3] = 0.3  # DEBT_OVERDUE_CRISIS
            event_transition[current_state, 0] = 0.2  # Stay in NORMAL
            # Normalize the row
            event_transition[current_state, :] = event_transition[current_state, :] / event_transition[current_state, :].sum()
            
            current_state = np.random.choice(n_states, p=event_transition[current_state, :])
        else:
            # Use base transition probabilities
            current_state = np.random.choice(n_states, p=base_transition_matrix[current_state, :])
        
        true_state[t] = current_state
    
    # Generate observations based on states
    for t in range(T):
        state = true_state[t]
        params = state_params[state]
        
        # Add some seasonality and trend
        seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * t / 12)  # Yearly seasonality
        trend_factor = 1 - 0.002 * t  # Slight downward trend
        
        cash_inflow[t] = np.random.normal(
            params["cash_inflow_mean"] * seasonal_factor * trend_factor,
            params["cash_inflow_std"]
        )
        # Ensure cash inflow is positive
        cash_inflow[t] = max(cash_inflow[t], 10)
        
        overdue_ratio[t] = np.random.normal(
            params["overdue_ratio_mean"] * seasonal_factor,
            params["overdue_ratio_std"]
        )
        # Ensure overdue ratio is between 0 and 1
        overdue_ratio[t] = np.clip(overdue_ratio[t], 0, 1)
        
        avg_payment_delay_days[t] = np.random.normal(
            params["avg_payment_delay_mean"] * seasonal_factor,
            params["avg_payment_delay_std"]
        )
        # Ensure delay is positive
        avg_payment_delay_days[t] = max(avg_payment_delay_days[t], 1)
        
        contract_terms_index[t] = np.random.normal(
            params["contract_terms_mean"] * seasonal_factor,
            params["contract_terms_std"]
        )
        # Ensure index is between 0 and 100
        contract_terms_index[t] = np.clip(contract_terms_index[t], 0, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        "date": dates,
        "cash_inflow": cash_inflow,
        "overdue_ratio": overdue_ratio,
        "avg_payment_delay_days": avg_payment_delay_days,
        "contract_terms_index": contract_terms_index,
        "true_state": [states[i] for i in true_state]
    })
    
    return df, events

# Generate and save data
if __name__ == "__main__":
    df, events = generate_synthetic_data()
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    df.to_csv("data/synthetic_cashflow.csv", index=False)
    
    print("Synthetic data generated successfully!")
    print(f"Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head(10))
    
    print(f"\nState distribution:")
    print(df["true_state"].value_counts())
    
    print(f"\nEvents:")
    for date, event in events.items():
        print(f"  {date}: {event}")

