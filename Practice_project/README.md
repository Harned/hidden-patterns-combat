# HMM for Identifying Hidden Causes of Cash Flow Decline

## Project Structure
```
project/
├── economic_hmm_analysis.py     # Main HMM analyzer class
├── demo_full.py                # Full demonstration script
├── extended_validation_demo.py # Extended validation demo
└── requirements.txt            # Dependencies
```

## Dependencies
The project requires the following packages:
- numpy>=1.21.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- scikit-learn>=1.0.0
- hmmlearn>=0.2.8

Install them using:
```bash
pip install -r requirements.txt
```

## How to Run

### 1. Generate Synthetic Data (if not already done)
```bash
python generate_data.py
```

### 2. Run Full Demo
```bash
python -m project.demo_full
```

### 3. Run Extended Validation
```bash
python -m project.extended_validation_demo
```

## Key Features

### EconomicHMMAnalyzer Class
The main class provides the following functionality:

- `load_data(path)`: Load CSV data
- `preprocess_data(df, columns)`: Standardize features
- `fit_model(X, n_hidden_states, ...)`: Fit HMM model
- `decode_states(X)`: Get most likely state sequence
- `find_optimal_states(X, ...)`: Find optimal number of states using AIC/BIC
- `temporal_validation(X, ...)`: Perform temporal validation
- `stability_check(X, ...)`: Check model stability across runs
- `interpret_states(df, feature_columns)`: Interpret hidden states
- `compare_with_events(df, events)`: Compare state changes with known events
- `visualize_results(df, feature_columns)`: Create visualizations

## Synthetic Data Structure
The generated synthetic data has the following columns:
- `date`: Date in YYYY-MM format
- `cash_inflow`: Monthly cash inflows (in millions)
- `overdue_ratio`: Ratio of overdue debt (0-1)
- `avg_payment_delay_days`: Average payment delay in days
- `contract_terms_index`: Index of contract terms worsening (0-100)
- `true_state`: True hidden state (for validation)

## Hidden States
The model identifies 4 hidden economic states:
1. `NORMAL`: Stable inflows, low overdue, quick payments
2. `LIQUIDITY_DROP`: Reduced cash flow, increased delays
3. `TERMS_WORSENING`: Moderate decline, unfavorable contract terms
4. `DEBT_OVERDUE_CRISIS`: High overdue, significant delays, low inflows

## Example Output
When running the demo, you should see output similar to:

```
=== Economic HMM Analysis Demo ===

1. Loading data...
Data shape: (120, 6)

First few rows:
      date  cash_inflow  ...  contract_terms_index           true_state
0  2015-01   113.718621  ...             23.365628               NORMAL
1  2015-02   102.223698  ...             19.540237               NORMAL
2  2015-03    10.000000  ...             39.994459  DEBT_OVERDUE_CRISIS

[5 rows x 6 columns]

2. Preprocessing data...
Preprocessed data shape: (120, 4)

3. Finding optimal number of states...
Model selection results:
   n_states  log_likelihood      aic      bic  n_params  n_observations
0         2         -477.70  1019.39  1108.59        32             120
1         3         -465.29  1032.58  1174.74        51             120
2         4         -412.29   968.58  1169.28        72             120
3         5         -435.67  1061.35  1326.16        95             120

Optimal number of states based on BIC: 2

4. Fitting HMM model with 2 hidden states...
Log-likelihood of final model: -477.70

5. Decoding state sequence...

State sequence over time:
2015-01: State 0
2016-01: State 1
2017-01: State 0
2018-01: State 1
2019-01: State 0
2020-01: State 0
2021-01: State 1
2022-01: State 1
2023-01: State 1
2024-01: State 0

6. Transition matrix:
[[0.84516266 0.15483734]
 [0.16894738 0.83105262]]

7. Interpreting states...
State 0:
  - Cash inflow mean: 84.82
  - Overdue ratio mean: 0.065
  - Payment delay mean: 16.13
  - Contract terms mean: 31.74
  - Interpretation: State 0: MODERATE DECLINE - Low overdue, Quick payments, Favorable terms

State 1:
  - Cash inflow mean: 55.98
  - Overdue ratio mean: 0.255
  - Payment delay mean: 40.05
  - Contract terms mean: 33.27
  - Interpretation: State 1: SIGNIFICANT DECLINE - High overdue, Significant delays, Favorable terms

8. Performing temporal validation...
Temporal validation results:
Log-likelihood on test set (model trained on train data): -166.62
Log-likelihood on test set (model trained on full data): -122.92
Improvement of full model over train-only model: 43.70

9. Performing stability check...
Stability check results (over 10 successful runs):
Mean log-likelihood: -477.70
Std of log-likelihood: 0.00
Coefficient of variation: 0.000

10. Comparing with known events...
State changes around known events:
Date		State	Event
----------------------------------------
2016-02	1	
2016-03	1	контрагентский шок (EVENT)
2016-04	0	
2020-03	1	
2020-04	1	кризис/пандемия (EVENT)
2020-05	1	
2022-01	1	
2022-02	1	изменение внешних условий (EVENT)
2022-03	1	

11. Visualizing results...
Note: Visualization is commented out in this demo to avoid blocking execution.

=== Demo completed successfully! ===
```

## How the Synthetic Regimes Work

The synthetic data generator creates 4 distinct economic regimes:

1. **NORMAL**: High cash inflows (~100), low overdue ratios (~0.05), quick payments (~15 days), favorable contract terms (~20)
2. **LIQUIDITY_DROP**: Lower cash inflows (~70), higher overdue ratios (~0.15), longer delays (~35 days), moderate terms (~30)
3. **TERMS_WORSENING**: Moderate cash inflows (~85), moderate overdue (~0.08), moderate delays (~20 days), worse terms (~60)
4. **DEBT_OVERDUE_CRISIS**: Very low cash inflows (~50), very high overdue ratios (~0.40), very long delays (~50 days), moderate terms (~40)

The HMM model learns to identify these regimes by analyzing the relationships between the features. During periods around known events (like economic shocks), the model is more likely to transition between states, allowing it to detect changes in economic conditions that affect cash flow patterns.

The model uses Bayesian Information Criterion (BIC) to select the optimal number of states and includes various validation techniques to ensure robustness.
