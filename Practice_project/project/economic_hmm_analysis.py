import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

class EconomicHMMAnalyzer:
    """
    HMM Analyzer for identifying hidden causes of cash flow decline
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.transition_matrix = None
        self.state_sequence = None
        self.state_probabilities = None
        
    def load_data(self, path):
        """
        Load CSV data
        """
        df = pd.read_csv(path)
        return df
    
    def preprocess_data(self, df, columns):
        """
        Standardize the data
        """
        X = df[columns].values
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled
    
    def fit_model(self, X, n_hidden_states, covariance_type="full", n_iter=1000, tol=1e-4, random_state=42):
        """
        Fit Gaussian HMM model
        """
        model = hmm.GaussianHMM(
            n_components=n_hidden_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state
        )
        
        try:
            model.fit(X)
        except Exception as e:
            print(f"Error fitting model with {covariance_type} covariance. Trying \"diag\" instead.")
            model = hmm.GaussianHMM(
                n_components=n_hidden_states,
                covariance_type="diag",
                n_iter=n_iter,
                tol=tol,
                random_state=random_state
            )
            model.fit(X)
        
        self.model = model
        self.transition_matrix = model.transmat_
        
        # Get state sequence and probabilities
        state_sequence, _ = model.decode(X)
        self.state_sequence = state_sequence
        state_probabilities = model.predict_proba(X)
        self.state_probabilities = state_probabilities
        
        return model
    
    def decode_states(self, X):
        """
        Return the most likely state sequence
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit_model first.")
        
        state_sequence = self.model.predict(X)
        return state_sequence
    
    def find_optimal_states(self, X, min_states=2, max_states=8, covariance_type="full"):
        """
        Find optimal number of hidden states using AIC/BIC criteria
        """
        results = []
        
        for n_states in range(min_states, max_states + 1):
            try:
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type=covariance_type,
                    n_iter=1000,
                    tol=1e-4,
                    random_state=42
                )
                
                model.fit(X)
                log_likelihood = model.score(X)
                
                # Calculate number of parameters
                # For GaussianHMM: n_states^2 (transition matrix) + n_states * n_features (means) + n_states * n_features (variances for diag)
                # For full covariance: + n_states * n_features * (n_features + 1) / 2
                n_features = X.shape[1]
                if covariance_type == "full":
                    n_params = (n_states * n_states) + (n_states * n_features) + (n_states * n_features * (n_features + 1) // 2)
                else:  # diag
                    n_params = (n_states * n_states) + (n_states * n_features) + (n_states * n_features)
                
                # Check if model is at risk of overfitting
                n_observations = X.shape[0]
                if n_params >= n_observations:
                    print(f"Warning: Model with {n_states} states has {n_params} parameters, "
                          f"which is >= number of observations {n_observations}. Risk of overfitting.")
                
                aic = -2 * log_likelihood + 2 * n_params
                bic = -2 * log_likelihood + n_params * np.log(n_observations)
                
                results.append({
                    "n_states": n_states,
                    "log_likelihood": log_likelihood,
                    "aic": aic,
                    "bic": bic,
                    "n_params": n_params,
                    "n_observations": n_observations
                })
                
            except Exception as e:
                print(f"Error fitting model with {n_states} states: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        if results_df.empty:
            print("Could not fit models for any number of states.")
            return None, None
        
        # Select optimal number of states based on BIC (conservative)
        optimal_n_states = results_df.loc[results_df["bic"].idxmin(), "n_states"]
        
        print("Model selection results:")
        print(results_df.round(2))
        print(f"\nOptimal number of states based on BIC: {optimal_n_states}")
        
        return results_df, optimal_n_states
    
    def temporal_validation(self, X, train_ratio=0.8, n_hidden_states=4, covariance_type="full", random_state=42):
        """
        Perform temporal validation (train on early periods, test on later periods)
        """
        n_train = int(X.shape[0] * train_ratio)
        X_train = X[:n_train]
        X_test = X[n_train:]
        
        # Fit model on training data
        model_train = hmm.GaussianHMM(
            n_components=n_hidden_states,
            covariance_type=covariance_type,
            n_iter=1000,
            tol=1e-4,
            random_state=random_state
        )
        
        model_train.fit(X_train)
        
        # Score on test data
        log_likelihood_test = model_train.score(X_test)
        
        # Also fit model on full data and score test data
        model_full = hmm.GaussianHMM(
            n_components=n_hidden_states,
            covariance_type=covariance_type,
            n_iter=1000,
            tol=1e-4,
            random_state=random_state
        )
        
        model_full.fit(X)
        log_likelihood_test_full = model_full.score(X_test)
        
        print(f"Temporal validation results:")
        print(f"Log-likelihood on test set (model trained on train data): {log_likelihood_test:.2f}")
        print(f"Log-likelihood on test set (model trained on full data): {log_likelihood_test_full:.2f}")
        print(f"Improvement of full model over train-only model: {log_likelihood_test_full - log_likelihood_test:.2f}")
        
        return {
            "log_likelihood_test_train_only": log_likelihood_test,
            "log_likelihood_test_full_data": log_likelihood_test_full,
            "improvement": log_likelihood_test_full - log_likelihood_test
        }
    
    def stability_check(self, X, n_runs=20, n_hidden_states=4, covariance_type="full", n_iter=1000):
        """
        Check stability of model across multiple random initializations
        """
        log_likelihoods = []
        
        for i in range(n_runs):
            model = hmm.GaussianHMM(
                n_components=n_hidden_states,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=1e-4,
                random_state=i
            )
            
            try:
                model.fit(X)
                log_likelihood = model.score(X)
                log_likelihoods.append(log_likelihood)
            except Exception as e:
                print(f"Run {i+1} failed: {e}")
                continue
        
        if len(log_likelihoods) == 0:
            print("All stability runs failed.")
            return None
        
        mean_ll = np.mean(log_likelihoods)
        std_ll = np.std(log_likelihoods)
        cv = std_ll / abs(mean_ll) if mean_ll != 0 else 0  # Coefficient of variation
        
        print(f"Stability check results (over {len(log_likelihoods)} successful runs):")
        print(f"Mean log-likelihood: {mean_ll:.2f}")
        print(f"Std of log-likelihood: {std_ll:.2f}")
        print(f"Coefficient of variation: {cv:.3f}")
        
        if cv > 0.1:  # 10% threshold
            print("Warning: High coefficient of variation (>10%), model may be unstable.")
        
        return {
            "log_likelihoods": log_likelihoods,
            "mean": mean_ll,
            "std": std_ll,
            "cv": cv
        }
    
    def interpret_states(self, df, feature_columns):
        """
        Interpret hidden states based on feature means
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit_model first.")
        
        # Get original scale data for interpretation
        X_original = df[feature_columns].values
        X_scaled = self.scaler.transform(X_original)
        
        # Get state sequence
        state_sequence = self.decode_states(X_scaled)
        
        # Calculate mean values for each state
        interpretations = {}
        
        for state in range(self.model.n_components):
            state_mask = state_sequence == state
            if np.sum(state_mask) == 0:
                print(f"Warning: State {state} was never visited in the sequence.")
                continue
            
            state_data = df.loc[state_mask, feature_columns]
            state_means = state_data.mean()
            
            # Interpret based on feature values
            interpretation = f"State {state}: "
            
            # cash_inflow interpretation
            avg_cash = state_means["cash_inflow"]
            if avg_cash > 85:
                interpretation += "NORMAL - "
            elif avg_cash > 65:
                interpretation += "MODERATE DECLINE - "
            else:
                interpretation += "SIGNIFICANT DECLINE - "
            
            # overdue_ratio interpretation
            avg_overdue = state_means["overdue_ratio"]
            if avg_overdue < 0.1:
                interpretation += "Low overdue, "
            elif avg_overdue < 0.25:
                interpretation += "Moderate overdue, "
            else:
                interpretation += "High overdue, "
            
            # avg_payment_delay_days interpretation
            avg_delay = state_means["avg_payment_delay_days"]
            if avg_delay < 25:
                interpretation += "Quick payments, "
            elif avg_delay < 40:
                interpretation += "Moderate delays, "
            else:
                interpretation += "Significant delays, "
            
            # contract_terms_index interpretation
            avg_terms = state_means["contract_terms_index"]
            if avg_terms < 40:
                interpretation += "Favorable terms"
            elif avg_terms < 70:
                interpretation += "Moderate terms"
            else:
                interpretation += "Unfavorable terms"
            
            interpretations[state] = interpretation
            
            print(f"State {state}:")
            print(f"  - Cash inflow mean: {state_means['cash_inflow']:.2f}")
            print(f"  - Overdue ratio mean: {state_means['overdue_ratio']:.3f}")
            print(f"  - Payment delay mean: {state_means['avg_payment_delay_days']:.2f}")
            print(f"  - Contract terms mean: {state_means['contract_terms_index']:.2f}")
            print(f"  - Interpretation: {interpretation}")
            print()
        
        return interpretations
    
    def compare_with_events(self, df, events):
        """
        Compare state changes with known events
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit_model first.")
        
        # Get state sequence
        X_scaled = self.scaler.transform(df[["cash_inflow", "overdue_ratio", "avg_payment_delay_days", "contract_terms_index"]].values)
        state_sequence = self.decode_states(X_scaled)
        
        # Add state sequence to dataframe
        df_with_states = df.copy()
        df_with_states["predicted_state"] = state_sequence
        
        # Convert date column to datetime for comparison
        df_with_states["date"] = pd.to_datetime(df_with_states["date"])
        
        print("State changes around known events:")
        print("Date\t\tState\tEvent")
        print("-" * 40)
        
        for event_date_str, event_desc in events.items():
            event_date = pd.to_datetime(event_date_str)
            
            # Find state changes within ±1 month of event
            event_mask = (df_with_states["date"] >= event_date - pd.DateOffset(months=1)) & \
                         (df_with_states["date"] <= event_date + pd.DateOffset(months=1))
            
            event_df = df_with_states[event_mask].copy()
            event_df = event_df.sort_values("date")
            
            if len(event_df) > 0:
                for _, row in event_df.iterrows():
                    date_str = row["date"].strftime("%Y-%m")
                    state = row["predicted_state"]
                    if row["date"] == event_date.normalize():
                        print(f"{date_str}\t{state}\t{event_desc} (EVENT)")
                    else:
                        print(f"{date_str}\t{state}\t")
        
        return df_with_states
    
    def visualize_results(self, df, feature_columns):
        """
        Visualize HMM results
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit_model first.")
        
        # Get state sequence
        X_scaled = self.scaler.transform(df[feature_columns].values)
        state_sequence = self.decode_states(X_scaled)
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(feature_columns) + 2, 1, figsize=(15, 20))
        
        dates = pd.to_datetime(df["date"])
        
        # Plot each feature with state coloring
        for i, col in enumerate(feature_columns):
            axes[i].plot(dates, df[col], label=col, color="blue", alpha=0.7)
            
            # Color segments by state
            for state in range(self.model.n_components):
                state_mask = state_sequence == state
                if np.any(state_mask):
                    axes[i].scatter(dates[state_mask], df.loc[state_mask, col], 
                                   c=f"C{state}", label=f"State {state}", s=20, alpha=0.7)
            
            axes[i].set_title(f"{col} over time with states")
            axes[i].set_xlabel("Date")
            axes[i].set_ylabel(col)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Plot state sequence
        axes[len(feature_columns)].plot(dates, state_sequence, "o-", color="red")
        axes[len(feature_columns)].set_title("Predicted State Sequence")
        axes[len(feature_columns)].set_xlabel("Date")
        axes[len(feature_columns)].set_ylabel("State")
        axes[len(feature_columns)].grid(True, alpha=0.3)
        
        # Plot transition matrix heatmap
        im = axes[len(feature_columns)+1].imshow(self.transition_matrix, cmap="Blues", aspect="auto")
        axes[len(feature_columns)+1].set_title("Transition Matrix Heatmap")
        axes[len(feature_columns)+1].set_xlabel("To State")
        axes[len(feature_columns)+1].set_ylabel("From State")
        
        # Add text annotations to the heatmap
        for i in range(self.transition_matrix.shape[0]):
            for j in range(self.transition_matrix.shape[1]):
                axes[len(feature_columns)+1].text(j, i, f"{self.transition_matrix[i, j]:.2f}", 
                                                 ha="center", va="center", color="black")
        
        # Add colorbar
        plt.colorbar(im, ax=axes[len(feature_columns)+1])
        
        plt.tight_layout()
        plt.show()
        
        # Feature means by state (bar plot)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate mean values for each state
        feature_means = []
        for state in range(self.model.n_components):
            state_mask = state_sequence == state
            if np.sum(state_mask) > 0:
                state_means = df.loc[state_mask, feature_columns].mean()
                feature_means.append(state_means)
            else:
                feature_means.append(pd.Series([0] * len(feature_columns), index=feature_columns))
        
        feature_means_df = pd.DataFrame(feature_means).T
        feature_means_df.columns = [f"State {i}" for i in range(self.model.n_components)]
        
        feature_means_df.plot(kind="bar", ax=ax)
        ax.set_title("Feature Means by Predicted State")
        ax.set_xlabel("Features")
        ax.set_ylabel("Mean Value")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

