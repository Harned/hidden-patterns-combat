from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import default_state_colors, ensure_output_path, get_pyplot


def _state_col(df: pd.DataFrame) -> str:
    if "hidden_state_name" in df.columns:
        return "hidden_state_name"
    if "hidden_state" in df.columns:
        return "hidden_state"
    raise ValueError("No hidden-state column found in analysis dataframe.")


def _sequence_col(df: pd.DataFrame) -> str | None:
    for candidate in ("sequence_id", "metadata__sheet", "_sheet"):
        if candidate in df.columns:
            return candidate
    return None


def plot_hidden_state_sequence(
    analysis_df: pd.DataFrame,
    out_path: str | Path,
    state_col: str | None = None,
    episode_axis_label: str = "Episode index",
) -> Path:
    state_col = state_col or _state_col(analysis_df)
    out = ensure_output_path(out_path)

    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(13, 3.5))
    x = range(len(analysis_df))

    state_series = analysis_df[state_col].astype(str)
    categories = pd.Categorical(state_series)
    y = categories.codes
    labels = list(categories.categories)
    colors = default_state_colors(len(labels))

    ax.scatter(x, y, c=[colors[i] for i in y], s=8, alpha=0.9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Hidden-State Sequence Over Episodes")
    ax.set_xlabel(episode_axis_label)
    ax.set_ylabel("Hidden state")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_state_probability_profile(
    analysis_df: pd.DataFrame,
    out_path: str | Path,
    prob_prefix: str = "p_state_",
) -> Path:
    out = ensure_output_path(out_path)
    prob_cols = [c for c in analysis_df.columns if c.startswith(prob_prefix)]
    if not prob_cols:
        raise ValueError("No state-probability columns found.")

    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(13, 4))

    colors = default_state_colors(len(prob_cols))
    for i, col in enumerate(sorted(prob_cols)):
        ax.plot(analysis_df[col].values, label=col.replace(prob_prefix, "state_"), color=colors[i], lw=1.5)

    ax.set_title("State Probability Profile")
    ax.set_xlabel("Episode index")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.legend(ncol=min(4, len(prob_cols)), fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_athlete_comparative_profile(
    analysis_df: pd.DataFrame,
    out_path: str | Path,
    athlete_col: str = "athlete_name",
    result_col: str = "observed_result",
    top_n: int = 10,
) -> Path | None:
    if athlete_col not in analysis_df.columns:
        return None

    work = analysis_df.copy()
    state_col = _state_col(work)

    counts = work.groupby(athlete_col).size().sort_values(ascending=False)
    athletes = counts.head(top_n).index.tolist()
    work = work[work[athlete_col].isin(athletes)]

    quality = work.groupby(athlete_col)[result_col].mean().rename("mean_result")
    state_share = (
        work.groupby([athlete_col, state_col]).size().unstack(fill_value=0)
        .div(work.groupby(athlete_col).size(), axis=0)
    )

    out = ensure_output_path(out_path)
    plt = get_pyplot()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 2]})

    quality.sort_values().plot(kind="barh", ax=ax1, color="#1f77b4")
    ax1.set_title("Athlete Mean Result")
    ax1.set_xlabel("Mean observed result")
    ax1.set_ylabel("Athlete")

    state_share = state_share.loc[quality.index]
    bottom = pd.Series(0.0, index=state_share.index)
    colors = default_state_colors(len(state_share.columns))
    for i, col in enumerate(state_share.columns):
        ax2.barh(state_share.index, state_share[col], left=bottom, label=str(col), color=colors[i], alpha=0.9)
        bottom = bottom + state_share[col]

    ax2.set_title("State-Scenario Composition by Athlete")
    ax2.set_xlabel("Share of episodes")
    ax2.set_xlim(0.0, 1.0)
    ax2.legend(title="State", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_success_failure_scenarios(
    analysis_df: pd.DataFrame,
    out_path: str | Path,
    result_col: str = "observed_result",
    success_threshold: float = 0.0,
) -> Path:
    state_col = _state_col(analysis_df)

    work = analysis_df.copy()
    work["success_flag"] = work[result_col].fillna(0.0) > success_threshold
    freq = (
        work.groupby([state_col, "success_flag"]).size().unstack(fill_value=0)
        .rename(columns={False: "unsuccessful", True: "successful"})
    )
    freq = freq.reindex(columns=["successful", "unsuccessful"], fill_value=0)

    out = ensure_output_path(out_path)
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(11, 4.5))

    freq[["successful", "unsuccessful"]].plot(kind="bar", ax=ax, color=["#2ca02c", "#d62728"])
    ax.set_title("Successful vs Unsuccessful Scenario Frequencies")
    ax.set_xlabel("Hidden-state scenario")
    ax.set_ylabel("Episode count")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_transition_distribution(
    analysis_df: pd.DataFrame,
    out_path: str | Path,
    top_k: int = 12,
) -> Path:
    state_col = _state_col(analysis_df)
    seq_col = _sequence_col(analysis_df)

    if seq_col is None:
        seq_ids = pd.Series(["all"] * len(analysis_df), index=analysis_df.index)
    else:
        seq_ids = analysis_df[seq_col].astype(str)

    transitions: dict[str, int] = {}
    for _, group in analysis_df.assign(_seq=seq_ids).groupby("_seq", sort=False):
        states = group[state_col].astype(str).tolist()
        for i in range(len(states) - 1):
            tr = f"{states[i]} -> {states[i+1]}"
            transitions[tr] = transitions.get(tr, 0) + 1

    sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:top_k]
    labels = [x[0] for x in sorted_transitions][::-1]
    values = [x[1] for x in sorted_transitions][::-1]

    out = ensure_output_path(out_path)
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.barh(labels, values, color="#1f77b4")
    ax.set_title("Most Frequent Hidden-State Transitions")
    ax.set_xlabel("Transition count")
    ax.set_ylabel("Transition")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def create_analysis_charts(
    analysis_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}

    outputs["hidden_state_sequence"] = str(
        plot_hidden_state_sequence(analysis_df, out / "hidden_state_sequence.png")
    )
    outputs["state_probability_profile"] = str(
        plot_state_probability_profile(analysis_df, out / "state_probability_profile.png")
    )

    athlete_plot = plot_athlete_comparative_profile(
        analysis_df,
        out / "athlete_comparative_profile.png",
    )
    if athlete_plot:
        outputs["athlete_comparative_profile"] = str(athlete_plot)

    outputs["scenario_success_frequencies"] = str(
        plot_success_failure_scenarios(analysis_df, out / "scenario_success_frequencies.png")
    )
    outputs["transition_distribution"] = str(
        plot_transition_distribution(analysis_df, out / "transition_distribution.png")
    )

    return outputs
