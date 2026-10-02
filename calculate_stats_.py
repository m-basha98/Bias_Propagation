from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


# ============================================================
# PATHS
# ============================================================

INPUT = Path("metrics/propagation_metrics.csv")

OUT = Path("tables")
DESC = OUT / "descriptive"
STATS = OUT / "statistics"
MODEL_SUMMARIES = OUT / "model_summaries"

for directory in [DESC, STATS, MODEL_SUMMARIES]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

SCOPES = ["overall", "target"]

METRIC_COLUMNS = {
    "overall": {
        "survival": "overall_survival",
        "emergence": "overall_emergence",
        "amplification": "overall_amplification",
        "mitigation": "overall_mitigation",
    },
    "target": {
        "survival": "target_survival",
        "emergence": "target_emergence",
        "amplification": "target_amplification",
        "mitigation": "target_mitigation",
    },
}

STAGES = [
    "requirements",
    "code_generation",
    "refactoring",
    "testing",
    "documentation",
]

TRANSITIONS = [
    "Requirements → Code Generation",
    "Code Generation → Refactoring",
    "Refactoring → Testing",
    "Testing → Documentation",
]

PROPAGATION_TRANSITIONS = [
    "Code Generation → Refactoring",
    "Refactoring → Testing",
    "Testing → Documentation",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_expression(value):
    value = normalize_text(value).lower()

    if value == "explicit":
        return "Explicit"
    if value == "implicit":
        return "Implicit"

    return value.title()


def transition_label(row):
    start = (
        normalize_text(row["from_stage"])
        .replace("_", " ")
        .title()
    )

    end = (
        normalize_text(row["to_stage"])
        .replace("_", " ")
        .title()
    )

    return f"{start} → {end}"


def mean_sd(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return ""

    if len(values) == 1:
        return f"{values.iloc[0]:.3f}"

    return (
        f"{values.mean():.3f} ± "
        f"{values.std(ddof=1):.3f}"
    )


def holm_adjust(p_values):
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    if len(p_values) == 0:
        return np.array([])

    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0

    for rank, index in enumerate(order):
        adjusted_value = (
            len(p_values) - rank
        ) * p_values[index]

        running_max = max(
            running_max,
            adjusted_value,
        )

        adjusted[index] = min(
            running_max,
            1.0,
        )

    return adjusted


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Could not find {INPUT}. "
        f"Run calculate_bias_metrics_updated.py first."
    )

df = pd.read_csv(INPUT)

df.columns = [
    column.strip()
    for column in df.columns
]


# ============================================================
# NORMALIZE VARIABLES
# ============================================================

categorical_columns = [
    "bias_type",
    "bias_expression",
    "domain",
    "sub_domain",
    "attribute",
    "cgt",
    "model",
    "from_stage",
    "to_stage",
    "transition_kind",
]

for column in categorical_columns:
    if column in df.columns:
        df[column] = df[column].map(normalize_text)

df["bias_expression"] = (
    df["bias_expression"]
    .map(normalize_expression)
)

df["run_number"] = pd.to_numeric(
    df["run_number"],
    errors="coerce",
)

numeric_columns = [
    "overall_previous_count",
    "overall_current_count",
    "overall_presence_previous",
    "overall_presence_current",
    "overall_survival",
    "overall_emergence",
    "overall_amplification",
    "overall_mitigation",
    "target_previous_count",
    "target_current_count",
    "target_presence_previous",
    "target_presence_current",
    "target_survival",
    "target_emergence",
    "target_amplification",
    "target_mitigation",
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )


# ============================================================
# ANALYSIS DATA
# ============================================================

analysis = df[
    df["bias_type"].str.lower().ne("neutral")
].copy()

analysis["transition"] = (
    analysis.apply(
        transition_label,
        axis=1,
    )
)

analysis["transition"] = pd.Categorical(
    analysis["transition"],
    categories=TRANSITIONS,
    ordered=True,
)

# Main inferential propagation analyses use only
# artifact-to-artifact transitions. Requirements -> Code
# is a target-bias introduction/realization step and is
# reported descriptively rather than mixed into the
# stage-to-stage GEE comparisons.

propagation_analysis = analysis[
    analysis["transition"].isin(
        PROPAGATION_TRANSITIONS
    )
].copy()


# ============================================================
# DATA CHECKS
# ============================================================

print("\n========================================")
print("DATASET")
print("========================================")

print("All non-neutral metric rows:", len(analysis))
print(
    "Propagation rows:",
    len(propagation_analysis),
)
print(
    "Experiments:",
    analysis["experiment_id"].nunique(),
)
print(
    "Runs:",
    sorted(
        analysis["run_number"]
        .dropna()
        .unique()
        .tolist()
    ),
)

print("\nBias expression:")
print(
    analysis["bias_expression"]
    .value_counts(dropna=False)
)

print("\nBias types:")
print(
    analysis["bias_type"]
    .value_counts(dropna=False)
)

print("\nTransitions:")
print(
    analysis["transition"]
    .value_counts()
    .reindex(TRANSITIONS, fill_value=0)
)


# ============================================================
# METRIC COMPLETENESS
# ============================================================

print("\n========================================")
print("METRIC COMPLETENESS")
print("========================================")

completeness_rows = []

for scope in SCOPES:
    for metric, column in METRIC_COLUMNS[scope].items():
        for transition in TRANSITIONS:
            subset = analysis[
                analysis["transition"]
                == transition
            ]

            completeness_rows.append({
                "scope": scope,
                "metric": metric,
                "column": column,
                "transition": transition,
                "N_nonmissing": subset[column].notna().sum()
                if column in subset.columns
                else 0,
                "N_total": len(subset),
            })

completeness = pd.DataFrame(completeness_rows)

completeness.to_csv(
    DESC / "metric_completeness.csv",
    index=False,
)

print(
    completeness.to_string(index=False)
)


# ============================================================
# DESCRIPTIVE TABLES
# ============================================================

def summarize(data, group_columns, scope):
    rows = []

    if data.empty:
        return pd.DataFrame()

    metric_map = METRIC_COLUMNS[scope]

    for keys, group in data.groupby(
        group_columns,
        dropna=False,
        observed=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_columns, keys))

        row["scope"] = scope
        row["N_rows"] = len(group)
        row["N_runs"] = group["run_number"].nunique()
        row["N_experiments"] = (
            group["experiment_id"].nunique()
        )

        for metric_name, column in metric_map.items():
            if column in group.columns:
                row[metric_name] = mean_sd(
                    group[column]
                )

        rows.append(row)

    return pd.DataFrame(rows)


def save_summary(
    data,
    group_columns,
    filename,
):
    all_tables = []

    for scope in SCOPES:
        table = summarize(
            data,
            group_columns,
            scope,
        )

        if not table.empty:
            all_tables.append(table)

    if all_tables:
        result = pd.concat(
            all_tables,
            ignore_index=True,
        )
    else:
        result = pd.DataFrame()

    result.to_csv(
        DESC / filename,
        index=False,
    )

    return result


run1 = analysis[
    analysis["run_number"] == 1
].copy()

RQ1_GROUPS = [
    "bias_expression",
    "transition",
]

RQ2_GROUPS = [
    "bias_type",
    "domain",
    "bias_expression",
]

MODEL_COLUMN = (
    "cgt"
    if "cgt" in analysis.columns
    else "model"
)

RQ4_GROUPS = [
    MODEL_COLUMN,
    "bias_expression",
]

save_summary(
    run1,
    RQ1_GROUPS,
    "RQ1_run1_both_scopes.csv",
)

save_summary(
    analysis,
    RQ1_GROUPS,
    "RQ1_all_runs_both_scopes.csv",
)

save_summary(
    run1,
    RQ2_GROUPS,
    "RQ2_run1_both_scopes.csv",
)

save_summary(
    analysis,
    RQ2_GROUPS,
    "RQ2_all_runs_both_scopes.csv",
)

save_summary(
    run1,
    RQ1_GROUPS,
    "RQ3_run1_both_scopes.csv",
)

save_summary(
    analysis,
    RQ1_GROUPS,
    "RQ3_all_runs_both_scopes.csv",
)

save_summary(
    run1,
    RQ4_GROUPS,
    "RQ4_run1_both_scopes.csv",
)

save_summary(
    analysis,
    RQ4_GROUPS,
    "RQ4_all_runs_both_scopes.csv",
)


# ============================================================
# TARGET INTRODUCTION: REQUIREMENTS -> CODE
# ============================================================

target_introduction = analysis[
    analysis["transition"]
    == "Requirements → Code Generation"
].copy()

if not target_introduction.empty:
    introduction_table = (
        target_introduction
        .groupby(
            [
                "bias_expression",
                "bias_type",
                "domain",
            ],
            dropna=False,
        )
        .agg(
            N_rows=("experiment_id", "size"),
            N_experiments=(
                "experiment_id",
                "nunique",
            ),
            target_code_count_mean=(
                "target_current_count",
                "mean",
            ),
            target_realization_rate=(
                "target_survival",
                "mean",
            ),
        )
        .reset_index()
    )

    introduction_table.to_csv(
        DESC / "target_introduction_requirements_to_code.csv",
        index=False,
    )


# ============================================================
# REPEATABILITY
# ============================================================

def calculate_delta(later_run, scope):
    run_a = propagation_analysis[
        propagation_analysis["run_number"] == 1
    ].copy()

    run_b = propagation_analysis[
        propagation_analysis["run_number"] == later_run
    ].copy()

    merge_columns = [
        "experiment_id",
        "transition",
    ]

    for column in [
        "bias_type",
        "bias_expression",
        "attribute",
        "domain",
        "sub_domain",
        "cgt",
        "model",
    ]:
        if column in propagation_analysis.columns:
            merge_columns.append(column)

    metric_columns = list(
        METRIC_COLUMNS[scope].values()
    )

    run_a = run_a[
        merge_columns + metric_columns
    ]

    run_b = run_b[
        merge_columns + metric_columns
    ]

    merged = run_a.merge(
        run_b,
        on=merge_columns,
        suffixes=(
            "_run1",
            f"_run{later_run}",
        ),
        how="inner",
    )

    for metric_name, metric_column in (
        METRIC_COLUMNS[scope].items()
    ):
        c1 = f"{metric_column}_run1"
        c2 = f"{metric_column}_run{later_run}"

        if c1 not in merged.columns or c2 not in merged.columns:
            continue

        merged[f"delta_{metric_name}"] = (
            pd.to_numeric(
                merged[c2],
                errors="coerce",
            )
            -
            pd.to_numeric(
                merged[c1],
                errors="coerce",
            )
        )

    return merged


repeat_stats = []

for scope in SCOPES:
    for later_run in [2, 3]:
        delta = calculate_delta(
            later_run,
            scope,
        )

        for metric_name in METRIC_COLUMNS[scope]:
            column = f"delta_{metric_name}"

            if column not in delta.columns:
                continue

            values = pd.to_numeric(
                delta[column],
                errors="coerce",
            ).dropna()

            if len(values) < 2:
                continue

            if np.allclose(values, 0):
                statistic = 0.0
                p_value = 1.0
            else:
                test = stats.wilcoxon(
                    values,
                    alternative="two-sided",
                )
                statistic = test.statistic
                p_value = test.pvalue

            repeat_stats.append({
                "scope": scope,
                "comparison": (
                    f"Run {later_run} vs Run 1"
                ),
                "metric": metric_name,
                "N_pairs": len(values),
                "mean_delta": values.mean(),
                "median_delta": values.median(),
                "wilcoxon_W": statistic,
                "p_value": p_value,
            })

repeat_stats = pd.DataFrame(repeat_stats)

if not repeat_stats.empty:
    repeat_stats["p_holm"] = holm_adjust(
        repeat_stats["p_value"].values
    )

    repeat_stats["significant"] = (
        repeat_stats["p_holm"] < 0.05
    )

    repeat_stats.to_csv(
        STATS / "repeatability_tests_both_scopes.csv",
        index=False,
    )


# ============================================================
# BINOMIAL DATA
# ============================================================

def prepare_rate_data(data, scope, metric_name):
    d = data.copy()

    metric_column = (
        METRIC_COLUMNS[scope][metric_name]
    )

    d["rate"] = pd.to_numeric(
        d[metric_column],
        errors="coerce",
    )

    if metric_name == "emergence":
        trials_column = (
            "overall_current_count"
            if scope == "overall"
            else "target_current_count"
        )
    else:
        trials_column = (
            "overall_previous_count"
            if scope == "overall"
            else "target_previous_count"
        )

    d["trials"] = pd.to_numeric(
        d[trials_column],
        errors="coerce",
    )

    d["successes"] = (
        d["rate"] * d["trials"]
    )

    d = d.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    d = d.dropna(
        subset=[
            "rate",
            "trials",
            "successes",
            "experiment_id",
        ]
    ).copy()

    d = d[d["trials"] > 0].copy()

    d["rate"] = d["rate"].clip(0, 1)
    d["successes"] = d["successes"].clip(
        0,
        d["trials"],
    )

    return d

# ============================================================
# NEUTRAL BASELINE ANALYSIS
# ============================================================
#
# Neutral conditions are retained separately as a baseline.
# They are not included in the explicit-vs-implicit analyses
# above. The neutral condition allows us to assess whether
# overall bias observed in biased workflows also occurs when
# the requirement contains no intended bias.
#
# Neutral conditions are used only for OVERALL bias metrics.
# Target-bias metrics are not applicable to neutral tasks.
# ============================================================

neutral_analysis = df[
    df["bias_type"].str.lower().eq("neutral")
].copy()

neutral_analysis["transition"] = (
    neutral_analysis.apply(
        transition_label,
        axis=1,
    )
)

neutral_analysis["transition"] = pd.Categorical(
    neutral_analysis["transition"],
    categories=TRANSITIONS,
    ordered=True,
)

neutral_propagation = neutral_analysis[
    neutral_analysis["transition"].isin(
        PROPAGATION_TRANSITIONS
    )
].copy()


print("\n========================================")
print("NEUTRAL BASELINE")
print("========================================")

print(
    "Neutral metric rows:",
    len(neutral_analysis),
)

print(
    "Neutral propagation rows:",
    len(neutral_propagation),
)

print(
    "Neutral experiments:",
    neutral_analysis["experiment_id"].nunique(),
)

print("\nNeutral transitions:")
print(
    neutral_analysis["transition"]
    .value_counts()
    .reindex(
        TRANSITIONS,
        fill_value=0,
    )
)


# ============================================================
# NEUTRAL DESCRIPTIVE TABLE
# ============================================================

neutral_descriptive = summarize(
    neutral_propagation,
    ["transition"],
    "overall",
)

if not neutral_descriptive.empty:
    neutral_descriptive.to_csv(
        DESC / "neutral_overall_propagation.csv",
        index=False,
    )


# ============================================================
# BIASED VS NEUTRAL OVERALL COMPARISON
# ============================================================
#
# This analysis compares overall bias metrics in biased
# requirements against their neutral counterparts.
#
# The comparison is performed separately from the main
# explicit-vs-implicit propagation analysis.
#
# "condition" is:
#   Neutral = neutral requirement
#   Biased  = explicit or implicit requirement
#
# The model includes condition, transition, and their
# interaction. This tests whether the overall propagation
# pattern differs between biased and neutral tasks.
# ============================================================

biased_for_baseline = propagation_analysis.copy()
biased_for_baseline["condition"] = "Biased"

neutral_for_baseline = neutral_propagation.copy()
neutral_for_baseline["condition"] = "Neutral"

baseline_analysis = pd.concat(
    [
        biased_for_baseline,
        neutral_for_baseline,
    ],
    ignore_index=True,
)

baseline_analysis["condition"] = pd.Categorical(
    baseline_analysis["condition"],
    categories=["Neutral", "Biased"],
    ordered=True,
)


# ============================================================
# BASELINE DESCRIPTIVES
# ============================================================

baseline_summary = summarize(
    baseline_analysis,
    ["condition", "transition"],
    "overall",
)

if not baseline_summary.empty:
    baseline_summary.to_csv(
        DESC / "neutral_vs_biased_overall.csv",
        index=False,
    )


# ============================================================
# BASELINE GEE
# ============================================================

baseline_results = []


def fit_baseline_binomial_gee(
    data,
    metric_name,
):
    d = prepare_rate_data(
        data,
        "overall",
        metric_name,
    )

    if len(d) < 10:
        print(
            f"Skipping neutral-vs-biased / "
            f"{metric_name}: insufficient data."
        )
        return

    print()
    print(
        f"Neutral vs Biased GEE "
        f"(overall / {metric_name}):"
    )

    print(
        d["condition"]
        .value_counts()
        .to_string()
    )

    print(
        d["transition"]
        .value_counts()
        .reindex(
            PROPAGATION_TRANSITIONS,
            fill_value=0,
        )
        .to_string()
    )

    try:
        model = smf.gee(
            "rate ~ C(condition) * C(transition)",
            groups="experiment_id",
            data=d,
            family=sm.families.Binomial(),
            weights=d["trials"],
            cov_struct=sm.cov_struct.Exchangeable(),
        )

        result = model.fit()

        summary_file = (
            MODEL_SUMMARIES
            / f"neutral_vs_biased_overall_{metric_name}.txt"
        )

        with open(
            summary_file,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(str(result.summary()))

        for term in result.params.index:
            baseline_results.append({
                "analysis": "neutral_vs_biased",
                "scope": "overall",
                "metric": metric_name,
                "term": term,
                "estimate": result.params[term],
                "SE": result.bse[term],
                "p_value": result.pvalues[term],
                "N": len(d),
                "N_experiments": (
                    d["experiment_id"].nunique()
                ),
            })

        print(
            f"✓ neutral_vs_biased / "
            f"overall / {metric_name}"
        )

    except Exception as error:
        print(
            f"FAILED neutral_vs_biased / "
            f"overall / {metric_name}: {error}"
        )


def fit_baseline_amplification_gee(data):

    d = data.copy()

    d["response"] = pd.to_numeric(
        d["overall_amplification"],
        errors="coerce",
    )

    d = d.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    d = d.dropna(
        subset=[
            "response",
            "experiment_id",
        ]
    ).copy()

    # Log amplification, consistent with the main analysis.
    d = d[d["response"] > 0].copy()

    print()
    print(
        "Neutral vs Biased GEE "
        "(overall / amplification):"
    )

    print(
        d["condition"]
        .value_counts()
        .to_string()
    )

    print(
        d["transition"]
        .value_counts()
        .reindex(
            PROPAGATION_TRANSITIONS,
            fill_value=0,
        )
        .to_string()
    )

    if len(d) < 10:
        print(
            "Skipping neutral-vs-biased / "
            "amplification: insufficient data."
        )
        return

    d["response"] = np.log(
        d["response"]
    )

    try:
        model = smf.gee(
            "response ~ C(condition) * C(transition)",
            groups="experiment_id",
            data=d,
            family=sm.families.Gaussian(),
            cov_struct=sm.cov_struct.Exchangeable(),
        )

        result = model.fit()

        summary_file = (
            MODEL_SUMMARIES
            / "neutral_vs_biased_overall_amplification.txt"
        )

        with open(
            summary_file,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(str(result.summary()))

        for term in result.params.index:
            baseline_results.append({
                "analysis": "neutral_vs_biased",
                "scope": "overall",
                "metric": "amplification",
                "term": term,
                "estimate": result.params[term],
                "SE": result.bse[term],
                "p_value": result.pvalues[term],
                "N": len(d),
                "N_experiments": (
                    d["experiment_id"].nunique()
                ),
            })

        print(
            "✓ neutral_vs_biased / "
            "overall / amplification"
        )

    except Exception as error:
        print(
            "FAILED neutral_vs_biased / "
            "overall / amplification:",
            error,
        )


# Run neutral-vs-biased models.
for metric_name in [
    "survival",
    "emergence",
    "mitigation",
]:
    fit_baseline_binomial_gee(
        baseline_analysis,
        metric_name,
    )

fit_baseline_amplification_gee(
    baseline_analysis
)


# ============================================================
# SAVE NEUTRAL BASELINE RESULTS
# ============================================================

baseline_results = pd.DataFrame(
    baseline_results
)

if not baseline_results.empty:

    baseline_results["p_holm"] = np.nan

    for metric_name, indices in baseline_results.groupby(
    "metric"
    ).groups.items():
        p_values = pd.to_numeric(
            baseline_results.loc[
                indices,
                "p_value",
            ],
            errors="coerce",
        ).values

        valid = ~np.isnan(p_values)

        if valid.any():

            adjusted = holm_adjust(
                p_values[valid]
            )

            valid_indices = (
                np.asarray(indices)[valid]
            )

            baseline_results.loc[
                valid_indices,
                "p_holm",
            ] = adjusted

    baseline_results["significant"] = (
        baseline_results["p_holm"] < 0.05
    )

    baseline_results.to_csv(
        STATS / "neutral_vs_biased_overall.csv",
        index=False,
    )

    print(
        "\nNeutral baseline statistical results:"
    )

    print(
        baseline_results[
            [
                "metric",
                "term",
                "estimate",
                "SE",
                "p_value",
                "p_holm",
                "significant",
            ]
        ].to_string(index=False)
    )


# ============================================================
# GEE
# ============================================================

model_results = []


def print_gee_transition_counts(data, scope, metric):
    print()
    print(
        f"Transitions entering GEE "
        f"({scope} / {metric}):"
    )

    print(
        data["transition"]
        .value_counts()
        .reindex(
            PROPAGATION_TRANSITIONS,
            fill_value=0,
        )
        .to_string()
    )


def fit_binomial_gee(
    data,
    scope,
    metric_name,
    formula,
    model_name,
    output_prefix,
):
    d = prepare_rate_data(
        data,
        scope,
        metric_name,
    )

    print_gee_transition_counts(
        d,
        scope,
        metric_name,
    )

    if len(d) < 10:
        print(
            f"Skipping {model_name} / "
            f"{scope} / {metric_name}: "
            f"insufficient data."
        )
        return

    try:
        model = smf.gee(
            formula,
            groups="experiment_id",
            data=d,
            family=sm.families.Binomial(),
            weights=d["trials"],
            cov_struct=sm.cov_struct.Exchangeable(),
        )

        result = model.fit()

        summary_file = (
            MODEL_SUMMARIES
            / f"{output_prefix}_{scope}_{metric_name}.txt"
        )

        with open(
            summary_file,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(str(result.summary()))

        for term in result.params.index:
            model_results.append({
                "scope": scope,
                "model": model_name,
                "metric": metric_name,
                "term": term,
                "estimate": result.params[term],
                "SE": result.bse[term],
                "p_value": result.pvalues[term],
                "N": len(d),
                "N_experiments": (
                    d["experiment_id"].nunique()
                ),
            })

        print(
            f"✓ {model_name} / "
            f"{scope} / {metric_name}"
        )

    except Exception as error:
        print(
            f"FAILED {model_name} / "
            f"{scope} / {metric_name}: {error}"
        )


def fit_amplification_gee(
    data,
    scope,
    formula,
    model_name,
    output_prefix,
):
    d = data.copy()

    d["response"] = pd.to_numeric(
        d[
            METRIC_COLUMNS[scope][
                "amplification"
            ]
        ],
        errors="coerce",
    )

    d = d.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    d = d.dropna(
        subset=[
            "response",
            "experiment_id",
        ]
    ).copy()

    print()
    print(
        f"Amplification rows "
        f"({scope}) before positive filtering:"
    )

    print(
        d["transition"]
        .value_counts()
        .reindex(
            PROPAGATION_TRANSITIONS,
            fill_value=0,
        )
        .to_string()
    )

    d = d[d["response"] > 0].copy()

    print()
    print(
        f"Amplification rows entering GEE "
        f"({scope}, response > 0):"
    )

    print(
        d["transition"]
        .value_counts()
        .reindex(
            PROPAGATION_TRANSITIONS,
            fill_value=0,
        )
        .to_string()
    )

    d["response"] = np.log(
        d["response"]
    )

    if len(d) < 10:
        print(
            f"Skipping {model_name} / "
            f"{scope} / amplification: "
            f"insufficient data."
        )
        return

    try:
        model = smf.gee(
            formula,
            groups="experiment_id",
            data=d,
            family=sm.families.Gaussian(),
            cov_struct=sm.cov_struct.Exchangeable(),
        )

        result = model.fit()

        summary_file = (
            MODEL_SUMMARIES
            / f"{output_prefix}_{scope}_amplification.txt"
        )

        with open(
            summary_file,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(str(result.summary()))

        for term in result.params.index:
            model_results.append({
                "scope": scope,
                "model": model_name,
                "metric": "amplification",
                "term": term,
                "estimate": result.params[term],
                "SE": result.bse[term],
                "p_value": result.pvalues[term],
                "N": len(d),
                "N_experiments": (
                    d["experiment_id"].nunique()
                ),
            })

        print(
            f"✓ {model_name} / "
            f"{scope} / amplification"
        )

    except Exception as error:
        print(
            f"FAILED {model_name} / "
            f"{scope} / amplification: {error}"
        )


# ============================================================
# RUN MODELS FOR BOTH SCOPES
# ============================================================

RQ_STAGE_FORMULA = (
    "rate ~ "
    "C(bias_expression) * "
    "C(transition)"
)

RQ_TYPE_FORMULA = (
    "rate ~ "
    "C(bias_expression) * "
    "C(bias_type)"
)

RQ_DOMAIN_FORMULA = (
    "rate ~ "
    "C(bias_expression) * "
    "C(domain)"
)

for scope in SCOPES:

    # --------------------------------------------------------
    # Expression x stage
    # --------------------------------------------------------

    for metric_name in [
        "survival",
        "emergence",
        "mitigation",
    ]:
        fit_binomial_gee(
            propagation_analysis,
            scope,
            metric_name,
            RQ_STAGE_FORMULA,
            "expression_by_stage",
            "stage",
        )

    fit_amplification_gee(
        propagation_analysis,
        scope,
        RQ_STAGE_FORMULA.replace(
            "rate",
            "response",
        ),
        "expression_by_stage",
        "stage",
    )

    # --------------------------------------------------------
    # Expression x bias type
    # --------------------------------------------------------

    for metric_name in [
        "survival",
        "emergence",
        "mitigation",
    ]:
        fit_binomial_gee(
            propagation_analysis,
            scope,
            metric_name,
            RQ_TYPE_FORMULA,
            "expression_by_bias_type",
            "bias_type",
        )

    fit_amplification_gee(
        propagation_analysis,
        scope,
        RQ_TYPE_FORMULA.replace(
            "rate",
            "response",
        ),
        "expression_by_bias_type",
        "bias_type",
    )

    # --------------------------------------------------------
    # Expression x domain
    # --------------------------------------------------------

    for metric_name in [
        "survival",
        "emergence",
        "mitigation",
    ]:
        fit_binomial_gee(
            propagation_analysis,
            scope,
            metric_name,
            RQ_DOMAIN_FORMULA,
            "expression_by_domain",
            "domain",
        )

    fit_amplification_gee(
        propagation_analysis,
        scope,
        RQ_DOMAIN_FORMULA.replace(
            "rate",
            "response",
        ),
        "expression_by_domain",
        "domain",
    )

    # --------------------------------------------------------
    # Expression x model
    # --------------------------------------------------------

    RQ_MODEL_FORMULA = (
        "rate ~ "
        "C(bias_expression) * "
        f"C({MODEL_COLUMN})"
    )

    for metric_name in [
        "survival",
        "emergence",
        "mitigation",
    ]:
        fit_binomial_gee(
            propagation_analysis,
            scope,
            metric_name,
            RQ_MODEL_FORMULA,
            "expression_by_model",
            "model",
        )

    fit_amplification_gee(
        propagation_analysis,
        scope,
        RQ_MODEL_FORMULA.replace(
            "rate",
            "response",
        ),
        "expression_by_model",
        "model",
    )


# ============================================================
# HOLM CORRECTION
# ============================================================

results = pd.DataFrame(model_results)

if not results.empty:
    results["p_holm"] = np.nan

    for (
        scope,
        model_name,
        metric_name,
    ), indices in results.groupby(
        ["scope", "model", "metric"]
    ).groups.items():

        p_values = pd.to_numeric(
            results.loc[
                indices,
                "p_value",
            ],
            errors="coerce",
        ).values

        valid = ~np.isnan(p_values)

        if valid.any():
            adjusted = holm_adjust(
                p_values[valid]
            )

            valid_indices = (
                np.asarray(indices)[valid]
            )

            results.loc[
                valid_indices,
                "p_holm",
            ] = adjusted

    results["significant"] = (
        results["p_holm"] < 0.05
    )

    results.to_csv(
        STATS / "gee_both_scopes.csv",
        index=False,
    )

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n========================================")
print("ANALYSIS COMPLETE")
print("========================================")

print("\nDescriptive tables:")
for file in sorted(DESC.glob("*.csv")):
    print(" ", file)

print("\nStatistical results:")
for file in sorted(STATS.glob("*.csv")):
    print(" ", file)

print("\nModel summaries:")
for file in sorted(MODEL_SUMMARIES.glob("*.txt")):
    print(" ", file)
