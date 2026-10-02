import json
import re
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BIAS_REPORTS_DIR = Path("bias_reports")
OUTPUTS_DIR = Path("outputs")
METRICS_DIR = Path("metrics")

METRICS_DIR.mkdir(parents=True, exist_ok=True)

STAGES = [
    "requirements",
    "code_generation",
    "refactoring",
    "testing",
    "documentation",
]

REPORT_FILES = {
    "code_generation": "01_code_generation_bias.json",
    "refactoring": "02_refactoring_bias.json",
    "testing": "03_testing_bias.json",
    "documentation": "04_documentation_bias.json",
}

MECHANISM_SIMILARITY_THRESHOLD = 0.50
TEXT_SIMILARITY_THRESHOLD = 0.55


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    value = str(value).lower()

    if value in {"nan", "none", "n/a"}:
        return ""

    value = value.replace('\\"', '"')
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_bias_type(value):
    value = normalize_text(value)

    aliases = {
        "documentation naming": "naming documentation",
        "documentation and naming": "naming documentation",
        "naming documentation": "naming documentation",
        "threshold cutoff": "threshold cutoff",
        "threshold cut off": "threshold cutoff",
        "demographic bias": "demographic",
        "stereotypical": "stereotype",
        "stereotypical bias": "stereotype",
        "proxy bias": "proxy",
        "neutral": "neutral",
    }

    return aliases.get(value, value)


def normalize_association(value):
    value = normalize_text(value)

    aliases = {
        "pos": "positive",
        "neg": "negative",
    }

    return aliases.get(value, value)


def text_similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


# ============================================================
# TARGET ATTRIBUTE DETECTION
# ============================================================

def attribute_in_instance(instance, target_attribute):
    target = normalize_text(target_attribute)

    if not target:
        return False

    fields = [
        instance.get("protected_attribute"),
        instance.get("biased_code"),
        instance.get("issue"),
        instance.get("impact_reasoning"),
    ]

    combined = " ".join(
        normalize_text(x)
        for x in fields
        if x is not None
    )

    return target in combined


def is_target_candidate(instance, metadata):
    """
    A current-stage BI is target-associated when it matches the
    experimentally specified bias type and target attribute.
    """

    expected_type = normalize_bias_type(
        metadata.get("bias_type")
    )

    expected_attribute = normalize_text(
        metadata.get("attribute")
    )

    detected_type = normalize_bias_type(
        instance.get("bias_type")
    )

    type_ok = (
        not expected_type
        or detected_type == expected_type
    )

    attribute_ok = (
        not expected_attribute
        or attribute_in_instance(
            instance,
            expected_attribute
        )
    )

    return type_ok and attribute_ok


# ============================================================
# BIAS IDENTITY CREATION
# ============================================================

def create_bias_identity(
    report_item,
    stage,
    experiment_metadata,
    instance_id,
):
    return {
        "instance_id": instance_id,
        "stage": stage,

        "bias_type": report_item.get("bias_type"),
        "bias_type_normalized": normalize_bias_type(
            report_item.get("bias_type")
        ),

        "association_type": report_item.get(
            "association_type"
        ),

        "association_normalized": normalize_association(
            report_item.get("association_type")
        ),

        "protected_attribute": report_item.get(
            "protected_attribute"
        ),

        "biased_code": report_item.get(
            "biased_code"
        ),

        "issue": report_item.get(
            "issue"
        ),

        "impact_reasoning": report_item.get(
            "impact_reasoning"
        ),

        "experiment_id": experiment_metadata[
            "experiment_id"
        ],

        "csv_row": experiment_metadata.get(
            "csv_row"
        ),

        "run_number": experiment_metadata[
            "run_number"
        ],

        "cgt": experiment_metadata[
            "cgt"
        ],

        "model": experiment_metadata.get(
            "model"
        ),

        "domain": experiment_metadata.get(
            "domain"
        ),

        "sub_domain": experiment_metadata.get(
            "sub_domain"
        ),

        "bias_expression": experiment_metadata.get(
            "bias_expression"
        ),

        "experiment_bias_type": experiment_metadata.get(
            "bias_type"
        ),

        "experiment_attribute": experiment_metadata.get(
            "attribute"
        ),
    }


# ============================================================
# LOAD TRACE
# ============================================================

def load_bias_report(report_path):
    with open(
        report_path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def load_trace(run_report_dir):
    relative_path = run_report_dir.relative_to(
        BIAS_REPORTS_DIR
    )

    output_run_dir = OUTPUTS_DIR / relative_path
    metadata_path = output_run_dir / "metadata.json"

    if not metadata_path.exists():
        print(
            f"WARNING: Missing metadata: "
            f"{metadata_path}"
        )
        return None

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    trace = {
        "experiment": metadata,
        "stages": {
            "requirements": []
        },
    }

    for stage, filename in REPORT_FILES.items():

        report_path = (
            run_report_dir / filename
        )

        if not report_path.exists():
            print(
                f"WARNING: Missing report: "
                f"{report_path}"
            )

            trace["stages"][stage] = []
            continue

        report = load_bias_report(
            report_path
        )

        report_items = report.get(
            "bias_report"
        )

        if not isinstance(
            report_items,
            list
        ):
            report_items = []

        instances = []

        for index, item in enumerate(
            report_items
        ):

            instance_id = (
                f"{metadata['experiment_id']}"
                f"_run{metadata['run_number']:02d}"
                f"_{stage}"
                f"_{index + 1:03d}"
            )

            instances.append(
                create_bias_identity(
                    item,
                    stage,
                    metadata,
                    instance_id,
                )
            )

        trace["stages"][stage] = instances

    return trace


# ============================================================
# BIAS IDENTITY MATCHING
# ============================================================

def calculate_match_score(
    instance_a,
    instance_b
):
    """
    Evaluate evidence that two findings represent the same
    underlying bias.

    Bias type, affected attribute, and association must match
    exactly after normalization.

    Mechanism and code similarity must both meet their
    configured thresholds for a pair to be eligible.
    """

    evidence = []

    type_a = instance_a.get(
        "bias_type_normalized",
        ""
    )

    type_b = instance_b.get(
        "bias_type_normalized",
        ""
    )

    type_match = bool(
        type_a
        and type_b
        and type_a == type_b
    )

    attr_a = normalize_text(
        instance_a.get(
            "protected_attribute"
        )
    )

    attr_b = normalize_text(
        instance_b.get(
            "protected_attribute"
        )
    )

    attribute_match = bool(
        attr_a
        and attr_b
        and attr_a == attr_b
    )

    association_a = normalize_association(
        instance_a.get(
            "association_type"
        )
    )

    association_b = normalize_association(
        instance_b.get(
            "association_type"
        )
    )

    association_match = bool(
        association_a
        and association_b
        and association_a == association_b
    )

    issue_a = (
        instance_a.get("issue")
        or ""
    )

    issue_b = (
        instance_b.get("issue")
        or ""
    )

    mechanism_similarity = text_similarity(
        issue_a,
        issue_b
    )

    reasoning_a = (
        instance_a.get(
            "impact_reasoning"
        )
        or ""
    )

    reasoning_b = (
        instance_b.get(
            "impact_reasoning"
        )
        or ""
    )

    reasoning_similarity = text_similarity(
        reasoning_a,
        reasoning_b
    )

    explanation_similarity = text_similarity(
        f"{issue_a} {reasoning_a}",
        f"{issue_b} {reasoning_b}",
    )

    code_similarity = text_similarity(
        instance_a.get(
            "biased_code"
        ),
        instance_b.get(
            "biased_code"
        ),
    )

    score = 0.0

    if type_match:
        score += 1.0
        evidence.append(
            "bias_type_exact_match"
        )

    if attribute_match:
        score += 1.0
        evidence.append(
            "attribute_exact_match"
        )

    if association_match:
        score += 1.0
        evidence.append(
            "association_exact_match"
        )

    if (
        mechanism_similarity
        >= MECHANISM_SIMILARITY_THRESHOLD
    ):
        score += mechanism_similarity
        evidence.append(
            "mechanism_similarity"
        )

    if (
        code_similarity
        >= TEXT_SIMILARITY_THRESHOLD
    ):
        score += code_similarity
        evidence.append(
            "code_similarity"
        )

    return {
        "score": score,
        "evidence": evidence,

        "type_match": type_match,
        "attribute_match": attribute_match,
        "association_match": association_match,

        "mechanism_similarity":
            mechanism_similarity,

        "reasoning_similarity":
            reasoning_similarity,

        "explanation_similarity":
            explanation_similarity,

        "code_similarity":
            code_similarity,
    }


def match_instances(
    previous_instances,
    current_instances
):
    """
    Greedy one-to-one matching.

    A pair is eligible only when:
      1. bias type matches exactly,
      2. affected attribute matches exactly,
      3. association matches exactly,
      4. mechanism similarity >= 0.50,
      5. code similarity >= 0.55.

    Among eligible pairs, candidates are ranked by mechanism
    similarity, code similarity, and overall match score.
    """

    candidates = []

    for previous in previous_instances:

        for current in current_instances:

            result = calculate_match_score(
                previous,
                current
            )

            semantic_match = (
                result["type_match"]
                and result["attribute_match"]
                and result["association_match"]
                and (
                    result["mechanism_similarity"]
                    >= MECHANISM_SIMILARITY_THRESHOLD
                )
                and (
                    result["code_similarity"]
                    >= TEXT_SIMILARITY_THRESHOLD
                )
            )

            if semantic_match:

                candidates.append({
                    "previous": previous,
                    "current": current,
                    "score": result["score"],
                    "details": result,
                })

    candidates.sort(
        key=lambda x: (
            x["details"][
                "mechanism_similarity"
            ],
            x["details"][
                "code_similarity"
            ],
            x["score"],
        ),
        reverse=True,
    )

    matched_previous = set()
    matched_current = set()

    matches = []

    for candidate in candidates:

        previous_id = candidate[
            "previous"
        ]["instance_id"]

        current_id = candidate[
            "current"
        ]["instance_id"]

        if previous_id in matched_previous:
            continue

        if current_id in matched_current:
            continue

        matched_previous.add(
            previous_id
        )

        matched_current.add(
            current_id
        )

        matches.append(
            candidate
        )

    return matches


# ============================================================
# TARGET BIAS IDENTIFICATION AT CODE GENERATION
# ============================================================

def identify_target_biases(
    biased_code_instances,
    neutral_code_instances,
    metadata,
):
    """
    Identify target BIs by comparing biased code generation
    against the neutral baseline, then constraining candidates
    to the experimentally intended bias type and attribute.
    """

    neutral_matches = match_instances(
        biased_code_instances,
        neutral_code_instances,
    )

    matched_biased_ids = {
        match["previous"]["instance_id"]
        for match in neutral_matches
    }

    candidates = [
        instance
        for instance in biased_code_instances
        if instance["instance_id"]
        not in matched_biased_ids
    ]

    target_instances = [
        instance
        for instance in candidates
        if is_target_candidate(
            instance,
            metadata,
        )
    ]

    return (
        target_instances,
        neutral_matches,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_presence(instances):
    return int(
        len(instances) > 0
    )


def calculate_survival(
    previous_instances,
    matches
):
    if len(previous_instances) == 0:
        return None

    return (
        len(matches)
        / len(previous_instances)
    )


def calculate_emergence(
    previous_instances,
    current_instances,
    matches,
):
    if len(current_instances) == 0:
        return 0.0

    newly_emerged = (
        len(current_instances)
        - len(matches)
    )

    return (
        newly_emerged
        / len(current_instances)
    )


def calculate_amplification(
    previous_instances,
    current_instances,
):
    return (
        len(current_instances)
        / max(
            1,
            len(previous_instances)
        )
    )


def calculate_mitigation(
    previous_instances,
    matches
):
    if len(previous_instances) == 0:
        return None

    removed = (
        len(previous_instances)
        - len(matches)
    )

    return (
        removed
        / len(previous_instances)
    )


# ============================================================
# MATCH RECORD
# ============================================================

def create_match_record(
    match,
    metadata,
    match_type,
    from_stage,
    to_stage,
):
    return {
        "experiment_id":
            metadata["experiment_id"],

        "csv_row":
            metadata.get("csv_row"),

        "run_number":
            metadata["run_number"],

        "cgt":
            metadata["cgt"],

        "model":
            metadata.get("model"),

        "domain":
            metadata.get("domain"),

        "sub_domain":
            metadata.get("sub_domain"),

        "experiment_bias_type":
            metadata.get("bias_type"),

        "bias_expression":
            metadata.get("bias_expression"),

        "experiment_attribute":
            metadata.get("attribute"),

        "match_type":
            match_type,

        "from_stage":
            from_stage,

        "to_stage":
            to_stage,

        "previous_instance_id":
            match["previous"]["instance_id"],

        "current_instance_id":
            match["current"]["instance_id"],

        "previous_bias_type":
            match["previous"]["bias_type"],

        "current_bias_type":
            match["current"]["bias_type"],

        "previous_attribute":
            match["previous"][
                "protected_attribute"
            ],

        "current_attribute":
            match["current"][
                "protected_attribute"
            ],

        "previous_code":
            match["previous"]["biased_code"],

        "current_code":
            match["current"]["biased_code"],

        "previous_issue":
            match["previous"]["issue"],

        "current_issue":
            match["current"]["issue"],

        "match_score":
            match["score"],

        "match_evidence":
            ";".join(
                match["details"]["evidence"]
            ),

        "code_similarity":
            match["details"][
                "code_similarity"
            ],

        "mechanism_similarity":
            match["details"][
                "mechanism_similarity"
            ],

        "explanation_similarity":
            match["details"][
                "explanation_similarity"
            ],
    }


# ============================================================
# ANALYZE ONE BIASED RUN
# ============================================================

def analyze_trace(
    biased_trace,
    neutral_traces
):
    metadata = biased_trace["experiment"]

    metrics = []
    match_records = []

    neutral_code = []

    for neutral_trace in neutral_traces:

        neutral_code.extend(
            neutral_trace[
                "stages"
            ].get(
                "code_generation",
                [],
            )
        )

    biased_code = (
        biased_trace[
            "stages"
        ].get(
            "code_generation",
            [],
        )
    )

    target_code, neutral_matches = (
        identify_target_biases(
            biased_code,
            neutral_code,
            metadata,
        )
    )

    for match in neutral_matches:

        match_records.append(
            create_match_record(
                match,
                metadata,
                "biased_vs_neutral",
                "code_generation",
                "neutral_baseline",
            )
        )

    # --------------------------------------------------------
    # Requirements -> Code Generation
    # --------------------------------------------------------

    target_previous_requirement_count = 1
    target_current_code_count = len(
        target_code
    )

    target_present = int(
        target_current_code_count > 0
    )

    metrics.append({
        "experiment_id":
            metadata["experiment_id"],

        "csv_row":
            metadata.get("csv_row"),

        "run_number":
            metadata["run_number"],

        "cgt":
            metadata["cgt"],

        "model":
            metadata.get("model"),

        "domain":
            metadata.get("domain"),

        "sub_domain":
            metadata.get("sub_domain"),

        "bias_type":
            metadata.get("bias_type"),

        "bias_expression":
            metadata.get("bias_expression"),

        "attribute":
            metadata.get("attribute"),

        "analysis_condition":
            "biased",

        "from_stage":
            "requirements",

        "to_stage":
            "code_generation",

        "transition_kind":
            "target_introduction",

        "overall_previous_count":
            None,

        "overall_current_count":
            len(biased_code),

        "overall_presence_previous":
            None,

        "overall_presence_current":
            calculate_presence(
                biased_code
            ),

        "overall_survival":
            None,

        "overall_emergence":
            None,

        "overall_amplification":
            None,

        "overall_mitigation":
            None,

        "target_previous_count":
            target_previous_requirement_count,

        "target_current_count":
            target_current_code_count,

        "target_presence_previous":
            1,

        "target_presence_current":
            target_present,

        "target_survival":
            float(target_present),

        "target_emergence":
            None,

        "target_amplification":
            float(
                target_current_code_count
            ),

        "target_mitigation":
            float(
                1 - target_present
            ),
    })

    # --------------------------------------------------------
    # Target propagation
    # --------------------------------------------------------

    target_by_stage = {
        "code_generation":
            target_code
    }

    target_matches_by_transition = {}

    target_previous = target_code

    for index in range(
        STAGES.index(
            "code_generation"
        ),
        len(STAGES) - 1,
    ):

        previous_stage = STAGES[
            index
        ]

        current_stage = STAGES[
            index + 1
        ]

        current_instances = (
            biased_trace[
                "stages"
            ].get(
                current_stage,
                [],
            )
        )

        matches = match_instances(
            target_previous,
            current_instances,
        )

        matched_current_ids = {
            match["current"]["instance_id"]
            for match in matches
        }

        newly_emerged_target = [
            instance
            for instance in current_instances
            if (
                instance["instance_id"]
                not in matched_current_ids
                and is_target_candidate(
                    instance,
                    metadata,
                )
            )
        ]

        target_current = (
            [
                match["current"]
                for match in matches
            ]
            + newly_emerged_target
        )

        target_by_stage[
            current_stage
        ] = target_current

        target_matches_by_transition[
            (
                previous_stage,
                current_stage,
            )
        ] = matches

        for match in matches:

            match_records.append(
                create_match_record(
                    match,
                    metadata,
                    "target_propagation",
                    previous_stage,
                    current_stage,
                )
            )

        target_previous = target_current

    # --------------------------------------------------------
    # Overall artifact-to-artifact propagation
    # --------------------------------------------------------

    for index in range(
        len(STAGES) - 1
    ):

        previous_stage = STAGES[
            index
        ]

        current_stage = STAGES[
            index + 1
        ]

        if previous_stage == "requirements":
            continue

        previous_all = (
            biased_trace[
                "stages"
            ].get(
                previous_stage,
                [],
            )
        )

        current_all = (
            biased_trace[
                "stages"
            ].get(
                current_stage,
                [],
            )
        )

        all_matches = match_instances(
            previous_all,
            current_all,
        )

        for match in all_matches:

            match_records.append(
                create_match_record(
                    match,
                    metadata,
                    "overall_propagation",
                    previous_stage,
                    current_stage,
                )
            )

        overall_survival = calculate_survival(
            previous_all,
            all_matches,
        )

        overall_emergence = calculate_emergence(
            previous_all,
            current_all,
            all_matches,
        )

        overall_amplification = (
            calculate_amplification(
                previous_all,
                current_all,
            )
        )

        overall_mitigation = calculate_mitigation(
            previous_all,
            all_matches,
        )

        target_previous = target_by_stage.get(
            previous_stage,
            [],
        )

        target_current = target_by_stage.get(
            current_stage,
            [],
        )

        target_matches = (
            target_matches_by_transition.get(
                (
                    previous_stage,
                    current_stage,
                ),
                [],
            )
        )

        target_survival = calculate_survival(
            target_previous,
            target_matches,
        )

        matched_target_current_ids = {
            match["current"]["instance_id"]
            for match in target_matches
        }

        target_new = [
            instance
            for instance in target_current
            if instance["instance_id"]
            not in matched_target_current_ids
        ]

        if len(target_current) == 0:
            target_emergence = 0.0
        else:
            target_emergence = (
                len(target_new)
                / len(target_current)
            )

        target_amplification = (
            calculate_amplification(
                target_previous,
                target_current,
            )
        )

        target_mitigation = (
            calculate_mitigation(
                target_previous,
                target_matches,
            )
        )

        metrics.append({
            "experiment_id":
                metadata["experiment_id"],

            "csv_row":
                metadata.get("csv_row"),

            "run_number":
                metadata["run_number"],

            "cgt":
                metadata["cgt"],

            "model":
                metadata.get("model"),

            "domain":
                metadata.get("domain"),

            "sub_domain":
                metadata.get("sub_domain"),

            "bias_type":
                metadata.get("bias_type"),

            "bias_expression":
                metadata.get("bias_expression"),

            "attribute":
                metadata.get("attribute"),

            "analysis_condition":
                "biased",

            "from_stage":
                previous_stage,

            "to_stage":
                current_stage,

            "transition_kind":
                "artifact_propagation",

            "overall_previous_count":
                len(previous_all),

            "overall_current_count":
                len(current_all),

            "overall_presence_previous":
                calculate_presence(
                    previous_all
                ),

            "overall_presence_current":
                calculate_presence(
                    current_all
                ),

            "overall_survival":
                overall_survival,

            "overall_emergence":
                overall_emergence,

            "overall_amplification":
                overall_amplification,

            "overall_mitigation":
                overall_mitigation,

            "target_previous_count":
                len(target_previous),

            "target_current_count":
                len(target_current),

            "target_presence_previous":
                calculate_presence(
                    target_previous
                ),

            "target_presence_current":
                calculate_presence(
                    target_current
                ),

            "target_survival":
                target_survival,

            "target_emergence":
                target_emergence,

            "target_amplification":
                target_amplification,

            "target_mitigation":
                target_mitigation,
        })

    return (
        metrics,
        match_records,
        target_by_stage,
    )


# ============================================================
# ANALYZE NEUTRAL RUN
# ============================================================

def analyze_overall_trace(trace):
    """
    Calculate overall artifact-to-artifact propagation metrics
    for a neutral workflow.

    Neutral workflows have no target-bias metrics because there
    is no experimentally specified target bias.
    """

    metadata = trace["experiment"]

    metrics = []

    for index in range(
        len(STAGES) - 1
    ):

        previous_stage = STAGES[
            index
        ]

        current_stage = STAGES[
            index + 1
        ]

        # Requirements -> Code Generation is not treated as
        # conventional artifact-to-artifact propagation.
        if previous_stage == "requirements":
            continue

        previous_all = (
            trace[
                "stages"
            ].get(
                previous_stage,
                [],
            )
        )

        current_all = (
            trace[
                "stages"
            ].get(
                current_stage,
                [],
            )
        )

        all_matches = match_instances(
            previous_all,
            current_all,
        )

        metrics.append({
            "experiment_id":
                metadata["experiment_id"],

            "csv_row":
                metadata.get("csv_row"),

            "run_number":
                metadata["run_number"],

            "cgt":
                metadata["cgt"],

            "model":
                metadata.get("model"),

            "domain":
                metadata.get("domain"),

            "sub_domain":
                metadata.get("sub_domain"),

            "bias_type":
                "Neutral",

            "bias_expression":
                "Neutral",

            "attribute":
                None,

            "analysis_condition":
                "neutral",

            "from_stage":
                previous_stage,

            "to_stage":
                current_stage,

            "transition_kind":
                "artifact_propagation",

            "overall_previous_count":
                len(previous_all),

            "overall_current_count":
                len(current_all),

            "overall_presence_previous":
                calculate_presence(
                    previous_all
                ),

            "overall_presence_current":
                calculate_presence(
                    current_all
                ),

            "overall_survival":
                calculate_survival(
                    previous_all,
                    all_matches,
                ),

            "overall_emergence":
                calculate_emergence(
                    previous_all,
                    current_all,
                    all_matches,
                ),

            "overall_amplification":
                calculate_amplification(
                    previous_all,
                    current_all,
                ),

            "overall_mitigation":
                calculate_mitigation(
                    previous_all,
                    all_matches,
                ),

            "target_previous_count":
                None,

            "target_current_count":
                None,

            "target_presence_previous":
                None,

            "target_presence_current":
                None,

            "target_survival":
                None,

            "target_emergence":
                None,

            "target_amplification":
                None,

            "target_mitigation":
                None,
        })

    return metrics


# ============================================================
# DISCOVERY / NEUTRAL BASELINES
# ============================================================

def discover_run_directories():

    if not BIAS_REPORTS_DIR.exists():
        raise FileNotFoundError(
            f"Bias reports directory not found: "
            f"{BIAS_REPORTS_DIR}"
        )

    run_directories = []

    for cgt_dir in sorted(
        BIAS_REPORTS_DIR.iterdir()
    ):

        if not cgt_dir.is_dir():
            continue

        for run_dir in sorted(
            cgt_dir.iterdir()
        ):

            if run_dir.is_dir():
                run_directories.append(
                    run_dir
                )

    return run_directories


def is_neutral_trace(trace):

    metadata = trace["experiment"]

    return (
        normalize_bias_type(
            metadata.get("bias_type")
        )
        == "neutral"
    )


def build_neutral_baselines(traces):

    baselines = {}

    for trace in traces:

        if not is_neutral_trace(trace):
            continue

        metadata = trace["experiment"]

        key = (
            normalize_text(
                metadata.get(
                    "sub_domain"
                )
            ),
            normalize_text(
                metadata.get("cgt")
            ),
        )

        baselines.setdefault(
            key,
            []
        ).append(trace)

    for key in baselines:

        baselines[key].sort(
            key=lambda trace:
                trace["experiment"][
                    "run_number"
                ]
        )

    return baselines


# ============================================================
# MAIN
# ============================================================

def run_metrics():

    run_directories = (
        discover_run_directories()
    )

    print(
        f"Found {len(run_directories)} runs."
    )

    all_traces = []

    for run_dir in run_directories:

        print(
            f"\nLoading: {run_dir}"
        )

        trace = load_trace(
            run_dir
        )

        if trace is not None:
            all_traces.append(
                trace
            )

    print(
        f"\nLoaded {len(all_traces)} traces."
    )

    neutral_baselines = (
        build_neutral_baselines(
            all_traces
        )
    )

    print(
        f"Found "
        f"{len(neutral_baselines)} "
        f"neutral baseline groups."
    )

    for key, traces in (
        neutral_baselines.items()
    ):

        print(
            f"  {key[0]} / {key[1]}: "
            f"{len(traces)} neutral runs"
        )

    all_metrics = []
    all_matches = []
    all_target_traces = []

    # --------------------------------------------------------
    # Separate neutral and biased traces
    # --------------------------------------------------------

    neutral_traces = [
        trace
        for trace in all_traces
        if is_neutral_trace(trace)
    ]

    biased_traces = [
        trace
        for trace in all_traces
        if not is_neutral_trace(trace)
    ]

    print(
        f"\nNeutral runs for overall analysis: "
        f"{len(neutral_traces)}"
    )

    print(
        f"Biased runs to analyze: "
        f"{len(biased_traces)}"
    )

    # --------------------------------------------------------
    # NEUTRAL OVERALL METRICS
    # --------------------------------------------------------

    for neutral_trace in neutral_traces:

        metadata = neutral_trace[
            "experiment"
        ]

        print(
            "\nProcessing neutral: "
            f"{metadata.get('experiment_id')} "
            f"run {metadata.get('run_number')}"
        )

        neutral_metrics = (
            analyze_overall_trace(
                neutral_trace
            )
        )

        all_metrics.extend(
            neutral_metrics
        )

    # --------------------------------------------------------
    # BIASED METRICS + TARGET TRACES
    # --------------------------------------------------------

    for biased_trace in biased_traces:

        metadata = biased_trace[
            "experiment"
        ]

        baseline_key = (
            normalize_text(
                metadata.get(
                    "sub_domain"
                )
            ),
            normalize_text(
                metadata.get("cgt")
            ),
        )

        neutral_traces_for_baseline = (
            neutral_baselines.get(
                baseline_key,
                [],
            )
        )

        if not neutral_traces_for_baseline:

            print(
                "WARNING: No neutral baseline "
                f"for {baseline_key} "
                f"({metadata.get('experiment_id')})"
            )

            continue

        if len(
            neutral_traces_for_baseline
        ) != 3:

            print(
                "WARNING: Expected 3 neutral "
                "runs for "
                f"{baseline_key}, "
                f"found "
                f"{len(neutral_traces_for_baseline)}."
            )

        print(
            "\nProcessing: "
            f"{metadata.get('experiment_id')} "
            f"run {metadata.get('run_number')}"
        )

        metrics, matches, target_trace = (
            analyze_trace(
                biased_trace,
                neutral_traces_for_baseline,
            )
        )

        all_metrics.extend(
            metrics
        )

        all_matches.extend(
            matches
        )

        all_target_traces.append({
            "experiment_id":
                metadata["experiment_id"],

            "csv_row":
                metadata.get("csv_row"),

            "run_number":
                metadata["run_number"],

            "cgt":
                metadata["cgt"],

            "model":
                metadata.get("model"),

            "domain":
                metadata.get("domain"),

            "sub_domain":
                metadata.get("sub_domain"),

            "bias_type":
                metadata.get("bias_type"),

            "bias_expression":
                metadata.get(
                    "bias_expression"
                ),

            "attribute":
                metadata.get("attribute"),

            "neutral_baseline_runs": [
                trace["experiment"][
                    "run_number"
                ]
                for trace
                in neutral_traces_for_baseline
            ],

            "target_trace":
                target_trace,
        })

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    traces_path = (
        METRICS_DIR
        / "bias_traces.json"
    )

    with open(
        traces_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_traces,
            f,
            indent=4,
            ensure_ascii=False,
        )

    target_path = (
        METRICS_DIR
        / "target_bias_traces.json"
    )

    with open(
        target_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_target_traces,
            f,
            indent=4,
            ensure_ascii=False,
        )

    metrics_path = (
        METRICS_DIR
        / "propagation_metrics.csv"
    )

    pd.DataFrame(
        all_metrics
    ).to_csv(
        metrics_path,
        index=False,
    )

    matches_path = (
        METRICS_DIR
        / "bias_instance_matches.csv"
    )

    pd.DataFrame(
        all_matches
    ).to_csv(
        matches_path,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "METRICS COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Bias traces: {traces_path}"
    )

    print(
        f"Target traces: {target_path}"
    )

    print(
        f"Metrics: {metrics_path}"
    )

    print(
        f"Instance matches: {matches_path}"
    )

    print(
        f"\nTotal runs loaded: "
        f"{len(all_traces)}"
    )

    print(
        f"Neutral runs: "
        f"{len(neutral_traces)}"
    )

    print(
        f"Neutral baseline groups: "
        f"{len(neutral_baselines)}"
    )

    print(
        f"Biased runs analyzed: "
        f"{len(biased_traces)}"
    )

    print(
        f"Metric rows: "
        f"{len(all_metrics)}"
    )

    print(
        f"Matched instances: "
        f"{len(all_matches)}"
    )


if __name__ == "__main__":
    run_metrics()