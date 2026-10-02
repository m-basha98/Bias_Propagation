import json
import re
import time

import pandas as pd
import yaml

from pathlib import Path

from cgt.gemini import GeminiCGT
from cgt.ollama import OllamaCGT


# ============================================================
# CGT INITIALIZATION
# ============================================================

def initialize_cgts(config):

    cgts = []

    if config["cgts"]["gemini"]["enabled"]:

        cgts.append(
            (
                "gemini_2_5_pro",
                GeminiCGT(
                    config["cgts"]["gemini"]["model"]
                )
            )
        )

    if config["cgts"]["qwen"]["enabled"]:

        cgts.append(
            (
                "qwen3_coder",
                OllamaCGT(
                    model_name=config["cgts"]["qwen"]["model"],
                    host=config["cgts"]["qwen"]["host"]
                )
            )
        )

    return cgts


# ============================================================
# FILE UTILITIES
# ============================================================

def save_text(path, content):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(content)


def load_text(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


# ============================================================
# CODE EXTRACTION
# ============================================================

def extract_python_code(response):

    """
    Extract Python code from an LLM response.

    If the response contains a fenced Python block, the contents
    of the first Python block are returned.

    If no code fence is present, the entire response is returned.
    """

    # Look specifically for ```python ... ```
    python_blocks = re.findall(
        r"```python\s*(.*?)```",
        response,
        flags=re.DOTALL | re.IGNORECASE
    )

    if python_blocks:

        return python_blocks[0].strip()

    # Look for generic ``` ... ```
    generic_blocks = re.findall(
        r"```\s*(.*?)```",
        response,
        flags=re.DOTALL
    )

    if generic_blocks:

        return generic_blocks[0].strip()

    # If there are no code fences, assume the entire response
    # is the requested code.
    return response.strip()


# ============================================================
# PROMPT CONSTRUCTION
# ============================================================

def build_code_prompt(row):

    return f"""
{row['Code Generation Prompt']}

Here is the complete software specification:

--- BEGIN SOFTWARE SPECIFICATION ---

{row['Full Requirements']}

--- END SOFTWARE SPECIFICATION ---
""".strip()


def build_refactoring_prompt(row, code):

    return f"""
{row['Refactoring Prompt']}

Here is the original software specification:

--- BEGIN SOFTWARE SPECIFICATION ---

{row['Full Requirements']}

--- END SOFTWARE SPECIFICATION ---

Here is the generated implementation:

--- BEGIN GENERATED CODE ---

{code}

--- END GENERATED CODE ---
""".strip()


def build_testing_prompt(row, refactored_code):

    return f"""
{row['Testing Prompt']}

Here is the original software specification:

--- BEGIN SOFTWARE SPECIFICATION ---

{row['Full Requirements']}

--- END SOFTWARE SPECIFICATION ---

Here is the refactored implementation:

--- BEGIN REFACTORED CODE ---

{refactored_code}

--- END REFACTORED CODE ---
""".strip()


def build_documentation_prompt(row, refactored_code):

    return f"""
{row['Documentation Prompt']}

Here is the original software specification:

--- BEGIN SOFTWARE SPECIFICATION ---

{row['Full Requirements']}

--- END SOFTWARE SPECIFICATION ---

Here is the refactored implementation:

--- BEGIN REFACTORED CODE ---

{refactored_code}

--- END REFACTORED CODE ---
""".strip()


# ============================================================
# CHECKPOINTED GENERATION
# ============================================================

def generate_or_load(
    cgt,
    prompt,
    raw_output_path,
    processed_output_path,
    stage_name,
    processor=None
):

    # --------------------------------------------------------
    # If the processed artifact already exists, use it.
    # This allows the experiment to resume after a crash.
    # --------------------------------------------------------

    if processed_output_path.exists():

        print(
            f"  ✓ {stage_name} already completed. "
            f"Skipping generation."
        )

        return load_text(
            processed_output_path
        )

    # --------------------------------------------------------
    # If raw output exists but processed output does not,
    # recover from the raw output without calling the model.
    # --------------------------------------------------------

    if raw_output_path.exists():

        print(
            f"  ↻ Raw {stage_name} found. "
            f"Reconstructing processed artifact."
        )

        raw_response = load_text(
            raw_output_path
        )

    else:

        print(
            f"  → Generating {stage_name}..."
        )

        raw_response = cgt.generate(
            prompt
        )

        # Save raw response IMMEDIATELY.
        save_text(
            raw_output_path,
            raw_response
        )

        print(
            f"  ✓ Raw {stage_name} saved."
        )

    # --------------------------------------------------------
    # Process the response.
    # --------------------------------------------------------

    if processor is not None:

        processed_response = processor(
            raw_response
        )

    else:

        processed_response = raw_response

    # --------------------------------------------------------
    # Save processed artifact IMMEDIATELY.
    # --------------------------------------------------------

    save_text(
        processed_output_path,
        processed_response
    )

    print(
        f"  ✓ {stage_name} saved."
    )

    return processed_response


# ============================================================
# METADATA
# ============================================================

def create_metadata(
    row,
    experiment_id,
    csv_row,
    cgt_name,
    cgt,
    run_number
):

    return {
        "experiment_id": experiment_id,
        "csv_row": csv_row,
        "run_number": run_number,

        "cgt": cgt_name,
        "model": cgt.model_name,

        "domain": row["Domain"],
        "sub_domain": row["Sub-Domain"],
        "bias_type": row["Bias Type"],
        "bias_expression": row["Implicit_Explicit"],
        "attribute": row["Attribute"],

        "status": {
            "code_generation": "not_started",
            "refactoring": "not_started",
            "testing": "not_started",
            "documentation": "not_started"
        }
    }


def save_metadata(path, metadata):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# SINGLE EXPERIMENTAL RUN
# ============================================================

def run_single_instance(
    row,
    cgt,
    cgt_name,
    experiment_id,
    csv_row,
    run_number,
    output_dir
):

    run_id = (
        f"{experiment_id}_"
        f"RUN{run_number:02d}"
    )

    run_dir = (
        output_dir /
        run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\n  Output directory: {run_dir}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_path = (
        run_dir /
        "metadata.json"
    )

    if metadata_path.exists():

        metadata = json.loads(
            load_text(metadata_path)
        )

    else:

        metadata = create_metadata(
            row=row,
            experiment_id=experiment_id,
            csv_row=csv_row,
            cgt_name=cgt_name,
            cgt=cgt,
            run_number=run_number
        )

        save_metadata(
            metadata_path,
            metadata
        )

    # --------------------------------------------------------
    # Save requirements
    # --------------------------------------------------------

    requirements_path = (
        run_dir /
        "00_requirements.md"
    )

    if not requirements_path.exists():

        save_text(
            requirements_path,
            row["Full Requirements"]
        )

    # ========================================================
    # STAGE 1: CODE GENERATION
    # ========================================================

    code_prompt = build_code_prompt(
        row
    )

    save_text(
        run_dir /
        "01_code_generation_prompt.txt",
        code_prompt
    )

    code = generate_or_load(
        cgt=cgt,
        prompt=code_prompt,

        raw_output_path=(
            run_dir /
            "01_code_generation_raw.txt"
        ),

        processed_output_path=(
            run_dir /
            "01_code_generation.py"
        ),

        stage_name="code generation",

        processor=extract_python_code
    )

    metadata["status"]["code_generation"] = (
        "completed"
    )

    save_metadata(
        metadata_path,
        metadata
    )

    # ========================================================
    # STAGE 2: REFACTORING
    # ========================================================

    refactor_prompt = build_refactoring_prompt(
        row,
        code
    )

    save_text(
        run_dir /
        "02_refactoring_prompt.txt",
        refactor_prompt
    )

    refactored_code = generate_or_load(
        cgt=cgt,
        prompt=refactor_prompt,

        raw_output_path=(
            run_dir /
            "02_refactoring_raw.txt"
        ),

        processed_output_path=(
            run_dir /
            "02_refactoring.py"
        ),

        stage_name="refactoring",

        processor=extract_python_code
    )

    metadata["status"]["refactoring"] = (
        "completed"
    )

    save_metadata(
        metadata_path,
        metadata
    )

    # ========================================================
    # STAGE 3: TESTING
    # ========================================================

    testing_prompt = build_testing_prompt(
        row,
        refactored_code
    )

    save_text(
        run_dir /
        "03_testing_prompt.txt",
        testing_prompt
    )

    tests = generate_or_load(
        cgt=cgt,
        prompt=testing_prompt,

        raw_output_path=(
            run_dir /
            "03_testing_raw.txt"
        ),

        processed_output_path=(
            run_dir /
            "03_testing.py"
        ),

        stage_name="testing",

        processor=extract_python_code
    )

    metadata["status"]["testing"] = (
        "completed"
    )

    save_metadata(
        metadata_path,
        metadata
    )

    # ========================================================
    # STAGE 4: DOCUMENTATION
    # ========================================================

    documentation_prompt = (
        build_documentation_prompt(
            row,
            refactored_code
        )
    )

    save_text(
        run_dir /
        "04_documentation_prompt.txt",
        documentation_prompt
    )

    documentation = generate_or_load(
        cgt=cgt,
        prompt=documentation_prompt,

        raw_output_path=(
            run_dir /
            "04_documentation_raw.txt"
        ),

        processed_output_path=(
            run_dir /
            "04_documentation.md"
        ),

        stage_name="documentation"
    )

    metadata["status"]["documentation"] = (
        "completed"
    )

    save_metadata(
        metadata_path,
        metadata
    )

    print(
        f"  ✓ Completed {run_id}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    with open(
        "config.yaml",
        "r",
        encoding="utf-8"
    ) as f:

        config = yaml.safe_load(f)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        config["dataset"]
    )

    print(
        f"Loaded {len(df)} experimental "
        f"data points."
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_root = Path(
        config["output_directory"]
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Initialize CGTs
    # --------------------------------------------------------

    cgts = initialize_cgts(
        config
    )

    runs = config[
        "runs_per_experiment"
    ]

    # --------------------------------------------------------
    # Execute experiment
    # --------------------------------------------------------

       # --------------------------------------------------------
    # Execute experiment
    # --------------------------------------------------------

    for cgt_name, cgt in cgts:

        print(
            "\n"
            + "#" * 70
        )

        print(
            f"Starting CGT: {cgt_name}"
        )

        print(
            "#" * 70
        )

        cgt_output = (
            output_root /
            cgt_name
        )

        for index, row in df.iterrows():

            experiment_id = (
                f"EXP{index + 1:03d}"
            )

            # CSV row number including header
            csv_row = index + 2

            for run_number in range(
                1,
                runs + 1
            ):

                print(
                    "\n"
                    + "=" * 70
                )

                print(
                    f"{experiment_id} | "
                    f"{cgt_name} | "
                    f"RUN {run_number}"
                )

                print(
                    "=" * 70
                )

                try:

                    run_single_instance(
                        row=row,
                        cgt=cgt,
                        cgt_name=cgt_name,
                        experiment_id=experiment_id,
                        csv_row=csv_row,
                        run_number=run_number,
                        output_dir=cgt_output
                    )

                except Exception as e:

                    print(
                        f"\nERROR: "
                        f"{experiment_id} | "
                        f"{cgt_name} | "
                        f"RUN {run_number}"
                    )

                    print(
                        f"{type(e).__name__}: {e}"
                    )

                    print(
                        "The completed stages have "
                        "been preserved."
                    )

                    print(
                        "The experiment can be "
                        "restarted safely."
                    )

                    continue

        print(
            "\n"
            + "#" * 70
        )

        print(
            f"Finished CGT: {cgt_name}"
        )

        print(
            "#" * 70
        )


if __name__ == "__main__":
    main()