#!/usr/bin/env python3
"""
Validate all judgment files for completeness, duplicates, and integrity.
This is critical since judgment files are expensive to generate and stored in git-lfs.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple

EXPECTED_QUESTIONS = 80
BENCH = "ja_mt_bench"


def validate_judgment_file(filepath: Path) -> Dict:
    """
    Validate a single judgment file and return detailed statistics.

    Returns:
        Dict with validation results including errors, warnings, and stats
    """
    result = {
        "file": str(filepath),
        "exists": filepath.exists(),
        "errors": [],
        "warnings": [],
        "models": {},
        "total_lines": 0,
        "invalid_json_lines": [],
    }

    if not filepath.exists():
        result["errors"].append(f"File does not exist: {filepath}")
        return result

# Track model -> (question_id, turn) -> list of line numbers (to detect duplicates)
model_question_turns: Dict[str, Dict[Tuple[int, int], List[int]]] = defaultdict(lambda: defaultdict(list))
# Track model -> set of question_ids (to check completeness regardless of turn count)
model_question_ids: Dict[str, Set[int]] = defaultdict(set)

    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                result["total_lines"] += 1

                # Parse JSON
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    result["invalid_json_lines"].append(line_num)
                    result["errors"].append(f"Line {line_num}: Invalid JSON - {e}")
                    continue

                # Extract model ID (can be 'model' or 'model_id')
                model = data.get("model") or data.get("model_id")
                if not model:
                    result["errors"].append(f"Line {line_num}: Missing 'model' or 'model_id' field")
                    continue

                # Extract question ID
                question_id = data.get("question_id")
                if question_id is None:
                    result["errors"].append(f"Line {line_num}: Missing 'question_id' field")
                    continue

                # Extract turn (default to 1 if missing)
                turn = data.get("turn", 1)

                # Track this model/question/turn combination
                key = (int(question_id), int(turn))
                model_question_turns[model][key].append(line_num)
                model_question_ids[model].add(int(question_id))

    except Exception as e:
        result["errors"].append(f"Failed to read file: {e}")
        return result

    # Analyze each model's completeness and duplicates
    for model, questions in model_question_turns.items():
        model_info = {
            "question_count": len(model_question_ids[model]),
            "is_complete": False,
            "missing_questions": [],
            "duplicate_questions": {},
            "unexpected_questions": [],
        }

        # Check for duplicates
        for (qid, turn), line_nums in questions.items():
            if len(line_nums) > 1:
                model_info["duplicate_questions"][(qid, turn)] = line_nums
                result["errors"].append(
                    f"Model '{model}' has DUPLICATE entries for question {qid} turn {turn} "
                    f"at lines: {line_nums}"
                )

        # Check completeness
        expected_ids = set(range(1, EXPECTED_QUESTIONS + 1))
        actual_ids = set(model_question_ids[model])

        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids

        if missing:
            model_info["missing_questions"] = sorted(missing)
            result["errors"].append(
                f"Model '{model}' is INCOMPLETE: missing {len(missing)} questions: "
                f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
            )

        if extra:
            model_info["unexpected_questions"] = sorted(extra)
            result["errors"].append(
                f"Model '{model}' has UNEXPECTED question IDs: {sorted(extra)}"
            )

        if len(actual_ids) == EXPECTED_QUESTIONS and not missing and not extra:
            model_info["is_complete"] = True

        result["models"][model] = model_info

    return result


def main():
    # Find all judgment files
    judgment_dir = Path(f"data/{BENCH}/model_judgment")

    if not judgment_dir.exists():
        print(f"ERROR: Judgment directory not found: {judgment_dir}")
        sys.exit(1)

    judgment_files = sorted(judgment_dir.glob("*_single.jsonl"))

    if not judgment_files:
        print(f"WARNING: No judgment files found in {judgment_dir}")
        sys.exit(0)

    print(f"🔍 Validating {len(judgment_files)} judgment files...\n")
    print("=" * 80)

    all_errors = []
    all_warnings = []
    file_summaries = []

    for filepath in judgment_files:
        judge_name = filepath.stem.replace("_single", "")
        print(f"\n📋 {judge_name}")
        print("-" * 80)

        result = validate_judgment_file(filepath)

        # Display results
        if result["errors"]:
            print(f"❌ ERRORS: {len(result['errors'])}")
            for error in result["errors"][:5]:  # Show first 5 errors
                print(f"   {error}")
            if len(result["errors"]) > 5:
                print(f"   ... and {len(result['errors']) - 5} more errors")
            all_errors.extend(result["errors"])

        if result["warnings"]:
            print(f"⚠️  WARNINGS: {len(result['warnings'])}")
            for warning in result["warnings"][:3]:  # Show first 3 warnings
                print(f"   {warning}")
            if len(result["warnings"]) > 3:
                print(f"   ... and {len(result['warnings']) - 3} more warnings")
            all_warnings.extend(result["warnings"])

        # Show model summary
        complete_models = [m for m, info in result["models"].items() if info["is_complete"]]
        incomplete_models = [m for m, info in result["models"].items() if not info["is_complete"]]
        duplicate_models = [m for m, info in result["models"].items() if info["duplicate_questions"]]

        print(f"📊 Models: {len(result['models'])} total")
        print(f"   ✓ Complete: {len(complete_models)}")
        if incomplete_models:
            print(f"   ⚠ Incomplete: {len(incomplete_models)}")
        if duplicate_models:
            print(f"   ❌ With duplicates: {len(duplicate_models)}")
            for model in duplicate_models:
                dup_count = len(result["models"][model]["duplicate_questions"])
                print(f"      • {model}: {dup_count} duplicate questions")

        file_summaries.append({
            "judge": judge_name,
            "total_models": len(result["models"]),
            "complete_models": len(complete_models),
            "incomplete_models": len(incomplete_models),
            "duplicate_models": len(duplicate_models),
            "errors": len(result["errors"]),
            "warnings": len(result["warnings"]),
        })

    # Overall summary
    print("\n" + "=" * 80)
    print("📈 OVERALL SUMMARY")
    print("=" * 80)

    total_errors = len(all_errors)
    total_warnings = len(all_warnings)

    print(f"\nTotal errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")

    if total_errors > 0:
        print("\n🚨 CRITICAL ISSUES FOUND!")
        print("   Your judgment files have ERRORS that need immediate attention.")
        print("   Duplicates or corrupt data could invalidate your published results!")
    elif total_warnings > 0:
        print("\n⚠️  Some judgment files are incomplete.")
        print("   This is OK if you haven't generated all judgments yet.")
    else:
        print("\n✅ All judgment files are valid and complete!")

    # Detailed summary table
    print("\n" + "-" * 80)
    print(f"{'Judge':<30} {'Models':<8} {'Complete':<10} {'Incomplete':<12} {'Duplicates':<12}")
    print("-" * 80)
    for summary in file_summaries:
        print(
            f"{summary['judge']:<30} "
            f"{summary['total_models']:<8} "
            f"{summary['complete_models']:<10} "
            f"{summary['incomplete_models']:<12} "
            f"{summary['duplicate_models']:<12}"
        )

    # Exit code based on errors
    if total_errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
