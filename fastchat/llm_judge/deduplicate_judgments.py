#!/usr/bin/env python3
"""
Clean judgment files with uneven/partial duplicates without discarding valid turns.

Uniform duplicates (every question/turn appears the same number of times) are GOOD -
they act like multiple annotations and improve reliability.

This script only cleans PARTIAL duplicates where specific question/turn pairs repeat
unevenly and would skew the mean scores. Completeness is checked, but missing data
is not auto-fixed; it is reported so you can regenerate.

Always shows statistics and requires interactive approval per file.
"""

import json
import sys
import shutil
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Set, Tuple, List

BENCH = "ja_mt_bench"
EXPECTED_QUESTIONS = 80


def analyze_judgment_file(filepath: Path) -> Dict:
    """
    Analyze a judgment file for duplicates.

    Distinguishes between:
    - Complete duplicates: All questions appear the same number of times (GOOD - like multiple annotators)
    - Partial duplicates: Questions have uneven repetition (BAD - skews results)

    Returns:
        Dict with detailed statistics
    """
    result = {
        "file": str(filepath),
        "total_lines": 0,
        "models": {},
        "error": None,
        "needs_cleaning": False,
        "has_missing": False,
    }

    if not filepath.exists():
        result["error"] = f"File does not exist: {filepath}"
        return result

    # Track model -> (question_id, turn) -> count
    model_question_turn_counts: Dict[str, Dict[Tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))
    # Track model -> question_id -> list of line data (for deduplication)
    model_question_lines: Dict[str, Dict[Tuple[int, int], List]] = defaultdict(lambda: defaultdict(list))
    # Track model -> set of question_ids (to check completeness)
    model_question_ids: Dict[str, Set[int]] = defaultdict(set)

    try:
        lines = []
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                result["total_lines"] += 1
                lines.append(line)

                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    result["error"] = f"Line {line_num}: Invalid JSON - {e}"
                    return result

                # Extract model ID
                model = data.get("model") or data.get("model_id")
                if not model:
                    result["error"] = f"Line {line_num}: Missing 'model' or 'model_id' field"
                    return result

                # Extract question ID
                question_id = data.get("question_id")
                if question_id is None:
                    result["error"] = f"Line {line_num}: Missing 'question_id' field"
                    return result

                turn = data.get("turn", 1)

                question_key = (int(question_id), int(turn))

                model_question_turn_counts[model][question_key] += 1
                model_question_lines[model][question_key].append((line_num, line, data))
                model_question_ids[model].add(int(question_id))

    except Exception as e:
        result["error"] = f"Failed to read file: {e}"
        return result

    # Analyze each model's duplication pattern
    for model, question_counts in model_question_turn_counts.items():
        counts = list(question_counts.values())
        count_distribution = Counter(counts)

        model_info = {
            "total_questions": len(model_question_ids[model]),
            "count_distribution": dict(count_distribution),
            "min_count": min(counts),
            "max_count": max(counts),
            "is_uniform": len(count_distribution) == 1,
            "duplicate_type": None,
            "needs_cleaning": False,
            "uneven_questions": [],
            "missing_questions": [],
            "unexpected_questions": [],
        }

        # Completeness (per question id, ignoring turn)
        expected_ids = set(range(1, EXPECTED_QUESTIONS + 1))
        actual_ids = set(model_question_ids[model])
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids

        if missing:
            model_info["missing_questions"] = sorted(missing)
            result["has_missing"] = True
        if extra:
            model_info["unexpected_questions"] = sorted(extra)
        # Determine duplication type (per question+turn)
        if model_info["is_uniform"]:
            if model_info["min_count"] == 1:
                model_info["duplicate_type"] = "no_duplicates"
            else:
                model_info["duplicate_type"] = "complete_uniform"
                # Complete duplicates are GOOD - multiple annotations
        else:
            model_info["duplicate_type"] = "partial_uneven"
            model_info["needs_cleaning"] = True
            result["needs_cleaning"] = True

            # Find which questions/turns have uneven counts
            for (qid, turn), count in question_counts.items():
                if count != model_info["min_count"]:
                    model_info["uneven_questions"].append(((qid, turn), count))
            model_info["uneven_questions"].sort(key=lambda x: x[1], reverse=True)

        result["models"][model] = model_info

    # Store line data for potential cleaning
    result["_line_data"] = model_question_lines
    result["_all_lines"] = lines

    return result


def clean_partial_duplicates(filepath: Path, analysis: Dict) -> Dict:
    """
    Clean only partial/uneven duplicates by keeping first occurrence.
    Complete uniform duplicates are preserved.

    Returns:
        Dict with cleaning results
    """
    result = {
        "cleaned": False,
        "original_lines": analysis["total_lines"],
        "new_lines": 0,
        "lines_removed": 0,
    }

    if not analysis["needs_cleaning"]:
        return result

    model_question_lines = analysis["_line_data"]
    cleaned_lines = []

    # For each model, keep only first occurrence of each question+turn
    # This normalizes all models to 1x each question+turn
    seen_pairs = set()

    for line_num, line in enumerate(analysis["_all_lines"], 1):
        try:
            data = json.loads(line.strip())
            model = data.get("model") or data.get("model_id")
            question_id = data.get("question_id")
            turn = data.get("turn", 1)

            key = (model, question_id, turn)
            if key not in seen_pairs:
                seen_pairs.add(key)
                cleaned_lines.append(line)
            else:
                result["lines_removed"] += 1
        except:
            # Keep malformed lines as-is (shouldn't happen if analysis passed)
            cleaned_lines.append(line)

    result["new_lines"] = len(cleaned_lines)

    # Write cleaned file
    try:
        # Create backup
        backup_path = filepath.with_suffix(filepath.suffix + ".backup")
        shutil.copy2(filepath, backup_path)

        # Write cleaned file
        with open(filepath, 'w') as f:
            f.writelines(cleaned_lines)

        result["cleaned"] = True
        result["backup_path"] = str(backup_path)

    except Exception as e:
        result["error"] = f"Failed to write cleaned file: {e}"
        return result

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean judgment files with partial/uneven duplicates.\n\n"
                    "Complete uniform duplicates (all questions appear same # of times) are preserved\n"
                    "as they act like multiple annotations and improve reliability.\n\n"
                    "Only cleans partial duplicates that would skew mean scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="ja_mt_bench",
        help="Benchmark name (default: ja_mt_bench)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Specific judgment file to process (if not provided, processes all)"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts and clean all files that need it"
    )

    args = parser.parse_args()

    judgment_dir = Path(f"data/{args.bench_name}/model_judgment")

    if not judgment_dir.exists():
        print(f"ERROR: Judgment directory not found: {judgment_dir}")
        sys.exit(1)

    if args.file:
        judgment_files = [Path(args.file)]
    else:
        judgment_files = sorted(judgment_dir.glob("*_single.jsonl"))

    if not judgment_files:
        print(f"WARNING: No judgment files found in {judgment_dir}")
        sys.exit(0)

    print(f"🔍 Analyzing {len(judgment_files)} judgment file(s)...")
    print("\nℹ️  Note: Complete uniform duplicates are PRESERVED (they improve reliability)")
    print("   Only partial/uneven duplicates that would skew results are cleaned.\n")
    print("=" * 80)

    files_to_clean = []
    files_complete_uniform = []
    files_clean = []
    files_incomplete = []
    files_with_unexpected = []

    # Analyze all files first
    for filepath in judgment_files:
        judge_name = filepath.stem.replace("_single", "")

        analysis = analyze_judgment_file(filepath)

        if analysis["error"]:
            print(f"\n❌ {judge_name}: ERROR - {analysis['error']}")
            continue

        print(f"\n📋 {judge_name}")
        print("-" * 80)
        print(f"Total lines: {analysis['total_lines']}")
        print(f"Models: {len(analysis['models'])}")

        # Show per-model statistics
        for model, info in sorted(analysis['models'].items()):
            print(f"\n  • {model}")
            print(f"    Questions: {info['total_questions']}")
            print(f"    Repetition: {info['count_distribution']}")
            if info["missing_questions"]:
                print(f"    Missing questions: {len(info['missing_questions'])}")
            if info["unexpected_questions"]:
                print(f"    Unexpected questions: {len(info['unexpected_questions'])}")

            if info["duplicate_type"] == "no_duplicates":
                print(f"    Status: ✓ Clean (no duplicates)")
            elif info["duplicate_type"] == "complete_uniform":
                print(f"    Status: ✓ Complete uniform duplicates (GOOD - like {info['min_count']}x annotations per question/turn)")
            elif info["duplicate_type"] == "partial_uneven":
                print(f"    Status: ⚠ PARTIAL DUPLICATES (needs cleaning)")
                print(f"    Uneven question/turn pairs: {len(info['uneven_questions'])}")
                # Show a few examples
                for (qid, turn), count in info['uneven_questions'][:3]:
                    print(f"      - Question {qid} turn {turn}: {count}x (should be uniform)")
                if len(info['uneven_questions']) > 3:
                    print(f"      ... and {len(info['uneven_questions']) - 3} more")

        if analysis["needs_cleaning"]:
            files_to_clean.append((filepath, judge_name, analysis))
            print(f"\n  ⚠ This file NEEDS CLEANING (uneven duplicates)")
        elif any(m["duplicate_type"] == "complete_uniform" for m in analysis["models"].values()):
            files_complete_uniform.append(judge_name)
            print(f"\n  ✓ This file is OK (complete uniform duplicates preserved)")
        else:
            files_clean.append(judge_name)
            print(f"\n  ✓ This file is OK (no duplicates)")

        if any(m["missing_questions"] for m in analysis["models"].values()):
            files_incomplete.append(judge_name)
            print(f"  ⚠ Incomplete coverage detected (regenerate missing questions)")
        if any(m["unexpected_questions"] for m in analysis["models"].values()):
            files_with_unexpected.append(judge_name)
            print(f"  ⚠ Unexpected question IDs present (check bench consistency)")

    # Summary
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    print(f"Files analyzed: {len(judgment_files)}")
    print(f"  ✓ Clean (no duplicates): {len(files_clean)}")
    print(f"  ✓ Complete uniform duplicates (preserved): {len(files_complete_uniform)}")
    print(f"  ⚠ Needs cleaning (partial duplicates): {len(files_to_clean)}")
    print(f"  ⚠ Incomplete coverage: {len(files_incomplete)}")
    print(f"  ⚠ Unexpected question IDs: {len(files_with_unexpected)}")

    if not files_to_clean:
        print(f"\n✅ All files are OK! No cleaning needed.")
        # Incomplete/unexpected are still problematic for reliability
        if files_incomplete or files_with_unexpected:
            sys.exit(1)
        sys.exit(0)

    # Interactive cleaning
    print(f"\n" + "=" * 80)
    print("🧹 CLEANING PHASE")
    print("=" * 80)

    files_cleaned = 0
    files_skipped = 0

    for filepath, judge_name, analysis in files_to_clean:
        print(f"\n📋 {judge_name}")
        print("-" * 80)

        # Show what will be cleaned
        models_affected = [m for m, info in analysis["models"].items() if info["needs_cleaning"]]
        print(f"Models with partial duplicates: {len(models_affected)}")
        for model in models_affected:
            info = analysis["models"][model]
            print(f"  • {model}: {len(info['uneven_questions'])} uneven questions")

        print(f"\nThis will reduce all models to 1x each question (keep first occurrence)")
        print(f"Original lines: {analysis['total_lines']}")

        # Ask for confirmation
        if args.yes:
            response = 'y'
        else:
            response = input(f"\nClean this file? [y/N]: ").strip().lower()

        if response == 'y':
            print(f"Cleaning...")
            clean_result = clean_partial_duplicates(filepath, analysis)

            if clean_result.get("error"):
                print(f"  ❌ ERROR: {clean_result['error']}")
                files_skipped += 1
            elif clean_result["cleaned"]:
                print(f"  ✓ Cleaned successfully")
                print(f"    New lines: {clean_result['new_lines']}")
                print(f"    Lines removed: {clean_result['lines_removed']}")
                print(f"    Backup: {clean_result['backup_path']}")
                files_cleaned += 1
            else:
                print(f"  • No changes needed")
                files_skipped += 1
        else:
            print(f"  ⊘ Skipped")
            files_skipped += 1

    # Final summary
    print(f"\n" + "=" * 80)
    print("✅ COMPLETE")
    print("=" * 80)
    print(f"Files cleaned: {files_cleaned}")
    print(f"Files skipped: {files_skipped}")

    if files_cleaned > 0:
        print(f"\n✓ Cleaning complete! Backup files created with .backup extension.")
        print(f"  Review the changes and delete .backup files when satisfied.")

    exit_code = 0
    if files_incomplete or files_with_unexpected:
        print("\n⚠ Incomplete or unexpected question IDs remain. Regenerate those judgments before scoring.")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
