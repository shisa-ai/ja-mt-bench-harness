#!/usr/bin/env python3
"""
Canonicalize judgment scores by computing mean per (question_id, turn, model).

This ensures:
- Equal weight per question (duplicates don't shift overall mean)
- Duplicates only reduce variance within each question
- Missing questions are visible but don't distort weights
- No need to constantly clean/validate duplicates

Usage:
    from canonicalize_scores import canonicalize_judgments

    # Load raw judgments
    raw_judgments = [json.loads(line) for line in f]

    # Canonicalize: one score per (model, question_id, turn)
    canonical = canonicalize_judgments(raw_judgments)

    # Use canonical for all downstream aggregation
"""

import json
from collections import defaultdict
from typing import List, Dict, Any
import numpy as np


def canonicalize_judgments(
    judgments: List[Dict[str, Any]],
    judge_name: str = None,
    show_stats: bool = False,
) -> List[Dict[str, Any]]:
    """
    Canonicalize judgments by averaging scores per (model, question_id, turn).

    For each unique (model, question_id, turn) combination:
    - Compute mean of all scores (if multiple judgments exist)
    - Keep first judgment's metadata
    - Replace score with the mean

    Args:
        judgments: List of judgment dicts (one per line from JSONL)
        judge_name: Optional judge name for logging
        show_stats: If True, print detailed statistics

    Returns:
        List of canonical judgments (one per unique question/turn/model)
    """
    # Group by (model, question_id, turn)
    grouped = defaultdict(lambda: {"scores": [], "judgment": None})

    skipped_missing_fields = 0
    skipped_invalid_scores = 0

    for judgment in judgments:
        # Determine model key
        model = judgment.get("model") or judgment.get("model_id")
        if not model:
            skipped_missing_fields += 1
            continue

        question_id = judgment.get("question_id")
        if question_id is None:
            skipped_missing_fields += 1
            continue

        turn = judgment.get("turn", 1)  # Default to turn 1 if not specified
        score = judgment.get("score")

        if score is None or score == -1:  # Skip invalid scores
            skipped_invalid_scores += 1
            continue

        key = (model, question_id, turn)

        # Store score for averaging
        grouped[key]["scores"].append(score)

        # Keep first judgment as template
        if grouped[key]["judgment"] is None:
            # Copy and normalize to always have "model" populated for downstream code
            template = judgment.copy()
            template["model"] = model
            grouped[key]["judgment"] = template

    # Build canonical judgments and collect statistics
    canonical = []
    duplicate_counts = defaultdict(int)
    per_model_stats = defaultdict(lambda: {
        "total_questions": 0,
        "duplicated_questions": 0,
        "duplication_counts": defaultdict(int),  # count -> how many questions
        "max_duplicates": 0,
        "total_variance_reduced": [],  # std devs for duplicated questions
        "example_duplicates": []  # (question_id, turn, n_judgments, std)
    })

    for (model, question_id, turn), data in grouped.items():
        scores = data["scores"]
        judgment = data["judgment"]

        if len(scores) == 0:
            continue

        # Compute mean score
        mean_score = np.mean(scores)
        n_scores = len(scores)

        # Update statistics
        stats = per_model_stats[model]
        stats["total_questions"] += 1
        stats["duplication_counts"][n_scores] += 1

        # Track duplicates for reporting
        if n_scores > 1:
            duplicate_counts[model] += 1
            stats["duplicated_questions"] += 1
            stats["max_duplicates"] = max(stats["max_duplicates"], n_scores)

            score_std = float(np.std(scores))
            stats["total_variance_reduced"].append(score_std)

            # Keep some examples
            if len(stats["example_duplicates"]) < 5:
                stats["example_duplicates"].append((question_id, turn, n_scores, score_std))

        # Update judgment with mean score
        judgment["score"] = mean_score

        # Optionally add metadata about averaging
        if n_scores > 1:
            judgment["_n_judgments"] = n_scores
            judgment["_score_std"] = float(np.std(scores))

        canonical.append(judgment)

    # Log basic statistics
    if duplicate_counts or skipped_missing_fields or skipped_invalid_scores:
        print(f"Canonicalized scores{' for ' + judge_name if judge_name else ''}:")
        print(f"  Total canonical (unique question,turn,model): {len(canonical)}")
        if duplicate_counts:
            for model, count in sorted(duplicate_counts.items()):
                print(f"  {model}: averaged {count} duplicated question/turn pairs")
        if skipped_missing_fields:
            print(f"  Skipped {skipped_missing_fields} rows missing model/question_id")
        if skipped_invalid_scores:
            print(f"  Skipped {skipped_invalid_scores} rows with invalid score")

    # Show detailed statistics if requested
    if show_stats and per_model_stats:
        print(f"\n{'='*80}")
        print(f"DETAILED CANONICALIZATION STATISTICS")
        print(f"{'='*80}")

        for model in sorted(per_model_stats.keys()):
            stats = per_model_stats[model]
            print(f"\n📊 {model}")
            print(f"{'─'*80}")
            print(f"  Total (question, turn) pairs: {stats['total_questions']}")
            print(f"  Pairs with duplicates: {stats['duplicated_questions']}")
            print(f"  Pairs with no duplicates: {stats['total_questions'] - stats['duplicated_questions']}")

            # Show duplication distribution
            print(f"\n  Duplication distribution:")
            for count in sorted(stats['duplication_counts'].keys()):
                n_questions = stats['duplication_counts'][count]
                pct = 100 * n_questions / stats['total_questions']
                if count == 1:
                    print(f"    {count}x (no duplicates): {n_questions} questions ({pct:.1f}%)")
                else:
                    print(f"    {count}x judgments: {n_questions} questions ({pct:.1f}%)")

            # Show variance reduction
            if stats['total_variance_reduced']:
                avg_std = np.mean(stats['total_variance_reduced'])
                print(f"\n  Variance reduction (for duplicated questions):")
                print(f"    Mean std dev: {avg_std:.3f}")
                print(f"    Max std dev: {max(stats['total_variance_reduced']):.3f}")
                print(f"    → Extra judgments reduced uncertainty by averaging")

            # Show examples
            if stats['example_duplicates']:
                print(f"\n  Example duplicated questions:")
                for qid, turn, n, std in stats['example_duplicates']:
                    print(f"    Q{qid} Turn{turn}: {n} judgments, std={std:.3f}")

        print(f"\n{'='*80}")
        print(f"✓ All duplicates averaged - equal weight per question maintained!")
        print(f"{'='*80}\n")

    return canonical


def canonicalize_judgment_file(input_path: str, output_path: str = None, judge_name: str = None, show_stats: bool = False) -> int:
    """
    Canonicalize a judgment file by averaging duplicate question/turn scores.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file (optional, if None just returns count)
        judge_name: Judge name for logging
        show_stats: If True, print detailed statistics

    Returns:
        Number of canonical judgments
    """
    # Load judgments
    with open(input_path, 'r') as f:
        judgments = [json.loads(line) for line in f]

    # Canonicalize
    canonical = canonicalize_judgments(judgments, judge_name, show_stats)

    # Write if output specified
    if output_path:
        with open(output_path, 'w') as f:
            for judgment in canonical:
                f.write(json.dumps(judgment, ensure_ascii=False) + '\n')
        print(f"Wrote {len(canonical)} canonical judgments to {output_path}")

    return len(canonical)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Canonicalize judgment files by averaging duplicate question/turn scores."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input JSONL judgment file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output JSONL file (optional, for inspection)"
    )
    parser.add_argument(
        "--judge",
        type=str,
        help="Judge name for logging"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed statistics about canonicalization"
    )

    args = parser.parse_args()

    canonicalize_judgment_file(args.input, args.output, args.judge, args.stats)
