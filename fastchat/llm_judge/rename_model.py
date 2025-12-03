#!/usr/bin/env python3
"""
Rename a model across all answer and judgment files.
Useful for promoting ablation experiments to final model names.
"""

import json
import sys
import shutil
from pathlib import Path
from typing import Set, List, Dict
from collections import defaultdict

BENCH = "ja_mt_bench"


def list_models(bench_name: str) -> Dict[str, Set[str]]:
    """List all models found in answer and judgment files."""
    models = {"answers": set(), "judgments": set()}

    # Find models in answer files
    answer_dir = Path(f"data/{bench_name}/model_answer")
    if answer_dir.exists():
        for answer_file in answer_dir.glob("*.jsonl"):
            try:
                with open(answer_file, 'r') as f:
                    first_line = f.readline()
                    if first_line:
                        data = json.loads(first_line)
                        model_id = data.get("model_id")
                        if model_id:
                            models["answers"].add(model_id)
            except Exception as e:
                print(f"Warning: Could not read {answer_file}: {e}", file=sys.stderr)

    # Find models in judgment files
    judgment_dir = Path(f"data/{bench_name}/model_judgment")
    if judgment_dir.exists():
        for judgment_file in judgment_dir.glob("*_single.jsonl"):
            try:
                with open(judgment_file, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        model = data.get("model") or data.get("model_id")
                        if model:
                            models["judgments"].add(model)
            except Exception as e:
                print(f"Warning: Could not read {judgment_file}: {e}", file=sys.stderr)

    return models


def rename_in_answer_file(filepath: Path, old_name: str, new_name: str, dry_run: bool = False) -> Dict:
    """Rename model in an answer file."""
    result = {
        "file": str(filepath),
        "lines_modified": 0,
        "error": None
    }

    try:
        modified_lines = []
        modified = False

        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if data.get("model_id") == old_name:
                        data["model_id"] = new_name
                        modified_lines.append(json.dumps(data, ensure_ascii=False) + '\n')
                        result["lines_modified"] += 1
                        modified = True
                    else:
                        modified_lines.append(line)
                except json.JSONDecodeError as e:
                    result["error"] = f"Line {line_num}: Invalid JSON - {e}"
                    return result

        if modified and not dry_run:
            # Create backup
            backup_path = filepath.with_suffix(filepath.suffix + ".backup")
            shutil.copy2(filepath, backup_path)

            # Write modified file
            with open(filepath, 'w') as f:
                f.writelines(modified_lines)

        result["modified"] = modified

    except Exception as e:
        result["error"] = f"Failed to process file: {e}"

    return result


def rename_in_judgment_file(filepath: Path, old_name: str, new_name: str, dry_run: bool = False) -> Dict:
    """Rename model in a judgment file."""
    result = {
        "file": str(filepath),
        "lines_modified": 0,
        "error": None
    }

    try:
        modified_lines = []
        modified = False

        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    # Check both 'model' and 'model_id' fields
                    if data.get("model") == old_name:
                        data["model"] = new_name
                        result["lines_modified"] += 1
                        modified = True
                    if data.get("model_id") == old_name:
                        data["model_id"] = new_name
                        result["lines_modified"] += 1
                        modified = True

                    modified_lines.append(json.dumps(data, ensure_ascii=False) + '\n')
                except json.JSONDecodeError as e:
                    result["error"] = f"Line {line_num}: Invalid JSON - {e}"
                    return result

        if modified and not dry_run:
            # Create backup
            backup_path = filepath.with_suffix(filepath.suffix + ".backup")
            shutil.copy2(filepath, backup_path)

            # Write modified file
            with open(filepath, 'w') as f:
                f.writelines(modified_lines)

        result["modified"] = modified

    except Exception as e:
        result["error"] = f"Failed to process file: {e}"

    return result


def sanitize_model_name(name: str) -> str:
    """Convert model name to safe filename format (matches generate-answers.sh)."""
    return name.replace("/", "__")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rename a model across all answer and judgment files.",
        epilog="Examples:\n"
               "  List all models:\n"
               "    python rename_model.py --list\n\n"
               "  Rename a model (dry run):\n"
               "    python rename_model.py --old shisa-ai/168-llama3.3-70b-v2.1-sft --new shisa-ai/shisa-v3-70b --dry-run\n\n"
               "  Rename a model (actual):\n"
               "    python rename_model.py --old shisa-ai/168-llama3.3-70b-v2.1-sft --new shisa-ai/shisa-v3-70b\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--bench-name",
        type=str,
        default="ja_mt_bench",
        help="Benchmark name (default: ja_mt_bench)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all models found in answer and judgment files"
    )
    parser.add_argument(
        "--old",
        type=str,
        help="Old model name to rename from"
    )
    parser.add_argument(
        "--new",
        type=str,
        help="New model name to rename to"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually modify files, just report what would be done"
    )

    args = parser.parse_args()

    # List models if requested
    if args.list:
        print("📋 Listing all models found in benchmark data...")
        print("=" * 80)

        models = list_models(args.bench_name)

        print(f"\n🔹 Models in answer files ({len(models['answers'])}):")
        for model in sorted(models['answers']):
            print(f"  • {model}")

        print(f"\n🔹 Models in judgment files ({len(models['judgments'])}):")
        for model in sorted(models['judgments']):
            print(f"  • {model}")

        # Show models only in one location
        only_answers = models['answers'] - models['judgments']
        only_judgments = models['judgments'] - models['answers']

        if only_answers:
            print(f"\n⚠ Models ONLY in answers (no judgments):")
            for model in sorted(only_answers):
                print(f"  • {model}")

        if only_judgments:
            print(f"\n⚠ Models ONLY in judgments (no answers):")
            for model in sorted(only_judgments):
                print(f"  • {model}")

        sys.exit(0)

    # Validate rename arguments
    if not args.old or not args.new:
        parser.error("--old and --new are required for renaming (or use --list to see available models)")

    old_name = args.old
    new_name = args.new

    print(f"{'DRY RUN - ' if args.dry_run else ''}Renaming model:")
    print(f"  Old: {old_name}")
    print(f"  New: {new_name}")
    print("=" * 80)

    # Process answer files
    answer_dir = Path(f"data/{args.bench_name}/model_answer")
    old_safe = sanitize_model_name(old_name)
    new_safe = sanitize_model_name(new_name)

    target_answer_file = answer_dir / f"{new_safe}.jsonl"
    answer_results: List[Dict] = []
    matching_answer_files: List[Path] = []
    answer_file_renamed = False
    multiple_answer_matches = False
    answer_rename_blocked = False

    if not answer_dir.exists():
        print(f"\n⚠ Answer directory not found: {answer_dir}")
    else:
        print(f"\n📄 Scanning answer files in {answer_dir} ...")
        for answer_file in sorted(answer_dir.glob("*.jsonl")):
            result = rename_in_answer_file(answer_file, old_name, new_name, args.dry_run)
            answer_results.append(result)

            if result["error"]:
                print(f"  ❌ {answer_file.name}: {result['error']}")
                continue

            if result["lines_modified"] > 0:
                matching_answer_files.append(answer_file)
                print(f"  ✓ {answer_file.name}: Modified {result['lines_modified']} lines")

        if not matching_answer_files:
            print("  • No answer files contained the old model")
        elif len(matching_answer_files) > 1:
            multiple_answer_matches = True
            print(f"  ⚠ Found {len(matching_answer_files)} answer files for '{old_name}'.")
            print("    Updated contents, but skipped file renames to avoid collisions.")
        else:
            source_path = matching_answer_files[0]
            if args.dry_run:
                if source_path.name != target_answer_file.name:
                    print(f"  • Would rename {source_path.name} -> {target_answer_file.name}")
                else:
                    print(f"  • Answer file already named {target_answer_file.name}")
            else:
                if target_answer_file.exists() and target_answer_file != source_path:
                    answer_rename_blocked = True
                    print(f"  ⚠ Target answer file already exists ({target_answer_file.name});")
                    print(f"    Skipped renaming {source_path.name} to avoid overwrite.")
                elif source_path == target_answer_file:
                    print(f"  • Answer file already uses the target name")
                    answer_file_renamed = True
                else:
                    source_path.rename(target_answer_file)
                    print(f"  ✓ Renamed file to: {target_answer_file.name}")
                    print(f"  ✓ Backup created: {source_path.name}.backup")
                    answer_file_renamed = True

    # Process judgment files
    judgment_dir = Path(f"data/{args.bench_name}/model_judgment")
    judgment_results: List[Dict] = []

    if not judgment_dir.exists():
        judgment_files: List[Path] = []
    else:
        judgment_files = [
            p for p in sorted(judgment_dir.glob("*.jsonl"))
            if not p.name.endswith(".backup")
        ]

    if not judgment_files:
        print(f"\n⚠ No judgment files found in {judgment_dir}")
    else:
        print(f"\n📋 Processing {len(judgment_files)} judgment file(s)...")

        files_modified = 0
        total_lines_modified = 0

        for judgment_file in judgment_files:
            result = rename_in_judgment_file(judgment_file, old_name, new_name, args.dry_run)

            if result["error"]:
                print(f"  ❌ {judgment_file.name}: ERROR - {result['error']}")
            elif result["lines_modified"] > 0:
                print(f"  ✓ {judgment_file.name}: Modified {result['lines_modified']} lines")
                files_modified += 1
                total_lines_modified += result["lines_modified"]
                if not args.dry_run:
                    print(f"    Backup created: {judgment_file.name}.backup")

            judgment_results.append(result)

        print(f"\n  Summary: {files_modified}/{len(judgment_files)} files modified, {total_lines_modified} total lines")

    # Final summary
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)

    modified_answer_files = sum(1 for r in answer_results if r.get("modified"))
    modified_judgment_files = sum(1 for r in judgment_results if r.get("modified"))

    if args.dry_run:
        print("⚠ This was a DRY RUN. Run without --dry-run to actually rename.")
        print(f"  • Would update {modified_answer_files} answer file(s)")
        print(f"  • Would update {modified_judgment_files} judgment file(s)")
    else:
        print("✓ Rename complete!")
        if modified_answer_files or modified_judgment_files:
            print(f"\nThe model '{old_name}' has been renamed to '{new_name}' in:")
        else:
            print("\nNo files required changes.")

        if modified_answer_files:
            if multiple_answer_matches:
                detail = "content updated; multiple answer files matched (filenames unchanged)"
            elif answer_rename_blocked:
                detail = f"content updated; filename rename skipped because {target_answer_file.name} exists"
            elif answer_file_renamed:
                detail = f"content updated and filename set to {target_answer_file.name}"
            else:
                detail = "content updated"
            print(f"  • {modified_answer_files} answer file(s) ({detail})")

        if modified_judgment_files:
            print(f"  • {modified_judgment_files} judgment file(s)")

        print("\nBackup files have been created with .backup extension.")

    sys.exit(0)


if __name__ == "__main__":
    main()
