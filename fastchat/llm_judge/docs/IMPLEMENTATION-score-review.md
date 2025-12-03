# Score Review Implementation Notes

This doc records how the scoring/analysis scripts behave when judgment data contain duplicates or gaps, and what we should do to keep results trustworthy.

## Validation first
- Run `python validate_judgment_files.py` before any analysis. It now flags as errors:
  - Duplicate entries per `(question_id, turn)`
  - Missing question IDs (incomplete coverage)
  - Unexpected question IDs
  - It tolerates multiple turns per question (turns are not treated as duplicates).
- Use `python deduplicate_judgments.py --dry-run` to see how many lines would be dropped; rerun without `--dry-run` only after confirming backups are acceptable. It:
  - Cleans only uneven duplicates per `(model, question_id, turn)` by keeping the first instance of each pair.
  - Preserves uniform duplicate coverage (e.g., full double-annotation) as “good” data.
  - Reports missing or unexpected question IDs but does not auto-fix them.
- Do not publish or compare scores produced from files that still report validator errors.

## Script behavior with bad inputs
- `judge-comparison-stats.py`  
  - Loads one score per `(question_id, model, turn)` by building a dict; if a file has duplicates, the **last** occurrence silently overwrites earlier ones.  
  - Uses the set intersection of all judges’ keys, so any missing entry from any judge deletes that item from the DataFrame entirely. Incomplete coverage shrinks the dataset with no warning and skews correlation/mean calculations toward the overlap only.
- `compare-judges.py`  
  - Aggregates all scores in lists, so duplicate lines are counted multiple times in the averages (duplicates overweight a model/question/turn).  
  - By default it filters to models present in **all** judges; `--include-incomplete` lets partial models through but still leaves per-category/turn gaps as `N/A` without flagging data loss.  
  - Missing categories or turns for a model/judge appear as `N/A`, and overall averages are still reported for that judge, which can hide uneven denominators.
- `results-table.py`  
  - Also appends every score into lists, so duplicates increase that category’s weight.  
  - Missing categories/turns become `nan` in the category rows; overall averages are computed over the remaining valid categories, so different judges can be averaged over different counts without a warning.  
  - Models not present in the requested list are ignored, but there is no explicit signal that a requested model was dropped for lack of data.
- `visualize-results.py`  
  - Duplicates are averaged in (they stay in the list, so repeated rows change means).  
  - Missing turns/categories default to `0`, which drags radar plots downward and makes absence look like a true low score rather than missing data.  
  - Per-turn averages default to `0` when a turn is absent, and the “overall” average uses that zero unless turn 2 is missing (then it uses only turn 1). There is no warning when a model is only partially judged.

## What we should do
- Treat validation as mandatory: block score generation until `validate_judgment_files.py` reports zero errors and zero duplicates (after `deduplicate_judgments.py` if needed).
- Make missing coverage explicit: fail fast or at least warn loudly in the four scripts when a model/judge is missing questions, turns, or categories instead of silently shrinking datasets or inserting zeros.
- Keep denominators consistent: when averages require the full 80 questions (or both turns), skip publishing numbers until coverage is complete; if partial analysis is necessary, label it as such and avoid mixing it with complete runs.
- Avoid overweighting duplicates: deduplicate before running `compare-judges.py`, `results-table.py`, or `visualize-results.py`, or add a guard that enforces one score per `(model, question_id, turn)` while reading.
- Prefer overlap-aware analysis: for exploratory comparisons on incomplete data, note that `judge-comparison-stats.py` already restricts to the intersection; document the reduced sample size alongside any plots/tables so readers know the coverage.
