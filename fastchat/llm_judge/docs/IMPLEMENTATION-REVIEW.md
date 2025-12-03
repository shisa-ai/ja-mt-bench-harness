# Implementation Review: Canonicalization Changes

## Summary

Your colleague's review was **excellent** and you addressed most issues correctly. Here's the status:

## ✅ What You Fixed Correctly

### 1. Model Field Normalization ✅
**Issue**: Template kept first judgment; if it had `model_id` but not `model`, downstream code would KeyError.

**Your fix**: 
```python
template = judgment.copy()
template["model"] = model  # Always normalize to "model"
grouped[key]["judgment"] = template
```

**Status**: ✅ **CORRECT** - This ensures `df[["model", "score", "turn"]]` always works.

### 2. Logging Skipped Rows ✅
**Issue**: Silent data loss when dropping invalid entries.

**Your fix**: Track and log `skipped_missing_fields` and `skipped_invalid_scores`:
```python
if duplicate_counts or skipped_missing_fields or skipped_invalid_scores:
    print(f"Canonicalized scores for {judge_name}:")
    print(f"  Total canonical: {len(canonical)}")
    if skipped_missing_fields:
        print(f"  Skipped {skipped_missing_fields} rows missing model/question_id")
    if skipped_invalid_scores:
        print(f"  Skipped {skipped_invalid_scores} rows with invalid score")
```

**Status**: ✅ **CORRECT** - Users now see data loss immediately.

### 3. Canonicalization in All Scripts ✅
**Issue**: compare-judges.py, judge-comparison-stats.py, visualize-results.py were still using raw judgments.

**Your fix**: Added imports and canonicalization to all three:
```python
from canonicalize_scores import canonicalize_judgments
# ... then use canonicalize_judgments(raw, judge_name)
```

**Status**: ✅ **CORRECT** - All scoring paths now consistent.

## ⚠️ What Still Needs Work

### 1. visualize-results.py: 0 Defaults ⚠️

**Remaining issue**: Missing turns still default to 0:
```python
turn1_avg = avg_scores_by_judge_model_turn[judge_name][model].get(1, 0)  # ❌ 0 default
turn2_avg = avg_scores_by_judge_model_turn[judge_name][model].get(2, 0)  # ❌ 0 default
```

**Problem**:
- Missing Turn 2 → appears as score of 0.0
- Radar plots show missing data as terrible performance
- Was flagged in IMPLEMENTATION-score-review.md

**Fix needed**:
```python
turn1_avg = avg_scores_by_judge_model_turn[judge_name][model].get(1, None)
turn2_avg = avg_scores_by_judge_model_turn[judge_name][model].get(2, None)

if turn1_avg is not None and turn2_avg is not None:
    overall_avg = (turn1_avg + turn2_avg) / 2
elif turn1_avg is not None:
    overall_avg = turn1_avg
    print(f"  ⚠️ Turn 2 missing for {model}, using Turn 1 only")
else:
    print(f"  ⚠️ No turns available for {model}, skipping")
    continue
```

Also check radar plot code for similar `.get(..., 0)` patterns.

### 2. Completeness Warnings (Acknowledged as TODO) ℹ️

**Your notes say**: "Canonicalization still doesn't enforce completeness; keep running the validator if coverage matters."

**Status**: ✅ **ACCEPTABLE** - This is acknowledged and correct. Canonicalization is for correct aggregation, not coverage validation.

**Optional enhancement** (not required):
Add basic completeness check in canonicalization:
```python
# After building canonical
expected = 80  # questions
models_with_incomplete = []
for model in per_model_stats:
    if per_model_stats[model]["total_questions"] < expected:
        models_with_incomplete.append(
            (model, per_model_stats[model]["total_questions"])
        )

if models_with_incomplete and show_stats:
    print(f"\n⚠️ Incomplete coverage:")
    for model, count in models_with_incomplete:
        print(f"  {model}: {count}/{expected} questions")
```

## Testing Recommendations

### Test 1: Model Field Normalization
```bash
# Create a test judgment with only model_id
echo '{"model_id": "test", "question_id": 1, "turn": 1, "score": 8.5}' > /tmp/test.jsonl

python3 -c "
from canonicalize_scores import canonicalize_judgments
import json
with open('/tmp/test.jsonl') as f:
    raw = [json.loads(line) for line in f]
canonical = canonicalize_judgments(raw, 'test')
print('Has model field:', 'model' in canonical[0])
print('Model value:', canonical[0].get('model'))
"
# Should print: Has model field: True, Model value: test
```

### Test 2: Skipped Row Logging
```bash
# File with invalid scores
python3 canonicalize_scores.py \
  data/ja_mt_bench/model_judgment/gpt-4-turbo_single.jsonl \
  --judge test | grep -i skip
# Should show: "Skipped X rows with invalid score" if any exist
```

### Test 3: Consistency Across Scripts
```bash
# All should give same scores for same model
python3 show_result.py --judge-model gpt-4-turbo --model-list "Model-X"
python3 results-table.py --model-list "Model-X" --judge gpt-4-turbo
python3 compare-judges.py --judges gpt-4-turbo --models "Model-X"
# Scores should match across all three
```

### Test 4: visualize-results.py with Missing Turns
```bash
# If any model has incomplete turn coverage, check output
python3 visualize-results.py --models "Model-With-Missing-Turn2" --judges gpt-4-turbo
# Currently: Will show Turn 2 as 0.0 (wrong!)
# Should: Warn and skip or use NaN
```

## What You Didn't Miss (Good!)

✅ Canonicalization key `(model, question_id, turn)` - correct  
✅ Handles both `model` and `model_id` in input - correct  
✅ Preserves `_n_judgments` and `_score_std` metadata - correct  
✅ Works with `--stats` flag for detailed analysis - correct  
✅ All three previously-uncovered scripts updated - correct  

## Priority

**HIGH**: Fix visualize-results.py 0 defaults  
- This silently makes missing data look like bad scores
- Misleading for users looking at radar plots
- Was specifically flagged in IMPLEMENTATION-score-review.md

**MEDIUM**: Add optional completeness warnings  
- Nice to have but not critical
- Validator handles this if needed

**LOW**: Other enhancements  
- Coverage stats in all scripts
- Per-turn breakdown in output
- (These are UX improvements)

## Overall Assessment

**Grade: A-**

You addressed the core correctness issues perfectly:
- ✅ Model field normalization
- ✅ Logging skipped rows
- ✅ Consistent canonicalization across all scripts

One remaining issue (visualize 0 defaults) prevents a perfect score, but it's a straightforward fix.

Your implementation is **production-ready** for show_result.py, results-table.py, compare-judges.py, and judge-comparison-stats.py. Just fix visualize-results.py before using it for published plots/reports.
