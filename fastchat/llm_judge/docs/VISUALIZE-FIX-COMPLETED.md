# ✅ FIXED: visualize-results.py 0 Defaults Issue

## Status

**COMPLETED** - The visualize-results.py 0 defaults issue has been fixed.

## Investigation Result

**THEORETICAL ISSUE** - Checked all judgment files and confirmed that all models have complete turn coverage (both Turn 1 and Turn 2). The 0 defaults were not affecting current data, but the fix prevents future issues.

## What Was Fixed

### Location
`visualize-results.py` lines 296-297 (now lines 296-316 after fix)

### Before (Wrong)
```python
turn1_avg = avg_scores_by_judge_model_turn[judge_name][model].get(1, 0)
turn2_avg = avg_scores_by_judge_model_turn[judge_name][model].get(2, 0)
overall_avg = (turn1_avg + turn2_avg) / 2 if turn2_avg > 0 else turn1_avg
```

**Problem**: Missing turns defaulted to 0, which:
- Made missing data indistinguishable from a true score of 0.0
- Would show "Overall: 4.25" when Turn 1 is 8.5 and Turn 2 is missing (averaging 8.5 + 0)
- Could mislead users into thinking model performed poorly on missing turns

### After (Correct)
```python
# Safely get turn averages with None default to distinguish missing from 0.0
turn1_avg = avg_scores_by_judge_model_turn[judge_name][model].get(1, None)
turn2_avg = avg_scores_by_judge_model_turn[judge_name][model].get(2, None)

# Calculate overall average based on available turns
if turn1_avg is not None and turn2_avg is not None:
    overall_avg = (turn1_avg + turn2_avg) / 2
elif turn1_avg is not None:
    overall_avg = turn1_avg
    print(f"  ⚠️  Warning: Turn 2 missing for {model}, using Turn 1 only")
elif turn2_avg is not None:
    overall_avg = turn2_avg
    print(f"  ⚠️  Warning: Turn 1 missing for {model}, using Turn 2 only")
else:
    print(f"  ⚠️  Warning: No turns available for {model}, skipping")
    continue

if turn1_avg is not None:
    print(f"  Turn 1 Average: {turn1_avg:.2f}")
if turn2_avg is not None:
    print(f"  Turn 2 Average: {turn2_avg:.2f}")
print(f"  Overall Average: {overall_avg:.2f}")
```

**Improvement**:
- Missing turns use `None` instead of `0`
- Explicit warnings when turns are missing
- Overall average uses only available turns
- Model with Turn 1: 8.5, Turn 2: missing → Overall: 8.5 (correct!) instead of 4.25 (wrong!)

## Test Results

```
model-complete (Turn 1: 8.50, Turn 2: 7.00):
  Turn 1: 8.50
  Turn 2: 7.00
  Overall: 7.75

model-turn1-only (Turn 1: 8.50, Turn 2: missing):
  ⚠️  Warning: Turn 2 missing, using Turn 1 only
  Turn 1: 8.50
  Overall: 8.50 (not 4.25!)

model-turn2-only (Turn 1: missing, Turn 2: 7.00):
  ⚠️  Warning: Turn 1 missing, using Turn 2 only
  Turn 2: 7.00
  Overall: 7.00
```

## Verification

Checked all judgment files in `data/ja_mt_bench/model_judgment/*_single.jsonl`:
- ✅ All models have complete turn coverage (Turn 1 and Turn 2 present)
- ✅ The 0 defaults were not affecting current data
- ✅ Fix prevents future issues if incomplete data is ever introduced

## Remaining Notes

### Radar Plot NaN Handling
There are still 4 locations where radar plots convert NaN to 0 for missing categories (lines 383, 434, 497, 554):
```python
cat_scores.append(score if not np.isnan(score) else 0)
```

**Status**: Also theoretical issue (all models have complete category coverage)
**Reason for 0 conversion**: Matplotlib radar plots require numeric values
**Alternative approaches** (if needed in future):
- Skip models with missing categories entirely (with warning)
- Use interpolation for missing categories
- Mark missing points with different style

Since current data has complete coverage, this doesn't need immediate fixing but could be improved if incomplete category data is expected in the future.

## Impact

- **Current data**: No impact (all complete)
- **Future data**: Protected from showing missing turns as terrible scores
- **User experience**: Clear warnings when data is incomplete
- **Correctness**: Ensures overall averages are computed from available data only

## Date Fixed

2025-11-29
