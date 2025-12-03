# visualize-results.py: Required Fixes for 0 Defaults

## Locations to Fix

Run this to find all problematic patterns:
```bash
grep -n "\.get.*0)" visualize-results.py | grep -E "turn|category"
```

## Pattern to Search For

Look for any of these patterns:
```python
.get(1, 0)
.get(2, 0)  
.get(turn, 0)
.get(category, 0)
```

## Required Changes

### For Turn Averages

**Before**:
```python
turn1_avg = avg_scores_by_judge_model_turn[judge_name][model].get(1, 0)
turn2_avg = avg_scores_by_judge_model_turn[judge_name][model].get(2, 0)
overall_avg = (turn1_avg + turn2_avg) / 2 if turn2_avg > 0 else turn1_avg
```

**After**:
```python
turn1_avg = avg_scores_by_judge_model_turn[judge_name][model].get(1, None)
turn2_avg = avg_scores_by_judge_model_turn[judge_name][model].get(2, None)

if turn1_avg is not None and turn2_avg is not None:
    overall_avg = (turn1_avg + turn2_avg) / 2
elif turn1_avg is not None:
    overall_avg = turn1_avg
    print(f"  ⚠️ Warning: Turn 2 missing for {model}, using Turn 1 only")
elif turn2_avg is not None:
    overall_avg = turn2_avg  
    print(f"  ⚠️ Warning: Turn 1 missing for {model}, using Turn 2 only")
else:
    print(f"  ⚠️ Warning: No turns available for {model}, skipping")
    continue
```

### For Category Averages in Radar Plots

Search for code building radar plot data - likely has:
```python
category_scores = [scores_by_category.get(cat, 0) for cat in CATEGORIES]
```

Should be:
```python
category_scores = []
missing_categories = []
for cat in CATEGORIES:
    score = scores_by_category.get(cat, None)
    if score is not None:
        category_scores.append(score)
    else:
        category_scores.append(np.nan)  # or skip the model entirely
        missing_categories.append(cat)

if missing_categories:
    print(f"  ⚠️ {model}: missing categories {missing_categories}")
```

## Why This Matters

**Current behavior** (wrong):
- Model has Turn 1: 8.5, Turn 2: missing
- Code shows: Turn 1: 8.5, Turn 2: 0.0, Overall: 4.25
- **User sees**: Model performs terribly on Turn 2!

**Correct behavior**:
- Model has Turn 1: 8.5, Turn 2: missing  
- Code shows: Turn 1: 8.5, Turn 2: (missing), Overall: 8.5
- **User sees**: Model only evaluated on Turn 1, warning displayed

## Testing After Fix

```bash
# Create test file with missing Turn 2
cat > /tmp/test_missing_turn.jsonl << 'EOF'
{"model": "test-model", "question_id": 1, "turn": 1, "score": 8.5}
{"model": "test-model", "question_id": 2, "turn": 1, "score": 7.0}
EOF

# Run visualize
python3 visualize-results.py \
  --models "test-model" \
  --judges test \
  --input-file /tmp/test_missing_turn.jsonl

# Should see:
# ⚠️ Warning: Turn 2 missing for test-model, using Turn 1 only
# Turn 1 Average: 7.75
# Overall: 7.75 (not 3.875!)
```

## Priority: HIGH

This is the **only remaining correctness issue** in your canonicalization implementation. Everything else is working correctly!
