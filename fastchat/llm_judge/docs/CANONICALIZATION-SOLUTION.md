# Score Canonicalization: The Elegant Solution

## Problem Statement

When judgment files have duplicate entries for some questions:
- **Naive averaging**: Questions with more duplicates get overweighted in the overall mean
- **Deduplication**: Throws away extra judgments, losing variance reduction
- **Constant validation**: Need to clean/check files before every calculation

## The Elegant Solution: Per-Question Averaging

Instead of cleaning duplicates, **canonicalize scores before aggregation**:

```
For each (judge, model, question_id, turn):
  1. Compute mean of all judgment scores
  2. Create one canonical score
  3. Run all existing logic on canonical scores
```

### Why This Works

**Equal weight preserved**:
- Each question contributes exactly once to overall mean
- Doesn't matter if Q1 has 3 judgments and Q2 has 1
- Distribution stays as intended (equal weight per question)

**Variance reduction retained**:
- Extra judgments → lower variance for that question's score
- More confident in per-question mean
- Like having multiple annotators for harder questions

**No data loss**:
- Don't throw away extra judgments
- Don't need to constantly clean/validate
- Works with any duplication pattern

### Implementation

We've added canonicalization to both scoring scripts:

**`show_result.py`**:
```python
from canonicalize_scores import canonicalize_judgments

# Load raw judgments
with open(input_file, 'r') as f:
    raw_judgments = [json.loads(line) for line in f]

# Canonicalize: one score per (model, question_id, turn)
canonical = canonicalize_judgments(raw_judgments, judge_name)

# Use canonical for all downstream calculations
df = pd.DataFrame(canonical)
```

**`results-table.py`**: Same pattern

**Also canonicalized now**: `compare-judges.py`, `judge-comparison-stats.py`, `visualize-results.py` so every scoring/plot path respects equal per-question weighting.

### Example

**Before (naive averaging)**:
```
Model A:
  Q1: [8.5, 8.5, 8.5]  (3 judgments)
  Q2: [7.0]            (1 judgment)

Naive mean = (8.5 + 8.5 + 8.5 + 7.0) / 4 = 8.0  ❌ Q1 overweighted!
```

**After (canonicalization)**:
```
Model A (canonical):
  Q1: 8.5  (mean of 3 judgments)
  Q2: 7.0  (mean of 1 judgment)

Canonical mean = (8.5 + 7.0) / 2 = 7.75  ✓ Equal weight!
```

### Benefits

1. **Correct weighting**: Each question contributes equally
2. **Variance reduction**: Extra judgments improve confidence per question
3. **No data loss**: All judgments used, none thrown away
4. **No maintenance**: Don't need to clean/validate duplicates
5. **Transparent**: Can see `_n_judgments` and `_score_std` in canonical data

### What About Missing Questions?

- Still reduced coverage (shows in completeness checks)
- But doesn't distort weights of present questions
- Reports still surface missing coverage

### Testing

```bash
cd /root/ja-mt-bench-harness/fastchat/llm_judge

# Test canonicalization standalone
python3 canonicalize_scores.py \
  data/ja_mt_bench/model_judgment/gpt-4-turbo_single.jsonl \
  --judge gpt-4-turbo \
  -o /tmp/canonical.jsonl

# Shows which models had duplicates averaged
# Output includes _n_judgments and _score_std for transparency

# Use updated scoring scripts (canonicalization automatic)
python3 show_result.py --judge-model gpt-4-turbo
python3 results-table.py --model-list "shisa-ai/shisa-v2-llama3.3-70b"
python3 compare-judges.py
python3 judge-comparison-stats.py
python3 visualize-results.py --model-list "shisa-ai/shisa-v2-llama3.3-70b"
```

### Philosophy

**Duplicates aren't inherently bad or good** - it depends on how you aggregate:

- ✓ **With canonicalization**: Extra judgments = reduced variance, equal weight
- ❌ **Without canonicalization**: Extra judgments = distorted weights
- ℹ️ Canonicalization reports skipped rows (missing model/question/score or invalid score) so you see any data that was dropped.

The solution isn't to clean the data, it's to **aggregate correctly**!

---

This approach is mathematically equivalent to standard practice in multi-annotator scenarios where you:
1. Compute inter-annotator agreement per item
2. Average per-item scores
3. Report overall metrics

The duplication just happens to be from multiple API calls rather than multiple human annotators.
