# Final Solution Summary: Score Canonicalization

## The Insight (Thanks to You & Your Colleague!)

**The problem isn't duplicate data - it's how we aggregate!**

Your colleague's insight was brilliant:
> "If you aggregate by averaging raw rows, extra judgments on some questions will overweight those questions... 
> Safer: compute the mean per question (or per question/turn) then average those means."

## What We Implemented

### 1. **canonicalize_scores.py** (New Utility)

Preprocessing step that computes mean per `(model, question_id, turn)`:

```python
from canonicalize_scores import canonicalize_judgments

# Load raw judgments (may have duplicates)
raw_judgments = [json.loads(line) for line in f]

# Canonicalize: one score per question/turn
canonical = canonicalize_judgments(raw_judgments, judge_name)

# Now all downstream logic uses canonical scores
```

**What it does**:
- Groups judgments by `(model, question_id, turn)`
- Computes mean score for each group
- Returns one canonical judgment per group
- Adds `_n_judgments` and `_score_std` for transparency

### 2. **Updated Scoring Scripts**

Both `show_result.py` and `results-table.py` now automatically canonicalize:

```bash
# Just use them normally - canonicalization happens automatically!
python3 show_result.py --judge-model gpt-4-turbo
python3 results-table.py --model-list "your-model-name"
```

Output shows which models had duplicates averaged:
```
Canonicalized scores for gpt-4-turbo:
  Total unique questions: 7831
  Qwen/Qwen3-235B-A22B: averaged 159 duplicate questions
```

## Why This Is Better Than Deduplication

| Approach | Duplicate Handling | Variance Reduction | Equal Weighting | Maintenance |
|----------|-------------------|-------------------|----------------|-------------|
| **Naive** (old) | Keeps all | ✓ But distorts weights | ❌ | None |
| **Deduplication** | Removes extras | ❌ Throws away data | ✓ | Constant cleaning |
| **Canonicalization** (new) | Averages per question | ✓ Within-question | ✓ Across questions | None! |

## The Math

**Naive averaging** (wrong):
```
Q1: [8.5, 8.5, 8.5]  (3 judgments)
Q2: [7.0]            (1 judgment)

Mean = (8.5 + 8.5 + 8.5 + 7.0) / 4 = 8.0  ❌ Q1 has 3x weight!
```

**Canonicalization** (correct):
```
Q1: 8.5  (mean of [8.5, 8.5, 8.5])
Q2: 7.0  (mean of [7.0])

Mean = (8.5 + 7.0) / 2 = 7.75  ✓ Equal weight per question!
```

## Benefits

1. ✅ **Mathematically correct**: Equal weight per question
2. ✅ **No data loss**: All judgments contribute to per-question means
3. ✅ **Variance reduction**: Extra judgments → more confident per-question scores
4. ✅ **No maintenance**: Don't need to clean/validate duplicates anymore
5. ✅ **Transparent**: Can see `_n_judgments` per question in output
6. ✅ **Works with any pattern**: Uniform duplicates, partial duplicates, no duplicates

## What About the Other Scripts?

### Still Useful:

**`validate_judgment_files.py`**:
- Good for checking data completeness
- Surfaces missing questions
- JSON integrity validation

**`20251129-generate-*.sh`**:
- Fixed syntax errors
- Robust skip-existing logic prevents overwrites
- Still important for safe data generation

**`rename_model.py`**:
- For promoting ablation experiments
- Renames across all files safely

### Less Critical Now:

**`deduplicate_judgments.py`**:
- Still works, but not needed for scoring correctness
- Only use if you want to reduce file sizes in git-lfs
- Canonical solution makes this optional rather than required

### Now Canonicalized Too:
- `compare-judges.py`
- `judge-comparison-stats.py`
- `visualize-results.py`

All scoring/plotting paths now average duplicates per `(model, question_id, turn)` first, so weights are consistent across scripts. Canonicalization also logs skipped rows (missing model/question/score or invalid score) so any data loss is visible.

## Migration Path

**Already done for you!**

The scoring scripts (`show_result.py`, `results-table.py`) already have canonicalization integrated. Just use them normally:

```bash
cd /root/ja-mt-bench-harness/fastchat/llm_judge

# Your existing commands work - just better now!
python3 show_result.py --judge-model gpt-4-turbo --bench-name ja_mt_bench

python3 results-table.py \
  --model-list "shisa-ai/shisa-v2-llama3.3-70b" \
  --bench-name ja_mt_bench
```

## Key Takeaway

**You don't need to clean your data - you need to aggregate it correctly!**

Duplicates are only a problem if you aggregate naively. With per-question averaging (canonicalization), duplicates become a **feature** (variance reduction) rather than a bug.

---

## Files Modified/Created

### Modified:
- `show_result.py` - Added automatic canonicalization
- `results-table.py` - Added automatic canonicalization
- `20251129-generate-answers.sh` - Fixed syntax (still useful)
- `20251129-generate-judgements.sh` - Fixed syntax (still useful)

### Created:
- `canonicalize_scores.py` - Core canonicalization logic
- `docs/CANONICALIZATION-SOLUTION.md` - Detailed explanation
- `docs/FINAL-SOLUTION-SUMMARY.md` - This document
- (Previous) `validate_judgment_files.py`, `deduplicate_judgments.py`, `rename_model.py`

### Philosophy Shift:

**Before**: "We have duplicates, we need to clean them!"
**After**: "We have extra data, let's use it correctly!"

This is the elegant solution your colleague envisioned. 🎉
