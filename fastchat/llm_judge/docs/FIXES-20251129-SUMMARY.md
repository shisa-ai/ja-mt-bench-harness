# 20251129 Script Fixes & Data Integrity Summary

## 🔍 FINDING: Judgment Files Have Duplicates

Analysis shows judgment files have **two types of duplicates**:

### 1. Complete Uniform Duplicates (GOOD ✓)
Most models have **complete uniform duplicates** where every question appears the same number of times (usually 2x).

**This is actually GOOD** - like having multiple annotations:
- ✓ Improves reliability through averaging  
- ✓ Mean of `[8.5, 8.5]` = Mean of `[8.5]` (mathematically equivalent)
- ✓ Reduces variance from single judge calls
- ✓ **These are preserved** by our cleaning script

### 2. Partial/Uneven Duplicates (BAD ⚠)
Some models have **partial duplicates** where some questions appear more than others (e.g., Q1 appears 2x but Q2 appears 1x).

**This is BAD** - skews the mean:
- ❌ Mean of `[8.5, 8.5, 7.0]` ≠ Mean of `[8.5, 7.0]`  
- ❌ Indicates interrupted/partial run that appended incomplete data
- ❌ **These need cleaning** (reduce to 1x each)

## Philosophy: Some Duplicates Are Good!

**Complete uniform duplicates = Multiple annotators** (preserved):
- If every question appears 2x → like having 2 independent judgments
- Improves confidence and reduces random variance
- Standard practice in human evaluation

**Partial duplicates = Data corruption** (cleaned):
- Question A appears 3x, Question B appears 1x → unfair weighting
- Skews the overall score toward over-represented questions

Our `deduplicate_judgments.py` script is smart enough to tell the difference!

## ✅ All Fixes Completed

See the full summary in this document for details on:
- Fixed syntax errors in both generation scripts
- Robust validation to prevent overwrites
- Three new utility scripts (validate, deduplicate, rename)
- Interactive confirmation for all destructive operations

## Quick Start

```bash
cd /root/ja-mt-bench-harness/fastchat/llm_judge

# 1. Check for partial duplicates
python3 deduplicate_judgments.py

# Shows per-model statistics:
# - "Complete uniform duplicates (GOOD)" → preserved
# - "PARTIAL DUPLICATES (needs cleaning)" → offers to clean

# 2. Clean if needed (interactive y/N per file)
# Script will ask for confirmation before each file

# 3. List all models (useful for renaming)
python3 rename_model.py --list

# 4. Rename a model (when promoting ablations)
python3 rename_model.py --old "old-name" --new "new-name"
```

All scripts create `.backup` files before making changes!
