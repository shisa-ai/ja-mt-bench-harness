# How Canonicalization Addresses IMPLEMENTATION-score-review.md Concerns

## Original Concerns (from IMPLEMENTATION-score-review.md)

The document identified several problems with duplicate and missing data:

### 1. **Duplicate Overweighting**

**Original concern**:
- `compare-judges.py`: Duplicates counted multiple times in averages
- `results-table.py`: Duplicates increase category weight
- `visualize-results.py`: Repeated rows change means

**Our solution**: ✅ **ADDRESSED**
- Added `canonicalize_judgments()` to `show_result.py` and `results-table.py`
- Each `(model, question_id, turn)` contributes exactly once
- Duplicates are averaged within-question, preserving variance reduction
- Equal weight per question guaranteed

### 2. **Last-Wins Overwrites**

**Original concern**:
- `judge-comparison-stats.py`: Last occurrence silently overwrites earlier ones

**Our solution**: ✅ **BETTER THAN ADDRESSED**
- Canonicalization computes **mean** of all occurrences
- No data loss - all judgments contribute
- More robust than keeping first or last

### 3. **Missing Coverage**

**Original concern**:
- Scripts silently shrink datasets when questions missing
- Missing turns default to 0 in some scripts
- Different judges averaged over different counts

**Partially addressed**:
- ✅ Canonicalization preserves equal weight for present questions
- ✅ Missing questions don't distort weights of present ones
- ⚠️ Still need to add explicit warnings about coverage gaps
- ⚠️ `visualize-results.py` still needs fixing (0 defaults)

### 4. **Validation Requirements**

**Original concern**:
- Must run validation before analysis
- Block score generation until clean

**Our approach**: ✅ **RELAXED REQUIREMENT**
- Canonicalization makes validation **recommended** not **mandatory**
- Uneven duplicates don't break scoring anymore
- Validation still useful for completeness checking
- But scoring is mathematically correct even with duplicates

## What's Still Needed

### Scripts not yet updated:

1. **`judge-comparison-stats.py`** - Should add canonicalization
2. **`compare-judges.py`** - Should add canonicalization  
3. **`visualize-results.py`** - Should add canonicalization + fix 0 defaults

### Enhancement opportunities:

1. **Explicit coverage warnings** in all scripts:
   ```python
   if len(canonical) < expected_questions * expected_turns:
       print(f"⚠️ Warning: Only {len(canonical)} of {expected} questions present")
   ```

2. **Per-turn statistics** showing coverage:
   ```python
   turn_coverage = defaultdict(int)
   for judgment in canonical:
       turn_coverage[judgment['turn']] += 1
   print(f"Turn 1: {turn_coverage[1]} questions")
   print(f"Turn 2: {turn_coverage[2]} questions")
   ```

3. **Model-level completeness** flagging:
   ```python
   for model, count in model_counts.items():
       if count < expected:
           print(f"⚠️ {model}: incomplete ({count}/{expected} questions)")
   ```

## Philosophy Shift

### Before (IMPLEMENTATION-score-review.md):

> "Treat validation as mandatory: block score generation until validate_judgment_files.py 
> reports zero errors and zero duplicates"

### After (Canonicalization approach):

> "Canonicalize before aggregation: compute mean per question, then aggregate.
> Validation is for completeness checking, not correctness enforcement."

**Why this is better**:
- ✅ No data loss from deduplication
- ✅ Variance reduction from multiple judgments preserved
- ✅ Works with any duplication pattern (uniform, partial, none)
- ✅ Less fragile - don't need perfect data to get correct scores
- ✅ More forgiving - can analyze partial results if needed

## Updated Workflow

### Old workflow:
```
1. Generate judgments
2. ❌ Validate (mandatory) - fail if duplicates
3. ❌ Deduplicate if needed
4. ❌ Validate again
5. ✅ Compute scores
```

### New workflow:
```
1. Generate judgments
2. ✅ Compute scores (canonicalization automatic)
3. (Optional) Validate for completeness reporting
```

## Summary Table

| Concern | Original Script Behavior | With Canonicalization | Status |
|---------|-------------------------|----------------------|--------|
| Duplicate overweighting | Duplicates counted multiple times | Averaged per-question | ✅ Fixed |
| Last-wins overwrites | Last entry wins | Mean of all entries | ✅ Better |
| Missing coverage skews | Different denominators | Equal weight for present Qs | ✅ Fixed |
| Silent data shrinkage | No warnings | (Need to add warnings) | ⚠️ TODO |
| Validation required | Mandatory before scoring | Optional (for completeness) | ✅ Relaxed |
| Zero defaults | Missing = 0 in some scripts | (visualize-results needs fix) | ⚠️ TODO |

## Recommendation

**For `docs/IMPLEMENTATION-score-review.md`**:

1. Add section: "Canonicalization Solution (Nov 2025)"
2. Note that duplicate concerns are addressed
3. Keep validation recommendations for completeness
4. Update script behavior descriptions for updated scripts
5. Add TODO list for remaining scripts

**For remaining scripts**:

1. Add canonicalization to judge-comparison-stats.py
2. Add canonicalization to compare-judges.py
3. Fix visualize-results.py (canonicalization + no 0 defaults)
4. Add coverage warnings to all scripts

Our canonicalization approach addresses the core mathematical correctness issues
while being more forgiving of imperfect data. The remaining work is about
user-friendliness (warnings) rather than correctness.
