# MATH evaluator audit

## Why this audit was performed

The first uniformly sampled MATH-500 development run reported 26/30. The four
reported failures were manually compared with the frozen references and official
solutions before treating the benchmark as non-saturated.

## Original four reported failures

| Task | Model answer | Reference | Finding |
|---|---|---|---|
| `test/geometry/880.json` | `2\sqrt{106}` | `2\sqrt{113}` | Genuine reasoning error. The model omitted the shorter unfolding that crosses a side wall, the ceiling, and the front/back wall. |
| `test/precalculus/779.json` | `-2+7i` | `-2 + 7i` | Evaluator false negative. |
| `test/number_theory/417.json` | `11111111100` | `11,\! 111,\! 111,\! 100` | Evaluator false negative caused by LaTeX thousands separators. |
| `test/intermediate_algebra/128.json` | `\text{even}` | `\text{even}` | Evaluator false negative. |

Math-Verify parsed unanchored references incorrectly; for example,
`2\sqrt{113}` became the scalar `2`, while the full model response was parsed as
`2\sqrt{106}`. The evaluator now:

1. extracts the final boxed prediction;
2. checks a conservative normalized exact match;
3. wraps both the normalized reference and prediction in `\boxed{}` before
   symbolic verification;
4. removes commas only when the complete answer is numeric.

Regression tests cover radicals, complex numbers, formatted large integers,
text categories, and symbolic fractions. After the correction, the original
run regrades to 29/30 (96.7%); only the geometry answer remains wrong.

## Difficulty adjustment

The pinned MATH-500 revision contains 134 Level 5 problems, enough for the
fixed 30-development/50-test design. The pilot therefore filters to Level 5
before deterministic splitting.

The revised Level 5 development baseline scored 25/30 (83.3%). Its five
failures were checked against the official solutions and were all substantive
answer/reasoning errors rather than extraction errors. The run is saved under
`logs/baseline-math-level5/`, including its config and manifest snapshots.
