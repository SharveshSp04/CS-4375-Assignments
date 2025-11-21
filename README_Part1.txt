CS 4375 - ASSIGNMENT 3 - PART I
THEORETICAL SOLUTIONS
==================================================

GROUP MEMBERS:
- [Your Name]
- [Partner's Name (if applicable)]

FILES INCLUDED:
- part_I_solutions.txt (this file)
- part_I_theoretical_solutions.py
- part_I_detailed_explanations.py 
- part_I_summary.py

OVERVIEW:
This document contains complete theoretical solutions for Assignment 3 Part I,
covering ensemble methods including bagging and AdaBoost.

================================================================================
QUESTION 1: BAGGING ERROR ANALYSIS (10 POINTS)
================================================================================

PROBLEM STATEMENT:
Prove that E_agg = (1/M) * E_avg under the assumptions:
1. E[ε_i(x)] = 0 for all i (zero mean errors)
2. E[ε_i(x)ε_j(x)] = 0 for all i ≠ j (uncorrelated errors)

COMPLETE PROOF:

Step 1: Define terms
ε_i(x) = f(x) - h_i(x)
E_avg = (1/M) * Σ_{i=1}^M E[ε_i(x)^2]
h_agg(x) = (1/M) * Σ_{i=1}^M h_i(x)
E_agg(x) = E[(h_agg(x) - f(x))^2]

Step 2: Express E_agg in terms of errors
E_agg = E[( (1/M) * Σ h_i(x) - f(x) )^2]
     = E[( (1/M) * Σ (f(x) - ε_i(x)) - f(x) )^2]
     = E[( f(x) - (1/M) * Σ ε_i(x) - f(x) )^2]
     = E[( - (1/M) * Σ ε_i(x) )^2]
     = (1/M^2) * E[(Σ ε_i(x))^2]

Step 3: Expand the squared sum
E[(Σ ε_i(x))^2] = E[Σ ε_i(x)^2 + Σ Σ_{i≠j} ε_i(x)ε_j(x)]
                = Σ E[ε_i(x)^2] + Σ Σ_{i≠j} E[ε_i(x)ε_j(x)]

Step 4: Apply uncorrelated errors assumption
Since E[ε_i(x)ε_j(x)] = 0 for i ≠ j:
E[(Σ ε_i(x))^2] = Σ E[ε_i(x)^2]

Step 5: Final substitution
E_agg = (1/M^2) * Σ E[ε_i(x)^2]
     = (1/M) * (1/M) * Σ E[ε_i(x)^2]
     = (1/M) * E_avg

Q.E.D.

KEY INSIGHT:
Bagging reduces error by factor M because averaging uncorrelated models
eliminates covariance terms, leaving only variance reduction.

================================================================================
QUESTION 2: JENSEN'S INEQUALITY APPLICATION (10 POINTS)
================================================================================

PROBLEM STATEMENT:
Using Jensen's inequality, prove that E_agg ≤ E_avg without the 
uncorrelated errors assumption.

COMPLETE PROOF:

Step 1: Recall Jensen's inequality
For any convex function f: f(Σ λ_i x_i) ≤ Σ λ_i f(x_i)
where λ_i ≥ 0 and Σ λ_i = 1

Step 2: Identify convex function
f(x) = x^2 is convex because f''(x) = 2 > 0

Step 3: Apply Jensen's inequality
Let λ_i = 1/M and x_i = ε_i(x)
Then: [(1/M) * Σ ε_i(x)]^2 ≤ (1/M) * Σ ε_i(x)^2

Step 4: Take expectations
E[(1/M * Σ ε_i(x))^2] ≤ E[(1/M) * Σ ε_i(x)^2]

Step 5: Recognize definitions
Left side = E_agg
Right side = (1/M) * Σ E[ε_i(x)^2] = E_avg

Therefore: E_agg ≤ E_avg

Q.E.D.

KEY INSIGHT:
Even with correlated models, aggregation never hurts performance due to
the convexity of the square function. This provides a safety guarantee
for ensemble methods.

================================================================================
QUESTION 3: ADABOOST TRAINING ERROR BOUND (10 POINTS)
================================================================================

PROBLEM STATEMENT:
Prove that AdaBoost training error is bounded by exp(-2 * Σ γ_t^2)

COMPLETE PROOF:

Step 1: Weight evolution
After T iterations:
D_{T+1}(i) = (1/N) * exp(-y_i Σ α_t h_t(x_i)) / [Π Z_t]

Step 2: Misclassification bound
If H(x_i) ≠ y_i, then y_i Σ α_t h_t(x_i) ≤ 0
Thus: [H(x_i) ≠ y_i] ≤ exp(-y_i Σ α_t h_t(x_i))

Step 3: Training error bound
Training error = (1/N) * Σ [H(x_i) ≠ y_i]
               ≤ (1/N) * Σ exp(-y_i Σ α_t h_t(x_i))

Step 4: Relate to normalization constants
From weight equation:
(1/N) * Σ exp(-y_i Σ α_t h_t(x_i)) = Π Z_t

Therefore: Training error ≤ Π Z_t

Step 5: Analyze Z_t
Z_t = Σ D_t(i) exp(-α_t y_i h_t(x_i))
    = (1 - ε_t) e^{-α_t} + ε_t e^{α_t}

Step 6: Optimal α_t
AdaBoost chooses α_t = (1/2) * ln((1 - ε_t)/ε_t)
Substituting: Z_t = 2 * √(ε_t(1 - ε_t))

Step 7: Express in terms of γ_t
Given ε_t = 1/2 - γ_t
Z_t = 2 * √((1/2 - γ_t)(1/2 + γ_t)) = √(1 - 4γ_t^2)

Step 8: Apply inequality
√(1 - 4γ_t^2) ≤ exp(-2γ_t^2)

Step 9: Final bound
Training error ≤ Π Z_t ≤ Π exp(-2γ_t^2) = exp(-2 Σ γ_t^2)

Q.E.D.

KEY INSIGHT:
AdaBoost achieves exponential error reduction because each weak learner
that's slightly better than random (γ_t > 0) multiplicatively reduces
the error bound.

================================================================================
HOW TO RUN THE CODE FILES
================================================================================

REQUIREMENTS:
- Python 3.6 or higher
- No external libraries needed

EXECUTION:
1. For complete mathematical proofs:
   python part_I_theoretical_solutions.py

2. For step-by-step explanations:
   python part_I_detailed_explanations.py

3. For concise summary:
   python part_I_summary.py

EXPECTED OUTPUT:
Each file will display:
- Problem statements
- Step-by-step solutions
- Final results and conclusions
- Mathematical insights


================================================================================
END OF PART I SOLUTIONS
================================================================================
