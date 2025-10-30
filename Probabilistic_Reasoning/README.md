# Probabilistic Reasoning & Decision Making

## 1. Project Description

(a) This project enables exact inference to predict student exam performance given observed health and tutor support using Bayesian Networks  
(b) It supports temporal reasoning by modeling health transitions across time slices via Dynamic Bayesian Networks  
(c) It recommends optimal study hour decisions (low/medium/high) by computing expected utility under uncertainty in a Decision Network  

## 2. Tech Stack / Tools Used

(a) Python 3.12+  
(b) NumPy  
(c) NetworkX  
(d) Matplotlib  

## 3. Objectives / Tasks

(a) Generate reproducible CPTs using ID-based deterministic seeding  
(b) Perform exact inference via enumeration and variable elimination  
(c) Simulate two-slice DBNs with temporal evidence propagation  
(d) Compute expected utilities for decision alternatives to identify optimal action  

## 4. Implementation / Methods

### 4.1 ID & Seeding

(a) Read ID and derived `idHash` using SHA256  
(b) Seeded `numpy` and `random` to ensure identical CPTs across runs for same ID  

### 4.2 Bayesian Network Setup

(a) Defined binary domains for six variables: Health, Sleep, Attendance, Tutor, ExamPrep, Performance  
(b) Generated normalized CPTs for root nodes (`H`, `T`) and conditional nodes (`S`, `A`, `E`, `P`)  

### 4.3 Exact Inference: Enumeration

(a) Implemented `enumerateAll` to compute joint probability by recursive variable summation  
(b) Used `parentsMap` and `localProb` for efficient CPT lookup  
(c) `enumerationAsk` returns normalized posterior P(X | evidence)  

### 4.4 Variable Elimination

(a) Defined `Factor` class supporting `restrict`, `multiply`, and `sumOut`  
(b) `variableElimination` converts CPTs to factors, applies evidence, eliminates in order, normalizes result  
(c) Improved factor construction with explicit parent-to-child assignment mapping  

### 4.5 Dynamic Bayesian Networks

(a) Adapted `dbnVariableElim` for two-slice temporal model (StudyHours → S1 → H1 → P1)  
(b) Handled time-indexed variables and evidence across slices  

### 4.6 Decision Networks

(a) Defined CPTs linking study hours to sleep, health, and performance  
(b) Defined `utilityTable` assigning 20 to low performance, 100 to high  
(c) `computeExpUtil` enumerates all outcome paths, computes joint probabilities, accumulates expected utility per decision  

## 5. Results / Outputs

(a) **Expected Utilities**: `{'low': 47.6, 'medium': 0.0, 'high': 56.0}`  
(b) **Recommended Decision**: `high` study hours (maximizes expected utility)  

## 6. Conclusion / Insights

(a) Variable elimination dramatically reduces inference cost vs. brute-force enumeration  
(b) DBNs enable modeling of evolving states (e.g., health over time)  
(c) Decision networks support rational choice under uncertainty via expected utility maximization  
(d) ID-based seeding ensures fairness and reproducibility in assessments  

## 7. Acknowledgements / References

(a) NetworkX Documentation: https://networkx.org/documentation/stable/  
(b) NumPy Documentation: https://numpy.org/doc/  

(c) Russell, S. J., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*. <br>

(d) Dechter, R. (1999). Bucket elimination: A unifying framework for processing hard and soft constraints. *International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems*. <br>

 (e) Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*. Morgan Kaufmann.<br>