---
title: "Multidimensional Skills, Sorting, and Human Capital Accumulation"
authors: Jeremy Lise, Fabien Postel-Vinay
year: 2020
type: article
journal: American Economic Review
URL: https://pubs.aeaweb.org/doi/10.1257/aer.20162002
DOI: 10.1257/aer.20162002
citekey: lise2020
tags: [ ]
---

# Multidimensional Skills, Sorting, and Human Capital Accumulation

[Open in Zotero](zotero://select/items/@lise2020)

## Abstract
We construct a structural model of on-the-job search in which workers differ in skills along several dimensions and sort themselves into jobs with heterogeneous skill requirements along those same dimensions. Skills are accumulated when used, and depreciate when not used. We estimate the model combining data from O\*NET with the NLSY79. We use the model to shed light on the origins and costs of mismatch along heterogeneous skill dimensions. We highlight the deficiencies of relying on a unidimensional model of skill when decomposing the sources of variation in the value of lifetime output between initial conditions and career shocks. (JEL J24, J41, J64)

## Notes
## Summary

### 1. Overview
- **Model Focus:** A dynamic on‐the‐job search model where workers possess multidimensional skills.
- **Skill Dynamics:** Skills accumulate when used and depreciate when idle.
- **Data Integration:** Combines detailed occupational measures from O\*NET with individual data from NLSY79.
- **Policy Relevance:** Explores the costs of skill mismatch across cognitive, manual, and interpersonal dimensions.

### 2. Introduction and Motivation
- **Traditional Limitations:** Conventional models use a unidimensional human capital measure, oversimplifying the diversity of skills.
- **Multifaceted Skills:** Recognizes that workers have varying strengths in areas such as cognitive problem-solving, manual tasks, and interpersonal communication.
- **Core Questions:** Investigates how differing rates of skill adjustment across dimensions affect wages, job matching, and overall productivity.

### 3. Theoretical Model
- **Worker Representation:** Each worker has a skill bundle:
  - $x = (x_C, x_M, x_I, x_T)$, where
    - $x_C$: Cognitive skills
    - $x_M$: Manual skills
    - $x_I$: Interpersonal skills
    - $x_T$: General efficiency
- **Job Representation:** Jobs are characterized by a requirements vector:
  - $y = (y_C, y_M, y_I)$ for cognitive, manual, and interpersonal requirements.
- **Skill Dynamics:** Skills adjust toward the job requirements with:
  - **Underqualification:** When $x_k < y_k$, skills increase at rate $\gamma^u_k$.
  - **Overqualification:** When $x_k > y_k$, skills decay at rate $\gamma^o_k$.
- **Wage Determination:** Wages are determined via surplus sharing, factoring in search frictions and renegotiation dynamics.

### 4. Empirical Strategy and Data
- **Data Sources:**
  - **O\*NET:** Provides quantitative measures of occupational skill requirements.
  - **NLSY79:** Offers panel data on individual career paths and wages.
- **Parameter Estimation:** Involves calibrating rates of skill accumulation, depreciation, and the returns to specific skills by matching simulated model moments with empirical observations such as wage profiles and occupational transitions.

### 5. Results and Findings
- **Differential Returns:**
  - **Cognitive Skills:** Yield high returns but adjust slowly.
  - **Manual Skills:** Offer moderate returns and adjust rapidly.
  - **Interpersonal Skills:** Are relatively constant over a career.
- **Mismatch Costs:** Underqualification—especially in cognitive skills—imposes significantly higher costs than overqualification.
- **Implications:** Unidimensional models overstate unobserved heterogeneity and understate the impact of career shocks on lifetime output.

### 6. Detailed View: Classifying Multidimensional Skills

#### a) Identification of Skill Dimensions
- **Primary Dimensions:** Cognitive ($x_C$), Manual ($x_M$), and Interpersonal ($x_I$).
- **General Efficiency:** ($x_T$) captures overall productivity not tied to a specific task.

#### b) Mapping Job Requirements to Skill Dimensions
- **Job Vector:** Each job is represented as $y = (y_C, y_M, y_I)$.
- **Utilization of O\*NET Data:** Provides detailed, quantitative measures that allow mapping of job tasks directly onto these skill dimensions, forming the basis for quantifying “skill mismatch.”

#### c) Skill Accumulation and Adjustment Process
- **Adjustment Equation:**
  $$
  \dot{x}_k = \gamma^u_k \cdot \max\{y_k - x_k, 0\} + \gamma^o_k \cdot \min\{y_k - x_k, 0\}, \quad k \in \{C, M, I\}.
  $$
- **Mechanism:**
  - When a worker is underqualified ($x_k < y_k$), they learn at a rate $\gamma^u_k$.
  - When a worker is overqualified ($x_k > y_k$), skills may decay at a rate $\gamma^o_k$.
- This setup ensures that over time, a worker’s skills adjust toward meeting job requirements at differing speeds.

#### d) Integration into the Production Function
- **Production Function Specification:**
  $$
  f(x,y) = x_T \times \Bigl( \alpha_T + \sum_{k\in\{C,M,I\}} \alpha_k y_k - \kappa^u_k\,\min\{x_k - y_k, 0\}^2 + \alpha_k\,k\,x_k\,y_k \Bigr).
  $$
- **Interpretation:**
  - The linear terms ($\alpha_k y_k$) reflect inherent productivity tied to each skill.
  - The quadratic penalty terms ($-\kappa^u_k \min\{x_k - y_k, 0\}^2$) capture the losses due to skill shortages, particularly highlighting the costliness of cognitive underqualification.

#### e) Estimation Strategy
- **Data Matching:** The merging of NLSY79 career data with O\*NET occupational metrics.
- **Moment Matching:** Model-simulated moments (wage profiles, job transitions) are aligned with empirical observations to estimate:
  - The learning (accumulation) rates $\gamma^u_k$.
  - The depreciation (decay) rates $\gamma^o_k$.
  - Production function parameters.
- This comprehensive strategy quantifies how specific skill mismatches influence wages and career trajectories.

### 7. Policy Implications and Conclusions
- **Targeted Interventions:** Suggests designing policy measures that address specific skill deficiencies instead of using a one-size-fits-all approach to human capital.
- **Practical Insights:** Emphasizes the need for tailored training programs and job placement strategies that mitigate the costs of mismatch and enhance productivity.
- **Overall Impact:** The multidimensional approach provides deeper insights into career dynamics and wage determination, prompting a rethinking of traditional human capital models.

# Construction of the Skill Requirement Vector (Detailed Explanation)

In Section 4.1, the authors detail how they compute the skill requirement vector, y, for each occupation using O\*NET data. The process can be summarized in the following steps:

### 1. Extraction of O\*NET Descriptors
- **Relevance Identification:**  
  Identify the set of O\*NET descriptors that best capture the key skill dimensions—typically cognitive, manual, and interpersonal skills.
- **Variable Selection:**  
  Choose quantitative measures such as task complexity, activity importance, and specific skill ratings from the O\*NET database.

### 2. Aggregation of Raw Measures
- **Grouping by Skill Dimension:**  
  Organize the selected descriptors into groups corresponding to each dimension:
  - **Cognitive Requirements ($y_C$):**  
    Variables related to problem-solving, information processing, reasoning, etc.
  - **Manual Requirements ($y_M$):**  
    Measures of physical demands, motor skills, and dexterity.
  - **Interpersonal Requirements ($y_I$):**  
    Indicators of communication, teamwork, and interaction with others.
- **Composite Score Formation:**  
  For each group, compute a composite score that represents the overall requirement level for that dimension. This is typically achieved through:
  - **Averaging or Weighted Averaging:**  
    Each descriptor may be normalized and then aggregated, often using weights derived from prior literature or calibration exercises.
  - **Dimensional Reduction Techniques:**  
    In some cases, factor analysis or principal component analysis (PCA) is employed to distill multiple descriptors into a single score per dimension.

### 3. Normalization and Standardization
- **Adjusting for Heterogeneity:**  
  Normalize the computed scores to ensure comparability across occupations. This involves standardizing the composite scores to have common units or scale.
- **Calibration to Data:**  
  The normalized scores are further calibrated to align with observed occupational outcomes in the NLSY79. This helps ensure that the computed vectors are consistent with real-world metrics of job requirements.

### 4. Constructing the Final Skill Requirement Vector
- **Assembly of the Vector y:**  
  Once the composite scores for cognitive, manual, and interpersonal dimensions are computed and standardized, they are assembled into the vector:  
  $$
  y = (y_C, y_M, y_I)
  $$
- **Use in the Model:**  
  This vector y is then used in the structural model to assess the matching between a worker’s skill bundle and the corresponding job requirements. The degree of mismatch is computed as the difference between a worker’s skill vector and the job’s requirement vector, which then feeds into the analysis of wage dynamics and career progression.

### 5. Technical Considerations
- **Measurement Error:**  
  The authors account for potential errors in O\*NET’s self-reported or aggregated measures by cross-validating with external benchmarks or using statistical corrections during estimation.
- **Employment Heterogeneity:**  
  Additional controls may be incorporated to adjust for occupational heterogeneity—ensuring that the requirements vector accurately reflects the task intensity across diverse industries or job types.

This detailed procedure allows the authors to systematically compute the occupation-specific skill requirement vector, which is a crucial input for quantifying skill mismatches and analyzing their implications in the labor market.