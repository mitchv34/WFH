---
title: "Bartik Instruments: What, When, Why, and How"
authors: Paul Goldsmith-Pinkham, Isaac Sorkin, Henry Swift
year: 2020
type: article
journal: American Economic Review
URL: https://pubs.aeaweb.org/doi/10.1257/aer.20181047
DOI: 10.1257/aer.20181047
citekey: goldsmith-pinkhamBartikInstrumentsWhat2020
tags: [ ]
---

# Bartik Instruments: What, When, Why, and How

[Open in Zotero](zotero://select/items/@goldsmith-pinkhamBartikInstrumentsWhat2020)

## Abstract
The Bartik instrument is formed by interacting local industry shares and national industry growth rates. We show that the typical use of a Bartik instrument assumes a pooled exposure research design, where the shares measure differential exposure to common shocks, and identification is based on exogeneity of the shares. Next, we show how the Bartik instrument weights each of the exposure designs. Finally, we discuss how to assess the plausibility of the research design. We illustrate our results through two applications: estimating the elasticity of labor supply, and estimating the elasticity of substitution between immigrants and natives. (JEL C51, F14, J15, J22, L60, R23, R32)

## Notes
- [x] Take notes about Bartik Instruments: What, When, Why, and How ✅ 2025-04-01
# Bartik Instruments: What, When, Why, and How

[Open in Zotero](zotero://select/items/@goldsmith-pinkhamBartikInstrumentsWhat2020)

## Summary of the Paper

- **Core Idea**:  
  The paper shows that a Bartik instrument is created by interacting local “exposure” measures (e.g. industry shares) with national “shock” measures (e.g. industry growth rates). This creates an instrument that leverages differential exposure to common shocks.

- **Identification Strategy**:  
  - The typical use assumes a *pooled exposure design*. That is, researchers use pre-determined shares that capture differential exposure across areas to a common shock, with the key identifying assumption being that these shares are exogenous with respect to other shocks affecting the outcome.  
  - For example, in their canonical application—estimating the elasticity of labor supply—the shares are fixed at a baseline period so that any subsequent outcome variation reflects the impact of the national shock rather than pre-existing differences.

- **Decomposition and Diagnostics**:  
  - The paper formally shows that the Bartik estimator is equivalent to a GMM estimator using the underlying industry shares as instruments with a certain set of weights (commonly referred to as **Rotemberg weights**).  
  - These weights help diagnose which sectors are driving the overall estimates, and they serve as a sensitivity measure to potential mis-specification of the assumptions.
  
- **Robustness and Heterogeneity**:  
  - The authors discuss how, in settings with differential or heterogeneous treatment effects, the overall estimate is a weighted average of the “just-identified” local estimates.  
  - They emphasize checking pre-trends, running over-identification tests, and considering alternative estimators as key diagnostics to assess the credibility of the research design.

- **Useful Links within the Paper**:  
  - [Identification via Differential Exposure](#): Discusses why fixing shares in a baseline period is crucial.  
  - [Rotemberg Weights and Sensitivity Analysis](#): Explains the decomposition of the estimator and how to interpret the importance of specific instruments.

## Potential Application for Estimating Hybrid Work Shares

- **Context & Data Challenges**:  
  - I have two datasets: one that identifies whether workers work remote (or not) with excellent occupational information, and a second dataset where you have a nuanced measure of hybrid work.
  - The challenge: the dataset with the nuanced hybrid measure lacks detailed occupational breakdowns, while the other has rich occupational data but only a binary remote/not indicator.

- **Bartik-Like Construction for Hybrid Work**:  
  - **Exposure Component (Shares)**:  
    Use the pre-determined occupational composition from the dataset with better occupation info as your “exposure” measure. For each location (or worker group), you can calculate the share of workers in occupation $o$.
  
  - **Shock Component (Hybrid Trends)**:  
    From the second dataset, compute the national (or aggregate) trend in hybrid work for each occupation. This provides the “shock” reflecting how hybrid work is evolving differently by occupation.
  
  - **Instrument Construction**:  
    Form a Bartik-style hybrid instrument as:  
    $$ \text{Hybrid Instrument}_l = \sum_{o} \text{Occupation Share}_{l,o} \times \text{Hybrid Trend}_o $$  
    This instrument measures the differential exposure of each location to hybrid work, as determined by its occupational composition and the common shift in hybrid work trends.

- **Key Considerations & Diagnostics (Inspired by the Paper)**:  
  - **Exogeneity of Occupational Shares**:  
    Ensure that the occupational shares are taken from a period prior to the expansion of hybrid work, or that they are otherwise plausibly exogenous to changes in hybrid work outcomes (similar to the baseline period in the paper).
    
  - **Weight Analysis**:  
    Following the approach on [Rotemberg weights](#), compute the contribution of each occupation to the overall hybrid instrument. This will help you diagnose if a few occupations dominate the instrument and whether those occupations are likely channels for the hybrid work phenomenon.
    
  - **Testing the Identification**:  
    - Check correlations between the occupation shares and other pre-existing trends or characteristics that might affect work outcomes.  
    - If possible, compare alternative estimators that use sector-specific instruments, and run overidentification tests to assess the validity of your instrument.

- **Conclusion for My Application**:  
  By leveraging a Bartik-like strategy, you can overcome data limitations by combining the strengths of both datasets. The resulting instrument would capture how local occupational compositions predispose different areas towards hybrid work, using the nuanced hybrid trends as the common shock. 