---
title: "Discretizing Unobserved Heterogeneity"
authors: Stéphane Bonhomme, Thibaut Lamadon, Elena Manresa
year: 2022
type: article
journal: Econometrica
URL: https://www.econometricsociety.org/doi/10.3982/ECTA15238
DOI: 10.3982/ECTA15238
citekey: bonhomme2022
tags: [ ]
---

# Discretizing Unobserved Heterogeneity

[Open in Zotero](zotero://select/items/@bonhomme2022)

## Abstract
We study discrete panel data methods where unobserved heterogeneity is revealed in a first step, in environments where population heterogeneity is not discrete. We focus on two‐step grouped fixed‐effects (GFE) estimators, where individuals are first classified into groups using kmeans clustering, and the model is then estimated allowing for group‐specific heterogeneity. Our framework relies on two key properties: heterogeneity is a function—possibly nonlinear and time‐varying—of a low‐dimensional continuous latent type, and informative moments are available for classification. We illustrate the method in a model of wages and labor market participation, and in a probit model with time‐varying heterogeneity. We derive asymptotic expansions of two‐step GFE estimators as the number of groups grows with the two dimensions of the panel. We propose a data‐driven rule for the number of groups, and discuss bias reduction and inference.

## Notes
Below is a brief summary of the paper and a discussion of how its methods can be applied to estimating your model.

────────────────────────────
Summary of [[2022 - Stéphane Bonhomme, Thibaut Lamadon, Elena Manresa - Discretizing Unobserved Heterogeneity]]:

• Purpose and Framework:  
 The paper develops a two‐step grouped fixed‑effects (GFE) method designed for panel data applications where unobserved heterogeneity is in fact continuous (or even time‑varying) but can be approximated using a limited number of discrete types. Instead of assuming discrete types a priori, the method “discretizes” the continuous latent heterogeneity for estimation purposes.

• Step‑by‑Step Procedure:  
 1. First, informative individual‑specific moments are computed (these moments may be nonlinear functions of outcomes and/or covariates). Under the idea that the heterogeneity is driven by a low‑dimensional latent type, the moments are assumed to be injective in the latent type.  
 2. A clustering algorithm (typically k‑means) is applied to these moments to partition individuals into K groups. In this step the latent continuous variation is approximated by group means.  
 3. In the second stage the model is estimated by allowing group‑specific parameters (instead of individual‑specific ones) while keeping the common structural parameters of interest.

• Theoretical Contributions:  
 • The authors derive large‑sample asymptotic expansions of the GFE estimator, showing that when the number of groups K increases appropriately (according to a data‑driven rule they propose), the approximation error is of lower order relative to the incidental parameter bias that is also comparable with that in conventional fixed‑effects estimators.  
 • They discuss bias reduction methods (e.g. split‑panel jackknife and two‑way grouping) and provide extensions to settings with time‑varying heterogeneity as well as models for which conditional moments can target only parts of the latent heterogeneity.

• Practical Implications:  
 The approach is particularly useful if you suspect that the unobserved heterogeneity in your panel data is driven by a low‑dimensional latent factor (possibly entering nonlinearly in the model) but is not actually discrete. By “discretizing” the latent types you obtain a simpler structure for estimation while still allowing for flexible forms of heterogeneity.

────────────────────────────
How to Use This Technique in Your Model:

1. Identification of Informative Moments:  
 • Determine (or construct) individual‑specific moments that are informative about the latent heterogeneous component(s) in your model.  
 • For example, if your model involves wages and labor market participation, joint moments combining outcomes (e.g. wages) and participation indicators can reveal the underlying latent type.

2. Grouping via Clustering:  
 • Use these moments as input to a k‑means algorithm (or a related clustering procedure) to classify individuals into a finite number K of groups.  
 • Use the data‐driven rule suggested in the paper (see Equation (6) in the text) to choose K so that the grouping approximation error is of the same or lower order than your estimation uncertainty.

3. Two‑Step Estimation:  
 • In the second step, re‑estimate your model by allowing for group‑specific fixed effects. That is, replace the continuous individual heterogeneity by a set of group indicators and estimate the common parameters of interest jointly with group‑specific parameters.  
 • This “grouped FE” method can reduce the dimensionality compared to having an individual‑specific parameter for each observation and may reduce the incidental parameter bias.

4. Bias Correction and Inference:  
 • Be aware that—as with standard fixed‑effects—the GFE estimator has an asymptotic bias of order 1/T. The paper discusses corrections (such as split panel jackknife or two‑way grouping if time‑variation is present) that can be applied to make inference more robust.
 • Once you’ve implemented GFE, you can use the provided asymptotic expansions to obtain standard errors for your structural parameters.

5. Extensions to Time‑Varying Heterogeneity or Conditioning on Covariates:  
 • If your model’s unobserved heterogeneity changes over time, the paper extends the GFE method to allow for simultaneous grouping over individuals and time.  
 • In models with additional covariates that also contain heterogeneity, you may consider constructing conditional moments (e.g. moments of the outcome given covariate values) so that the clustering focuses on the latent type relevant for the outcome equation only.

────────────────────────────
In Practice:

If you are estimating, say, a dynamic panel model where the individual effect enters nonlinearly and you suspect that its variation is essentially one‑dimensional, then the GFE approach lets you mimic the flexibility of a nonparametric heterogeneity model while simplifying estimation:
 – Compute an average or other informative moment for each individual.
 – Cluster the individuals into a manageable number of groups using k‑means.
 – Run your model with group-specific effects rather than individual-specific intercepts.
 – Optionally, implement a bias correction in finite panels.

This approach can help if traditional fixed‑effects estimation yields biased or overly variable estimates due to the large number of nuisance parameters.

────────────────────────────
Summary Points:

• The paper proposes a two‑step procedure to “discretize” continuous unobserved heterogeneity via clustering.
• It delivers asymptotic theory, data‑driven procedures for selecting the number of groups, and extensions for time‑varying cases.
• In your model, by choosing suitable informative moments and grouping individuals, you can estimate the structural parameters efficiently while reducing the dimensionality of unobserved heterogeneity and mitigating potential biases.

I hope this summary and discussion help you see how to apply the GFE method in the estimation of your model. Let me know if you need any more details or further assistance with implementation within your Obsidian notes!
