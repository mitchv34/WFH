# The Value of Flexibility: Teleworkability, Sorting,and the Work‑From‑Home Wage Gap

# **1. Introduction & Motivation**

Remote work has surged, accompanied by a persistent and evolving wage premium for remote employees compared to their on-site counterparts. This premium exists even within detailed occupations (Stylized Fact II) and correlates with occupational teleworkability (Stylized Fact I). While factors like productivity differences (Bloom et al., 2015; Davis et al., 2021) and amenity value (Mas & Pallais, 2017) are discussed, existing reduced-form analyses cannot fully disentangle these effects or account for general equilibrium sorting. Understanding the drivers of this premium is crucial for labor market policy, firm strategy, and understanding inequality. **This project investigates the structural determinants of the WFH wage premium, focusing on the role of firm heterogeneity in remote work efficiency and worker heterogeneity in skill.**

**2. Research Question**

What mechanisms drive the observed wage gap between remote and non-remote workers, and how does the sorting of heterogeneous firms and workers contribute to this premium?

**3. Proposed Methodology: Directed Search Model with Heterogeneity**

We develop and analyze a structural **directed search model** (Menzio & Shi, 2010). Key features include:

*   **Heterogeneity:** Workers differ in skill ($h$), affecting productivity. Firms differ in remote work efficiency ($\psi$), influencing their ability to utilize remote labor effectively ($g(\psi, h)$).
*   **Directed Search:** Workers search across submarkets defined by utility offers ($x$, bundling wage $w$ and remote work share $\alpha$). Firms post utility offers targeted at specific worker types ($h$).
*   **Probabilistic Choice & Sorting:** Firms probabilistically choose which submarket $(h, x)$ to enter based on expected profits ($\Pi_{post} = qJ - \kappa(x)$), where $\kappa(x)$ is a utility-dependent posting cost. This generates **endogenous sorting**.
*   **Equilibrium:** Solved numerically for equilibrium vacancy filling rates ($q^*$), tightness ($\theta^*$), choice probabilities ($P^*$), conditional firm distributions ($f^*(\psi|h,x)$), and worker/firm value functions ($U(h), W(\psi,h,x), J(\psi,h,x)$).

This framework explicitly models competition via contract offers and allows sorting dynamics to determine wages and remote work arrangements, going beyond simple correlations.

**4. Data and Empirical Strategy**

We utilize a rich combination of datasets:

*   American Community Survey **ACS (2013-2023):** Individual worker characteristics, wages, remote work status.
*   **O\*NET:** Detailed occupation characteristics (skills, abilities, tasks, context).
*   Bureau of Labor and Statistics **BLS:** Employment weights, industry productivity, ground-truth labels for remote work feasibility.
*  Business Dynamics Statistics **BDS:** Firm creation data for weighting distributions.

Our empirical strategy involves:
1.  **Estimating a novel Occupation Teleworkability Index:** Using a two-stage Random Forest model (Classifier + Regressor) trained on ORS labels and O\*NET features to predict remote work feasibility for all occupations.
2.  **Estimating Production Function Components:** Regressing industry-level productivity on skill, remote work rates, teleworkability, and interactions using panel data methods (Fixed Effects).
3.  **Mapping & Distribution Estimation:** Mapping regression coefficients to structural parameters ($A_1, \psi_0, \psi_1, \phi$) and estimating distributions $F(h)$ and $F(\psi)$ using KDE weighted by employment/firm creation.
4.  **Calibration:** Setting remaining parameters ($\beta, \delta, \gamma, b, \kappa_0, \xi$) based on literature values and matching aggregate moments (e.g., $v/u$ ratio, wage dispersion).

**5. Expected Outcomes & Contribution**

The calibrated model is expected to generate:
*   Positive assortative matching between worker skill $h$ and firm efficiency $\psi$.
*   Higher expected wages and higher expected remote work shares ($\alpha^*$) for higher-skilled workers due to sorting into better matches.
*   A positive WFH wage premium arising endogenously from these sorting dynamics.

This research contributes by:
*   Providing a structural interpretation of the WFH premium, disentangling sorting effects from average productivity/amenity effects.
*   Developing and utilizing a novel, data-driven teleworkability index.
*   Quantifying the role of firm heterogeneity in remote work efficiency in shaping labor market outcomes.
*   Offering a framework to analyze counterfactuals related to remote work policies or technological changes.


---