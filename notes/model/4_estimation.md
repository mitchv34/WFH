
This section details the procedure used to discipline the model and obtain plausible values for key structural parameters governing firm production, worker skills, and firm efficiency distributions. Our approach combines econometric estimation using detailed industry-occupation data with calibration techniques to map estimated coefficients to the model's parameters. We leverage data from the Bureau of Labor Statistics (OES, Productivity), the American Community Survey (ACS), the Census Bureau's Business Dynamics Statistics (BDS), and the O\*NET database.

### Data Sources and Variable Construction

We construct a panel dataset primarily at the 3-digit NAICS industry level, spanning the period form 2013 to 2023 . Key variables include:

*   **Output per Hour:** Derived from BLS sectoral productivity data, adjusted for price deflation. This serves as our measure of $Y/L$ in the data.
*   **Industry Skill Index:** Constructed by weighting occupation-level skill indices derived from O\*NET data (combining normalized importance and level ratings for relevant skills and abilities) by their employment shares within each industry, using BLS OES data. This proxies for the average $h$ within an industry.
*   **Industry Teleworkability Index:** Self-constructed index based on the Teleworkability Index at the occupation level. The procedure is described in the next section, weighted by employment shares within industries.
*   **Observe Remote Work Rates:** Measured using data from the ACS, representing the observed share of remote work in each industry-year.

### Estimating the Occupation Teleworkability Index {#sec-teleworkability}

A key input into our structural model is the heterogeneity in firms' remote work efficiency ($\psi$). We require an empirical counterpart to discipline the distribution of this parameter. Existing indices of occupation-level remote work feasibility, such as the binary classification by [@dingelHowManyJobs2020] or the index by [@mongey2020], provide valuable benchmarks. These approaches typically involve selecting key occupational characteristics based on theoretical considerations or researcher assessment to determine teleworkability. To leverage a richer, data-driven approach using a high-dimensional feature set, we construct a novel **Teleworkability Index** using machine learning techniques on detailed occupational data. This index aims to capture both whether an occupation *can* be performed remotely (extensive margin) and the *degree* to which it is feasible (intensive margin).

**Data:** Our primary feature set consists of detailed occupation-level characteristics from the O\*NET database, encompassing a wide range of variables related to skills, abilities, work activities, and work context. For supervised learning, we utilize data from the Occupational Requirements Survey, which reports the share of workers in each occupation for whom telework is available. In 2024, telework was available to 14.9% of civilian workers. We normalize this variable to a [0, 1] scale. As ORS coverage is incomplete, this provides labels for a subset of occupations.

**Methodology:** We employ a two-stage modeling approach using Random Forest models, leveraging their ability to handle high-dimensional data and capture non-linear relationships. Both models' hyperparameters are tuned using cross-validation techniques on the training data.

1. **Stage 1: Classification (Extensive Margin):** We first train a *Random Forest Classifier* to distinguish occupations where telework is impossible (ORS label = 0) from those where it is at least partially possible (ORS label > 0). The model is trained on the labeled subset of occupations using the binary indicator derived from the ORS label. This stage identifies occupations fundamentally unsuited for remote work, which is particularly important because of the large amounts of zeros in the training data.

2.  **Stage 2: Regression (Intensive Margin):** For occupations identified as having *non-zero* teleworkability by the ORS labels, we train a *Random Forest Regressor*. To ensure predictions remain within the [0, 1] bound and to better handle the distribution of the target variable, we apply a logit transformation ($\log(y / (1-y))$) to the non-zero ORS labels before training. The regressor learns the relationship between ONET features and the *degree* of teleworkability for jobs where remote work is feasible.

![](figures/SCR-20250417-t92.png){#fig-dist_labeled_unlabeled width=85%}

**Validation and Prediction:** The performance of the tuned two-stage model is evaluated using bootstrap validation on held-out test data, assessing metrics such as Accuracy and F1-score for the classifier, and Mean Squared Error (MSE) and Correlation for the regressor . The classifier achieved an accuracy of 90.7% along with an F1 score of 91.9%, while the regression model recorded a mean squared error (MSE) of 0.1 and a correlation of 0.71; @fig-dist_labeled_unlabeled shows the distribution of predicted teleworkability indices for both labeled and unlabeled data. The final trained model is then used to predict the teleworkability index for *all* occupations in the O\*NET database, including those without ORS labels. For a given occupation:

*   The classifier predicts the probability of it being a zero-teleworkability job. If this probability exceeds a threshold, the index is set to 0.
*   Otherwise, the regressor predicts the logit-transformed level of teleworkability, which is then converted back to the [0, 1] scale using the inverse logit function.

This resulting continuous index represents our measure of occupation-level teleworkability.

**Linking to the Model:** This estimated occupation-level Teleworkability Index forms the basis for constructing our industry-level $\text{TELE}$ variable (by weighting occupational indices by employment shares within industries). Furthermore, the relationship estimated that we estimate allows us to use the distribution of this index across industries (weighted by firm creation) to empirically inform the aggregate distribution of firm remote work efficiency, $F(\psi)$, that we supply externally to the structural model.

### Estimation of Production Function Components

To inform the parameters of the model's production function, $Y = A(h) ((1-\alpha) + \alpha g(\psi, h))$, we estimate a reduced-form relationship between productivity, skills, remote work, and teleworkability using our panel data. We assume that the industry-level data reflects the aggregate outcomes of the firm-level production specified in the model. Specifically, we relate the firm's remote work efficiency $g(\psi, h)$ to the observed industry teleworkability ($\text{TELE}$) and potentially worker skill ($\text{SKILL}$), and the baseline productivity $A(h)$ to worker skill ($\text{SKILL}$).

Our preferred specification is:  
\begin{equation}\label{eq:reducedFormProd}
\begin{aligned}
    \log(\text{OUTPUT}_{it}) ={}& \beta_0 \cdot \text{SKILL}_{it} \\
    & + \beta_1 \cdot (\text{ALPHA}_{it} \times \text{SKILL}_{it}) \\
    & + \beta_2 \cdot (\text{TELE}_{it} \times \text{ALPHA}_{it} \times \text{SKILL}_{it}) \\
    & + \beta_3 \cdot (\text{SKILL}_{it} \times \log(\text{SKILL}_{it})) \\
    & + \mu_i + \lambda_t + \epsilon_{it}
\end{aligned}
\end{equation}


where $i$ denotes industry and $t$ denotes year, $\mu_{i}$ are sector fixed effects, and $\lambda_{t}$ are year fixed effects. The regression results are summarized in Table 4.

\begin{table}[h]  
\centering  
\fontsize{8}{10}\selectfont  
\begin{tabular}{lrrrrr}
\toprule
                                   &                   \multicolumn{5}{c}{OUTPUT}                   \\ 
\cmidrule(lr){2-6} 
                                   &        (1) &        (2) &        (3) &        (4) &        (5) \\ 
\midrule
(Intercept)                        &   16.739** &            &            &            &            \\ 
                                   &    (6.424) &            &            &            &            \\ 
SKILL                              &  25.708*** &  23.910*** &  26.279*** &  37.024*** &  39.948*** \\ 
                                   &    (2.353) &    (2.266) &    (2.495) &    (2.922) &    (3.158) \\ 
ALPHA $\times$ SKILL               &   -37.564* &   -34.714* &   -53.873* &    -29.745 &  -84.807** \\ 
                                   &   (15.298) &   (15.366) &   (26.421) &   (15.311) &   (26.700) \\ 
SKILL $\times$ log(SKILL)          &  64.152*** &  15.974*** &  63.632*** & 149.791*** & 157.405*** \\ 
                                   &   (18.611) &    (2.148) &   (18.837) &   (22.199) &   (22.526) \\ 
TELE $\times$ ALPHA $\times$ SKILL & 315.199*** & 286.139*** & 356.831*** &   176.199* &  318.733** \\ 
                                   &   (70.115) &   (69.705) &   (89.545) &   (78.090) &   (96.738) \\ 
\midrule
YEAR Fixed Effects                 &            &            &        Yes &            &        Yes \\ 
SECTOR Fixed Effects               &            &            &            &        Yes &        Yes \\ 
\midrule
$N$                                &        418 &        418 &        418 &        418 &        418 \\ 
$R^2$                              &      0.263 &      0.510 &      0.265 &      0.378 &      0.388 \\ 
Within-$R^2$                       &            &            &      0.264 &      0.292 &      0.303 \\ 
\bottomrule
\end{tabular} 
\caption{Regression results for industry-level output estimation}  
\label{tab:regressionResults}  
\end{table}



\begin{table}
\centering
\begin{tabular}{lcc}
\hline
\textbf{Parameter} & \textbf{Mapping} & \textbf{Value} \\
\hline
$A$ & $\beta_0$ & 25.71 \\
$\psi_0$ & $(\beta_1 + 1) / A$ & -1.42 \\
$\psi_{1}$ & $\beta_2 / A$ & 12.26 \\
$\phi$ & $\beta_3 / A$ & 2.50 \\
$C$ & \text{(Intercept)} & 16.74 \\
\hline
\end{tabular}
\caption{Model Parameters, Mappings to Regression Coefficients, and Values}
\label{tab:parameters}
\end{table}

**Mapping Estimates to Structural Parameters**

We map the estimated coefficients from \eqref{eq:reducedFormProd} (specifically, using the results from model $(1)$) to the parameters of our assumed functional forms for $A(h)$ and $g(\psi, h)$. with the parametric assumption $$Y = A(h) (1 + \alpha (g(\psi, h) - 1))\qquad \text{with} \quad g(\psi, h) \approx \psi_0 + \psi + \phi \log h \:\text{ and }\: A(h)=A_{1} h$$

We derive the structural parameters $A_1, \phi, \psi_0, \psi_1$ The estimated values used in the model simulation are presented in Table 5.


**Estimating Distributions**

![](/figures/model_figures/productivity_density.pdf){#fig-remote-productivity-density width=50%}

*   **Firm Type Distribution $F(\psi)$:** We use the relationship derived from the regression, $\psi \approx \psi_1 \times \text{TELE}_{i}$, to obtain an estimate of the average firm type $\psi_i$ for each industry $i$. Using industry weights based on job creation from new firms (from BDS), we estimate the aggregate distribution of firm types $f(\psi)$ using Kernel Density Estimation (KDE). The resulting distribution @fig-remote-productivity-density captures the heterogeneity in remote work efficiency across the economy. We then discretize this distribution onto the model's $\psi_{grid}$.
   
![](/figures/model_figures/worker_type_distribution.pdf){#fig-skill-density width=50%}

*   **Worker Skill Distribution $F(h)$:** We construct the aggregate distribution of worker skills $f(h)$ similarly. We use the occupation-level skill indices derived from O\*NET and weight them by their aggregate employment. We then fit a KDE to these weighted skill indices to obtain the distribution $f(h)$ @fig-skill-density and discretize it onto the model's $h_{grid}$.

**Calibration of Remaining Parameters**
Several model parameters are not directly identified by the estimation procedure above and are calibrated using standard values from the literature or set to match specific aggregate targets.

*   **Discount Factor ($\beta$):** Set to $0.9615$ corresponding to an annual discount rate of 4%.
*   **Exogenous Separation Rate ($\delta$):** Set to $0.07$.
*   **Matching Function Exponent ($\gamma$):** Set to $0.6$ following [@menzio2010a]
*   **Unemployment Benefit ($b$):** Set the unemployment benefit as  $0.6$ of the potential wage the lowest skill worker
