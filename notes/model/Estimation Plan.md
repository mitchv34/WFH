# 1 **Parameter and Model Structure Overview**
## 1.1 List of parameters and their roles in the model.
- **Production Function Parameters:**
	* *(calibrated)* $A_1$: Skill productivity gain in $A(h) = A_0 + A_1h$.
		* $A_0 = 0$ by assumption.
	* *(calibrated)* $\phi$: Sensitivity of remote work efficiency to skill in $g(\psi, h) = \psi - \psi_0 + \phi \log(h)$.
	* *(calibrated)* $\psi_0$: Overall economic conduciveness for remote work.
- **Utility Function Parameters:**
	- Utility Function: $U(w, \alpha) = a_0 + a_1w - c_0(1 - \alpha)^\chi$
	* $a_1$: Marginal utility of wage.
	* $c_0$: Disutility from on-site work.
	* $\chi$: Concavity of on-site disutility.
		* *To Estimate:* $\chi$ and $\frac{c_0}{a_{1}}$.
- **Economic Environment Parameters:**
	*  *(calibrated)* $\beta$: Discount factor.
		* Calibrated to match a $r$ discount factor.
	* *(To Estimate)* $\kappa$: Vacancy posting cost.  
	* *(calibrated)* $\delta$: Job separation rate.
## 1.2 Flowchart of how parameters feed into the model
```mehrmaid
graph TD

    subgraph Economic Environment
		list_params["$\beta$ Discount Factor - $\kappa$ Vacancy Posting Cost - $\delta$ Job Separation Rate - $h$  Worker Skill continous variable - $\psi$  Firm Remote Work Efficiency continous variable - $x$ Promised utility workers search for continuous variable.
        "]:::big
    end

    subgraph Production Function
        A0["$A_0$"]:::big
        A1["$A_1$"]:::big
        phi["$\phi$"]:::big
        psi0["$\psi_0$"]:::big
    end

    subgraph Utility Function
        a0["$a_0$"]:::big
        a1["$a_1$"]:::big
        c0["$c_0$"]:::big
        chi["$\chi$"]:::big
    end

    A0 & A1 --> A_h["$A(h) = A_0 + A_1 h$"]:::big

    phi & psi0--> g_psi_h["$g(\psi,h) = \psi - \psi_0 + \phi \cdot \log(h)$"]:::big

    a0 & a1 & c0 & chi --> U_w_alpha["$U(w,\\alpha)= a_0  +  a_1 \cdot w + c_0 \cdot (1-\alpha)^\chi$"]:::big

    A_h & g_psi_h --> Y_alpha_psi_h["$Y(\alpha \mid \psi, h) = A(h) \cdot ( (1-\alpha) + \alpha \cdot g(\psi,h) )$"]:::big

    Y_alpha_psi_h & U_w_alpha --> FirmProblem["Firm's Profit Maximization Problem: $\max_{\alpha}\{Y(\alpha \mid \psi,h) - w(\alpha) \: \text{s.t.} \: x = U(w(\alpha),\alpha)\Bigr\}$"]:::big

    FirmProblem --> FOC["First Order Condition (FOC)"]:::big
    FirmProblem --> Thresholds["Threshold Conditions: $\underline{\psi}(h)$, $\overline{\psi}(h)$"]:::big

    FOC --> alpha_star["$\alpha^*(\psi,h,x)$: Optimal Remote Work Choice (Interior Solution)"]:::big

    Thresholds --> alpha_star

    alpha_star --> Output{"Optimal Remote Work Share $(\alpha^*)$"}:::big

    style Output fill:#f9f,stroke:#333,stroke-width:2px;

classDef big font-size:20px,stroke:#000,stroke-width:1px;
```    
---
# 2 **Target Moments and Data**  
## 2.1 List of (potential) empirical moments to match.

1. **Distribution of Remote-Work Shares by Skill**
    - _Moment:_ Fraction of workers in corner solutions ($\alpha=0$ or $\alpha = 1$) vs. interior solution ($0<\alpha<1$), across different skill bins.
    - _Identification:_
        - $\chi$ will affect how sharply workers bunch into corner solutions or partial solutions.
        - $\frac{c_{0}}{a_{1}}$​​ scales how large the remote wage discount is in partial-remote submarkets.
            
2. **Average Wages by Remote-Share Category**
    - _Moment:_ Compare average (or median) wages for purely on-site, partially remote, and fully remote workers (potentially interacted with skill bins).
    - _Identification:_
        - High $\frac{c_{0}}{a_{1}}$​​​​ means a bigger wage discount for remote, all else equal.
        - $\chi$ also shapes how the partial-remote group’s wages deviate from those of corner groups because it influences how “costly” on-site time is.
            
3. **Skill–Remote Correlation**
    - _Moment:_ Regression or correlation coefficient between skill $h$ and actual remote share $α$.
    - _Identification:_
        - Cross-check the calibration of $A_1, \psi_0$ and $\phi$ against the skill–remote correlation.
        - It also provides an indirect check on whether $\frac{c_{0}}{a_{1}}$​​ and $\chi$ are consistent with the shape of the skill–remote adoption pattern (e.g., high-skilled workers are more likely to be remote if remote is more productive and if disutility parameters are consistent with that distribution).
            
4. **Vacancy–Unemployment Ratios or Matching Rates**
    - _Moment:_ Overall job-filling rates, or submarket tightness $θ(h,x)$, or the fraction of vacancies that remain unfilled after a certain time.
    - _Identification:_
        - This is crucial for $\kappa$. Need some measure of how _costly_ it is to post vacancies, which in equilibrium determines how many jobs are posted
        - Micro-data or partial data on job postings and acceptance rates, can help pin down $\kappa$.
5. **Unemployment Spells by Skill Level**
	- *Rich Cross-Sectional Variation:*  In our model, worker skill $h$ affects both the production side and, indirectly, the search environment (via submarket tightness $\theta(h, x)$ and the matching probabilities $p(\theta)$). Differences in unemployment duration across skill levels be used to asses whether the model correctly captures these heterogeneous matching frictions.
	- *Indirect Identification of Vacancy Posting Costs ($\kappa$):*  The equilibrium unemployment spell durations—reflected in job-finding rates $p(\theta(h, x))$—are connected to the vacancy posting cost through the free-entry condition. 
		- If firms face higher $\kappa$, they post fewer vacancies in equilibrium, which then increases the time a worker (of a given skill level) is likely to remain unemployed.
		- Variation by skill level can help identify $\kappa$ as the matching efficiency might differ systematically across submarkets targeting different $h$.
	- *Data Construction:*
	    - **Empirical Measurement:** Use longitudinal labor force data to measure the average or distribution of unemployment spell lengths segmented by skill level $h$.
	-  *Mapping to the Model:*
	    - **Skill-Level Heterogeneity:** Since $\theta(h, x) = v(h, x)/u(h, x)$ and firms’ and workers’ decisions may vary with skill $h$, the model’s predicted unemployment duration should also vary by $h$. 
	- *Identification:*
		- **Matching Component:** If your model predicts a systematic change in $p(θ(h,x))$ with skill (for instance, high-skilled workers might have a different matching probability due to a better fit with jobs or more active search behavior), then mismatches between the predicted and observed unemployment durations can signal mis-specification in the matching component or in the equilibrium behavior arising from $\kappa$.
        - **Utility Verification:** Differences between predicted unemployment durations and actual data can also point to issues with the assumed trade-offs in the utility function if these lead to unintended distortions in workers’ job search intensity.
        

---
