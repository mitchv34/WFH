### Model Overview

This model follows a directed search framework in the spirit of [@menzio2010a]. The model incorporates two key sources of heterogeneity: firms differ in their remote-work efficiencies, while workers vary in their skill levels. The key mechanisms in this framework are that workers value the flexibility provided by remote work arrangements, high-skilled workers are more productive and better suited for remote work, and firms treat remote and on-site work as substitutable inputs in their production processes.

**Workers** are characterized by their productivity $h$. They incur disutility from on-site work, which is partially compensated by their wage. We denote by $u(w,\alpha)$ the utility of a worker earning wage $w$ with remote work arrangement $\alpha\in[0,1]$, where $\alpha$ represents the fraction of time working remotely. We assume that utility continuously differentiable, concave and  increasing in consumption $u_{w}(\cdot) > 0$ and in remote work $u_{\alpha}(\cdot) > 0$. We further assume that the marginal rate of substitution between remote work and consumption, $-{u_\alpha\big(w,\alpha\big)}/{u_w\big(w,\alpha\big)}$ is increasing in $\alpha$. This condition means that as workers spend more time working remotely, they place an increasing relative value on additional remote work compared to higher wages. Intuitively, this captures increasing marginal willingness to trade off wage for flexibility: the more accustomed a worker becomes to remote work, the more they value the ability to work remotely even further.  We assume that workers supply their unit of labor inelastically.

**Firms** are characterized by their remote-work efficiency parameter $\psi$, which determines how effectively they can implement remote work arrangements.  The output of a firm-worker match depends on firm remote productivity, worker skill and the fraction of remote work determined by the arrangement $\alpha$.  We asume that $A(h)$,  ($A'(h) > 0$) captures the contribution to output from worker skill $h$. The firm split worker's labor between in-person work and remote work. However, these components aren’t perfectly substitutable, remote work is adjusted by a factor $g(\psi, h)$. The function $g(\psi, h)$ captures the efficiency of remote work, with $g_{\psi}(\psi, h) \geq 0$ indicating that remote work efficiency increases with firm type $\psi$ due to better technology, management practices, or the nature of the occupation. Similarly, $g_{h}(\psi, h) \geq 0$ suggests that remote work efficiency increases with worker skill level due to greater autonomy and technological ability.  The production function of the firm is given by:
$$Y(\alpha \mid \psi, h) = A(h)\left((1 - \alpha) + \alpha g(h,\psi)\right)$$ {#eq-prod}

**Profits** are determined by the difference between the output produced and the wage paid to the worker. Let $x$ denote the total utility level that a working arrangement delivers to the worker, $x$ is derived from both the wage received and the work arrangement (i.e., the fraction of remote work $\alpha$).  Since workers care only about their total utility, the firm is constrained to ensure that the worker obtains at least the utility level $x$. This creates a trade-off for the firm: while offering higher remote work (a larger $\alpha$) may decrease productivity it might also allow the firm to pay a lower wage to meet the utility guarantee. A type $\psi$ firm with a worker of type $h$ chooses $\alpha$ to maximize the following expression: $$\Pi(\alpha \mid \psi, h, x)=\max_{\alpha\in[0,1]}\left\{Y(\alpha \mid \psi, h)-w(\alpha) \:\mid\: x=u(w(\alpha),\alpha)\right\}$$ {#eq-firmProblem}

Since $g(\psi, h)$ captures the adjustment of productivity to remote work in @eq-prod and is heterogeneous across firms and workers, the optimal remote work policy will also be heterogeneous.

### Optimal Remote Work Choice
The firm’s problem implies an optimal choice of remote work $\alpha^*(\psi, h, x)$ that satisfies the first-order condition:

**Interior Solution:** Let's start with the interior solution, where the firm chooses a positive fraction of remote work. This optimal choice $\alpha^*(\psi, h, x)$ must satisfy:
1. **First Order Condition:**
$$A(h)\,\big(g(\psi,h)-1\big) = -\frac{u_\alpha\big(w(\alpha^*),\alpha^*\big)}{u_w\big(w(\alpha^*),\alpha^*\big)}.$$ {#eq-interiorSolutionOptimalRemote}
2. **Promise Keeping Constraint:**$$
    x = u(w(\alpha^*),\alpha^*).$$ {#eq-promiseKeepingConstraint}

The assumptions $u_w > 0$ and $u_\alpha > 0$ ensure that the utility function is strictly increasing in the wage argument and continuously differentiable, which in turn guarantees that the inversion $w(\alpha) = u^{-1}(x,\alpha)$ is well defined and unique. Details of the derivation are on [Appendix @sec-appendix-remote-policy-general]

**Economic Interpretation:** The term $A(h)\,\big(g(\psi,h)-1\big)$ represents the marginal change in production when increasing the share of remote work $\alpha$. In particular, if remote work is less productive (i.e., $g(\psi,h) < 1$), then increasing $\alpha$ improves output; otherwise, the opposite is true. The term $u_\alpha/u_w$ represents the trade-off from the worker's utility perspective, indicating how much the wage can be reduced (in marginal terms) for an increase in $\alpha$ while still maintaining the worker’s promised utility level $x$. The firm chooses $\alpha$ such that the marginal benefit from the production side exactly offsets the marginal saving in wage cost, as captured by the worker's indifference curve. Importantly, no assumption is made the magnitude of $g(\psi, h)$. Should remote work be more productive for any worker skill level ($g(\psi, h) > 1$), the firm optimally selects a corner solution with $\alpha=1$. This arises because the marginal benefit from increasing $\alpha$ includes both enhanced production _and_ potential wage savings ($u_\alpha/u_w > 0$). Consequently, the trade-off described earlier—balancing potential output changes against wage cost adjustments—no longer constrains the firm from maximizing remote work.

**Corner Solution:** Notice that @eq-interiorSolutionOptimalRemote describes only the interior solutions. The firm may also choose to offer either full remote work ($\alpha = 1$) or no remote work $(\alpha = 0)$. The conditions for these corner solutions are as follows:
- \[$\underline{\psi}(h)$\] *Threshold for Offering Some Remote Work ($\alpha^*>0$):*$$g(\underline{\psi}(h),h)=1-\frac{1}{A(h)}\left[\frac{u_\alpha\big(w(0),0\big)}{u_w\big(w(0),0\big)}\right]$$ {#eq-lower-threshold}
- \[$\overline{\psi}(h)$\] *Threshold for Full Remote Work ($\alpha^*=1$):*$$g(\overline{\psi}(h),h)=1-\frac{1}{A(h)}\left[\frac{u_\alpha\big(w(1),1\big)}{u_w\big(w(1),1\big)}\right]$$ {#eq-upper-threshold}

Both thresholds are obtained by evaluating $$\frac{d\Pi(\alpha)}{d\alpha}\Big|_{\alpha=0} \qquad \text{and} \qquad  \frac{d\Pi(\alpha)}{d\alpha}\Big|_{\alpha=1}$$the idea is to find for each skill level the region in the remote productivity space where even a slight deviation in remote work (from full remote or full in person) do not increase profits from the firm. Details on [Appendix @sec-appendix-remote-policy-general]

The optimal remote work policy is given by: $$\alpha^*(\psi, h, x) = \begin{cases}
    \hspace{1cm} 0 & \text{if} \quad \psi \leq \underline{\psi}(h) \\
    \alpha^*(\psi, h, x) &  \text{if} \quad \underline{\psi}(h) < \psi < \overline{\psi}(h) \\
    \hspace{1cm} 1 & \text{if} \quad \overline{\psi}(h) \leq \psi
\end{cases}$$ {#eq-optimalRemoteWorkPolicy}
The interior solution $\alpha^*(\psi, h, x)$ satisfies the first-order condition in @eq-interiorSolutionOptimalRemote.

**Properties of the Optimal Remote Policy**

1. If an increase in worker skill $h$ raises remote productivity by more than it raises baseline productivity. In other words, then the interior solution $\alpha^*(h)$ increases with $h$ (firms offer more remote work to more skilled workers). At the same time, the thresholds $\underline{\psi}(h)$ and $\overline{\psi}(h)$, decrease with worker skill; meaning that the requirements on firm remote productivity for offering some (and full) remote work are lower.
2. If an increase in worker skill $h$ raises remote productivity and baseline productivity equally at the margin, then the interior solution $\alpha^*(h)$ is constant with respect to $h$. The thresholds $\underline{\psi}(h)$ and $\overline{\psi}(h)$ are locally invariant (flat) with respect to worker skill.
3. If an increase in worker skill $h$ raises baseline productivity more than remote productivity, then the interior solution $\alpha^*(h)$ decreases with $h$ higher worker skill implies lower remote work. The thresholds $\underline{\psi}(h)$ and $\overline{\psi}(h)$ increase with worker skill; meaning that the requirements on firm remote productivity for offering some (and full) remote work are higher.
Details of the derivations of this properties are on [Appendix @sec-appendix-properties-remote-policy-general]

#### Parametric Example

In this example, we detail a specific parameterization that illustrates the general structure of our model. The productivity function is given by
$$
A(h)=A_0+A_1h,
$$
with the assumption that $A'(h)=A_1>0$, ensuring that productivity increases with skill level $h$. The effectiveness of remote work is modeled as$$
g(\psi,h)=\psi-\psi_0+\phi\log(h) \quad \text{ with }\phi\ge0.$$ Worker utility is specified by$$
U(w,\alpha)=a_0+a_1w-c_0(1-\alpha)^\chi,$$with parameters satisfying $a_1>0$, $c_0>0$, and $\chi>1$. This formulation guarantees that utility increases with the remote work share $\alpha$ and that the marginal trade-off ratio is rising in $\alpha$. 

The firm's profit maximization problem, subject to a fixed utility level for the worker, leads to the following optimal remote work share:$$\alpha^{*}(\psi,h,x)=\begin{cases}
		0 & \text{if } \psi \leq \underline{\psi}(h), \\
		1-\left[\frac{a_1A(h)(1-g(\psi,h))}{c_0\chi}\right]^{\frac{1}{\chi-1}} & \text{if } \underline{\psi}(h) < \psi < \overline{\psi}(h), \\
		1 & \text{if } \overline{\psi}(h) \leq \psi,
\end{cases}$$With:$$\overline{\psi}(h)=1+\psi_0-\phi\log(h),
\qquad \text{and}\qquad \underline{\psi}(h)=1+\psi_0-\phi\log(h)-\frac{c_0\chi}{a_1(A_0+A_1h)}.
$$Regarding their monotonicity in $h$, if $\phi=0$ the upper threshold remains constant, but if $\phi>0$, $\overline{\psi}(h)$ is strictly decreasing. In contrast, with $\phi=0$, the lower threshold $\underline{\psi}(h)$ increases with $h$. When $\phi>0$, the monotonicity of $\underline{\psi}(h)$ depends on the value of $A_0$. If $A_0=0$, $\underline{\psi}(h)$ increases for $h<h_2$ and decreases for $h>h_2$, where 
$$
h_2=\frac{c_0\chi}{\phi a_1A_1}.
$$
For $A_0>0$, if $c_0\chi\le4\phi a_1A_0$, $\underline{\psi}(h)$ is strictly decreasing; if $c_0\chi>4\phi a_1A_0$, it decreases for $h\in(0,h_1)$, increases for $h\in(h_1,h_2)$, and decreases again for $h>h_2$, with the turning points determined by 
$$
h_{1,2}=\frac{(c_0\chi-2\phi a_1A_0)\pm\sqrt{c_0\chi(c_0\chi-4\phi a_1A_0)}}{2\phi a_1A_1}.
$$

::: {layout-ncol=2}
![This panel maps the optimal remote work share $(\alpha^{*})$ for different combinations of firm efficiency and worker skill. The graph visually represents the trade-off between productivity adjustments and wage savings in determining the remote work arrangement.](../figures/model_figures/plot_policy_for_skill.pdf)

![This panel illustrates how the lower $(\underline{\psi}(h))$ and upper $\overline{\psi}(h)$ thresholds  vary with worker skill, highlighting the conditions under which firms move from no remote work to an interior solution or full remote work.](../figures/model_figures/plot_thresholds.pdf)
:::

### Labor Market Search

Both firms and workers discount the future at rate $\beta$. Workers are characterized by their type $h$ and direct their search toward submarkets distinguished by the promised utility level $x$. A worker of type $h$ evaluates the different utility promises available in each submarket and chooses to search in the one that maximizes their expected value. This expected value incorporates not only the probability of being hired but also the future discounted value of the job. At the same time, firms target workers of a particular type $h$ by posting job offers (or contracts) that promise a specific utility level $x$. This setup allows the market to be segmented into different submarkets. The tightness of a submarket $(h, x)$  is defined as:$$
\theta(h, x) = \frac{v(h, x)}{u(h, x)},$$where $v(h, x)$ denotes the number of vacancies posted by firms in the submarket and $u(h, x)$ represents the number of unemployed workers actively searching within that particular submarket. This measure of tightness directly influences the probabilities of matching: the vacancy filling rate $q(\theta(h, x))$ and the job finding rate $p(\theta(h, x))$ are both functions of $\theta$. In our equilibrium, free entry of firms ensures that the expected profit from posting a vacancy is zero, after incurring a cost $\kappa\in\mathbb{R}_{++}$. Matches are exogenously broken at a rate $\delta$.

Once a firm and a worker are matched, the firm delivers the promised utility $x$ to the worker by applying the firm's optimal remote work policy. Before posting vacancies, firms face uncertainty about their remote-work efficiency parameter $\psi$. However, the distribution $F(\psi)$ is common knowledge among all agents in the economy. Because firms are ex-ante identical in this dimension, any worker searching in a given submarket faces the same probability of being matched with a firm having a particular productivity level $\psi$.

For firms, the value of posting a vacancy in a submarket characterized by $(h, x)$ is given by$$
V(h, x) = -\kappa + q(\theta(h,x))\int J(\psi, h, x)\, dF(\psi),
$$ {#eq-valueFirmEntry}
where $\kappa$ is the vacancy posting cost and $J(\psi, h, x)$ is the value from an ongoing match with a firm of productivity $\psi$. The match value is determined by the current payoff—expressed as the output minus the wage cost plus the discounted expected continuation value:$$
J(\psi, h, x) = Y\bigl(\alpha^*(\psi, h)\mid \psi, h\bigr) - w\bigl(x, \alpha^*(\psi, h)\bigr) + \beta\Bigl[(1-\delta) J(\psi, h, x) + \delta\, V(h, x)\Bigr].
$$ {#eq-valueFirmMatch}

Notice that free-entry guarantee that $V(h,x)=0$, this means that the  value function described in @eq-valueFirmMatch can be computed independently of the distributions of workers and vacancies across submarkets. Furthermore the value of matches pin-down the meeting rates and thus the submarket tightness. Notice that free entry condition is binding if the submarket is active in equilibrium (i.e. $\theta(h,x)>0$), then from @eq-valueFirmEntry:$$\theta(h,x) = q^{-1}\left(\frac{\kappa}{\int J(\psi, h, x)\, dF(\psi)}\right) \quad \text{ if } \theta(h,x)>0$${#eq-submarketTightnessPinDownEquation}
For workers, the value functions capture the trade-off between being unemployed and employed. The value of unemployment for a worker of type $h$ is$$
U(h) = b + \max_{x} \Biggl\{ p\bigl(\theta(h,x)\bigr) \int W(\psi, h, x)\, dF(\psi) + \Bigl(1 - p\bigl(\theta(h,x)\bigr)\Bigr) U(h)\Biggr\},$${#eq-unemployedWorkerValue}
where $b$ denotes the unemployment benefit. Once employed, the worker's value is given by$$
W(\psi, h, x) = x + \beta\Bigl[(1-\delta)W(\psi, h, x) + \delta\,U(h)\Bigr].$${#eq-employedWorkerValue}
This recursive formulation encapsulates the idea that a worker receives the promised utility $x$ while also facing the possibility of job separation.

