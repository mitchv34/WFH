>[!asm] Assumptions
>- The production parameters satisfy:
>	- $A'(h)>0$, so that worker skill $h$ positively affects overall productivity
>	- For each $\psi$, $\displaystyle \frac{\partial g(\psi, h)}{\partial h} \ge 0$, so that higher-skilled workers benefit more from remote work.
>- The worker's utility function $u(w,\alpha)$ is such that
>	- $u_w(w,\alpha)>0$ and $u_\alpha(w,\alpha)>0$ (utility is increasing in both wage and remote work);
>	- $u(w, \alpha)$ is a concave function.
>	- The mapping $w\mapsto u(w,\alpha)$ is strictly increasing, ensuring that the inversion $w=\omega(x,\alpha)$ exists;
>	- The marginal trade-off ratio $$-\dfrac{u_\alpha(w,\alpha)}{u_w(w,\alpha)}$$ is strictly increasing in $\alpha$.

---

>[!prp] Properties of the optimal remote work policy:
>For any range of values of worker skill $h$ and remote work productivity $\psi$, for which the production functions satisfy: $$
\frac{\partial}{\partial h}\Bigl[A(h)\,g(\psi,h)\Bigr] \ge \frac{\partial A(h)}{\partial h}.$$ 
> The optimal remote work policy $$\alpha^*(\psi,h,x)=
\begin{cases}
0 & \text{if } \psi \le \underline{\psi}(h), \\
\alpha^*(\psi,h,x) & \text{if } \underline{\psi}(h) < \psi < \overline{\psi}(h), \\
1 & \text{if } \overline{\psi}(h) \le \psi,
\end{cases}
$$ with the interior solution $\alpha^*(\psi,h,x)$ satisfying the first-order condition $$A(h)\bigl(g(\psi,h)-1\bigr) = -\frac{u_\alpha\bigl(w(\alpha^*),\alpha^*\bigr)}{u_w\bigl(w(\alpha^*),\alpha^*\bigr)},$$
has the following properties: 
>1. **Higher-skilled workers get more remote work:**  
   The interior solution $\alpha^*(h)$ is (weakly) increasing in $h$; that is, holding $\psi$ and $x$ constant, firms assign a higher remote work share to higher-skilled workers. 
>2. **Continuity and Differentiability of Thresholds:**     The thresholds $\underline{\psi}(h)$ and $\overline{\psi}(h)$ —defined respectively as the minimal and maximal remote productivity levels for which it is optimal to offer some and full remote work—are characterized implicitly by: $$\begin{align*}g(\underline{\psi}(h),h)&=1-\frac{1}{A(h)}\left[\frac{u_\alpha\big(w(0),0\big)}{u_w\big(w(0),0\big)}\right]\\g(\overline{\psi}(h),h)&=1-\frac{1}{A(h)}\left[\frac{u_\alpha\big(w(1),1\big)}{u_w\big(w(1),1\big)}\right]\end{align*}$$
 
**Proof:**
We are only concerned with the interior solution $\alpha^*(h)$, which satisfies the first order condition (FOC):$$ 
F\big(\alpha^*(h),\psi,h,x\big) \equiv A(h)\big(g(\psi,h)-1\big) + \frac{u_\alpha\big(w(\alpha^*(h)),\alpha^*(h)\big)}{u_w\big(w(\alpha^*(h)),\alpha^*(h)\big)} = 0
$$
our objective is to show that $\partial \alpha^*(h)/\partial h > 0$.  Differentiating $F$ with respect to $h$ yields $$ 
F_h + F_{\alpha}\frac{\partial \alpha^*(h)}{\partial h} = 0 \qquad \implies \qquad \frac{\partial \alpha^*(h)}{\partial h} = -\frac{F_h}{F_\alpha}
$$Notice that $$-F_\alpha = -\frac{\partial}{\partial \alpha}\Bigg[\frac{u_\alpha\big(w(\alpha),\alpha\big)}{u_w\big(w(\alpha),\alpha\big)}\Bigg] > 0 \qquad \text{by assumption}$$Taking the partial derivative of $F$ with respect to $h$, we obtain:$$
\begin{align*}
F_h &= \frac{\partial}{\partial h}\Big[A(h)\big(g(\psi,h)-1\big)\Big] + \frac{\partial}{\partial h}\Bigg[\frac{u_\alpha\big(w(\alpha),\alpha\big)}{u_w\big(w(\alpha),\alpha\big)}\Bigg] \\
&= \frac{\partial A(h)}{\partial h}\big(g(\psi,h)-1\big) + A(h)\,\frac{\partial g(\psi,h)}{\partial h}
\end{align*}
$$
We assume that that $h$ enters only through the production side so the inversion $w = \omega(x,\alpha)$ is independent of $h$, so that derivative is zero.

Recall that by assumption the production functions satisfy: $$
\frac{\partial}{\partial h}\Bigl[A(h)\,g(\psi,h)\Bigr] \ge \frac{\partial A(h)}{\partial h}.
$$ Using the product rule, this condition can be equivalently written as $$
A'(h)\bigl(g(\psi,h)-1\bigr) + A(h)\,\frac{\partial g(\psi,h)}{\partial h} \ge 0.
$$ This assumption guarantees that the marginal effect of an increase in worker skill on effective remote productivity (i.e. $A(h)g(\psi,h)$) is at least as large as its marginal effect on overall productivity $A(h)$ alone. This condition ensures that $F_h$ is positive.
$$
\frac{\partial \alpha^*(h)}{\partial h} = -\frac{F_h}{F_\alpha} \ge 0.
$$
**Proof of Property 2: Continuity and Differentiability of Thresholds**

The thresholds $\underline{\psi}(h)$ and $\overline{\psi}(h)$ define the boundaries between the corner solutions ($\alpha=0$ or $\alpha=1$) and the interior solution.

The lower threshold $\underline{\psi}(h)$ is the value of $\psi$ where the firm is indifferent between $\alpha=0$ and $\alpha>0$. This occurs when the derivative of the profit function with respect to $\alpha$ is zero at $\alpha=0$. The profit is $\pi(\alpha) = A(h)[1 + \alpha(g(\psi, h) - 1)] - \omega(x, \alpha)$.
$$ \frac{d\pi}{d\alpha} = A(h)(g(\psi, h) - 1) - \frac{\partial \omega(x, \alpha)}{\partial \alpha} = A(h)(g(\psi, h) - 1) + \frac{u_\alpha(\omega(x, \alpha), \alpha)}{u_w(\omega(x, \alpha), \alpha)} $$
Setting $\frac{d\pi}{d\alpha} = 0$ at $\alpha=0$ for $\psi = \underline{\psi}(h)$:
$$ A(h)(g(\underline{\psi}(h), h) - 1) + \frac{u_\alpha(\omega(x, 0), 0)}{u_w(\omega(x, 0), 0)} = 0 $$
Letting $w(0) = \omega(x, 0)$, this rearranges to:
$$ g(\underline{\psi}(h), h) = 1 - \frac{1}{A(h)} \left[ \frac{u_\alpha(w(0), 0)}{u_w(w(0), 0)} \right] $$

The upper threshold $\overline{\psi}(h)$ is the value of $\psi$ where the firm is indifferent between $\alpha=1$ and $\alpha<1$. This occurs when $\frac{d\pi}{d\alpha} = 0$ at $\alpha=1$ for $\psi = \overline{\psi}(h)$:
$$ A(h)(g(\overline{\psi}(h), h) - 1) + \frac{u_\alpha(\omega(x, 1), 1)}{u_w(\omega(x, 1), 1)} = 0 $$
Letting $w(1) = \omega(x, 1)$, this rearranges to:
$$ g(\overline{\psi}(h), h) = 1 - \frac{1}{A(h)} \left[ \frac{u_\alpha(w(1), 1)}{u_w(w(1), 1)} \right] $$
These implicitly define the thresholds $\underline{\psi}(h)$ and $\overline{\psi}(h)$.

For continuity and differentiability, we assume:
1.  $A(h)$ is continuously differentiable ($C^1$).
2.  $g(\psi, h)$ is $C^1$ in both arguments.
3.  $u(w, \alpha)$ is $C^1$, which implies $\omega(x, \alpha)$, $u_w$, and $u_\alpha$ are continuous (and $w(0)$, $w(1)$ are well-defined constants for a fixed $x$).
4.  $\frac{\partial g(\psi, h)}{\partial \psi} \ne 0$ (typically assumed positive, meaning higher $\psi$ increases remote productivity).

Let the defining equation for $\underline{\psi}(h)$ be $G_0(\psi, h) = g(\psi, h) - \left( 1 - \frac{1}{A(h)} \left[ \frac{u_\alpha(w(0), 0)}{u_w(w(0), 0)} \right] \right) = 0$.
By the Implicit Function Theorem, if $\frac{\partial G_0}{\partial \psi} = \frac{\partial g(\psi, h)}{\partial \psi} \ne 0$ at $(\underline{\psi}(h), h)$, then $\underline{\psi}(h)$ is a $C^1$ function of $h$.

Similarly, let $G_1(\psi, h) = g(\psi, h) - \left( 1 - \frac{1}{A(h)} \left[ \frac{u_\alpha(w(1), 1)}{u_w(w(1), 1)} \right] \right) = 0$.
If $\frac{\partial G_1}{\partial \psi} = \frac{\partial g(\psi, h)}{\partial \psi} \ne 0$ at $(\overline{\psi}(h), h)$, then $\overline{\psi}(h)$ is a $C^1$ function of $h$.

Thus, under standard smoothness assumptions and the assumption that remote productivity $g$ responds to $\psi$, the thresholds $\underline{\psi}(h)$ and $\overline{\psi}(h)$ are continuous and differentiable functions of $h$.

---

> [!prp] Splitting Point for the Lower Threshold
There exists a unique splitting point $\hat{h}_0$ in the skill space characterized by $$
g_h\Bigl(g^{-1}\!\left(1-\frac{c_0}{A(\hat{h}_0)}; \hat{h}_0\right),\hat{h}_0\Bigr)= c_0\,\frac{A'(\hat{h}_0)}{A(\hat{h}_0)^2}.$$
Then:
>- For $h < \hat{h}_0$,  $$
  g_h\bigl(\underline{\psi}(h),h\bigr) < c_0\,\frac{A'(h)}{A(h)^2} \quad \Longrightarrow \quad \underline{\psi}'(h) > 0.$$
>- For $h > \hat{h}_0$,  $$
  g_h\bigl(\underline{\psi}(h),h\bigr) > c_0\,\frac{A'(h)}{A(h)^2} \quad \Longrightarrow \quad \underline{\psi}'(h) < 0.$$

**Interpretation:**
The splitting point for the lower threshold, which we denote as $\hat{h}_0$, captures the critical worker skill level where the effect of an increase in skill on the firm's willingness to offer remote work changes its direction. In economic terms, $\hat{h}_0$ marks the boundary between two regimes:
1. For workers with skills below $\hat{h}_0$: an increase in skill raises overall productivity $A(h)$ significantly relative to the direct impact of skill on the remote work productivity function $g$. In this region, the threshold $\underline{\psi}(h)$ – that is, the minimal remote productivity level required for the firm to consider offering any remote work – actually increases with $h$. In other words, when workers are relatively less skilled, only a higher remote work productivity level would justify offering remote work; slight skill improvements demand an even higher productivity level because the benefit from increased $A(h)$ dominates.
2. For workers with skills above $\hat{h}_0$: the direct effect of skill on remote productivity (captured by $g_h$) becomes stronger relative to the productivity gain $A'(h)$. Now, as skill increases further, the threshold $\underline{\psi}(h)$ declines. That is, higher-skilled workers need a lower remote productivity level (in a relative sense) to justify offering remote work since their intrinsic capacity for remote work is already high.

**Proof:**
We already know that $\underline{\psi}(h)$ is continuously differentiable and that its derivative is given by:$$
\underline{\psi}'(h) = -\frac{F_h\bigl(\underline{\psi}(h),h\bigr)}{F_\psi\bigl(\underline{\psi}(h),h\bigr)} 
= \frac{\frac{c_0\,A'(h)}{A(h)^2} - g_h\bigl(\underline{\psi}(h),h\bigr)}{g_\psi\bigl(\underline{\psi}(h),h\bigr)},\text{ with } c_0 = \frac{u_\alpha(w(0),0)}{u_w(w(0),0)}>0;$$Because $g_\psi(\cdot,\cdot) > 0$ by assumption, the sign of $\underline{\psi}'(h)$ is determined entirely by the numerator. In particular, the derivative $\underline{\psi}'(h)$ changes sign precisely when
$$
g_h\bigl(\underline{\psi}(h),h\bigr) - \frac{c_0\,A'(h)}{A(h)^2} = 0.
$$
Thus, if we define the splitting point $\hat{h}_0$ as the unique value of $h$ such that
$$
g_h\bigl(\underline{\psi}(\hat{h}_0),\hat{h}_0\bigr) = \frac{c_0\,A'(\hat{h}_0)}{A(\hat{h}_0)^2},
$$
then for $h < \hat{h}_0$ the numerator is negative (so that $\underline{\psi}'(h)>0$) and for $h > \hat{h}_0$ the numerator is positive (so that $\underline{\psi}'(h)<0$). This establishes not only that $\underline{\psi}(h)$ is continuously differentiable but also identifies $\hat{h}_0$ as the critical point at which the sensitivity of the remote productivity function $g$ to skill exactly balances the weighted sensitivity of overall productivity $A$ with respect to $h$.


---

>[!prp] Splitting Point for the Upper Threshold
>There exists a unique splitting point $\hat{h}_1$ in the skill space defined by$$
g_h\Bigl(g^{-1}\!\left(1-\frac{c_1}{A(\hat{h}_1)}; \hat{h}_1\right),\hat{h}_1\Bigr)= c_1\,\frac{A'(\hat{h}_1)}{A(\hat{h}_1)^2}.$$
> Then:
> - For $h < \hat{h}_1$,$$
  g_h\bigl(\overline{\psi}(h),h\bigr) < c_1\,\frac{A'(h)}{A(h)^2} \quad \Longrightarrow \quad \overline{\psi}'(h) > 0.$$
>- For $h > \hat{h}_1$, $$g_h\bigl(\overline{\psi}(h),h\bigr) > c_1\,\frac{A'(h)}{A(h)^2} \quad \Longrightarrow \quad \overline{\psi}'(h) < 0.  $$

**Economic Intuition:**

The upper threshold $\overline{\psi}(h)$ sets the productivity level above which full remote work becomes optimal. Its value depends on both the overall productivity factor $A(h)$ and the remote work quality captured by $g$. For lower-skilled workers (i.e. $h<\hat{h}_1$), increases in skill boost overall productivity $A(h)$ enough (through its derivative $A'(h)$) that a higher remote productivity level is required for the firm to justify a full remote arrangement. In this region, $\overline{\psi}(h)$ increases with $h$. In contrast, above the splitting point ($h>\hat{h}_1$), the direct effect of skill on the remote work technology—as measured by $g_h$—dominates. Thus, further improvements in skill reduce the required threshold (i.e. $\overline{\psi}(h)$ falls), making full remote work attractive to high-skilled workers.

This critical value $\hat{h}_1$ therefore partitions the worker skill space into two regions with qualitatively different responses of the full remote work threshold to changes in skill.

**Proof:**
Similar to the proof for $\underline{\psi}(h)$.

