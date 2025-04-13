### Proof of Properties of the Optimal Remote Policy {#sec-appendix-properties-remote-policy-general}

::: {#prp-optimal-remote-policy-general}
###### Properties of the Optimal Remote Work Policy
Consider the optimal remote work policy $$\alpha^*(\psi,h,x)=
\begin{cases}
0 & \text{if } \psi \le \underline{\psi}(h), \\
\alpha^*(\psi,h,x) & \text{if } \underline{\psi}(h) < \psi < \overline{\psi}(h), \\
1 & \text{if } \overline{\psi}(h) \le \psi,
\end{cases}
$$ with the interior solution $\alpha^*(\psi,h,x)$ satisfying the first-order condition $$A(h)\bigl(g(\psi,h)-1\bigr) = -\frac{u_\alpha\bigl(w(\alpha^*),\alpha^*\bigr)}{u_w\bigl(w(\alpha^*),\alpha^*\bigr)}.$$ and the thresholds $\underline{\psi}(h)$ and $\overline{\psi}(h)$ defined implicitly by:$$
A(h)\Bigl(g\bigl(\underline{\psi}(h),h\bigr) -1\Bigr) =  C_{0}, \quad \text{and} \quad
A(h)\Bigl(g\bigl(\overline{\psi}(h),h\bigr) -1\Bigr) =  C_{1},
$$Then the following results hold:
1. If worker skill enhances relative remote productivity:$$
   \frac{\partial}{\partial h}\Bigl[A(h)\,g(\psi,h)\Bigr] > \frac{\partial A(h)}{\partial h},
	   $${#eq-condition-1-prop-monotonicity}
	   then the interior solution $\alpha^*(h)$ is *strictly increasing* in $h$. And the thresholds $\underline{\psi}(h)$ and $\overline{\psi}(h)$ are *strictly decreasing* in $h$.
2. If worker skill affects remote and baseline productivity equally at the margin:$$
	 \frac{\partial}{\partial h}\Bigl[A(h)\,g(\psi,h)\Bigr] = \frac{\partial A(h)}{\partial h},$${#eq-condition-2-prop-monotonicity}
	   then the interior solution $\alpha^*(h)$ is *constant* with respect to $h$. And the thresholds $\underline{\psi}(h)$ and $\overline{\psi}(h)$ are *locally invariant* with respect to $h$. 
   3. If worker skill affects remote and baseline productivity equally at the margin:$$
	 \frac{\partial}{\partial h}\Bigl[A(h)\,g(\psi,h)\Bigr] < \frac{\partial A(h)}{\partial h},$${#eq-condition-3-prop-monotonicity}
	   then the interior solution $\alpha^*(h)$ is *strictly decreasing* in $h$. And the thresholds $\underline{\psi}(h)$ and $\overline{\psi}(h)$ are *strictly increasing* in $h$.
:::

:::{.proof} 

**Monotonicity of the Interior Solution:**
We are concerned with the interior solution $\alpha^*(h)$, which satisfies the first order condition (FOC):
$$
F\big(\alpha^*(h),\psi,h,x\big) \equiv A(h)\big(g(\psi,h)-1\big) + \frac{u_\alpha\big(w(\alpha^*(h)),\alpha^*(h)\big)}{u_w\big(w(\alpha^*(h)),\alpha^*(h)\big)} = 0
$$
Our objective is to determine the sign of $\partial \alpha^*(h)/\partial h$. Differentiating the identity $F\big(\alpha^*(h),\psi,h,x\big) = 0$ with respect to $h$ using the chain rule yields:
$$
F_h + F_{\alpha}\frac{\partial \alpha^*(h)}{\partial h} = 0 \qquad \implies \qquad \frac{\partial \alpha^*(h)}{\partial h} = -\frac{F_h}{F_\alpha}
$$
Notice that: 
$$
-F_\alpha = -\frac{\partial}{\partial \alpha}\Bigg[\frac{u_\alpha\big(w(\alpha),\alpha\big)}{u_w\big(w(\alpha),\alpha\big)}\Bigg] > 0 \qquad \text{(by assumption)}
$$
The sign of $\partial \alpha^*(h)/\partial h$ therefore depends directly on the sign of $F_h$. Taking the partial derivative of $F$ with respect to $h$, we obtain:$$F_h = \frac{\partial}{\partial h}\Big[A(h)\big(g(\psi,h)-1\big)\Big] + \frac{\partial}{\partial h}\Bigg[\frac{u_\alpha\big(w(\alpha),\alpha\big)}{u_w\big(w(\alpha),\alpha\big)}\Bigg] \\
$$
We assume that skill $h$ enters the firm's problem only through the production side ($A(h)$ and $g(\psi, h)$) and not through the worker's utility function or the wage determination mechanism $\omega(x, \alpha)$ directly for a given $x$ and $\alpha$. Thus, the second term involving the marginal rate of substitution is zero with respect to $h$: $\frac{\partial}{\partial h}\Bigg[\frac{u_\alpha}{u_w}\Bigg] = 0$. This simplifies $F_h$ to:$$F_h = \frac{\partial}{\partial h}\Bigl[A(h)\,g(\psi,h)\Bigr] - \frac{\partial A(h)}{\partial h}$${#eq-condition-monotonicity-interior-solution}
From this expression is clear that the conditions in the proposition directly determine the sign of $F_h$ thus determining the monotonicity of $\alpha^*(h)$.

**Monotonicity of the Thresholds:**
We begin with the implicit definition of $\psi(h)$ (which may denote $\underline{\psi}(h)$ or $\overline{\psi}(h)$):
$$
A(h)\Bigl(g(\psi(h),h)-1\Bigr) = C,
$$
where $C$ is a constant (either $C_0$ or $C_1$). Differentiating both sides with respect to $h$, we have$$\frac{d}{dh}\Bigr[A(h)\Bigl(g(\psi(h),h)-1\Bigr)\Bigr] = A'(h)\Bigl(g(\psi(h),h)-1\Bigr) + A(h)\Bigr(g_{\psi}(\psi(h),h)\cdot \psi'(h) + g_{h}(\psi(h),h)\Bigr)= 0.$$Solving for $\psi'(h)$:
$$
\psi'(h) = -\frac{ A'(h)\Bigl(g(\psi(h),h)-1\Bigr) + A(h)g_{h}(\psi(h),h)}{A(h)g_{\psi}(\psi(h),h)} = - \frac{\frac{\partial}{\partial h}\Bigl[A(h)\,g(\psi(h),h)\Bigr] - \frac{\partial A(h)}{\partial h}}{A(h)g_{\psi}(\psi(h),h)},
$$
Because $A(h)>0$ and $g_{\psi}(\psi,h)>0$, the denominator is positive. Hence, the sign of $\psi'(h)$ is determined by the numerator. From here is clear that the conditions in the proposition directly determine the monotonicity of the thresholds $\underline{\psi}(h)$ and $\overline{\psi}(h)$.
:::