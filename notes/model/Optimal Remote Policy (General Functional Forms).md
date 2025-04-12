# Problem Statement:

We consider the following optimization problem:
$$
\max_{\alpha \in [0,1]} \Big\{ Y(\alpha \mid \psi, h) - w(\alpha) \;\Big|\; x = u(w(\alpha), \alpha) \Big\}
$$
**Invertibility Check:**
The key condition needed here is that the mapping $w \mapsto u(w,\alpha)$ is **strictly increasing**. Since we assume $u_w(w,\alpha) > 0$ for every $(w,\alpha)$ in the domain of $u(\cdot)$, this means that it can be uniquely inverted in $w$.  Define $\omega(x,\alpha)$ as the inverse function of $u(w,\alpha)$:
$$
w(\alpha) = \omega(x,\alpha) \qquad \implies \quad x = u(w(\alpha),\alpha) \quad \text{for } w \in [0,\infty).
$$
Continuous differentiability of $u$ ensures that $\omega(x,\alpha)$ is well-behaved.

Substitute $w(\alpha)$ with $\omega(x,\alpha)$ in the objective function:
$$
\max_{\alpha \in [0,1]} \; \Pi(\alpha) \quad \text{where} \quad \Pi(\alpha) = Y(\alpha \mid \psi, h) - \omega(x, \alpha)
$$
Applying this substitution and considering considering any potential $\alpha \in \mathbb{R}$, the firm's problem becomes an unconstrained maximization in $\alpha$:
$$
\max_{\alpha \in \mathbb{R}} \; \Pi(\alpha) \quad \text{where} \quad \Pi(\alpha) = Y(\alpha \mid \psi, h) - \omega(x, \alpha)
$$
The first order condition (FOC) for this maximization problem is given by:
$$
\frac{d\Pi(\alpha)}{d\alpha} = A(h)\,\big(g(\psi,h)-1\big) - \frac{\partial \omega(x,\alpha)}{\partial \alpha} = 0
$$
Recall:
$$
x = u\big(\omega(x,\alpha), \alpha\big).
$$
Differentiate both sides with respect to $\alpha$:
$$
0 = \frac{d}{d\alpha}\, u\big(\omega(x,\alpha), \alpha\big)
$$
Applying the chain rule:
$$
0 = u_w\big(\omega(x,\alpha),\alpha\big) \, \frac{\partial \omega(x,\alpha)}{\partial \alpha} + u_\alpha\big(\omega(x,\alpha),\alpha\big)
$$
Solving for $\partial \omega(x,\alpha)/\partial \alpha$, we obtain:
$$
\frac{\partial \omega(x,\alpha)}{\partial \alpha} = -\frac{u_\alpha\big(\omega(x,\alpha),\alpha\big)}{u_w\big(\omega(x,\alpha),\alpha\big)}
$$

Substituting back into the derivative of $\Pi(\alpha)$:
$$
\frac{d\Pi(\alpha)}{d\alpha} = A(h)\,\big(g(\psi,h)-1\big) + \frac{u_\alpha\big(\omega(x,\alpha),\alpha\big)}{u_w\big(\omega(x,\alpha),\alpha\big)}
$$
For an interior optimum $\alpha^*$, the first order condition is given by:
$$
A(h)\,\big(g(\psi,h)-1\big) + \frac{u_\alpha\big(\omega(x,\alpha^*),\alpha^*\big)}{u_w\big(\omega(x,\alpha^*),\alpha^*\big)} = 0
$$
Since by definition $\omega(x,\alpha^*) = w(\alpha^*)$ (because the substitution recovers the original wage from the constraint $x = u(w(\alpha^*),\alpha^*)$), we can equivalently write:
$$
A(h)\,\big(g(\psi,h)-1\big) = -\frac{u_\alpha\big(w(\alpha^*),\alpha^*\big)}{u_w\big(w(\alpha^*),\alpha^*\big)}
$$
In addition, the original promise-keeping constraint still holds:
$$
x = u\big(w(\alpha^*),\alpha^*\big).
$$

**Economic Interpretation:**

- The term $A(h)\,\big(g(\psi,h)-1\big)$ represents the marginal change in production when increasing the share of remote work $\alpha$. In particular, if remote work is less productive (i.e., $g(\psi,h) < 1$), then increasing $\alpha$ improves output; otherwise, the opposite is true.
- The term $u_\alpha/u_w$ represents the trade-off from the worker's utility perspective, indicating how much the wage can be reduced (in marginal terms) for an increase in $\alpha$ while still maintaining the worker’s promised utility level $x$.
- The firm chooses $\alpha$ such that the marginal benefit from the production side exactly offsets the marginal saving in wage cost, as captured by the worker's indifference curve.

**Threshold Conditions:**
In order to fully characterize the solution, we now derive threshold conditions for two boundary cases:

- When the firm decides to offer some remote work (i.e., an optimal $\alpha^*>0$).
- When the firm decides to go all remote (i.e., $\alpha^*=1$).

*Threshold for Offering Some Remote Work ($\alpha^*>0$):*
For the firm to choose a strictly positive level of remote work, a marginal increase in $\alpha$ from zero must yield an increase in profit. In other words, the derivative of $\Pi(\alpha)$ evaluated at $\alpha=0$ must be **positive**:
$$
\frac{d\Pi(\alpha)}{d\alpha}\Big|_{\alpha=0} = A(h)\big(g(\psi,h)-1\big)+\frac{u_\alpha\big(w(0),0\big)}{u_w\big(w(0),0\big)}>0
$$
Rearranging the inequality, the threshold condition is:
$$
A(h)\big(g(\psi,h)-1\big)>-\frac{u_\alpha\big(w(0),0\big)}{u_w\big(w(0),0\big)}
$$
If this condition holds, then even a slight increase in $\alpha$ from zero improves profits and the firm will offer some remote work, i.e. $\alpha^*>0$. We can solve for the threshold $\underline{\psi}(h)$ such that:$$g(\underline{\psi}(h),h)=1-\frac{1}{A(h)}\left[\frac{u_\alpha\big(w(0),0\big)}{u_w\big(w(0),0\big)}\right]$$*Threshold for Full Remote Work ($\alpha^*=1$):*
Similarly, for the firm to choose full remote work, the derivative of $\Pi(\alpha)$ at the upper boundary $\alpha=1$ must be non-negative. That is,
$$
\frac{d\Pi(\alpha)}{d\alpha}\Big|_{\alpha=1} = A(h)\big(g(\psi,h)-1\big)+\frac{u_\alpha\big(w(1),1\big)}{u_w\big(w(1),1\big)}\geq 0
$$
When this condition holds, increasing $\alpha$ further (beyond values arbitrarily close to 1) does not raise profits, so the firm optimally opts for a full remote work arrangement, i.e. $\alpha^*=1$.   Similarly, we can solve for the threshold $\overline{\psi}(h)$:$$g(\overline{\psi}(h),h)=1-\frac{1}{A(h)}\left[\frac{u_\alpha\big(w(1),1\big)}{u_w\big(w(1),1\big)}\right]$$