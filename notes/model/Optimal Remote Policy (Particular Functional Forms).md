Consider the following parametrization of the model:

**1. Functional Forms & Parameters:**
*   Productivity: $A(h) = A_0 + A_1 h$
*   Remote Work Effectiveness: $g(\psi, h) = \psi - \psi_0 + \phi \log(h)$
*   Utility: $U(w, \alpha) = a_0 + a_1 w - c_0 (1 - \alpha)^\chi$
    *   Note: Utility increases as $\alpha$ increases because the negative term $-c_0(1-\alpha)^\chi$ becomes less negative (assuming $c_0, \chi > 0$).

**2. Checking Model Assumptions:**
Let's verify the assumptions stated in the prompt using these functional forms.
*   **Assumption: $A'(h) > 0$**$$A'(h) = \frac{d}{dh}(A_0 + A_1 h) = A_1.$$
	* *Condition:* This assumption holds if $A_1 > 0$.

*   **Assumption: $\frac{\partial g(\psi, h)}{\partial h} \ge 0$**$$\frac{\partial g}{\partial h} = \frac{\partial}{\partial h}(\psi - \psi_0 + \phi \log(h)) = \frac{\phi}{h}.$$
	* *Condition:* For $h>0$, this assumption holds if $\phi \ge 0$.

*   **Assumption: $u_w(w,\alpha) = a_1> 0$**
*   **Assumption: $u_\alpha(w,\alpha) > 0$**$$u_\alpha = \frac{\partial U}{\partial \alpha} = - c_0 \chi (1 - \alpha)^{\chi - 1} (-1) = c_0 \chi (1 - \alpha)^{\chi - 1}.$$
	* *Condition:*  For $\alpha \in [0, 1)$, this holds if $c_0 \chi > 0$, if $c_0>0$ then $\chi > 0.$

*   **Assumption:** $w \mapsto u(w,\alpha)$ strictly increasing
	* This is guaranteed if $u_w > 0$, which holds if $a_1 > 0$.

*   **Assumption:** Marginal trade-off ratio $M(\alpha) = -\dfrac{u_\alpha(w,\alpha)}{u_w(w,\alpha)}$ $\uparrow\alpha$.
	* First, let's find the wage $w = \omega(x, \alpha)$ that achieves utility $x$:$$x = (a_0 + a_1 w) - c_0 (1 - \alpha)^\chi \quad \implies \quad w(\alpha) =  \frac{x - a_0 + c_0 (1 - \alpha)^\chi}{a_1}$$
		* Notice that no-negative wages only if $x - a_0 + c_0(1-\alpha)^\chi \geq 0$.
	* Next we compute the trade-off ratio:$$M(\alpha) = -\frac{u_\alpha}{u_w} = -\frac{c_0 \chi (1-\alpha)^{\chi-1}}{a_1}$$
	    - **Derivative of $M(\alpha)$:**$$M'(\alpha) = \frac{c_0 \chi (\chi-1)}{a_1} (1-\alpha)^{\chi-2}$$
		* *Condition:* For $M'(\alpha) > 0$, we require $(\chi-1) > 0$. This means $\chi > 1$.
**3. Interior Solution (First-Order Condition):**
Okay, let's derive the interior solution $\alpha^*$ by explicitly setting up the firm's optimization problem and performing the substitution of the wage function $w(\alpha)$ derived from the utility constraint.

**1. The Firm's Optimization Problem**

The firm chooses the remote work share $\alpha \in [0, 1]$ to maximize its profit, which is the output produced minus the wage paid to the worker. The firm must ensure the worker receives a utility level of exactly $x$.
*   **Output:** The production function depends on skill $h$, remote productivity $\psi$, and the remote work share $\alpha$. On-site productivity is normalized to 1.$$Y(\alpha, \psi, h) = A(h) [\alpha g(\psi, h) + (1-\alpha) \cdot 1]$$Substituting the functional forms for $A(h)$ and $g(\psi, h)$:$$Y(\alpha, \psi, h) = (A_0 + A_1 h) [1 + \alpha (\psi - \psi_0 + \phi \log(h) - 1)]$$
*   **Wage Constraint:** The wage $w$ must be set such that the worker's utility $U(w, \alpha)$ equals the target level $x$.
*   **Profit Function:** $\Pi(\alpha) = Y(\alpha, \psi, h) - w(\alpha)$
**2. Finding the Interior Solution via First-Order Condition (FOC)**
To find the optimal interior solution $\alpha^* \in (0, 1)$, we differentiate the profit function $\Pi$ with respect to $\alpha$ and set the derivative equal to zero.
$$\frac{d\Pi}{d\alpha} = \frac{d}{d\alpha} [ \text{Output Term} ] - \frac{d}{d\alpha} [ \text{Wage Term} ]$$
* **Derivative of Output Term:** $$\frac{d}{d\alpha}Y(\alpha, \psi, h)=\frac{d}{d\alpha}\left(A(h) [\alpha g(\psi, h) + (1-\alpha)]\right) = A(h)(g(\psi, h) -1)$$
* **Derivative of Wage Term:** 
$$\frac{d}{d\alpha} w(\alpha) = \frac{d}{d\alpha} \left(\frac{x - a_0 + c_0 (1 - \alpha)^\chi}{a_1}\right)=-\frac{c_0\chi}{a_1}(1-\alpha)^{\chi-1}$$
* **Setting the FOC:** $\frac{d\Pi}{d\alpha} = 0$ $$ A(h)(g(\psi, h) - 1) = -\frac{c_0\chi}{a_1}(1-\alpha)^{\chi-1} $$thus $$\alpha^*=1-\left[\frac{a_1 A(h) (g(\psi, h)-1)}{c_0 \chi}\right]^{\frac{1}{\chi-1}}$$
**Solving for $\alpha^*$ :**$$ (1-\alpha)^{\chi-1} = \frac{a_1 A(h) (1 - g(\psi, h))}{c_0 \chi} $$Taking both sides to the power of $1/(\chi-1)$ (note $\chi-1 > 0$):  $$ \alpha^* = 1 - \left[\frac{a_1 A(h) (1 - g(\psi, h))}{c_0 \chi}\right]^{\frac{1}{\chi-1}} $$
*Condition for Interior Solution:* For $\alpha^*$ to be in $(0, 1)$, we need $0 < \alpha^* < 1$. 
- $\alpha^* < 1$ requires the term being subtracted to be positive: $\left[\frac{a_1 A(h) (1 - g(\psi, h))}{c_0 \chi}\right]^{\frac{1}{\chi-1}} > 0$. Since $a_1, A(h), c_0, \chi$ are positive, this requires $1 - g(\psi, h) > 0$, or $g(\psi, h) < 1$.
- $\alpha^* > 0$ requires $1 > \left[\frac{a_1 A(h) (1 - g(\psi, h))}{c_0 \chi}\right]^{\frac{1}{\chi-1}}$. This means $\frac{a_1 A(h) (1 - g(\psi, h))}{c_0 \chi} < 1^{\chi-1} = 1$.
So, an interior solution exists only if $g(\psi, h) < 1$ and $$a_1 A(h) (1 - g(\psi, h)) < c_0 \chi.$$
The optimal interior remote work share $\alpha^*(\psi, h, x)$ is given by:    $$ \alpha^*(\psi, h, x) = 1 - \left[\frac{a_1 (A_0 + A_1 h) (1 - (\psi - \psi_0 + \phi \log(h)))}{c_0 \chi}\right]^{\frac{1}{\chi-1}} $$
This solution is valid when $0 < \alpha^* < 1$.

**Interpretation of FOC:**

The FOC $A(h)(g(\psi, h) - 1) = -\frac{c_0\chi}{a_1}(1-\alpha)^{\chi-1}$ equates the marginal change in output from increasing $\alpha$ (left side) to the marginal change in wage cost (right side).
- $A(h)(g-1)$ is the marginal output change. Since $g<1$ for an interior solution, this is negative (marginal output loss).
*   $-\frac{c_0\chi}{a_1}(1-\alpha)^{\chi-1}$ is the marginal wage change ($dw/d\alpha$). Since $c_0, \chi, a_1 > 0$ and $\chi>1$, this term is negative. Increasing $\alpha$ allows the firm to pay a lower wage while maintaining utility $x$.
*   The FOC states that at the optimum, the marginal loss in output from increasing remote work must exactly equal the marginal savings in wage cost.

**4. Thresholds $\underline{\psi}(h)$ and $\overline{\psi}(h)$:**
For any $h$ consider when the optimal solution is in the $(0,1)$ interval:
1. $\alpha^*(\psi, h, x) \geq 0$ if $$1 - \left[\frac{a_1 (A_0 + A_1 h) (1 - (\psi - \psi_0 + \phi \log(h)))}{c_0 \chi}\right]^{\frac{1}{\chi-1}}\geq 0 $$Solve for $\psi$:
$$ \boxed{\psi \ge 1 + \psi_0 - \phi \log(h) - \frac{c_0 \chi}{a_1 (A_0 + A_1 h)}} $$
2. $\alpha^*(\psi, h, x) \le 1$ if $$ \boxed{\psi \le 1 + \psi_0 - \phi \log(h)} $$
Comparing with the theoretical solution. The defining equations are:
$$ g(\underline{\psi}(h),h) = 1 + \frac{M(0)}{A(h)} $$
$$ g(\overline{\psi}(h),h) = 1 + \frac{M(1)}{A(h)} $$
Let's calculate $M(0)$ and $M(1)$:
$$M(0) = -\frac{c_0 \chi (1-0)^{\chi-1}}{a_1}=-\frac{c_0 \chi}{a_1}$$
$$M(1) = -\frac{c_0 \chi (1-1)^{\chi-1}}{a_1} = 0$$this implies that:$$ g(\underline{\psi}(h),h) = 1 - \frac{c_0 \chi}{a_1 A(h)} \quad \implies \quad \underline{\psi}(h) = 1 + \psi_0 - \psi\log(h)- \frac{c_0 \chi}{a_1 A(h)}$$and
$$ g(\overline{\psi}(h),h) = 1 \quad \implies \quad \underline{\psi}(h) = 1 + \psi_0 - \psi\log(h)$$

---
Conditions determining the monotonicity of the thresholds $\overline{\psi}(h)$ and $\underline{\psi}(h)$ with respect to $h$, under the assumptions:
$h>0$, $A_1>0$, $a_1>0$, $\chi>1$, $c_0>0$, $A_0 \ge 0$, $\phi \ge 0$.

**1. Upper Threshold $\overline{\psi}(h) = 1 + \psi_0 - \phi \log(h)$**

*   **If $\phi = 0$**: $\overline{\psi}(h)$ is **constant**.
*   **If $\phi > 0$**: $\overline{\psi}(h)$ is **strictly decreasing** for all $h>0$.

**2. Lower Threshold $\underline{\psi}(h) = 1 + \psi_0 - \phi \log(h) - \frac{c_0 \chi}{a_1 (A_0 + A_1 h)}$**

*   **If $\phi = 0$**: $\underline{\psi}(h)$ is **strictly increasing** for all $h>0$.
*   **If $\phi > 0$**:
    *   **Case (i): $A_0 = 0$**
        *   $\underline{\psi}(h)$ is **strictly increasing** on $(0, h_2)$.
        *   $\underline{\psi}(h)$ is **strictly decreasing** on $(h_2, \infty)$.
        *   (Local maximum at $h_2 = \frac{c_0 \chi}{\phi a_1 A_1}$)
    *   **Case (ii): $A_0 > 0$**
        *   **Subcase (a): $c_0 \chi \le 4 \phi a_1 A_0$**
            *   $\underline{\psi}(h)$ is **strictly decreasing** for all $h>0$.
            *   (If $c_0 \chi = 4 \phi a_1 A_0$, there is a stationary inflection point at $h = A_0/A_1$).
        *   **Subcase (b): $c_0 \chi > 4 \phi a_1 A_0$**
            *   $\underline{\psi}(h)$ is **strictly decreasing** on $(0, h_1)$.
            *   $\underline{\psi}(h)$ is **strictly increasing** on $(h_1, h_2)$.
            *   $\underline{\psi}(h)$ is **strictly decreasing** on $(h_2, \infty)$.
            *   (Local minimum at $h_1$, local maximum at $h_2$, where $h_{1,2} = \frac{(c_0 \chi - 2 \phi a_1 A_0) \pm \sqrt{c_0 \chi ( c_0 \chi - 4 \phi a_1 A_0 )}}{2 \phi a_1 A_1}$ are the two positive roots).