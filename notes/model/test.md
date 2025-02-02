# Optimal Work-from-Home Model with Non-Linear Transition

This note details a model where firms choose the optimal fraction of remote work (\(\alpha^*\)) based on their remote efficiency (\(\psi\)), incorporating **non-linear transitions** in \(\alpha^*\) as \(\psi\) increases.

---

## **Model Components**

### **1. Production Function**
The firm’s productivity depends on:
- **On-site work**: \(1 - \alpha\)
- **Remote work**: \(\alpha(\psi - \psi_0)^\gamma\), where:
  - \(\psi \in [0, 1]\): Firm’s remote efficiency.
  - \(\psi_0\): Threshold efficiency below which remote work is unproductive.
  - \(\gamma > 0\): Curvature parameter for non-linearity.

Total productivity:  
\[
\text{Production} = A\left[(1 - \alpha) + \alpha(\psi - \psi_0)^\gamma\right]
\]

### **2. Worker Utility and Wage**
Workers incur a disutility from On-site work, compensated by wage \(w\):  
\[
w = x + c(1 - \alpha)^\eta
\]
- \(x\): Baseline utility guaranteed by the firm.
- \(c > 0\): Disutility scaling factor.
- \(\eta > 1\): Curvature parameter for On-site work disutility.

### **3. Profit Function**
Profit = Productivity - Wage:  
\[
\text{Profit} = A\left[(1 - \alpha) + \alpha(\psi - \psi_0)^\gamma\right] - \left[x + c(1 - \alpha)^\eta\right]
\]

---

## **Optimal \(\alpha^*\) Derivation**

### **First-Order Condition (FOC)**
Maximize profit by differentiating with respect to \(\alpha\):  
\[
\frac{d\text{Profit}}{d\alpha} = A\left[(\psi - \psi_0)^\gamma - 1\right] + c\eta(1 - \alpha)^{\eta - 1} = 0
\]

### **Solution**
Solving the FOC gives \(\alpha^*\):  
\[
\alpha^* = 1 - \left[\frac{A\left(1 - (\psi - \psi_0)^\gamma\right)}{c\eta}\right]^{\frac{1}{\eta - 1}}
\]

---

## **Key Properties**

### **1. Corner Solutions**
- **Low \(\psi\)**: If \(\psi \leq \psi_0 + \left(\frac{c\eta}{A}\right)^{\frac{1}{\gamma}}\), then \(\alpha^* = 0\).  
  *Rationale*: Remote work is too inefficient to justify wage savings.
- **High \(\psi\)**: If \(\psi \geq \psi_0 + 1\), then \(\alpha^* = 1\).  
  *Rationale*: Remote work is more productive than On-site work.

### **2. Non-Linear Transition**
For intermediate \(\psi\):  
\[
\alpha^* \propto \left[1 - (\psi - \psi_0)^\gamma\right]^{\frac{1}{\eta - 1}}
\]
- **Convex transition** (\(\alpha^*\) rises rapidly) if \(\gamma > 1\) or \(\eta < 2\).
- **Concave transition** (\(\alpha^*\) rises slowly) if \(\gamma < 1\) or \(\eta > 2\).

---

## **Interpretation**

### **Parameter Roles**
| Parameter | Role |
|-----------|------|
| \(\gamma\) | Controls how sharply remote productivity rises with \(\psi\). |
| \(\eta\) | Controls how sharply wage costs rise with On-site work. |
| \(A\) | Scales the productivity of remote work relative to wages. |
| \(c\) | Determines the cost of compensating On-site work disutility. |

### **Intuition**
- Firms with low \(\psi\) (\(\leq \psi_0 + \text{threshold}\)) choose full On-site work (\(\alpha^* = 0\)) to avoid productivity losses.
- As \(\psi\) crosses a threshold, firms adopt hybrid work, with \(\alpha^*\) increasing **non-linearly** depending on \(\gamma\) and \(\eta\).
- High-\(\psi\) firms (\(\geq \psi_0 + 1\)) go fully remote (\(\alpha^* = 1\)).

---

## **Example Cases**

### **Case 1: S-Shaped Adoption (\(\gamma = 2, \eta = 3\))**
\[
\alpha^* = 1 - \left[\frac{A(1 - (\psi - \psi_0)^2)}{3c}\right]^{0.5}
\]
- Slow adoption at low \(\psi\), rapid near mid-\(\psi\), then slows again.

### **Case 2: Concave Adoption (\(\gamma = 0.5, \eta = 3\))**
\[
\alpha^* = 1 - \left[\frac{A(1 - \sqrt{\psi - \psi_0})}{3c}\right]^{0.5}
\]
- Gradual increase in \(\alpha^*\) across \(\psi\).

---

## **Final Answer**
The optimal remote work fraction is:  
\[
\boxed{
\alpha^* = 
\begin{cases} 
0 & \text{if } \psi \leq \psi_0 + \left(\frac{c\eta}{A}\right)^{\frac{1}{\gamma}}, \\
1 - \left[\frac{A\left(1 - (\psi - \psi_0)^\gamma\right)}{c\eta}\right]^{\frac{1}{\eta - 1}} & \text{otherwise}, \\
1 & \text{if } \psi \geq \psi_0 + 1.
\end{cases}
}
\]