### Equilibrium Computation Algorithm  {#sec-appendix-endogenousFirmDistribution}

This appendix details the iterative algorithm used to compute the equilibrium vacancy filling rates $q^*(h, x)$, the associated market tightnesses $\theta^*(h, x)$, the firm choice probabilities $P^*(x | \psi, h)$, and the conditional firm distributions $f^*(\psi | h, x)$ and $F^*(\psi | h, x)$ described in @sec-labor-market-search. The algorithm searches for a fixed point in the vacancy filling rates $q(h, x)$ by iterating on firm profit calculations, probabilistic choices, and the free entry condition.

- **Inputs:** Grids $\mathcal{H}, \Psi_{grid}, \mathcal{X}$; Match values $J(\psi, h, x)$; Density $f(\psi)$; Cost $\kappa(x)$; Matching functions $q(\theta), q^{-1}(q)$; Sensitivity $\xi$; Tolerance $\epsilon$; Max iterations $N_{\max}$.
- **Outputs:** Equilibrium rates $q^*(h, x), \theta^*(h, x)$; Choice probabilities $P^*(x | \psi, h)$; Conditional distributions $f^*(\psi | h, x), F^*(\psi | h, x)$.

\begin{algorithm}[H]
\caption{Equilibrium Computation (Probabilistic Choice)}
\label{alg:equilibrium_computation_concise}
\begin{algorithmic}[1]
    \State Initialize $q(h, x) \leftarrow q_{init}$, $n \leftarrow 0$, \textit{converged} $\leftarrow$ False.
    \While{\textit{not converged} \textbf{and} $n < N_{\max}$}
        \State $n \leftarrow n + 1$.
        \State $q_{old} \leftarrow q$.
        \State Initialize $P_{choice}(\psi, h, x) \leftarrow 0.0$, $q_{new}(h, x) \leftarrow 0.0$.
        \Statex \textit{// Calculate choice probabilities based on expected profit $\Pi_{post} = q_{old}J - \kappa(x)$}
        \ForAll{firm types $\psi \in \Psi_{grid}$}
            \ForAll{worker types $h \in \mathcal{H}$}
                \State Find $\Pi_{max}(\psi, h) \leftarrow \max_{x'} \{ \Pi_{post}(\psi, h, x') \}$.
                \If{$\Pi_{max}(\psi, h) \ge -\epsilon$}
                    \State Calculate $E_{x'} \leftarrow \exp(\xi \cdot (\Pi_{post}(\psi, h, x') - \Pi_{max}(\psi, h)))$ for all $x'$.
                    \State Calculate sum $S \leftarrow \sum_{x''} E_{x''}$.
                    \If{$S > \epsilon$} $P_{choice}(\psi, h, x') \leftarrow E_{x'} / S$ for all $x'$. \EndIf
                \EndIf
            \EndFor
        \EndFor
        \Statex \textit{// Update target $q_{new}$ based on free entry $q_{new} = \kappa(x) / \Exp[J|h,x]$}
        \ForAll{worker types $h \in \mathcal{H}$}
            \ForAll{utility levels $x \in \mathcal{X}$}
                \State Calculate total mass $M \leftarrow \sum_{\psi'} P_{choice}(\psi', h, x) f(\psi')$.
                \If{$M > \epsilon$}
                    \State Calculate $\Exp[J|h,x] \leftarrow (\sum_{\psi'} J(\psi', h, x) P_{choice}(\psi', h, x) f(\psi')) / M$.
                    \If{$\Exp[J|h,x] > \kappa(x) + \epsilon$}
                        \State $q_{target} \leftarrow \kappa(x) / \Exp[J|h,x]$.
                        \State $q_{new}(h, x) \leftarrow \max(0, \min(1, q_{target}))$.
                    \EndIf
                \EndIf
            \EndFor
        \EndFor
        \State Check convergence and update: $q \leftarrow \text{UpdateRule}(q_{old}, q_{new})$.
    \EndWhile
    \Statex
    \State \textbf{Post-Processing:}
    \State $q^* \leftarrow q$; $P^* \leftarrow P_{choice}$. and  $\theta^*(h,x) \leftarrow q^{-1}(q^*(h,x))$.
    \State Compute $f^*(\psi|h,x)$ and $F^*(\psi|h,x)$ using $P^*$ and $f(\psi)$.
    \Statex
    \State \Return $q^*, \theta^*, P^*, f^*, F^*$.
\end{algorithmic}
\end{algorithm}