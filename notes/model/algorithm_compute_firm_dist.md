
> [!quote]+ **Algorithm** Equilibrium Vacancy Filling Rates and Distributions Computation (Probabilistic Choice)
>
> > [!inputs]- **Inputs:**
> > *   Grids for worker types $h \in \mathcal{H}$, firm types $\psi \in \Psi_{grid}$, utility levels $x \in \mathcal{X}$.
> > *   Pre-computed match value function $J(\psi, h, x)$ for all $(\psi, h, x)$ combinations.
> > *   Underlying firm type density $f(\psi)$ (or probability mass $p(\psi)$ if discrete).
> > *   Vacancy posting cost function $\kappa(x)$.
> > *   Vacancy filling rate function $q(\theta)$ **and its inverse** $q^{-1}(q)$.
> > *   Probabilistic choice sensitivity parameter $\xi$.
> > *   Convergence tolerance $\epsilon$.
> > *   Maximum number of iterations $N_{\max}$.
> >
>
> > [!outputs]- **Outputs:**
> > *   Equilibrium vacancy filling rate array $q^*(h, x)$.
> > *   Equilibrium market tightness array $\theta^*(h, x)$ (computed from $q^*$).
> > *   Equilibrium choice probability array $P^*(x | \psi, h)$.
> > *   Conditional firm PDF $f^*(\psi | h, x)$.
> > *   Conditional firm CDF $F^*(\psi | h, x)$.
>
> > [!algorithm]- **Algorithm:**
> > 1.  **Initialization:**
> >     *   Initialize vacancy filling rate array: $q(h, x) \leftarrow q_{init}$ (e.g., 0.1) for all $h, x$.
> >     *   Initialize iteration counter: $n \leftarrow 0$.
> >     *   Initialize convergence flag: *converged* $\leftarrow$ *False*.
> >     *   Initialize adaptive damping parameter $\lambda \leftarrow \lambda_{initial}$.
> >     *   Initialize other damping state variables (`dist_min_so_far`, `dist_previous`).
> >     *   Pre-calculate cost vector $\kappa_x = [\kappa(x_1), \kappa(x_2), ...]$.
> > 2.  **Iteration Loop:**
> >     *   **While** *not converged* **and** $n < N_{\max}$:
> >         *   $n \leftarrow n + 1$.
> >         *   *Store previous rates:* $q_{old} \leftarrow q$.
> >         *   *Initialize probability array:* $P_{choice}( \psi, h, x) \leftarrow 0.0$.
> >         *   *Initialize target rate array:* $q_{new}(h, x) \leftarrow 0.0$.
> >         *   **(Step 1: Calculate Choice Probabilities)**
> >             *   **For each** firm type $\psi \in \Psi_{grid}$:
> >                 *   Calculate potential profit matrix for firm $\psi$: $\Pi_{post}(\psi, h', x') \leftarrow q_{old}(h', x') J(\psi, h', x') - \kappa(x')$ for all $h', x'$.
> >                 *   **For each** worker type $h \in \mathcal{H}$:
> >                     *   Find max profit for this $(\psi, h)$: $\Pi_{max}(\psi, h) \leftarrow \max_{x'} \{ \Pi_{post}(\psi, h, x') \}$.
> >                     *   **If** $\Pi_{max}(\psi, h) \ge -\epsilon$:
> >                         *   Calculate exponentiated terms (stable): $E_{x'} \leftarrow \exp(\xi \cdot (\Pi_{post}(\psi, h, x') - \Pi_{max}(\psi, h)))$.
> >                         *   Calculate sum: $S \leftarrow \sum_{x''} E_{x''}$.
> >                         *   **If** $S > \epsilon$:
> >                             *   Calculate probabilities: $P_{choice}(\psi, h, x') \leftarrow E_{x'} / S$ for all $x'$.
> >         *   **(Step 2: Update Target `q_new`)**
> >             *   **For each** worker type $h \in \mathcal{H}$:
> >                 *   **For each** utility level $x \in \mathcal{X}$:
> >                     *   Calculate total mass choosing $(h,x)$: $M \leftarrow \sum_{\psi'} P_{choice}(\psi', h, x) f(\psi')$.
> >                     *   **If** $M > \epsilon$:
> >                         *   Calculate expected $J$ numerator: $N \leftarrow \sum_{\psi'} J(\psi', h, x) P_{choice}(\psi', h, x) f(\psi')$.
> >                         *   Calculate conditional expectation: $\mathbb{E}[J|h,x] \leftarrow N / M$.
> >                         *   Check profitability: **If** $\mathbb{E}[J|h,x] > \kappa(x) + \epsilon$:
> >                             *   Calculate target rate: $q_{target} \leftarrow \kappa(x) / \mathbb{E}[J|h,x]$.
> >                             *   $q_{new}(h, x) \leftarrow \text{clamp}(q_{target}, 0, 1)$.
> >                         *   **Else:** $q_{new}(h, x) \leftarrow 0$.
> >                     *   **Else:** $q_{new}(h, x) \leftarrow 0$.
> >         *   **(Step 3: Check Convergence)**
> >             *   Calculate distance: $d \leftarrow \text{distance}(q_{new}, q_{old})$.
> >             *   **If** $d < \epsilon$: *converged* $\leftarrow$ *True*.
> >
> >         *   **(Step 5: Update `q`)** $q \leftarrow (1-\lambda)q_{old} + \lambda q_{new}$.
> >         *   *Update `dist_previous`*.
> > 3.  **Post-Processing:**
> >     *   $q^* \leftarrow q$.
> >     *   $P^* \leftarrow P_{choice}$ (from last iteration).
> >     *   Compute $\theta^*(h,x) \leftarrow q^{-1}(q^*(h,x))$.
> >     *   Compute $f^*(\psi|h,x) \leftarrow \frac{P^*(x | \psi, h) f(\psi)}{\sum_{\psi'} P^*(x | \psi', h) f(\psi')}$.
> >     *   Compute $F^*(\psi|h,x) \leftarrow \sum_{\psi'' \le \psi} f^*(\psi'' | h, x)$.
> > 4.  **Return:** $q^*, \theta^*, P^*, f^*, F^*$.