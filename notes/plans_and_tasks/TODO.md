# Empiric
- [ ] Explore additional controls for skill and occupation groups. 🆔 K76cwI
  - [x] Read [[2020 - Jeremy Lise, Fabien Postel-Vinay - Multidimensional Skills, Sorting, and Human Capital Accumulation||2020 - Jeremy Lise, Fabien Postel-Vinay]] with a focus on Section 4.1 to understand their aggregation of multidimensional skills by occupation. ⏫ 📅 2025-04-01 ✅ 2025-04-01
  - [x] List and document the steps required to implement the skill measure: 🆔 eD9GcI ⏫ ⏳ 2025-04-02 📅 2025-04-02 ✅ 2025-04-08
    - [x] Verify extraction of relevant O\*NET descriptors. 🆔 UT9C4R ✅ 2025-04-08
    - [x] Aggregate raw measures into composite scores for cognitive, manual, and interpersonal requirements. 🆔 7f0oVq ✅ 2025-04-08
    - [x] Normalize and standardize the composite scores. 🆔 D1seRD ✅ 2025-04-08
    - [x] Assemble the final occupation-specific skill requirement vector, $y = (y_C, y_M, y_I)$. 🆔 amSN1w ✅ 2025-04-08
  - [x] Identify additional control variables based on the multidimensional skills framework and assess their impact in the estimation strategy. 🆔 ze1u3z ⏫ ⏳ 2025-04-13 📅 2025-04-13 ✅ 2025-04-20
- [ ] Try to approximate the hybrid work shares using demographics and aggregates. 🆔 EAHI3W
	- [x] 23:15 - 23:30 Read [[2020 - Paul Goldsmith-Pinkham, Isaac Sorkin, Henry Swift - Bartik Instruments What, When, Why, and How|2020 - Paul Goldsmith-Pinkham, Isaac Sorkin, Henry Swift]] to understand a strategy similar to Bartik instruments. ⏫ 📅 2025-04-01 ✅ 2025-04-01
  - [ ] Investigate which demographic and aggregate measures could proxy for hybrid work shares. 🆔 5jlbi5 ⏫ ⏳ 2025-04-13 📅 2025-04-13
  - [ ] Develop a preliminary plan to integrate these proxies into the empirical analysis. 🆔 ucdavd ⏫ ⏳ 2025-04-13 📅 2025-04-13
# Model
- [x] Write down the [[Model.qmd|model note]]. 🆔 137KEa ⏫ ⏳ 2025-04-12 📅 2025-04-16 ✅ 2025-04-20
      [clock::2025-04-08T13:20:42--2025-04-08T17:28:46]
	- [x] Write the search section of the model. ⏳ 2025-04-09 📅 2025-04-09 ✅ 2025-04-09
		      [clock::2025-04-09T09:20:07--2025-04-09T11:00:30]
	- [ ] Try to identify what is missing in the model note (take a look at literature) 🆔 syumfl 🔼 ⏳ 2025-04-12 📅 2025-04-12
	- [x] Write an illustrative example with concrete functional forms to create counterparts to the notes [[Optimal Remote Policy (General Functional Forms)]] and [[Proposition (General Functional Forms)]] this should be a note called [[Optimal Remote Policy (Particular Functional Forms)]] with closed form solutions and graphs and [[Proposition (Particular Functional Forms)]] with particular examples of the proposition results and how some of the assumptions fails for some parameters. 🆔 eZTesQ ✅ 2025-04-11
	      [clock::2025-04-11T18:15:17--2025-04-12T00:30:45]
- [x] Proofs with general functions of propositions. 🆔 45hTeU ✅ 2025-04-09
  - [x] List main propositions to be proven. 🆔 Pplas5 ✅ 2025-04-09
  - [x] Outline the proof strategy for each proposition. 🆔 odSwJl ✅ 2025-04-09
# Manuscript
- [x] Start with outline. 🆔 391R1e ⏳ 2025-04-13 📅 2025-04-16 ✅ 2025-04-20
  - [x] Organize main sections and subsections. 🆔 kY7HTq ✅ 2025-04-20

# Estimation
- [x] Find paper that Carter told me and read it (I will then recall what was the name of the paper in question). 📅 2025-04-01 ✅ 2025-04-01
- [ ] Read [[2022 - Stéphane Bonhomme, Thibaut Lamadon, Elena Manresa - Discretizing Unobserved Heterogeneity||2022 - Stéphane Bonhomme, Thibaut Lamadon, Elena Manresa]] 🆔 51lkJg ⏫ ⏳ 2025-04-13 📅 2025-04-13
- [x] Write down an outline of an estimation strategy. 🆔 Wh3mx1 ⏳ 2025-04-12 📅 2025-04-12 ✅ 2025-04-20
- [x] Make sure that the model code and simulation delivers objects in the right format for the estimation. 🆔 3j7Jq1 ⏳ 2025-04-09 📅 2025-04-09 ✅ 2025-04-10
      [clock::2025-04-09T15:46:38--2025-04-09T17:00:28]
- [x] Write a `simulation.jl` module for my model. 🆔 jzyzyc 🔺 ⏳ 2025-04-10 📅 2025-04-10 ✅ 2025-04-11
	- [x] Pre-generate all stochastic components/shocks for the entire simulation ✅ 2025-04-10
	      [clock::2025-04-10T13:09:45--2025-04-10T15:21:04]
		- [x] Worker productivity shocks ✅ 2025-04-10
		- [x] Matching shocks ✅ 2025-04-10
		- [x] Any other model-specific random components ✅ 2025-04-10
		- [x] Store these shocks for reuse across parameter iterations ✅ 2025-04-10
	- [x] Initialize data structures to track $N$ workers over $T$ periods 🆔 lp7B5W ✅ 2025-04-20
		- [x] Set up containers for collecting moments for **SMM** estimation 🆔 Ccjrwc ✅ 2025-04-20
	- [x] Implement block recursive equilibrium simulation loop: ✅ 2025-04-10
	      [clock::2025-04-10T15:20:56--2025-04-10T18:26:35]
		- [x] Update worker states using pre-generated shocks ✅ 2025-04-10
		- [x] Record relevant state variables and transitions ✅ 2025-04-10
- [x] Work on speeding up/parallelizing the simulation and/or solution of the model 🆔 ic76cb ⛔ jzyzyc ⏫ ⏳ 2025-04-10 📅 2025-04-10 ✅ 2025-04-11
- [x] Write an `estimation.jl` module for my model. 🆔 28snw7 ⛔ jzyzyc 🔺 ⏳ 2025-04-12 📅 2025-04-16 ✅ 2025-04-20
      [clock::2025-04-11T14:40:53--2025-04-11T16:59:59]
- [x] Fix the selection of $\delta(\psi, x, h)$ in the simulation. Not important right now since we keep it fixed for now. 🆔 83ZlHJ 🔽 ✅ 2025-04-20
- [x] Check the distribution of worker type and how it is feed into the model. 🆔 6fu3yo 🔺 ⏳ 2025-04-11 📅 2025-04-11 ✅ 2025-04-11
      [clock::2025-04-11T10:12:12--2025-04-11T11:46:58]
- [x] Revise calibration and identify potential improvements. 📅 2025-04-11 ⏳ 2025-04-11 🔺🆔 usxg9s q1 ✅ 2025-04-20