
Before presenting our structural model, we establish key empirical patterns using individual-level data from the American Community Survey (ACS) for 2013-2022, combined with occupation-level data from O\*NET and BLS. Our sample includes civilian wage-employed individuals aged 22-70 working over 30 hours per week above the federal minimum wage. We identify remote workers based on the ACS question regarding commuting, where respondents indicate they "worked from home."

### Worker Characteristics

As shown in Table 1, remote workers differ significantly from their non-remote counterparts. Pre-pandemic (2013-2019), remote work was rare (3%), but its share increased substantially post-2020 (15%). Remote workers, on average, are similar in age but report considerably higher total income and hourly wages (both nominal and real). They are also significantly more likely to hold college (66% vs. 39%) or postgraduate (26% vs. 15%) degrees. These descriptive statistics highlight selection into remote work based on observable characteristics, particularly education and earnings potential.

\begin{table}[htpb]
\centering
\caption{Summary statistics by work arrangement}\label{tab:summary_stats}
\begin{tabular}{lcccc}
\hline
\hline
& \multicolumn{2}{c}{\textbf{Non-WFH}} & \multicolumn{2}{c}{\textbf{WFH}}\\
\cline{2-3}\cline{4-5}
& Mean & Sd & Mean & Sd \\
\hline
Share labor force 2013 - 2019 & 97\% & - & 3\% & - \\
Share labor force 2020 - 2023 & 85\% & - & 15\% & - \\
Age & 44.20 & 12.43 & 44.66& 11.88 \\
Total income & 67,536.4 & 69,200.87 & 106,556.2 & 97,919.89 \\
Hourly wage & 27.95 & 25.81 & 44.20 & 37.59 \\
Real hourly wage & 26.31 & 24.10 & 39.31 & 33.10\\
Commuting time & 26.81 & 23.50 & - & - \\
Share of College & 39\% & - & 66\% & - \\
Share of Postgraduate & 15\% & - & 26\% & - \\
\hline
\hline
Observations & \multicolumn{2}{c}{9025857} & \multicolumn{2}{c}{751654}\\
\hline
\hline
\end{tabular}
\end{table}

### Stylized Fact I: Remote Work Correlates with Higher Wages, Even Controlling for Teleworkability

A simple comparison suggests remote workers earn more. However, occupations differ in their suitability for remote work. We construct a novel, data-driven **Teleworkability Index** for occupations using machine learning on high-dimensional O\*NET data, validated against BLS data (see @sec-teleworkability for details). This index measures the feasibility of performing an occupation remotely.

Table 2 presents regressions of log real hourly wages on our Teleworkability Index and a WFH indicator. Column (1) shows a strong positive correlation between teleworkability and wages. Adding industry fixed effects (Column 2) and individual controls including detailed age-education interactions (Column 5) attenuates but does not eliminate this relationship. Crucially, when including both the Teleworkability Index and the WFH indicator (Column 6), both remain highly significant. Workers in more teleworkable occupations earn more, and *within* those occupations (controlling for teleworkability and other factors), those actually working remotely earn an additional premium (approx. 7.9%). This suggests both occupational sorting and a distinct remote work effect contribute to wage differences.

\begin{table}[htbp]
\label{tab:reg_teleworkability}
\centering
\footnotesize
\caption{Wage regressed on Teleworkability index and remote work indicator.}
\begin{tabular}{l c c c c c c}
\hline \hline
&\multicolumn{1}{c}{(1)} &\multicolumn{1}{c}{(2)} &\multicolumn{1}{c}{(3)} &\multicolumn{1}{c}{(4)} &\multicolumn{1}{c}{(5)} &\multicolumn{1}{c}{(6)} \\
\hline
Teleworkability Index & 33.58\sym{***}& 27.68\sym{***}& 20.89\sym{***}& 19.70\sym{***}& 15.36\sym{***}& 14.90\sym{***}\\
& (0.0522) & (0.0580) & (0.115) & (0.115) & (0.0569) & (0.0570) \\
WFH Indicator & & & & 6.365\sym{***}& & 3.203\sym{***}\\
& & & & (0.0506) & & (0.0487) \\
\hline
FE: Year \& Location & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ \\
FE: Industry & & $\checkmark$ & & $\checkmark$ & $\checkmark$ & $\checkmark$ \\
AgeCat $\times$ Educ & & & & & $\checkmark$ & $\checkmark$ \\
\hline
N & 9708029 & 9708029 & 9708029 & 9708029 & 9708028 & 9708028 \\
$R^2$ & 0.141 & 0.186 & 0.227 & 0.230 & 0.292 & 0.293 \\
\hline\hline
\multicolumn{7}{l}{\footnotesize Worker level data. All regressions include demographic controls: age, race, education, others.}\\
\multicolumn{7}{l}{\footnotesize Standard errors in parentheses: \sym{*} \(p<0.1\), \sym{**} \(p<0.05\), \sym{***} \(p<0.001\).}\\
\end{tabular}
\end{table}

### Stylized Fact II: Within Occupations, Remote Workers Earn More

To isolate the WFH premium from occupational characteristics like teleworkability or skill requirements, we regress log real hourly wages on the WFH indicator while including increasingly stringent fixed effects. Table 3 shows the results. Controlling for year, location, industry, demographics, and age-education interactions (Column 4) yields a WFH premium of approximately 13%. Adding detailed occupation fixed effects (Column 5), which absorbs time-invariant differences across occupations including average skill and teleworkability, reduces the premium to a still statistically and economically significant 8.8%. This confirms that even when comparing workers within the same detailed occupation, those working remotely tend to earn more.

\begin{table}[htbp]
\label{tab:reg_wfh_fe}
\centering
\footnotesize
\caption{Wage regressed on remote work indicator and controls. }
\begin{tabular}{l c c c c c}
\hline\hline
& \multicolumn{1}{c}{(1)} & \multicolumn{1}{c}{(2)} & \multicolumn{1}{c}{(3)} & \multicolumn{1}{c}{(4)} & \multicolumn{1}{c}{(5)} \\
\hline
WFH Indicator & 12.44\sym{***}& 7.702\sym{***}& 5.834\sym{***}& 5.031\sym{***}& 3.603\sym{***}\\
& (0.0530) & (0.0525) & (0.0494) & (0.0493) & (0.0471) \\
\hline
FE: Year \& Location & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ \\
FE: Industry & & $\checkmark$ & & $\checkmark$ & $\checkmark$ \\
FE: Occupation & & & $\checkmark$ & $\checkmark$ & $\checkmark$ \\
FE: Class of Worker & & & & $\checkmark$ & $\checkmark$ \\
AgeCat $\times$ Educ & & & & & $\checkmark$ \\
\hline
N & 9712293 & 9712293 & 9712293 & 9712293 & 9712292 \\
$R^2$ & 0.0711 & 0.153 & 0.289 & 0.307 & 0.364 \\
\hline\hline
\multicolumn{6}{l}{\footnotesize Worker level data. All regressions include demographic controls: age, race, education, others.}\\
\multicolumn{6}{l}{\footnotesize Standard errors in parentheses: \sym{*} \(p<0.1\), \sym{**} \(p<0.05\), \sym{***} \(p<0.001\).}\\
\end{tabular}
\end{table}

### Summary and Motivation for Model

The empirical evidence highlights several key facts: remote workers are positively selected on skill and earnings; occupations amenable to remote work command higher wages overall; a significant wage premium exists for remote workers even within detailed occupations; and this premium has evolved over time. While regressions control for observables, they cannot fully account for unobserved heterogeneity (e.g., firm productivity, worker preferences) or the general equilibrium effects of remote work adoption. These facts motivate our structural search and matching model, which incorporates heterogeneity in worker skill ($h$) and firm remote work efficiency ($\psi$) to explore the mechanisms driving these wage patterns and the sorting of workers and firms in the labor market.

