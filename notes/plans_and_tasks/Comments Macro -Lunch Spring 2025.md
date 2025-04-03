# Ken West

1.       You said conventional wisdom is that WFH wages are higher on average because occupations where it is possible to work from home have, on average, higher wages.  And your Tables 1 and 2 say that even controlling for teleworkability (Table 1) or with occupation dummies (Table 2), WFH wages are still higher.  
	1. Are you the first to introduce occupation dummies into a WFH wage regression?  
	2.  If my understanding is correct, you should have a slide somewhere that says “Therefore, paper X and paper Y are wrong when they speculate that high WFH wages solely capture the teleworkability of high-wage occupations.”
	3. Well, maybe that’s too strong since Carter and others did not seem to agree that you had properly controlled for the occupation effect.  I didn’t completely get their objections.  But once you account for those objections, if indeed you are the first to rule out that the WFH wage premium merely reflects relative teleworkability, then you should emphasize that.

2.       The audience seemed to like the explanation that WFH employees are more productive.  And now that you presented the model, that seems to be your explanation.  But maybe I’m confused because you did not seem eager to endorse that explanation.  

3.       The model: my impression is that it has changed in a major way.  As I understand it, you are no longer talking about firms paying higher wages to fill vacancies faster.  Instead, you have a mechanism in which higher ability are disproportionately likely to WFH.  This is the primitive assumption about $g(h)$ made on slide 12.  I think this is an improvement, and I guess that’s reasonable but ...

4.       Even under these assumptions, is it inevitable that WFH employees get paid more?  For parameters different from the ones you chose, is there an equilibrium in which
	1. person **A** would be paid more than person **B** if they both came to the office (i.e.,  $h$ is higher for **A** than **B**), but
	2. person **A** accepts a lower wage because person **A** likes to work at home.  I guess that’s a question about the calibration of  $c$ and $\chi$.

5.       Stepping outside your paper, I am surprised that it is a general finding that WFH pays more, given the amenity value of WFH.  My intuition–which, needless to say, is not to be trusted–is that in real life there are many WFH people who have accepted a slight wage cut in exchange for being allowed to work from home.  This would perhaps follow from heterogeneity in the amenity value of WFH rather than heterogeneity in productivity.  I’m not recommending modeling this, since the data you have say WFH gets paid more.  Just noting my intuition about my uninformed speculation vs. your data and model.

# Comments From Faculty

## Slide 2:
- **Carter:** Your discussion on point 2 should clarify that while some papers focus on occupations, your analysis addresses a wage premium tied to the ability to work from home. Emphasize that it’s not just about who works from home, but about the premium for having the capacity to do so. In this context, highlight the limitations of existing measures like the Dingel-Neiman index—which uses a binary (0 or 1) classification—and explain what important nuances might be missed by relying solely on such a measure.
- **Ken:** Clarify your unique contribution. Are you the first to observe that people working from home earn more, or is your primary contribution the innovative way you model this relationship? Your model, after all, stands as a significant contribution in itself.
- **Rishabh:** Note that the interpretation of the wage premium and your findings may depend heavily on the model specification you use.
- **Giselle:** Keep the details on the teleworkability measure to a minimum, focusing on the main points rather than getting bogged down in excessive technical detail.

## Slide 3:
- **Rishabh:** Questions how the ACS (or similar surveys) phrase the work-from-home query. Specifically, does it ask if a person ever works from home during the week or only capture if they worked from home on the interview day? This raises the issue of whether the question is set up to capture hybrid work arrangements—where employees split time between home and a commute.
- **Rasmus:** Inquires about the survey question regarding means of transportation. He asks if the question is conditional on commuting and, if so, whether it details the specific modes of transportation used.
- **Carter:** Seeks clarification on the source of the wage premium effect—whether it stems directly from working from home or from working in teleworkable occupations. This is important for understanding the underlying story of the wage premium.
- **Giselle:** Emphasizes that the current measure likely fails to capture hybrid work arrangements and suggests checking if any survey includes a third category to indicate hybrid working.

# Slide 5 - 7:
- **Simplify the Technical Explanation:**
    - Clarify the construction of your teleworkability index by providing a simple, concrete example or diagram. Several comments noted that the current explanation is too technical and that a more accessible illustration would help.
- **Rationale for Using Machine Learning:**
    - Explain why machine learning is used in your approach—specifically, how it helps predict which tasks determine teleworkability. However, keep the discussion high-level without delving too deeply into methodological details.
- **Choice of Data and Comparison to Existing Measures:**
    - Address why you don’t simply use BLS data. Since BLS measures are not binary (0 or 1) and may not capture the full range of occupational heterogeneity, clarify the limitations of existing measures.
    - Consider highlighting a comparative analysis between your index and the Dingel-Neiman (DN) index, discussing the advantages and potential shortcomings of each approach.
- **Accounting for Intra-Occupational Heterogeneity:**
    - Acknowledge that significant heterogeneity exists within occupations. Explain how your approach accounts for differences in individual workers’ likelihood of working from home—recognizing that workers may have very different remote work capabilities even within the same occupation.
- **Overall Presentation Focus:**
    - Reassess if this segment of your presentation is too technical for the audience. Rather than overwhelming the audience with technical details, emphasize the key insights and comparative advantages of your index over existing measures.

## Slide 9:
- **Carter:**
    - Explain how Stylized Fact II advances your narrative, especially given that Stylized Fact I already demonstrates that wages are positively correlated with both working from home and the feasibility of telework.
    - Clarify whether your focus is on the wage premium from teleworkability or on a broader story of telework potential.
    - Emphasize including occupation intensity (e.g., cognitive skills) to ensure your teleworkability estimates are not upward biased by inherently "smarter" workers being more likely to work from home.
    - For Stylized Fact III, adjust your presentation to plot teleworkability multiplied by year instead of simply using a binary WFH measure.
- **Rishabh:**
    - If you use a binary $\{0,1\}$ teleworkability index, address whether this might lead to an insignificant coefficient.
    - Demonstrate that the effect you capture is not just a proxy for a skill premium by properly controlling for worker skill levels.
- **Rasmus:**
    - Consider whether you have adequate data on worker heterogeneity—including unobservable factors—which could affect your conclusions about amenities and the broader relationship.
    - Ensure that your regression specifications, such as including occupation fixed effects, effectively control for high-skill intensity within occupations

## Slide 10:##
- **Teleworkability Interaction & Time Trends:**
    - **Carter** suggests interacting teleworkability with year to capture time trends—clarify whether you assume the underlying effect is constant or evolving over time.
- **Statistical Significance & Compensation Effects:**
    - **Rishabh** raises the concern that if higher-skilled (or “smart”) workers receive compensating benefits (e.g., image, additional perks), the teleworkability index might not show a significant coefficient. This implies that empirically, rejecting the null claim (i.e., that teleworkability has no effect) may be challenging.
- **Worker Heterogeneity:**
    - **Rasmus** emphasizes the importance of distinguishing between workers by comparing observable characteristics across individuals. He notes that while your model may capture differences in compensation linked to observable traits, it might not fully address unobservable heterogeneity that could influence the results.
- **Contextualizing with Conventional Wisdom:**
    - **Carter** also points out that previous research has not made within-occupation comparisons (e.g., comparing computer programmers who work remotely versus those who work on-site). Position your analysis as a novel approach that fills this gap.
    
## Slides 12–16:
- **Uncertainty Parameters & Functional Form:**
    - **Rasmus** questions the significance of the uncertainty in ψ and, on Slide 14, why there is uncertainty over $\psi$. Clarify that your model adopts a specific functional form for these uncertainty parameters and discuss the implications of this choice.
- **Utility Convexity & Disutility of Work:**
    - **Rishabh** advises you to address utility convexity in your framework.
    - **Rasmus** inquires how the disutility of going to work interacts with h and later, how it interacts with a worker's skill (Slide 16). He emphasizes that only the joint effect might matter and asks if you can separately identify the contributions from firms and workers. Explain how these interactions are modeled and whether separate identification is feasible.
- **Separating Ability from Amenity:**
    - **Agustin** asks how your model distinguishes between a worker’s inherent ability and the amenity (or benefit) of working from home. Provide insight into the assumptions or methods you use to disentangle these factors.
- **Simplifying the Model Explanation:**
    - **Rasmus** also suggests offering a more naive or simplified version of your model to make the underlying concepts clearer. Consider including an intuitive explanation that highlights the core mechanics without the full technical detail.

## Slide 21:
- **Agustin:** Clarify the model element that attributes a wage premium to being in a highly teleworkable occupation—even for workers who do not actually work from home. Explain whether this premium arises from inherent occupational characteristics (e.g., productivity or cognitive skill requirements) or other factors that signal higher value in the labor market.