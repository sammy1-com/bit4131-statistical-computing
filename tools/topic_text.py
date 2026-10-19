"""Theory (item 1) and mathematical formulation (item 2) for each topic. Used by build_readmes.py."""
T = {
"01_central_tendency_dispersion": ("Measures of central tendency and dispersion",
 "Central tendency summarises where data is centred (mean, median, mode). Dispersion summarises how spread out it is (range, variance, standard deviation, quartiles/IQR). The mean is sensitive to outliers; the median is robust. A boxplot displays the five-number summary and flags outliers.",
 r"""- Mean: $\bar x=\frac1n\sum x_i$
- Sample variance: $s^2=\frac{1}{n-1}\sum (x_i-\bar x)^2$, standard deviation $s=\sqrt{s^2}$
- IQR $=Q_3-Q_1$; outlier fences $Q_1-1.5\,IQR$ and $Q_3+1.5\,IQR$
- Coefficient of variation $CV=s/\bar x\times100\%$"""),
"02_probability_conditional": ("Probability and conditional probability",
 "Probability measures the likelihood of an event between 0 and 1. Conditional probability is the probability of an event given that another has occurred; events are independent when knowing one does not change the other. Simulation approximates probabilities by repeating a random experiment.",
 r"""- $P(A)=\frac{|A|}{|S|}$ for equally likely outcomes
- $P(A\mid B)=\frac{P(A\cap B)}{P(B)}$
- Multiplication rule: $P(A\cap B)=P(A\mid B)P(B)$; independence: $P(A\cap B)=P(A)P(B)$
- Bayes: $P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}$"""),
"03_probability_distributions": ("Probability distributions",
 "A distribution describes the probabilities of a random variable's values. Discrete examples: Binomial, Poisson. Continuous: Normal, Exponential, Lognormal. Fitting estimates the parameters from data and compares candidate models (histogram vs fitted curve, KS test, AIC).",
 r"""- Binomial: $P(X=k)=\binom nk p^k(1-p)^{n-k}$
- Poisson: $P(X=k)=\frac{\lambda^k e^{-\lambda}}{k!}$, mean $=$ variance $=\lambda$
- Normal: $f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$
- Exponential: $f(x)=\lambda e^{-\lambda x}$, mean $1/\lambda$
- AIC $=2k-2\ln L$ (lower is better)"""),
"04_sampling_clt": ("Sampling distributions and the Central Limit Theorem",
 "A sampling distribution is the distribution of a statistic (e.g. the mean) over repeated samples. The CLT says the distribution of the sample mean approaches a Normal distribution as n grows, whatever the shape of the population (finite variance). The standard error shrinks like $1/\\sqrt n$.",
 r"""- $E[\bar X]=\mu$, $SE(\bar X)=\frac{\sigma}{\sqrt n}$
- CLT: $\frac{\bar X-\mu}{\sigma/\sqrt n}\xrightarrow{d}N(0,1)$ as $n\to\infty$"""),
"05_confidence_intervals": ("Confidence intervals",
 "A confidence interval gives a range of plausible values for a population parameter. A 95% CI means that if the sampling were repeated many times, about 95% of such intervals would contain the true value. It quantifies uncertainty around a point estimate.",
 r"""- Mean (σ unknown): $\bar x\pm t_{\alpha/2,\,n-1}\frac{s}{\sqrt n}$
- Proportion (Wilson) is preferred to Wald $\hat p\pm z\sqrt{\hat p(1-\hat p)/n}$ for small n or extreme p
- Difference in means (Welch): $(\bar x_1-\bar x_2)\pm t\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}$"""),
"06_hypothesis_testing": ("Hypothesis testing",
 "A hypothesis test weighs evidence against a null hypothesis H0. The p-value is the probability of data at least as extreme as observed if H0 were true; if p < α (usually 0.05) we reject H0. Type I error = rejecting a true H0 (probability α); Type II error = failing to reject a false H0.",
 r"""- One-sample t: $t=\frac{\bar x-\mu_0}{s/\sqrt n}$, df $=n-1$
- Welch two-sample t: $t=\frac{\bar x_1-\bar x_2}{\sqrt{s_1^2/n_1+s_2^2/n_2}}$
- Paired t: one-sample t on differences $d_i$
- One-proportion z: $z=\frac{\hat p-p_0}{\sqrt{p_0(1-p_0)/n}}$"""),
"07_correlation_covariance": ("Correlation and covariance",
 "Covariance measures how two variables vary together (sign shows direction, but units make magnitude hard to read). Pearson correlation standardises it to [-1, 1] and measures linear association; Spearman uses ranks and captures monotonic association. Correlation does not imply causation.",
 r"""- $\operatorname{cov}(X,Y)=\frac{1}{n-1}\sum (x_i-\bar x)(y_i-\bar y)$
- $r=\frac{\operatorname{cov}(X,Y)}{s_Xs_Y}$
- Test: $t=r\sqrt{\frac{n-2}{1-r^2}}$, df $=n-2$"""),
"08_linear_regression": ("Linear regression and least squares",
 "Linear regression models a response as a linear function of predictors. Ordinary least squares chooses the coefficients that minimise the sum of squared residuals. R² is the share of variance explained; residual plots check linearity, constant variance and normality.",
 r"""- Model: $y=\beta_0+\beta_1x_1+\dots+\beta_px_p+\varepsilon$
- Simple: $b_1=\frac{S_{xy}}{S_{xx}}$, $b_0=\bar y-b_1\bar x$
- Matrix form: $\hat\beta=(X^\top X)^{-1}X^\top y$
- $R^2=1-\frac{SSE}{SST}$"""),
"09_matrix_algebra": ("Matrix algebra in statistics",
 "Statistics is written in matrices: data tables are matrices, covariance is a matrix and regression solves a linear system. Core operations: addition, multiplication, transpose, determinant, inverse and solving $Ax=b$. A near-zero determinant or huge condition number signals an unstable system.",
 r"""- $(AB)^\top=B^\top A^\top$, $AA^{-1}=I$
- $2\times2$ determinant: $ad-bc$
- Solve $Ax=b\Rightarrow x=A^{-1}b$ (use `solve`, not an explicit inverse)
- Normal equations: $X^\top X\beta=X^\top y$"""),
"10_eigen_pca": ("Eigenvalues, eigenvectors and PCA",
 "An eigenvector of a matrix keeps its direction under the transformation and is only scaled by its eigenvalue. PCA applies this to the covariance/correlation matrix: eigenvectors are the principal components (directions of greatest variance) and eigenvalues give the variance each carries.",
 r"""- $Av=\lambda v$; solve $\det(A-\lambda I)=0$
- Explained variance ratio of component $k$: $\lambda_k/\sum_j\lambda_j$
- Scores: $Z=X_{std}V$
- Kaiser rule: keep components with $\lambda>1$ on the correlation matrix"""),
"11_numerical_optimization": ("Numerical optimization",
 "Many statistical estimates (MLE, least squares) have no closed form and are found by iterative optimisation. Gradient-based methods (BFGS, CG) use derivatives; Nelder-Mead is derivative-free. Maximising a likelihood is minimising the negative log-likelihood.",
 r"""- Minimise $f(\theta)$; optimum satisfies $\nabla f(\theta^*)=0$
- Gradient descent: $\theta_{k+1}=\theta_k-\eta\nabla f(\theta_k)$
- MLE: $\hat\theta=\arg\max\sum\ln f(x_i;\theta)$
- Rosenbrock: $f(x,y)=(1-x)^2+100(y-x^2)^2$"""),
"12_monte_carlo": ("Random numbers and Monte Carlo methods",
 "Monte Carlo methods use random sampling to approximate quantities that are hard to compute exactly. Pseudo-random numbers are reproducible via a seed. By the law of large numbers the estimate converges, with error shrinking like $1/\\sqrt n$.",
 r"""- Integral: $\int_a^b g(x)dx\approx\frac{b-a}{n}\sum g(U_i)$, $U_i\sim U(a,b)$
- $\pi\approx4\cdot\frac{\#\text{points inside quarter circle}}{n}$
- Inverse transform: $X=F^{-1}(U)$, e.g. Exponential $X=-\ln(1-U)/\lambda$
- MC standard error $=s/\sqrt n$"""),
"13_time_series": ("Time-series analysis",
 "A time series is data ordered in time. It is decomposed into trend, seasonality and noise. Autocorrelation shows dependence on past values; stationarity (tested by ADF) is required by many models. Holt-Winters exponential smoothing forecasts trend and seasonality.",
 r"""- Additive model: $y_t=T_t+S_t+\varepsilon_t$
- ACF: $\rho_k=\frac{\operatorname{cov}(y_t,y_{t-k})}{\operatorname{var}(y_t)}$
- Moving average: $\frac1m\sum_{i}y_{t-i}$
- Accuracy: MAE, RMSE, MAPE $=\frac{100}{n}\sum|\frac{y_t-\hat y_t}{y_t}|$"""),
"14_statistical_visualization": ("Statistical data visualization",
 "The chart must match the data: histogram/density for one numeric variable, boxplot/violin for distributions by group, scatter for two numeric variables, bar for counts, line for time. Honest axes and clear labels prevent misleading graphics.",
 r"""- Histogram bins (Freedman-Diaconis): $h=2\,\frac{IQR}{n^{1/3}}$
- Boxplot: whiskers to $1.5\,IQR$ beyond the quartiles
- Kernel density: $\hat f(x)=\frac{1}{nh}\sum K\!\left(\frac{x-x_i}{h}\right)$"""),
"15_transformation_normalization": ("Data transformation and normalization",
 "Transformations reduce skewness (log, square root, Box-Cox). Scaling puts variables on comparable ranges (z-score standardisation, min-max normalisation), which matters for distance-based and gradient methods such as PCA, k-means and regularised models.",
 r"""- z-score: $z=\frac{x-\bar x}{s}$
- Min-max: $x'=\frac{x-x_{min}}{x_{max}-x_{min}}$
- Log: $y=\ln x$; Box-Cox: $y=\frac{x^\lambda-1}{\lambda}$ ($\lambda\neq0$), $\ln x$ ($\lambda=0$)"""),
"16_anova": ("ANOVA and experimental design",
 "One-way ANOVA tests whether three or more group means are equal by comparing between-group to within-group variability. If significant, post-hoc tests (Tukey HSD) locate the differing groups. Two-way ANOVA adds a second factor and an interaction. Assumptions: independence, normal residuals, equal variances.",
 r"""- $SST=SSB+SSW$
- $F=\frac{MSB}{MSW}=\frac{SSB/(k-1)}{SSW/(N-k)}$
- Effect size: $\eta^2=SSB/SST$"""),
"17_chi_square": ("Chi-square and categorical data analysis",
 "The chi-square test of independence checks whether two categorical variables are associated by comparing observed counts to those expected under independence. The goodness-of-fit version compares one variable to a hypothesised distribution. Fisher's exact test suits small counts.",
 r"""- Expected: $E_{ij}=\frac{R_i C_j}{N}$
- $\chi^2=\sum\frac{(O_{ij}-E_{ij})^2}{E_{ij}}$, df $=(r-1)(c-1)$
- Cramér's V: $\sqrt{\frac{\chi^2}{N(\min(r,c)-1)}}$"""),
"18_logistic_regression": ("Logistic regression",
 "Logistic regression models a binary outcome. The log-odds are linear in the predictors; exp(coefficient) is the odds ratio. Predicted probabilities are thresholded to classify, and performance is judged with the confusion matrix, accuracy, precision, recall and ROC-AUC.",
 r"""- $\ln\frac{p}{1-p}=\beta_0+\beta_1x_1+\dots$, so $p=\frac{1}{1+e^{-\eta}}$
- Odds ratio $=e^{\beta_j}$
- Precision $=\frac{TP}{TP+FP}$, Recall $=\frac{TP}{TP+FN}$"""),
"19_nonparametric": ("Non-parametric statistical methods",
 "Rank-based tests make no normality assumption, so they suit skewed data, ordinal data and small samples. Mann-Whitney (two independent groups), Kruskal-Wallis (3+ groups), Wilcoxon signed-rank (paired) and Spearman correlation replace their parametric counterparts.",
 r"""- Mann-Whitney: $U_1=R_1-\frac{n_1(n_1+1)}{2}$
- Kruskal-Wallis: $H=\frac{12}{N(N+1)}\sum\frac{R_i^2}{n_i}-3(N+1)$
- Spearman: $\rho=1-\frac{6\sum d_i^2}{n(n^2-1)}$"""),
"20_bootstrap": ("Bootstrap and resampling methods",
 "The bootstrap draws many samples with replacement from the observed data to approximate a statistic's sampling distribution. It yields standard errors and confidence intervals for statistics without a formula (median, correlation, ratios).",
 r"""- Resample $x^*_1..x^*_n$ with replacement, compute $\hat\theta^*_b$, $b=1..B$
- $SE_{boot}=\sqrt{\frac{1}{B-1}\sum(\hat\theta^*_b-\bar{\hat\theta}^*)^2}$
- Percentile CI: the 2.5% and 97.5% quantiles of $\hat\theta^*$"""),
"21_multivariate": ("Multivariate statistical analysis",
 "Multivariate methods analyse several variables together: mean vector and covariance structure, Mahalanobis distance for multivariate outliers, MANOVA for group differences on several outcomes, and clustering (k-means, hierarchical) to find groups.",
 r"""- Mahalanobis: $D^2=(x-\mu)^\top\Sigma^{-1}(x-\mu)\sim\chi^2_p$
- k-means minimises $\sum_k\sum_{x\in C_k}\|x-\mu_k\|^2$
- Silhouette: $s=\frac{b-a}{\max(a,b)}$"""),
"22_missing_data": ("Missing-data analysis and imputation",
 "Missingness is MCAR (completely random), MAR (depends on observed data) or MNAR (depends on the missing value itself). Listwise deletion wastes data and can bias results; imputation (mean, regression/iterative, multiple imputation) fills gaps. Mean imputation shrinks variance.",
 r"""- Mean imputation: $\hat x_{miss}=\bar x_{obs}$
- Iterative (chained) imputation regresses each variable on the others in turn
- Rubin's rules (multiple imputation): $\bar Q=\frac1m\sum Q_i$, $T=\bar U+(1+\frac1m)B$"""),
"23_reproducible_reporting": ("Statistical reporting and reproducible analysis",
 "A reproducible report combines code, tables, figures and interpretation so anyone can regenerate it from raw data. Key habits: fixed seeds, scripted data cleaning, recorded package versions, and literate programming (R Markdown / Jupyter).",
 r"""- No new formulas: the report reuses topics 1-22 (e.g. a $t$-test with a confidence interval and effect size)
- Report each result as: estimate, CI, test statistic, p-value, interpretation"""),
}
