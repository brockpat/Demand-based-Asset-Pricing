This project is the first chapter of my PhD thesis in quantitative finance.

# Demand-based-Asset-Pricing

This research rigorously evaluates the **Demand-Based Asset Pricing (DBAP)** framework introduced by Koijen & Yogo (2019). While traditional asset pricing (e.g., Fama-French) focuses on risk premia and expected returns, DBAP treats prices as the outcome of **market-clearing** (more buyers than sellers implies that asset prices rise)

In this paper, we bridge the gap between empirical factor investing and market microstructure. We investigate whether the 90% of stock return volatility attributed to "unexplained" **Latent Demand** can be resolved by incorporating the "Factor Zoo." Our findings provide critical warnings for practitioners regarding the use of 13F data, investor grouping assumptions, and the reliability of estimated price elasticities.

## Motivation

The central premise of DBAP is
> Prices move because *investor demand curves* shift — and markets clear at new equilibrium prices.

Koijen & Yogo (2019) find:  
**~90% of cross-sectional stock return volatility is driven by demand shocks**, not supply.

But almost all of that demand effect is attributed to an unexplained residual:
> **Latent demand**

In typical asset pricing stock prices are modelled as some function of stocks' characteristics.

Demand-based asset pricing instead states to model the demand for stocks in order to explain asset prices.

This raises a core question:

- If demand drives prices,
- and if demand can be modelled as a function of stock characteristics,
- can stock prices be better forecasted by forecasting the demand?

A key insight is that through the lens of DBAP, a powerful mechanism can be explored: **Price signals drive demand, and demand drives volatility.** If institutional investors (who manage the majority of AUM) aggregate around specific characteristics and trade inelastically, they dictate the volatility profile of those assets. 

---
## Core Idea

We revisit the DBAP framework and ask:

> Can the “Factor Zoo” explain latent demand?

Specifically, we:

- Extend the original 5-characteristic demand system to **65 characteristics**
  - Add **60 return-invariant anomalies** from Chen & Zimmermann (2022)
- Rebuild the entire estimation and market-clearing engine in Python
- Implement scalable IV-GMM, LASSO, adaptive LASSO, backward selection, and gradient boosting
- Recompute the full variance decomposition of stock returns

The goal:
> Replace “latent demand” with systematic, observable drivers.

---

## Key Result

**Investor Demand is hard to explain and forecast.**

Even after adding 60 additional characteristics:

- Latent demand still explains the majority of stock return volatility.
- The characteristics-based demand equation often performs worse than a simple mean benchmark.
- Penalized nonlinear estimation improves in-sample fit — but does **not** reduce latent demand in variance decomposition.


**SEC's 13F Data Is Structurally Problematic**

DBAP relies heavily on 13F holdings data.

We show:

- Most investors hold too few stocks to estimate demand individually.
- Investors must be pooled.
- But grouped investors do not have homogeneous preferences.

This creates:

- Biased partial effects
- Mis-measured price elasticities
- Artificial latent demand

---

## Technical Contributions

### 1. Fully Rewritten DBAP Engine (Python)

- ~2,000 lines of documented code
- 18,000+ IV-GMM estimations
- Newton-Krylov high-dimensional root finding
- Stable constrained optimization under exponential nonlinearity
- Cross-validation & penalized NLLS

The original Stata implementation frequently failed to converge; our version scales and is reproducible.

### 2. Extended Variance Decomposition

We decompose annual stock return variance into:

- Supply:
  - Shares outstanding
  - Characteristics
  - Dividends
- Demand:
  - AUM
  - Coefficient shifts
  - Latent demand (extensive & intensive margins)

Result (replication + extension):

| Component | Share of Variance |
|-----------|-------------------|
| Latent demand (intensive) | ~60% |
| Latent demand (extensive) | ~23% |
| Everything else combined | < 20% |

Adding 60 characteristics barely moves these numbers.

---

## Interpretation

There are two possible explanations:

1. **Institutions ignore anomalies.**
2. **The functional equation to fit demand is heavily misspecified.**

Our evidence strongly favors (2).

The exponential logit-style demand equation may be too restrictive to capture:

- Mandate-driven investing
- Benchmark tracking
- ESG taste heterogeneity
- Sentiment
- Liquidity preferences
- Institutional inertia

---

## Broader Conceptual Insight

Demand-based asset pricing is powerful because it links:

- Investor heterogeneity  
- Market clearing  
- Counterfactual returns  
- Structural elasticities  

But this paper suggests:

> We do not yet understand the true microstructure foundations of institutional demand.

Latent demand is not noise.  
It is a structural modeling failure.

---

## Future Research Directions

We propose:

1. **Better investor clustering**
   - Beyond AUM and type
   - Preference-based clustering
   - Behaviorally informed grouping

2. **Sentiment-based demand factors**
   - Flow-based measures
   - Attention metrics
   - Mandate proxies

3. **Endogenous supply modeling**
   - Firms adjust issuance
   - Equity compensation dynamics
   - Passive flow accommodation

4. **Improved modeling of zeros**
   - Active non-ownership contains information
   - Extensive margin deserves structural treatment

---
