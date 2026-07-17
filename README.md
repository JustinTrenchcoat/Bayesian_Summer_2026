# Bayesian Research Project (Summer 2026)

This repository documents my attempt to reproduce the experimental results from the paper

> _Nested_ $\widehat{R}$: _Assessing the Convergence of Markov Chain Monte Carlo When Running Many Short Chains_

by Charles Margossian, Matt Hoffman, Pavel Sountsov, Lionel Riou-Durand, Aki Vehtari, and Andrew Gelman. The [Link](https://projecteuclid.org/journals/bayesian-analysis/volume-20/issue-4/Nested-R%CB%86--Assessing-the-Convergence-of-Markov-Chain-Monte/10.1214/24-BA1453.full) is here.

The goal of this project is to better understand the proposed Nested $\widehat{R}$ diagnostic by reproducing the numerical experiments presented in the paper and exploring its behavior under additional simulation settings.

The original implementation appears to rely on several software packages that have since changed substantially, and some helper files (e.g., `utility.py`) are not publicly available (Update on 2026 Jul 16: Now it is!). As a result, part of this repository also documents the modifications and reimplementations required to reproduce the experiments.

## Repository Structure

The `.R` files contain my initial exploratory work for understanding the concepts and methodology presented in the paper. The primary reproduction and simulation work is implemented in the Jupyter notebooks. Computation is hosted on Google Colab with GPU A100.

### `EarlyStopSim_MVN.ipynb`

Evaluates the performance of Nested $\widehat{R}$ on multivariate normal targets. The notebook compares the standard $\widehat{R}$ and Nested $\widehat{R}$ under different simulation settings (e.g., dimensionality and covariance structure). Each configuration is repeated 100 times to estimate average performance.

### `EarlyStopSim_GMM.ipynb`

Performs similar experiments using Gaussian mixture model targets to study the behavior of Nested $\widehat{R}$ on multimodal distributions.

### `NumericalExperimentReproduction.ipynb`

Attempts to reproduce several numerical experiments and figures from Sections 4.2 and 4.3 of the paper.

### `BimodalExperiment.ipynb`

Attempts to assess the relationship between $$\widehat{R}$$ and distance between modes in bimodal distribution, and whether there would be a similar trend in "shifting" of $\widehat{R}$ and MSE.
