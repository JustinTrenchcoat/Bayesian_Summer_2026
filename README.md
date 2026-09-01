# Many-Short-Chain MCMC: An Investigation of Convergence Diagnostics and Initialization Strategies

This repository contains code and experiments for reproducing and extending the numerical experiments in

> _Nested_ $\widehat{R}$: _Assessing the Convergence of Markov Chain Monte Carlo When Running Many Short Chains_

by Charles Margossian, Matt Hoffman, Pavel Sountsov, Lionel Riou-Durand, Aki Vehtari, and Andrew Gelman. 

[Paper](https://projecteuclid.org/journals/bayesian-analysis/volume-20/issue-4/Nested-R%CB%86--Assessing-the-Convergence-of-Markov-Chain-Monte/10.1214/24-BA1453.full).
[Original Repository](https://github.com/charlesm93/nested-rhat)

## Project Overview 

The project has three main objectives:
1. Reproduce the numerical experiments from the original paper using the original TensorFlow-Probability workflow. 
2. Reimplement the many-chain sampling workflow using BlackJAX and assess whether the reported behavior can be reproduced in the newer ecosystem.
3. Investigate how initialization strategies (e.g., PathFinder Initialization) affect the behavior of nested $\widehat{R}$.

## Project Status

This project has been transitioned to my Degree Capstone Project! [Link to the new repository](https://github.com/JustinTrenchcoat/Capstone_Many_Short_Chain)

- [x] Reproduce original TFP experiments
- [x] Implement BlackJAX workflow
- [x] Implement PathFinder initialization
- [x] Run BlackJAX numerical experiments
- [x] Generate comparative visualizations
- [ ] Complete synthesis and analysis
- [ ] Complete technical report

### Original-paper reproduction
`paperReproduction/` contains the TFP-based (TensorFlow Probability) implementation used to reproduce the numerical experiments from the original paper.

### BlackJAX implementation
`BlackJAX_Files/` contains a reimplementation of the many-chain workflow using BlackJAX, following the computational strategy suggested by Dr. Margossian.

### Additional simulations
`AdditionalExperiments/` contains exploratory experiments investigating the behavior of Nested $\widehat{R}$ under additional target distributions and convariance structures.

| Directory  | Purpose |
| ------------- | ------------- |
| `paperReproduction\`  | Reproduction of the original TFP-based experiments |
| `BlackJAX_Files\`  | BlackJAX implementation and PathFinder experiments  |
| `AdditionalExperiments\`  | Additional simulation studies  |
| `RFolder\`  | Early exploratory R implementations |

The original experiments rely on TensorFlow Probability(TFP), whose APIs and dependencies have changed since the paper's implementation was developed. During the initial reproduction, the original paper's repository was also missing a helper file `utility.py`; this file was later added to the paper's repository. The TFP implementation is retained here as the baseline reproduction, while a separate BlackJAX implementation was developed to explore the same workflow in a more actively maintained JAX-based ecosystem.

## Computational Environment

The experiments were primarily run in Google Colab using GPU acceleration.
The main computational frameworks are:

- Python
- JAX
- BlackJAX
- TensorFlow Probability
- ArviZ (Will be used in future works)
- Pathfinder
- NumPy / pandas / Matplotlib

## Report
To Be Finished!
[Technical Report]()
