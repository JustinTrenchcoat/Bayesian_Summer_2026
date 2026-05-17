library(ggplot2)
library(dplyr)
library(coda)

log_joint_density <- function(theta) {
  t1 <- theta[1]
  t2 <- theta[2]
  log_p_t1 <- dnorm(t1, 0, 1, log = TRUE)
  log_p_t2_given_t1 <- dnorm(t2, 0.03 * (t1^2 - 100), 1, log = TRUE)
  return(log_p_t1 + log_p_t2_given_t1)
}
# function for R hat calculation:
G_RStat <- function(L, W, B){
  return((((L-1)/L)*W+(B/L))/W)
}
  
# func for metropolis
run_metropolis_chain <- function(n_samples, init_theta, proposal_sd = 0.5) {
  # initialization phase
  samples <- matrix(NA, nrow = n_samples, ncol = 2)
  current_theta <- init_theta
  samples[1, ] <- current_theta
  
  for (i in 2:n_samples) {
    # proposal is theta1 at s + x, where x ~norm(0, 0.5)
    proposal_theta1 <- current_theta
    proposal_theta1[1] <- current_theta[1] + rnorm(1, mean = 0, sd = proposal_sd)
    # check r
    log_acc_ratio <- log_joint_density(proposal_theta1) - log_joint_density(current_theta)
    if (log(runif(1)) < log_acc_ratio) {
      current_theta[1] <- proposal_theta1[1]
    }
    
    # update theta2 by Gibbs
    mean_t2 <- 0.03 * (current_theta[1]^2 - 100)
    current_theta[2] <- rnorm(1, mean = mean_t2, sd = 1)
    # update the theta vector
    samples[i, ] <- current_theta
  }
  return(samples[, 1]) # Return only theta1 samples
}
# actual sampling part:
set.seed(123)
n_samples <- 1100
n_chains_4 <- 4
n_chains_512 <- 512

theta1_matrix_4 <- matrix(NA, nrow = n_samples, ncol = n_chains_4)
theta1_matrix_512 <- matrix(NA, nrow = n_samples, ncol = n_chains_512)


init_point <- c(runif(1, -3, 3), runif(1, -10, 0)) 

for (j in 1:n_chains_4) {
  theta1_matrix_4[, j] <- run_metropolis_chain(n_samples, init_point, proposal_sd = 0.5)
}
for (j in 1:n_chains_512) {
  theta1_matrix_512[, j] <- run_metropolis_chain(n_samples, init_point, proposal_sd = 0.5)
}
# convert to mcmc.list
chains_4 <- list()
for (i in 1:n_chains_4){
  chains_4[[i]] <- theta1_matrix_4[100:1100,i]
}

mcmc_chains <- mcmc.list(lapply(chains_4,mcmc))
gelman.diag(mcmc_chains)
