# part II, replicate fig 1 with MCMC
# check the joint density
log_joint_density <- function(theta) {
  t1 <- theta[1]
  t2 <- theta[2]
  log_p_t1 <- dnorm(t1, 0, 1, log = TRUE)
  log_p_t2_given_t1 <- dnorm(t2, 0.03 * (t1^2 - 100), 1, log = TRUE)
  return(log_p_t1 + log_p_t2_given_t1)
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
B <- 100
n_samples <- 1100
n_chains_4 <- 4
n_chains_512 <- 512

theta1_matrix_4 <- matrix(NA, nrow = n_samples, ncol = n_chains_4)
theta1_matrix_512 <- matrix(NA, nrow = n_samples, ncol = n_chains_512)


init_point <- c(runif(1, -2, 2), runif(1, -10, 0)) 

for (j in 1:n_chains_4) {
  theta1_matrix_4[, j] <- run_metropolis_chain(n_samples, init_point, proposal_sd = 0.5)
}
for (j in 1:n_chains_512) {
  theta1_matrix_512[, j] <- run_metropolis_chain(n_samples, init_point, proposal_sd = 0.5)
}

# get squared error
se_matrix_4 <- matrix(NA, nrow = n_samples, ncol = n_chains_4)
for (j in 1:n_chains_4) {
  se_matrix_4[, j] <- cumsum(theta1_matrix_4[, j]) / (1:n_samples)
}
avg_squared_error_4 <- (rowMeans(se_matrix_4))^2 

se_matrix_512 <- matrix(NA, nrow = n_samples, ncol = n_chains_512)
for (j in 1:n_chains_512) {
  running_mean <- cumsum(theta1_matrix_512[, j]) / (1:n_samples)
  se_matrix_512[, j] <- (running_mean - 0)^2 
}

avg_squared_error_512 <- rowMeans(se_matrix_512)

avg_squared_error_4 <- avg_squared_error_4[B+1:n_samples]

plot(B+1:n_samples,avg_squared_error_4, 
     type = "l", col = "darkgreen", lwd = 2,
     #ylim = range(c(1.5, avg_squared_error_4)),
     main = "Avg Squared Error (Burn-in Not Included)",
     xlab = "Post-warmup sampling iterations", 
     ylab = "Squared Error for mean estimation",
     yaxt="n",
     log = 'y')
tick_positions <- c(1, 0.01, 0.0001)
tick_labels <- expression(10^0, 10^-2, 10^-4)
axis(side = 2, at = tick_positions, labels = tick_labels, las = 1)
# 2. Add the second line on top using lines()
lines(100:1100, avg_squared_error_512[100:1100], 
      col = "orange", lwd = 2)
abline(h = 1, lty = 2, col="red") 
# 3. Add a legend so you know which line is which
legend("topright", legend = c("4 Chains", "512 Chains", "Var theta / 100"),
       col = c("darkgreen", "orange", "red"), lwd = 2)


acf(theta1_matrix_4[,1], main = "ACF of Chain 1")
