# sample path of 4 chain MCMC
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
  return(samples) # Return both theta1 and theta 2
}

set.seed(123)
n_samples <- 110000
n_chains_4 <- 4

theta1_matrix_4 <- matrix(NA, nrow = n_samples, ncol = n_chains_4)
theta2_matrix_4 <- matrix(NA, nrow = n_samples, ncol = n_chains_4)

# Define 4 distinct colors for the 4 chains
chain_colors <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3") 

# Set up a single plot window first so all chains overlay on the same graph

plot(NULL, 
     xlim = c(-4, 4),     # Reasonable range for theta1
     ylim = c(-6, 5),     # Reasonable range for theta2
     main = expression("MCMC Sampling Track for " * theta[1] * " and " * theta[2]),   
     xlab = expression(theta[1]), 
     ylab = expression(theta[2]))

# Loop to run chains and plot them
for (j in 1:n_chains_4) {
  
  # 1. Generate a unique random starting point for EACH chain
  init_point <- c(runif(1, -3, 3), runif(1, -10, 0)) 
  
  # 2. FIX: Run the chain ONCE and store the 2D matrix
  chain_output <- run_metropolis_chain(n_samples, init_point, proposal_sd = 0.5)
  
  # 3. Separate the components into your matrices
  theta1_matrix_4[, j] <- chain_output[, 1]
  theta2_matrix_4[, j] <- chain_output[, 2]
  
  # 4. Draw the track for this chain
  # type = "b" draws both points and lines connecting them in order of the rows
  lines(theta1_matrix_4[, j], 
        theta2_matrix_4[, j],
        type = "b",
        col = chain_colors[j], 
        pch = 16,             # Solid circles for points
        cex = 0.5,            # Make points slightly smaller so it's less crowded
        lwd = 1)              # Line width
}

# Add a simple legend to identify the chains
legend("topright", 
       legend = paste("Chain", 1:4), 
       col = chain_colors, 
       lty = 1, 
       pch = 16)

# separate plots:
for (j in 1:n_chains_4) {
  plot(NULL, 
       xlim = c(-4, 4),     # Reasonable range for theta1
       ylim = c(-6, 2),     # Reasonable range for theta2
       main = bquote("MCMC Sampling Track for " ~ theta[1] ~ " and " ~ theta[2] ~ "- Chain" ~ .(j)),  
       xlab = expression(theta[1]), 
       ylab = expression(theta[2]))
  lines(theta1_matrix_4[, j], 
        theta2_matrix_4[, j],
        type = "b",
        col = chain_colors[j], 
        pch = 16,             # Solid circles for points
        cex = 0.5,            # Make points slightly smaller so it's less crowded
        lwd = 1) 
} 

