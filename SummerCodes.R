# AI codes
S <- 5000
t1 <- rnorm(S, 0, 1)
t2 <- rnorm(S, 0.03 * (t1^2 - 100), 1)

plot(t1,t2, type='p')

# 2. Use a 2D density estimator (standard way to visualize MC samples)
# This replaces your nested loop and matrix
dens <- MASS::kde2d(t1, t2, n = 100) # n=100 creates a 100x100 grid

# 3. Plot using image() or contour()
# Notice that 'dens' contains sorted x and y coordinates automatically
image(dens, col = terrain.colors(100), xlab = expression(theta[1]), ylab = expression(theta[2]))
contour(dens, add = TRUE) # Overlay level sets

##########################
library(plotly)

S1 <- 250
S2 <- 300
t1 <- seq(-4, 4, length=S1)
t2 <- seq(-5, -1, length=S2) 

# Initialize a clean matrix of zeros
post.grid <- matrix(0, nrow=S1, ncol=S2)

# 2. Build the stable log grid
for(the1 in 1:S1) {
  for(the2 in 1:S2) { 
    log_p_t1 <- dnorm(t1[the1], 0, 1, log = TRUE)
    log_p_t2_given_t1 <- dnorm(t2[the2], 0.03 * (t1[the1]^2 - 100), 1, log = TRUE)
    post.grid[the1, the2] <- log_p_t1 + log_p_t2_given_t1
  }
}

post.grid <- exp(post.grid)
post.grid <- post.grid / sum(post.grid)

plot_ly(x = t1, y = t2, z = post.grid, type = "surface") %>%
  layout(
    scene = list(
      xaxis = list(title = "Theta 1"),
      yaxis = list(title = "Theta 2"),
      zaxis = list(title = "Value")
    )
  )
