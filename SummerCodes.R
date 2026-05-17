# AI codes
S <- 5000
t1 <- rnorm(S, 0, 1)
t2 <- rnorm(S, 0.03 * (t1^2 - 100), 1)

# 2. Use a 2D density estimator (standard way to visualize MC samples)
# This replaces your nested loop and matrix
dens <- MASS::kde2d(t1, t2, n = 100) # n=100 creates a 100x100 grid

# 3. Plot using image() or contour()
# Notice that 'dens' contains sorted x and y coordinates automatically
image(dens, col = terrain.colors(100), xlab = expression(theta[1]), ylab = expression(theta[2]))
contour(dens, add = TRUE) # Overlay level sets

##########################
# original code:
S <- 500

theta1.mc <- rnorm(S, 0, 1)

theta2.mc <- rnorm(S, 0.03 * (theta1.mc^2 - 100), 1)

t1 <- seq(-2, 2, length=S)
t2 <- seq(-5, -1, length=S)

matx.mc <- matrix(0,S,S)

for (i in 1:S){
  for (j in 1:S){
    matx.mc[i,j] <- dnorm(t1[i], 0,1,log=TRUE)+
      dnorm(t2[j], 0.03*(t1[i]^2-100), 1, log=TRUE)
  }
}
image(t1, t2, exp(matx.mc), col=blues9, 
      xlab=expression(theta1),
      ylab=expression(theta2)
)
contour(t1, t2, exp(matx.mc), add=TRUE)