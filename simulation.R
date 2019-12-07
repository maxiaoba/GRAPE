library(dplyr)
library(MASS)
library(Matrix)
library(pls)
library(svd)
library(tidyr)
library(grpreg)
library(pracma)

# Setup
sim_no <- 4
n <- 200
p <- 50
n_ev <- 1
scenario <- "top"
#n_test <- 5000
simN <- 1

corr <- 0.8 # within-group correlation
snr <- 2 # signal-to-noise ratio
print(paste("SNR", snr, "correlation", corr))

# Hyperparameters
# here, change b
# coefficients for the eigenvectors to get signal in response
# to be specified as a vector of length n_groups * n_ev
b <- c(5, 2, rep(0, times = 3))

if (length(group_sizes) != n_groups) { 
  stop("group_sizes must have length == n_groups") }
if (sum(group_sizes) != p) { 
  stop("elements of group_sizes must add up to p") }
if (length(b) != n_groups * n_ev) { 
  stop("b must have length n_groups * n_ev") }

b_mat <- matrix(b, ncol = 1)
b <- list()
for (g in 1:n_groups) {
  b[[g]] <- b_mat[ ((g-1) * n_ev + 1):(g * n_ev) , , drop = FALSE]
}

groups_w_signal <- unlist(lapply(b, function(x) any(x != 0) ))
support_size <- sum(mapply(function(x, y) x * y, groups_w_signal, group_sizes))

# Simulation
set.seed(sim_no)
for (i in 1:simN) {
  print(i)
  ##############
  # Generate X #
  ##############
  group_cov <- lapply(group_sizes, 
                      function(x) corr * matrix(1, nrow = x, ncol = x) +
                        (1 - corr) * diag(x))
  X <- lapply(1:n_groups,
              function(i) mvrnorm(n, 
                                  mu = rep(0, times = group_sizes[i]),
                                  Sigma = group_cov[[i]]))
  X_mat <- do.call(cbind, X)
  write.csv(X_mat, "/Users/daisyding/Dropbox/StarterCode/simulated_data.csv")
  
  ##############
  # Generate y #
  ##############
  svdX <- lapply(X, svd)
  V <- lapply(svdX, function(x) x$v)
  
  if (scenario == "top") {
    newV <- lapply(V, function(x) x[, 1:n_ev, drop = FALSE])
  } else if (scenario == "bottom") {
    newV <- lapply(V, function(x) 
      x[, (ncol(x) - n_ev + 1):ncol(x), drop = FALSE])
  } else {
    newV <- lapply(V, function(x) 
      x[, sample.int(ncol(x), size = n_ev), drop = FALSE])
  }
  
  PC_mat <- do.call(cbind, mapply(function(x, y) x %*% y, 
                                  X, newV, SIMPLIFY = FALSE))
  signal <- PC_mat %*% b_mat
  noise_var <- sum(mapply(function(b, w, sigma) t(w %*% b) %*% sigma %*%
                            (w %*% b),
                          b, newV, group_cov)) / snr
  y <- signal + rnorm(n, sd = sqrt(noise_var))
  
  ######################
  # Generate test data #
  ######################
  #X_test <- lapply(1:n_groups, 
  #                 function(i) mvrnorm(n_test, 
  #                                     mu = rep(0, times = group_sizes[i]),
  #                                     Sigma = group_cov[[i]]))
  #X_test_mat <- do.call(cbind, X_test)
 
  #signal_test <- mapply(function(x, w, b) x %*% w %*% b, 
  #                      X_test, newV, b, SIMPLIFY = FALSE)
  #signal_test <- rowSums(matrix(unlist(signal_test), 
  #                              ncol = length(signal_test))) #here, why not add noise here
}
  