# Ensure the tolerance package is installed
# install.packages("tolerance")
library(tolerance)

#' Calculate the beta-content, gamma-confidence Tolerance Interval for the Reference Product
#' @param x_ref Numeric vector of reference product lot values
#' @param beta Content proportion (e.g., 0.99)
#' @param gamma Confidence level (e.g., 0.95)
#' @return A named numeric vector with Lower and Upper bounds
calc_ref_TI <- function(x_ref, beta, gamma) {
  # The tolerance package uses 'alpha' as the significance level (1 - gamma)
  # and 'P' as the content proportion (beta)
  ti_res <- normtol.int(x = x_ref, alpha = 1 - gamma, P = beta, side = 2,
            method="EXACT") #or method="HE"
  return(c(Lower = ti_res$lower, Upper = ti_res$upper))
}

#' Calculate the beta Prediction Interval for the Test Product
#' @param x_test Numeric vector of test product lot values
#' @param beta Prediction proportion (e.g., 0.99)
#' @return A named numeric vector with Lower and Upper bounds
calc_test_PI <- function(x_test, beta) {
  n_T <- length(x_test)
  mean_T <- mean(x_test)
  sd_T <- sd(x_test)
  
  # Critical value from the Student's t-distribution
  t_crit <- qt((1 + beta) / 2, df = n_T - 1)
  
  # The sqrt(1 + 1/n_T) term accounts for the variance of the new observation 
  # plus the variance of the estimated mean
  margin <- t_crit * sd_T * sqrt(1 + 1 / n_T)
  
  return(c(Lower = mean_T - margin, Upper = mean_T + margin))
}

# ==========================================
# Example Execution and Decision Logic
# ==========================================
set.seed(42)

# Simulate Reference Product (e.g., 15 lots, Mean=100, SD=1)
x_ref <- rnorm(15, mean = 100, sd = 1.0)

# Simulate Test Product (e.g., 10 lots, Mean=100.2, SD=1.1)
x_test <- rnorm(10, mean = 100.2, sd = 1.1)

# Define EFSPI recommended parameters (e.g., 99% PI within 99%/95% TI)
beta <- 0.99
gamma <- 0.95

ref_TI <- calc_ref_TI(x_ref, beta, gamma)
test_PI <- calc_test_PI(x_test, beta)

cat(sprintf("Reference %d%%-content %d%%-confidence TI: [%.3f, %.3f]\n", 
            beta*100, gamma*100, ref_TI["Lower"], ref_TI["Upper"]))
cat(sprintf("Test %d%% Prediction Interval: [%.3f, %.3f]\n", 
            beta*100, test_PI["Lower"], test_PI["Upper"]))

# Decision Rule: Is the Test PI entirely nested within the Reference TI?
is_comparable <- (test_PI["Lower"] >= ref_TI["Lower"]) & (test_PI["Upper"] <= ref_TI["Upper"])

cat(sprintf("\nDecision: Are the products analytically comparable? %s\n", 
            ifelse(is_comparable, "YES (PI is within TI)", "NO (PI exceeds TI bounds)")))