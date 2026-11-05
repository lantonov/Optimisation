# ==============================================================================
# R Implementation: QR and mQR (Effect Size & Plausibility Integral) Methods
# ==============================================================================

#' Evaluate Analytical Similarity using QR or mQR Methods
#' @param x_ref Numeric vector of reference product lot values
#' @param x_test Numeric vector of test product lot values
#' @param method Character: "QR", "ES_mQR", or "PI_mQR"
#' @param inclusion_thresh Numeric: Minimum proportion of test lots required in range (default 0.90)
#' @return A list containing the decision, bounds, and diagnostic metrics
evaluate_similarity <- function(x_ref, x_test, method = "QR", inclusion_thresh = 0.90) {
  
  n_R <- length(x_ref)
  n_T <- length(x_test)
  mean_R <- mean(x_ref)
  sd_R <- sd(x_ref)
  mean_T <- mean(x_test)
  sd_T <- sd(x_test)
  
  sd_ratio <- sd_T / sd_R
  mean_diff <- mean_T - mean_R
  
  # Preliminary Check: Similarity of Standard Deviations (Method 1)
  sd_similar <- (sd_ratio > 2/3) & (sd_ratio < 3/2)
  
  lower_bound <- NA
  upper_bound <- NA
  
  if (method == "QR") {
    # Standard Quality Range Method (k = 3)
    k <- 3
    lower_bound <- mean_R - k * sd_R
    upper_bound <- mean_R + k * sd_R
    
  } else if (method == "ES_mQR") {
    # Effect Size Modified QR
    k_base <- 2.23
    # Condition 1: Mean shift <= 1/8 sd_R
    # Condition 2: SD ratio within (2/3, 3/2)
    if (abs(mean_diff) <= (1/8) * sd_R && sd_similar) {
      # Conditions met, revert to standard QR logic but with k=2.23 as per text optimization
      k <- k_base 
    } else {
      # Apply Effect Size adjustment: k_adj = k - (C - 1) where C = sd_T / sd_R
      C <- sd_ratio
      k <- k_base - (C - 1)
      if (k <= 0) k <- 0.01 # Prevent negative bounds
    }
    lower_bound <- mean_R - k * sd_R
    upper_bound <- mean_R + k * sd_R
    
  } else if (method == "PI_mQR") {
    # Plausibility Integral Modified QR
    a <- 3.0 # Plausibility interval multiplier
    alpha <- 0.05 # False positive rate target
    
    # Preliminary Step 1: Check if 95% CI of difference is within Plausibility Interval [-a*sd_R, a*sd_R]
    # Using a simplified 2-sample t-interval for the difference as a proxy for the TI of difference
    se_diff <- sqrt(sd_R^2/n_R + sd_T^2/n_T)
    t_crit <- qt(1 - alpha/2, df = n_R + n_T - 2)
    ci_diff_lower <- mean_diff - t_crit * se_diff
    ci_diff_upper <- mean_diff + t_crit * se_diff
    
    pi_lower <- -a * sd_R
    pi_upper <- a * sd_R
    
    ti_within_pi <- (ci_diff_lower >= pi_lower) & (ci_diff_upper <= pi_upper)
    
    # Determine target probability and mean shift parameter (d)
    if (ti_within_pi) {
      # Focus on reducing false negative (Power)
      P_target <- 1 - alpha 
      d_val <- 0
    } else {
      # Focus on reducing false positive (Patient Risk)
      P_target <- alpha
      d_val <- (mean_R - mean_T) / sd_T
    }
    
    r_val <- sd_R / sd_T
    
    # Define the integral function f(M) = Phi(d + M*r) - Phi(d - M*r)
    f_M <- function(M) {
      pnorm(d_val + M * r_val) - pnorm(d_val - M * r_val)
    }
    
    # Solve for M using root finding: f(M) - P_target = 0
    # Since f(M) is monotonically increasing, uniroot is guaranteed to find the unique root
    root_obj <- tryCatch({
      uniroot(function(M) f_M(M) - P_target, interval = c(0, 10), tol = 1e-8)
    }, error = function(e) NULL)
    
    if (is.null(root_obj)) {
      M <- NA
      warning("Root finding for PI mQR failed to converge.")
    } else {
      M <- root_obj$root
    }
    
    # Preliminary Step 2: Variance Asymmetry Penalty
    # If sd_R > sd_T, use sd_T to prevent oversized range. 
    # If sd_T/sd_R >= 1.5, use sd_R.
    if (sd_R > sd_T) {
      scale_factor <- sd_T
    } else if (sd_ratio >= 1.5) {
      scale_factor <- sd_R
    } else {
      scale_factor <- sd_R
    }
    
    lower_bound <- mean_R - M * scale_factor
    upper_bound <- mean_R + M * scale_factor
  }
  
  # Calculate proportion of test lots within the calculated range
  in_range <- sum(x_test >= lower_bound & x_test <= upper_bound) / n_T
  decision <- ifelse(in_range >= inclusion_thresh, "PASS", "FAIL")
  
  return(list(
    Method = method,
    Decision = decision,
    Inclusion_Proportion = round(in_range, 4),
    Lower_Bound = round(lower_bound, 4),
    Upper_Bound = round(upper_bound, 4),
    SD_Ratio = round(sd_ratio, 4),
    Mean_Diff = round(mean_diff, 4),
    SD_Similar = sd_similar
  ))
}

# ==========================================
# Example Execution
# ==========================================
set.seed(123)
x_ref <- rnorm(15, mean = 100, sd = 1.0)
x_test <- rnorm(10, mean = 101.5, sd = 1.4) # Intentional mean shift and variance inflation

cat("--- QR Method ---\n")
print(evaluate_similarity(x_ref, x_test, method = "QR"))

cat("\n--- ES mQR Method ---\n")
print(evaluate_similarity(x_ref, x_test, method = "ES_mQR"))

cat("\n--- PI mQR Method ---\n")
print(evaluate_similarity(x_ref, x_test, method = "PI_mQR"))