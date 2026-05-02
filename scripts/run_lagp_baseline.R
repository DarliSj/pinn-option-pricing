# =============================================================================
# laGP baseline (A3) — walk-forward, strict features.
#
# Methodology:
#   For each fold f:
#     1. Read data/folds/<f>/{train,test}.csv and meta.json.
#     2. Compute bs_theoretical = BS(F, K, T, r_fixed, sigma_fixed) per row.
#     3. residual = mid_price - bs_theoretical
#     4. Standardize features (moneyness, time_to_exp, log_volume) on TRAIN.
#     5. Fit local approximate GP via aGP(start=6, end=50) — no daily_vol_proxy.
#     6. Final price = bs_theoretical + GP-predicted residual.
#     7. 95% predictive interval: gp_se = sqrt(latent_var + train_residual_sd^2)
#        PI = price ± 1.96 * gp_se   (saved for Stage 3 UQ comparison)
#     8. Per-fold + pooled metrics, same schema as bs_baseline.
#
# Output:
#   results/lagp_baseline/results.json    (tagged A3_lagp)
#
# Usage:
#   Rscript scripts/run_lagp_baseline.R
#
# Requires:
#   - Folds dumped to data/folds/  (run scripts/dump_folds.py first)
#   - R packages: laGP, jsonlite, dplyr, readr
# =============================================================================

suppressPackageStartupMessages({
  library(laGP)
  library(dplyr)
  library(readr)
  library(jsonlite)
})

# Same numerically-safe BS as PINN code
bs_call_price <- function(S, K, Tau, r, sigma) {
  Tau   <- pmax(Tau, 1e-4)
  sigma <- pmax(sigma, 1e-6)
  d1 <- (log(S / K) + (r + 0.5 * sigma^2) * Tau) / (sigma * sqrt(Tau))
  d2 <- d1 - sigma * sqrt(Tau)
  S * pnorm(d1) - K * exp(-r * Tau) * pnorm(d2)
}

stratify_metrics <- function(market, predicted, moneyness) {
  err <- market - predicted
  ssq <- err^2
  otm_mask <- moneyness < 0.97
  itm_mask <- moneyness > 1.03
  atm_mask <- !otm_mask & !itm_mask
  list(
    sum_sq_err_otm = sum(ssq[otm_mask]), n_otm = sum(otm_mask),
    sum_sq_err_atm = sum(ssq[atm_mask]), n_atm = sum(atm_mask),
    sum_sq_err_itm = sum(ssq[itm_mask]), n_itm = sum(itm_mask),
    rmse_otm = if (any(otm_mask)) sqrt(mean(ssq[otm_mask])) else NA_real_,
    rmse_atm = if (any(atm_mask)) sqrt(mean(ssq[atm_mask])) else NA_real_,
    rmse_itm = if (any(itm_mask)) sqrt(mean(ssq[itm_mask])) else NA_real_
  )
}


run_one_fold <- function(fold_dir, gp_start = 6, gp_end = 50) {
  fold_name <- basename(fold_dir)
  meta <- jsonlite::read_json(file.path(fold_dir, "meta.json"))
  train <- read_csv(file.path(fold_dir, "train.csv"), show_col_types = FALSE)
  test  <- read_csv(file.path(fold_dir, "test.csv"),  show_col_types = FALSE)

  sigma_fixed <- meta$sigma_fixed
  r_fixed     <- meta$r_fixed

  train <- train %>% mutate(
    log_volume     = log(pmax(volume, 1)),
    bs_theoretical = bs_call_price(forward_price, strike_price, time_to_exp,
                                    r_fixed, sigma_fixed),
    residual       = mid_price - bs_theoretical
  )
  test <- test %>% mutate(
    log_volume     = log(pmax(volume, 1)),
    bs_theoretical = bs_call_price(forward_price, strike_price, time_to_exp,
                                    r_fixed, sigma_fixed)
  )

  # Strict feature set — no daily_vol_proxy (matches PINN information set)
  gp_cols <- c("moneyness", "time_to_exp", "log_volume")

  # Standardize using TRAIN statistics only
  gp_mu  <- sapply(train[, gp_cols], mean, na.rm = TRUE)
  gp_sds <- sapply(train[, gp_cols], sd,   na.rm = TRUE)
  gp_sds[gp_sds == 0] <- 1.0

  X_train <- sweep(sweep(as.matrix(train[, gp_cols]), 2, gp_mu, "-"), 2, gp_sds, "/")
  X_test  <- sweep(sweep(as.matrix(test [, gp_cols]), 2, gp_mu, "-"), 2, gp_sds, "/")

  # Fit local approximate GP (aGP fits a separate local GP at each test point)
  cat(sprintf("  [%s] aGP on %d train, %d test (start=%d, end=%d)... ",
              fold_name, nrow(train), nrow(test), gp_start, gp_end))
  t0 <- Sys.time()
  set.seed(1)
  gp_fit <- aGP(X = X_train, Z = train$residual, XX = X_test,
                start = gp_start, end = gp_end, verb = 0)
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("done in %.1fs\n", elapsed))

  pred_resid     <- as.numeric(gp_fit$mean)
  latent_var     <- as.numeric(gp_fit$var)
  sigma_obs      <- sd(train$residual, na.rm = TRUE)
  pred_se        <- sqrt(latent_var + sigma_obs^2)
  predicted_price <- test$bs_theoretical + pred_resid

  # 95% predictive interval (for Stage 3 UQ comparison later)
  PI_lower <- predicted_price - 1.96 * pred_se
  PI_upper <- predicted_price + 1.96 * pred_se
  in_PI    <- (test$mid_price >= PI_lower) & (test$mid_price <= PI_upper)
  coverage_95 <- mean(in_PI)
  mean_PI_width <- mean(PI_upper - PI_lower)

  # Metrics
  err  <- test$mid_price - predicted_price
  ssq  <- sum(err^2)
  sabs <- sum(abs(err))
  n_test <- nrow(test)

  strat <- stratify_metrics(test$mid_price, predicted_price, test$moneyness)

  # Floor of $0.05 on half_spread matches src/training.py::validate (spread_floor)
  half_spread_test <- pmax((test$best_offer - test$best_bid) / 2, 0.05)
  ssq_spread       <- sum((err / half_spread_test)^2)
  spread_units     <- sqrt(mean((err / half_spread_test)^2))

  list(
    fold              = fold_name,
    n_train           = nrow(train),
    n_test            = n_test,
    sigma_fixed       = sigma_fixed,
    r_fixed           = r_fixed,
    rmse_mkt          = sqrt(ssq / n_test),
    mae_mkt           = sabs / n_test,
    sum_sq_err_mkt    = ssq,
    sum_abs_err_mkt   = sabs,
    rmse_otm          = strat$rmse_otm,
    rmse_atm          = strat$rmse_atm,
    rmse_itm          = strat$rmse_itm,
    n_otm             = strat$n_otm,
    n_atm             = strat$n_atm,
    n_itm             = strat$n_itm,
    sum_sq_err_otm    = strat$sum_sq_err_otm,
    sum_sq_err_atm    = strat$sum_sq_err_atm,
    sum_sq_err_itm    = strat$sum_sq_err_itm,
    rmse_spread       = spread_units,
    sum_sq_err_spread = ssq_spread,
    # UQ fields — feed Stage 3 comparison
    coverage_95       = coverage_95,
    mean_PI_width     = mean_PI_width,
    elapsed           = elapsed
  )
}


main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  fold_root  <- if (length(args) >= 1) args[1] else "data/folds"
  output_dir <- if (length(args) >= 2) args[2] else "results/lagp_baseline"

  if (!dir.exists(fold_root)) {
    stop("Fold dir not found: ", fold_root, "\nRun: python scripts/dump_folds.py")
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  fold_dirs <- list.dirs(fold_root, recursive = FALSE)
  cat(sprintf("Found %d folds in %s\n\n", length(fold_dirs), fold_root))

  fold_results <- list()
  for (fd in fold_dirs) {
    res <- run_one_fold(fd)
    fold_results[[res$fold]] <- res
    cat(sprintf("       rmse=%.3f  cov95=%.1f%%  meanPI=%.2f\n",
                res$rmse_mkt, 100 * res$coverage_95, res$mean_PI_width))
  }

  fold_df <- do.call(rbind, lapply(fold_results, as.data.frame))
  pooled_rmse_mkt <- sqrt(sum(fold_df$sum_sq_err_mkt)    / sum(fold_df$n_test))
  pooled_mae_mkt  <-       sum(fold_df$sum_abs_err_mkt)  / sum(fold_df$n_test)
  pooled_rmse_otm <- sqrt(sum(fold_df$sum_sq_err_otm)    / sum(fold_df$n_otm))
  pooled_rmse_atm <- sqrt(sum(fold_df$sum_sq_err_atm)    / sum(fold_df$n_atm))
  pooled_rmse_itm <- sqrt(sum(fold_df$sum_sq_err_itm)    / sum(fold_df$n_itm))
  pooled_rmse_spr <- sqrt(sum(fold_df$sum_sq_err_spread) / sum(fold_df$n_test))

  worst_idx <- which.max(fold_df$rmse_mkt)

  summary_block <- list(
    pooled_rmse_mkt    = pooled_rmse_mkt,
    pooled_mae_mkt     = pooled_mae_mkt,
    pooled_rmse_otm    = pooled_rmse_otm,
    pooled_rmse_atm    = pooled_rmse_atm,
    pooled_rmse_itm    = pooled_rmse_itm,
    pooled_rmse_spread = pooled_rmse_spr,
    mean_rmse_mkt      = mean(fold_df$rmse_mkt),
    std_rmse_mkt       = sd(fold_df$rmse_mkt),
    worst_rmse_mkt     = fold_df$rmse_mkt[worst_idx],
    worst_fold         = fold_df$fold[worst_idx],
    n_folds            = nrow(fold_df),
    total_n_test       = sum(fold_df$n_test),
    val_months         = 1,
    # UQ pooled
    mean_coverage_95   = mean(fold_df$coverage_95),
    mean_PI_width      = mean(fold_df$mean_PI_width)
  )

  out_obj <- list(
    summary  = summary_block,
    folds    = unname(fold_results),
    config   = list(
      method     = "laGP (aGP local approximate GP)",
      kernel     = "default Gaussian (laGP)",
      start      = 6,
      end        = 50,
      target     = "BS residual (mid_price - BS(F,K,T,r_fixed,sigma_fixed))",
      features   = "moneyness, time_to_exp, log_volume (standardized on train)",
      uses_regime = FALSE,
      val_months = 1
    )
  )

  out_path <- file.path(output_dir, "results.json")
  write_json(out_obj, out_path, auto_unbox = TRUE, pretty = TRUE, na = "null")

  cat("\n", strrep("=", 70), "\n", sep = "")
  cat(sprintf("Pooled RMSE: $%.4f   (BS_A1 baseline ≈ $8.795)\n", pooled_rmse_mkt))
  cat(sprintf("Mean fold RMSE: $%.4f   Std: $%.4f\n",
              summary_block$mean_rmse_mkt, summary_block$std_rmse_mkt))
  cat(sprintf("Mean 95%% PI coverage: %.1f%%   Mean PI width: $%.2f\n",
              100 * summary_block$mean_coverage_95, summary_block$mean_PI_width))
  cat(sprintf("Worst fold: %s ($%.4f)\n",
              summary_block$worst_fold, summary_block$worst_rmse_mkt))
  cat(sprintf("Wrote %s\n", out_path))
}

main()
