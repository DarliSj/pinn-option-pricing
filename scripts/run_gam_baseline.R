# =============================================================================
# GAM baseline (A2) — walk-forward, strict features.
#
# Methodology:
#   For each fold f:
#     1. Read data/folds/<f>/{train,test}.csv and meta.json.
#     2. Compute sigma_fixed and r_fixed from meta.json (same as BS_A1, same
#        as PINN PDE).
#     3. bs_theoretical = BS(F, K, T, r_fixed, sigma_fixed)  for train + test.
#     4. residual = mid_price - bs_theoretical.
#     5. Fit GAM on TRAIN residuals:
#          residual ~ te(moneyness, time_to_exp, k=c(8,5)) + ns(log_volume, df=2)
#        REML-tuned smoothing; "strict" feature set (no daily_vol_proxy
#        regime indicator — matches the PINN's information set exactly).
#     6. Predict residuals on TEST; final price = bs_theoretical + predicted residual.
#     7. Compute pooled and per-fold metrics matching the PINN schema.
#
# Output:
#   results/gam_baseline/results.json   (same schema as bs_baseline so
#                                        scripts/build_master_table.py reads
#                                        it without changes; tagged A2_gam)
#
# Usage:
#   Rscript scripts/run_gam_baseline.R
#
# Requires:
#   - Folds dumped to data/folds/  (run scripts/dump_folds.py first)
#   - R packages: mgcv, splines, jsonlite, dplyr, readr
# =============================================================================

suppressPackageStartupMessages({
  library(mgcv)
  library(splines)
  library(dplyr)
  library(readr)
  library(jsonlite)
})

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Same numerically-safe BS as the PINN code (src/bs_formulas.py::bs_call_price)
bs_call_price <- function(S, K, Tau, r, sigma) {
  Tau   <- pmax(Tau, 1e-4)
  sigma <- pmax(sigma, 1e-6)
  d1 <- (log(S / K) + (r + 0.5 * sigma^2) * Tau) / (sigma * sqrt(Tau))
  d2 <- d1 - sigma * sqrt(Tau)
  S * pnorm(d1) - K * exp(-r * Tau) * pnorm(d2)
}

# Stratified RMSE across moneyness bands
# Convention matches PINN code: m = forward/strike, ATM band = (0.97, 1.03)
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

# Per-fold spread-normalized RMSE: residuals expressed in units of half-spread.
# A value of 1 means the typical error equals one half-spread (= "inside the
# spread"); values below 1 mean errors are inside market microstructure noise.
# Floor of $0.05 on half_spread matches src/training.py::validate (spread_floor).
spread_rmse <- function(market, predicted, best_bid, best_offer, spread_floor = 0.05) {
  half_spread <- pmax((best_offer - best_bid) / 2, spread_floor)
  resid_norm  <- (market - predicted) / half_spread
  sqrt(mean(resid_norm^2))
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold runner
# ─────────────────────────────────────────────────────────────────────────────

run_one_fold <- function(fold_dir) {
  fold_name <- basename(fold_dir)
  meta <- jsonlite::read_json(file.path(fold_dir, "meta.json"))

  train <- read_csv(file.path(fold_dir, "train.csv"), show_col_types = FALSE)
  test  <- read_csv(file.path(fold_dir, "test.csv"),  show_col_types = FALSE)

  sigma_fixed <- meta$sigma_fixed
  r_fixed     <- meta$r_fixed

  # Feature engineering — STRICT set only
  train <- train %>% mutate(
    log_volume      = log(pmax(volume, 1)),
    bs_theoretical  = bs_call_price(forward_price, strike_price, time_to_exp,
                                     r_fixed, sigma_fixed),
    residual        = mid_price - bs_theoretical
  )
  test <- test %>% mutate(
    log_volume      = log(pmax(volume, 1)),
    bs_theoretical  = bs_call_price(forward_price, strike_price, time_to_exp,
                                     r_fixed, sigma_fixed)
  )

  # Fit GAM on training residuals
  # te(moneyness, time_to_exp, k=c(8,5))  — tensor product, smile×term-structure
  # ns(log_volume, df=2)                   — natural-spline liquidity control
  # REML auto-tunes the smoothing parameters for each fold
  t0 <- Sys.time()
  gam_fit <- gam(
    residual ~ te(moneyness, time_to_exp, k = c(8, 5)) + ns(log_volume, df = 2),
    data    = train,
    family  = gaussian(),
    method  = "REML"
  )
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  # Predict on test
  pred_resid <- predict(gam_fit, newdata = test, type = "response")
  test$predicted_price <- test$bs_theoretical + pred_resid

  # Metrics
  err     <- test$mid_price - test$predicted_price
  ssq     <- sum(err^2)
  sabs    <- sum(abs(err))
  n_test  <- nrow(test)

  strat <- stratify_metrics(test$mid_price, test$predicted_price, test$moneyness)

  half_spread_test <- pmax((test$best_offer - test$best_bid) / 2, 0.05)
  spread_units     <- spread_rmse(test$mid_price, test$predicted_price,
                                   test$best_bid, test$best_offer)
  ssq_spread       <- sum((err / half_spread_test)^2)

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
    elapsed           = elapsed,
    # Diagnostic fields
    gam_edf_total     = sum(gam_fit$edf),
    gam_sig2          = gam_fit$sig2
  )
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  fold_root  <- if (length(args) >= 1) args[1] else "data/folds"
  output_dir <- if (length(args) >= 2) args[2] else "results/gam_baseline"

  if (!dir.exists(fold_root)) {
    stop("Fold dir not found: ", fold_root,
         "\nRun: python scripts/dump_folds.py")
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  fold_dirs <- list.dirs(fold_root, recursive = FALSE)
  cat(sprintf("Found %d folds in %s\n\n", length(fold_dirs), fold_root))

  fold_results <- list()
  cat(sprintf("%-10s %8s %8s %7s %7s %7s %7s\n",
              "fold", "n_train", "n_test", "rmse", "rmse_otm", "rmse_atm", "rmse_itm"))
  cat(strrep("-", 70), "\n", sep = "")
  for (fd in fold_dirs) {
    res <- run_one_fold(fd)
    fold_results[[res$fold]] <- res
    cat(sprintf("%-10s %8d %8d %7.3f %7.3f %7.3f %7.3f\n",
                res$fold, res$n_train, res$n_test, res$rmse_mkt,
                res$rmse_otm, res$rmse_atm, res$rmse_itm))
  }

  # Pooled summary
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
    val_months         = 1
  )

  out_obj <- list(
    summary  = summary_block,
    folds    = unname(fold_results),
    config   = list(
      method         = "GAM",
      formula        = "residual ~ te(moneyness, time_to_exp, k=c(8,5)) + ns(log_volume, df=2)",
      smoothing      = "REML",
      target         = "BS residual (mid_price - BS(F,K,T,r_fixed,sigma_fixed))",
      features       = "moneyness, time_to_exp, log_volume",
      uses_regime    = FALSE,
      val_months     = 1
    )
  )

  out_path <- file.path(output_dir, "results.json")
  write_json(out_obj, out_path, auto_unbox = TRUE, pretty = TRUE, na = "null")

  cat("\n", strrep("=", 70), "\n", sep = "")
  cat(sprintf("Pooled RMSE: $%.4f   (BS_A1 baseline ≈ $8.795)\n", pooled_rmse_mkt))
  cat(sprintf("Mean fold RMSE: $%.4f   Std: $%.4f\n",
              summary_block$mean_rmse_mkt, summary_block$std_rmse_mkt))
  cat(sprintf("Worst fold: %s ($%.4f)\n",
              summary_block$worst_fold, summary_block$worst_rmse_mkt))
  cat(sprintf("Wrote %s\n", out_path))
}

main()
