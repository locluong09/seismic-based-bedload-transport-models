library("eseis")
library("caTools")

n_cores <- parallel::detectCores() - 1

par_model <- list(
  d_s  = 0.009,
  s_s  = 0.85,
  r_s  = 2650,
  w_w  = 10.0,
  a_w  = 0.0122,
  f    = c(30, 80),
  r_0  = 17.0,
  f_0  = 1,
  q_0  = 20,
  v_0  = 250,
  p_0  = 0.272,
  e_0  = 0.089,
  n_0  = c(0.5, 0.8),
  res  = 60
)

# Load PSD 
psd_mat  <- as.matrix(read.table("data/test_data/psd_matrix.txt"))
psd_freq <- as.numeric(read.table("data/test_data/psd_matrix_freq.txt")[, 1])
h_gauge  <- as.numeric(read.table("data/test_data/flow_depth_m.txt")[, 1])

n_minutes <- nrow(psd_mat)
time_min  <- seq_len(n_minutes)

psd_f_target <- seq(par_model$f[1], par_model$f[2], length.out = par_model$res)
psd_agg <- apply(psd_mat, 1, function(x) {
  spline(x = psd_freq, y = x, xout = psd_f_target)$y
})  # (res x n_minutes)

cat(sprintf("PSD: %d min x %d bins (%.0f-%.0f Hz)\n",
            n_minutes, nrow(psd_agg), par_model$f[1], par_model$f[2]))
stopifnot(nrow(psd_agg) == par_model$res, ncol(psd_agg) == n_minutes)

# Turbulence-only catalogue
ref_pars <- fmi_parameters(
  n     = 3000,
  d_s   = par_model$d_s,
  s_s   = par_model$s_s,
  r_s   = par_model$r_s,
  q_s   = 0,                  # turbulence only
  h_w   = c(0.015, 1.20),     # realistic Pinos range
  w_w   = par_model$w_w,
  a_w   = par_model$a_w,
  f_min = par_model$f[1],
  f_max = par_model$f[2],
  r_0   = par_model$r_0,
  f_0   = par_model$f_0,
  q_0   = par_model$q_0,
  v_0   = par_model$v_0,
  p_0   = par_model$p_0,
  e_0   = par_model$e_0,
  n_0_a = par_model$n_0[1],
  n_0_b = par_model$n_0[2],
  res   = par_model$res
)

cat("Computing reference spectra...\n")
ref_spectra <- fmi_spectra(parameters = ref_pars, n_cores = n_cores)
nan_count <- sum(sapply(ref_spectra, function(x) any(is.nan(x$spectrum))))
cat(sprintf("NaN entries: %d / %d\n", nan_count, length(ref_spectra)))
cat("Spectrum range:", range(sapply(ref_spectra, function(x) x$spectrum), na.rm=TRUE), "\n")

# Inversion 
cat("Running inversion...\n")
X_emp <- fmi_inversion(reference = ref_spectra,
                       data      = psd_agg,
                       n_cores   = n_cores)

#Plot 
png("pinos_turbulence_R.png", width = 3200, height = 1600, res = 300)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

plot(time_min,
     caTools::runmean(X_emp$parameters$h_w, 18),
     type = "l", lwd = 2,
     ylim = c(0, 1.5),
     xlab = "Time (min)", ylab = "Flow depth (m)",
     main = "Flow depth: FMI vs Gauge")
lines(time_min, h_gauge, col = 2, lwd = 2)
legend("topright", legend = c("FMI (turbulence)", "Gauge"),
       col = c(1, 2), lwd = 2, bty = "n")

plot(time_min, h_gauge,
     type = "l", lwd = 2, col = 2,
     xlab = "Time (min)", ylab = "Flow depth (m)",
     main = "Gauge only")

dev.off()
cat("Saved fmi_pinos_turbulence.png\n")