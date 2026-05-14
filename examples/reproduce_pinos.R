library("eseis")
library("caTools")

n_cores <- parallel::detectCores() - 1

# Site / model parameters  (Arroyo de los Pinos)
par_model <- list(
  d_s  = 0.009,          # median grain diameter [m]  (D50)
  s_s  = 0.85,           # grain size std (log-normal sigma) — adjust if needed
  r_s  = 2650,           # sediment density [kg/m3]
  w_w  = 10.0,           # channel width [m]
  a_w  = 0.0122,         # channel slope (dimensionless)
  f    = c(30, 80),      # inversion frequency band [Hz]
  r_0  = 17.0,           # source-receiver distance [m]
  f_0  = 1,              # reference frequency [Hz]
  q_0  = 20,             # seismic quality factor
  v_0  = 250,           # phase velocity [m/s] at reference frequency 1Hz
  p_0  = 0.272,          # seismic attenuation exponent  (= a in SeismicParams)
  e_0  = 0.089,          #               (= zeta)
  n_0  = c(0.5, 0.8),    # N11 N12
  res  = 60              # number of frequency bins (matches N_F = 10 in Python)
)


f <- seq(from = par_model$f[1], to = par_model$f[2], length.out = par_model$res)


PSD_PATH  <- "data/test_data/psd_matrix.txt"
FREQ_PATH <- "data/test_data/psd_matrix_freq.txt"
FLOW_PATH <- "data/test_data/flow_depth_m.txt"

psd_mat  <- as.matrix(read.table(PSD_PATH))   # (n_minutes, N_F)
psd_freq <- as.numeric(read.table(FREQ_PATH)[, 1])
h_gauge  <- as.numeric(read.table(FLOW_PATH)[, 1])   # m

n_minutes <- nrow(psd_mat)
time_min  <- seq_len(n_minutes)

# eseis expects (n_freq × n_time), rows = frequency bins, cols = time steps
# Python wrote (n_time × n_freq), so transpose:
psd_agg <- t(psd_mat)   # (N_F × n_minutes)
print(dim(psd_agg))

psd_f_target <- seq(par_model$f[1], par_model$f[2], length.out = par_model$res)

psd_agg <- apply(psd_mat, 1, function(x) {
  spline(x = psd_freq, y = x, xout = psd_f_target)$y
})

# apply over MARGIN=1 (rows = time steps) returns (res × n_minutes)
cat(sprintf("PSD loaded: %d minutes x %d freq bins (%.0f-%.0f Hz)\n",
            n_minutes, nrow(psd_agg),
            par_model$f[1], par_model$f[2]))

# Build Monte Carlo reference catalogue
#     ref_pars_a: combined turbulence + bedload (q_s free)
#     ref_pars_b: turbulence only            (q_s = 0)
ref_pars_a <- fmi_parameters(
  n     = 5000,
  d_s   = par_model$d_s,
  s_s   = par_model$s_s,
  r_s   = par_model$r_s,
  q_s   = c(0, 50) / par_model$r_s,   # volumetric flux range [m²/s]
  h_w   = c(0.015, 2.0),
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

ref_pars_b <- fmi_parameters(
  n     = 1000,
  d_s   = par_model$d_s,
  s_s   = par_model$s_s,
  r_s   = par_model$r_s,
  q_s   = 0,                           # turbulence-only entries
  h_w   = c(0.015, 2.0),
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

ref_pars    <- c(ref_pars_a, ref_pars_b)

cat("Computing reference spectra …\n")
ref_spectra <- fmi_spectra(parameters = ref_pars, n_cores = 4)
cat("Catalogue spectrum range: ", range(sapply(ref_spectra, function(x) x$spectrum)), "\n")

# inversion
cat("Running FMI inversion …\n")
X_emp <- fmi_inversion(reference = ref_spectra,
                       data      = psd_agg,
                       n_cores   = n_cores)

#plot
png("fmi_pinos_eseis_results.png", width = 4800, height = 1800, res = 300)

par(mfcol = c(1, 3))

# Panel 1: RMSE map
# fields::image.plot(X_emp$rmse,
#                    main = "Inversion RMSE",
#                    xlab = "Time (min)",
#                    ylab = "Catalogue entry")

plot(time_min,
     caTools::runmean(X_emp$parameters$h_w, 18),
     type = "l", lwd = 2,
     ylim = c(0, 1.5),
     xlab = "Time (min since flood start)",
     ylab = "Flow depth (m)",
     main = "Flow depth")
lines(time_min, h_gauge, col = 2, lwd = 2)
legend("topright",
       legend = c("FMI estimated", "Gauge"),
       col    = c(1, 2), lwd = 2, bty = "n")

plot(time_min,
     caTools::runmean(X_emp$parameters$q_s, 18) * par_model$r_s,
     type = "l", lwd = 2,
     ylim = c(0, 5),
     xlab = "Time (min since flood start)",
     ylab = expression(q[b] * " · " * rho[s] ~ "(kg m"^{-1} * " s"^{-1} * ")"),
     main = "Bedload flux")

dev.off()
cat("Figure saved to fmi_pinos_eseis_results.png\n")