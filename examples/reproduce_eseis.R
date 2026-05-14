library("eseis")
library("caTools")

n_cores <- parallel::detectCores() - 1

par_model <- list(d_s = 0.01,
                  s_s = 1.35,
                  r_s = 2650,
                  w_w = 5,
                  a_w = 0.0075,
                  f   = c(10, 70),
                  r_0 = 5.5,
                  f_0 = 1,
                  q_0 = 16.77,
                  v_0 = 859,
                  p_0 = 0.62,
                  e_0 = 0.07,
                  n_0 = c(0.5, 0.8),
                  res = 100)

f <- seq(from = par_model$f[1], to = par_model$f[2], length.out = par_model$res)

psd <- read.table(file = "/Users/locluong/Documents/FluvialSedimentTransport/dietze/supplementary_material_revised/spectrogram_nahal_eshtemoa_2016-02-22.txt",
                  sep = "\t")

psd_t <- as.POSIXct(x      = as.numeric(substr(x = colnames(psd), start = 2, stop = 12)),
                    origin = "1970-01-01",
                    tz     = "UTC")                    # ✓ already correct

psd_f <- as.numeric(rownames(psd))

i_cut     <- psd_f >= par_model$f[1] & psd_f <= par_model$f[2]
psd_cut   <- t(psd[i_cut,])
psd_f_cut <- psd_f[i_cut]

psd_f_agg <- seq(from = 10, to = 70, length.out = 100)
psd_t_agg <- psd_t
psd_agg   <- t(apply(X      = psd_cut,
                     MARGIN = 1,
                     FUN    = function(x, psd_f_cut, psd_f_agg) {
                       spline(x = psd_f_cut, y = x, xout = psd_f_agg)$y
                     }, psd_f_cut, psd_f_agg))
psd_agg <- t(psd_agg)
print(psd_agg[1:5, 1:5])
print(dim(psd_agg))
data_flood <- read.table(file   = "/Users/locluong/Documents/FluvialSedimentTransport/dietze/supplementary_material_revised/data_nahal_eshtemoa_flood_2016-02-22.txt",
                         sep    = "\t",
                         header = TRUE)

data_flood$time <- as.POSIXct(data_flood$time, tz = "UTC")  # ✓ fixed

data_flood_cut <- data_flood[1:300,]
t_emp <- data_flood_cut$time
h_emp <- data_flood_cut$h_w
q_emp <- data_flood_cut$q_s_median

ref_pars_a <- fmi_parameters(n     = 5000,
                             d_s   = par_model$d_s,
                             s_s   = par_model$s_s,
                             r_s   = par_model$r_s,
                             q_s   = c(0, 15) / 2650,
                             h_w   = c(0.015, 1.20),
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
                             res   = par_model$res)

ref_pars_b <- fmi_parameters(n     = 1000,
                             d_s   = par_model$d_s,
                             s_s   = par_model$s_s,
                             r_s   = par_model$r_s,
                             q_s   = 0,
                             h_w   = c(0.015, 1.20),
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
                             res   = par_model$res)

ref_pars    <- c(ref_pars_a, ref_pars_b)
ref_spectra <- fmi_spectra(parameters = ref_pars, n_cores = n_cores)  # ✓
cat("Catalogue spectrum range: ", range(sapply(ref_spectra, function(x) x$spectrum)), "\n")

X_emp <- fmi_inversion(reference = ref_spectra,
                       data      = psd_agg,
                       n_cores   = n_cores)            # ✓ fixed

png("fmi_eshtemoa_results.png", width = 4800, height = 1800, res = 300)  # ✓ fixed

par(mfcol = c(1, 3))
fields::image.plot(X_emp$rmse)

plot(psd_t, caTools::runmean(X_emp$parameters$h_w, 18),
     type = "l", ylim = c(0, 1.2))
lines(t_emp, h_emp, col = 2)

plot(psd_t, caTools::runmean(X_emp$parameters$q_s, 18) * 2650,
     type = "l", ylim = c(0, 5))
lines(t_emp, q_emp, col = 2)

dev.off()