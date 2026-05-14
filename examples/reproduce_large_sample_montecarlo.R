## load packages
library("eseis")
library("caTools")

n_cores <- parallel::detectCores() - 1
## define model parameters for inversion
par_model <- list(d_s = 0.01,
s_s = 1.35,
r_s = 2650,
w_w = 5,
a_w = 0.0075,
f = c(10, 70),
r_0 = 5.5,
f_0 = 1,
q_0 = 16.77,
v_0 = 859,
p_0 = 0.62,
e_0 = 0.07,
n_0 = c(0.5, 0.8),
res = 100)
## define frequency vector
f <- seq(from = par_model$f[1],
to = par_model$f[2],
length.out = par_model$res)
## load empiric PSD
psd <- read.table(file = "/Users/locluong/Documents/FluvialSedimentTransport/dietze/supplementary_material_revised/spectrogram_nahal_eshtemoa_2016-02-22.txt",
sep = "\t")
## convert column names to time stamps
psd_t <- as.POSIXct(x = as.numeric(substr(x = colnames(psd),
start = 2,
stop = 12)),
origin = "1970-01-01",tz = "UTC")
## convert row names to frequency vector
psd_f <- as.numeric(rownames(psd))
## truncate spectrogram by frequency
i_cut <- psd_f >= par_model$f[1] & psd_f <= par_model$f[2]
psd_cut <- t(psd[i_cut,])
psd_f_cut <- psd_f[i_cut]
## aggregate spectrogram by frequency
psd_f_agg <- seq(from = 10,
to = 70,
length.out = 100)
psd_t_agg <- psd_t
psd_agg <- t(apply(X = psd_cut,
MARGIN = 1,
FUN = function(x, psd_f_cut, psd_f_agg) {
spline(x = psd_f_cut,
y = x,
xout = psd_f_agg)$y
}, psd_f_cut, psd_f_agg))
psd_agg <- t(psd_agg)
## load observatory sensor data set
data_flood <- read.table(file = "/Users/locluong/Documents/FluvialSedimentTransport/dietze/supplementary_material_revised/data_nahal_eshtemoa_flood_2016-02-22.txt",
sep = "\t"
,
header = TRUE)

## convert time string to POSIX format
data_flood$time <- as.POSIXct(data_flood$time)
## truncate data set to period of interest
data_flood_cut <- data_flood[1:300,]
## isolate parameters of interest
t_emp <- data_flood_cut$time
h_emp <- data_flood_cut$h_w
q_emp <- data_flood_cut$q_s_median
## plot time series of water level
plot(x = t_emp, y = h_emp, type = "l")

## plot time series of average bedload flux
plot(x = t_emp, y = q_emp, type = "l")

## plot original spectrogram
fields::image.plot(x = psd_t, y = psd_f, z = t(psd))

## plot truncated and aggregated spectrogram
fields::image.plot(x = psd_t_agg,
y = psd_f_agg,
z = t(psd_agg))


ref_pars_a <- fmi_parameters(n       = 50000,
                             d_s     = par_model$d_s,
                             s_s     = par_model$s_s,
                             r_s     = par_model$r_s,
                             q_s     = c(0, 15) / 2650,
                             h_w     = c(0.015, 1.20),
                             w_w     = par_model$w_w,
                             a_w     = par_model$a_w,
                             f_min   = par_model$f[1],
                             f_max   = par_model$f[2],
                             r_0     = par_model$r_0,
                             f_0     = par_model$f_0,
                             q_0     = par_model$q_0,
                             v_0     = par_model$v_0,
                             p_0     = par_model$p_0,
                             e_0     = par_model$e_0,
                             n_0_a   = par_model$n_0[1],
                             n_0_b   = par_model$n_0[2],
                             res     = par_model$res)

ref_pars_b <- fmi_parameters(n       = 10000,
                             d_s     = par_model$d_s,
                             s_s     = par_model$s_s,
                             r_s     = par_model$r_s,
                             q_s     = 0,
                             h_w     = c(0.015, 1.20),
                             w_w     = par_model$w_w,
                             a_w     = par_model$a_w,
                             f_min   = par_model$f[1],
                             f_max   = par_model$f[2],
                             r_0     = par_model$r_0,
                             f_0     = par_model$f_0,
                             q_0     = par_model$q_0,
                             v_0     = par_model$v_0,
                             p_0     = par_model$p_0,
                             e_0     = par_model$e_0,
                             n_0_a   = par_model$n_0[1],
                             n_0_b   = par_model$n_0[2],
                             res     = par_model$res)

ref_pars    <- c(ref_pars_a, ref_pars_b)
ref_spectra <- fmi_spectra(parameters = ref_pars, n_cores = n_cores)


## invert empirical data set
X_emp <- fmi_inversion(reference = ref_spectra,
data = psd_agg,
n_cores = 28)

png("fmi_eshtemoa_results.png", width=1600, height=600, res = 300)

par(mfcol = c(1, 3))
fields::image.plot(X_emp$rmse)
plot(psd_t, caTools::runmean(X_emp$parameters$h_w, 180),
type = "l",
ylim = c(0, 1.2))
lines(t_emp + 3600, h_emp, col = 2)
plot(psd_t, caTools::runmean(X_emp$parameters$q_s, 180) * 2650,
type = "l",
ylim = c(0, 5))
lines(t_emp + 3600, q_emp, col = 2)

dev.off()  
