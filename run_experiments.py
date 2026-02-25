"""
Run wind-speed and simulation experiments for IDR PIT values.
"""

import numpy as np
import pandas as pd
import xarray as xr
from isodisreg import idr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dask.diagnostics import ProgressBar
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from crpsmixture import smooth_crps
from helper_functions import tune_gaussian_bandwidth_onefit, pit_gaussian_kernel

# Overall settings
DIGITS = 8
MAX_WORKERS = 40
BASE_SEED = 42
N_TUNE = 250

# Wind speed settings
VAR = "10m_wind_speed"
LEAD_TIME = np.timedelta64(3, "D")
TEST_YEAR = 2020

# Simulation settings
N = 5000
TRAIN_FRAC = 0.8
NUM_ITERATIONS = 2000
COL = "fore"
N_TRAIN = int(N * TRAIN_FRAC)
N_TEST = N - N_TRAIN

# Worker globals
G_WS = {}
G_S  = {}

# Wind speed workers
def init_ws(base_seed_, X_train_, Y_train_, var_, digits_, X_test_, Y_test_, h_):
    G_WS["base_seed"] = base_seed_
    G_WS["X_train"] = X_train_
    G_WS["Y_train"] = Y_train_
    G_WS["var"] = var_
    G_WS["digits"] = digits_
    G_WS["X_test"] = X_test_
    G_WS["Y_test"] = Y_test_
    G_WS["h"] = h_

def work_h_ws(idx):
    i, j = idx
    x_train = G_WS["X_train"][:, i, j]
    y_train = G_WS["Y_train"][:, i, j]

    fit = idr(y_train, pd.DataFrame({G_WS["var"]: x_train}))
    preds_train = fit.predict(pd.DataFrame({G_WS["var"]: x_train}), digits=G_WS["digits"])

    h, ll_train = tune_gaussian_bandwidth_onefit(preds_train, y_train)
    return h

def work_ws(idx):
    i, j = idx
    x_train = G_WS["X_train"][:, i, j]
    y_train = G_WS["Y_train"][:, i, j]
    x_test  = G_WS["X_test"][:, i, j]
    y_test  = G_WS["Y_test"][:, i, j]

    fit = idr(y_train, pd.DataFrame({G_WS["var"]: x_train}))
    preds_test = fit.predict(pd.DataFrame({G_WS["var"]: x_test}), digits=G_WS["digits"])

    pit = preds_test.pit(y_test, seed=G_WS["base_seed"] + 10_000 * i + j)
    pit_s = pit_gaussian_kernel(preds_test, y_test, G_WS["h"])

    c = np.mean(preds_test.crps(y_test))
    cs = smooth_crps(preds_test, y_test, G_WS["h"], df=None)

    return i, j, pit, pit_s, c, cs

# Simulation workers
def make_data(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, size=N)
    y = rng.gamma(
        shape=np.sqrt(x),
        scale=np.minimum(np.maximum(x, 2.0), 8.0),
        size=N,
    )
    return x[:N_TRAIN], y[:N_TRAIN], x[N_TRAIN:], y[N_TRAIN:]

def init_sim(base_seed_, digits_, col_, h_):
    G_S["base_seed"] = base_seed_
    G_S["digits"] = digits_
    G_S["col"] = col_
    G_S["h"] = h_ 

def work_h_sim(it):
    seed = G_S["base_seed"] + it
    x_train, Y_train, _, _ = make_data(seed)

    fit = idr(Y_train, pd.DataFrame({G_S["col"]: x_train}))
    preds_train = fit.predict(pd.DataFrame({G_S["col"]: x_train}), digits=G_S["digits"])

    h, ll_train = tune_gaussian_bandwidth_onefit(preds_train, Y_train)
    return h

def work_sim(it):
    seed = G_S["base_seed"] + 100_000 + it
    x_train, Y_train, X_test, Y_test = make_data(seed)

    fit = idr(Y_train, pd.DataFrame({G_S["col"]: x_train}))
    preds_test = fit.predict(pd.DataFrame({G_S["col"]: X_test}), digits=G_S["digits"])

    pit = preds_test.pit(Y_test, seed=G_S["base_seed"] + 200_000 + it)
    pit_s = pit_gaussian_kernel(preds_test, Y_test, G_S["h"])

    c = np.mean(preds_test.crps(Y_test))
    cs = smooth_crps(preds_test, Y_test, G_S["h"], df=None)

    return it, pit, pit_s, c, cs

# Plotting helpers
def add_fraction_guides(ax, x_values, x_shift=-0.021, y_frac=0.94):
    y_top = ax.get_ylim()[1]
    for f in x_values:
        x = float(f)
        ax.axvline(x, linestyle="--", color="tab:red", linewidth=1.5, zorder=5)
        ax.text(x + x_shift, y_top * y_frac, str(f), color="tab:red", ha="center", va="bottom", fontsize=18)

def pit_panel(ax, data, title, x_values, bins1=17, bins2=100):
    ax.hist(data, bins=bins1, density=True, color="dimgray", alpha=1, edgecolor="none", range=(0,1), zorder=1)
    ax.hist(data, bins=bins2, density=True, color="tab:cyan", alpha=0.5, edgecolor="none", range=(0,1), zorder=3)

    h1, e1 = np.histogram(data, bins=bins1, density=True, range=(0,1))
    ax.stairs(h1, e1, color="black", linewidth=1.0, zorder=2)

    h2, e2 = np.histogram(data, bins=bins2, density=True, range=(0,1))
    ax.stairs(h2, e2, color="black", linewidth=1.0, zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.9)
    ax.set_xlabel("PIT value")
    ax.set_title(title, pad=9)
    add_fraction_guides(ax, x_values)

def main():
    ctx = multiprocessing.get_context("fork")

    # Wind speed
    with ProgressBar():
        predictions = xr.open_zarr(
            store="gs://weatherbench2/datasets/pangu/2018-2022_0012_64x32_equiangular_conservative.zarr",
            decode_timedelta=True,
            storage_options={"token": "anon"},
        ).sel(prediction_timedelta=LEAD_TIME)[[VAR]].load()

    valid_time = predictions.time + predictions.prediction_timedelta

    with ProgressBar():
        targets = xr.open_zarr(
            store="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
            storage_options={"token": "anon"},
        ).sel(time=valid_time)[[VAR]].load()

    X = predictions[VAR].values
    Y = targets[VAR].values

    years = pd.DatetimeIndex(predictions.time.values).year.values
    train = years != TEST_YEAR
    test = years == TEST_YEAR

    X_train, Y_train = X[train], Y[train]
    X_test, Y_test = X[test], Y[test]

    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    
    # Tune h_ws on random subset of grid points
    rng = np.random.default_rng(BASE_SEED)

    nlon, nlat = X_train.shape[1], X_train.shape[2]
    indices = [(i, j) for i in range(nlon) for j in range(nlat)]
    sample_indices = [indices[k] for k in rng.choice(len(indices), size=N_TUNE, replace=False)]

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=init_ws,
        initargs=(BASE_SEED, X_train, Y_train, VAR, DIGITS, None, None, None),
        mp_context=ctx,
    ) as ex:
        hs = list(
            tqdm(
                ex.map(work_h_ws, sample_indices, chunksize=1),
                total=len(sample_indices),
                desc="Tuning h (wind speed)",
            )
        )

    hs = np.asarray(hs, dtype=float)
    h_ws = np.nanmedian(hs)
    print("Wind speed median h:", h_ws)
    
    # Full grid evaluation
    pit_ws = np.empty_like(X_test, dtype=np.float64)
    pit_ws_smooth = np.empty_like(X_test, dtype=np.float64)

    crps_ws = np.empty((nlon, nlat), dtype=np.float64)
    crps_ws_smooth = np.empty((nlon, nlat), dtype=np.float64)

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=init_ws,
        initargs=(BASE_SEED, X_train, Y_train, VAR, DIGITS, X_test, Y_test, h_ws),
        mp_context=ctx,
    ) as ex:
        for i, j, p, ps, c, cs in tqdm(
            ex.map(work_ws, indices, chunksize=1),
            total=len(indices),
            desc="WS grid",
        ):
            pit_ws[:, i, j] = p
            pit_ws_smooth[:, i, j] = ps
            crps_ws[i, j] = c
            crps_ws_smooth[i, j] = cs

    print("Mean CRPS (wind speed, basic):   ", np.mean(crps_ws))
    print("Mean CRPS (wind speed, smooth):", np.mean(crps_ws_smooth))

    # Simulation
    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=init_sim,
        initargs=(BASE_SEED, DIGITS, COL, None),
        mp_context=ctx,
    ) as ex:
        hs = list(
            tqdm(
                ex.map(work_h_sim, range(N_TUNE), chunksize=1),
                total=N_TUNE,
                desc="Tuning h (simulation)",
            )
        )

    hs = np.asarray(hs, dtype=float)
    h_sim = np.nanmedian(hs)
    print("Simulation median h:", h_sim)

    pit_sim = np.empty((NUM_ITERATIONS, N_TEST), dtype=np.float64)
    pit_sim_smooth = np.empty((NUM_ITERATIONS, N_TEST), dtype=np.float64)
    crps_sim = np.empty(NUM_ITERATIONS, dtype=np.float64)
    crps_sim_smooth = np.empty(NUM_ITERATIONS, dtype=np.float64)

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=init_sim,
        initargs=(BASE_SEED, DIGITS, COL, h_sim),
        mp_context=ctx,
    ) as ex:
        for it, p, ps, c, cs in tqdm(
            ex.map(work_sim, range(NUM_ITERATIONS), chunksize=1),
            total=NUM_ITERATIONS,
            desc="Simulation",
        ):
            pit_sim[it, :] = p
            pit_sim_smooth[it, :] = ps
            crps_sim[it] = c
            crps_sim_smooth[it] = cs

    print("Mean CRPS (simulation, IDR):   ", np.mean(crps_sim))
    print("Mean CRPS (simulation, smooth):", np.mean(crps_sim_smooth))

    # Plots
    plt.rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.labelsize": 22,
        "axes.titlesize": 26,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })
    
    x_values = [
        Fraction(1,2),
        Fraction(1,3), Fraction(2,3),
        Fraction(1,4), Fraction(3,4), 
        Fraction(1,5), Fraction(2,5), Fraction(3,5),  Fraction(4,5)
    ]
    
    # Figure 1: Basic PIT
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    pit_panel(axes[0], pit_ws.ravel(), "Wind Speed", x_values)
    pit_panel(axes[1], pit_sim.ravel(), "Simulation", x_values)
    axes[0].set_ylabel("Density")
    plt.tight_layout()
    plt.savefig("plots/pit.png", dpi=300, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close()
    
    # Figure 2: Unique PIT frequencies (simulation)
    s = pd.Series(pit_sim.ravel())
    value_counts = s.value_counts()

    tab = pd.DataFrame(
        {"Value": value_counts.index.to_numpy(), "Counts": value_counts.values}
    ).sort_values("Value")

    plt.figure(figsize=(9, 6))
    plt.plot(tab["Value"], tab["Counts"], ".", color="black", markersize=6)
    plt.xlabel("PIT value")
    plt.ylabel("Frequency (thousands)")
    plt.ylim(0, None)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v/1000:.0f}"))
    plt.savefig("plots/thomae.png", dpi=300, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close()
    
    # Figure 3: Smooth PIT
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    pit_panel(axes[0], pit_ws_smooth.ravel(), "Wind Speed", x_values)
    pit_panel(axes[1], pit_sim_smooth.ravel(), "Simulation", x_values)
    axes[0].set_ylabel("Density")
    plt.tight_layout()
    plt.savefig("plots/pit_smooth.png", dpi=300, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close()
    
if __name__ == "__main__":
    main()