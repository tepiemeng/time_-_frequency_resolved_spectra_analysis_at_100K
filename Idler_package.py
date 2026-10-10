"""
Pump-probe data analysis utilities.

This module contains functions for:

    - Loading and compiling MIR pump-probe data
    - Processing fluence-dependent measurements
    - Calculating scan-to-scan standard deviations
    - Loading polarization-dependent HDF5 measurements
    - Fitting transient reflectivity dynamics
    - Extracting coherent oscillations using Fourier transforms
    - Constructing phonon amplitude and phase maps
    - Extracting lmfit fitting parameters
    - Calculating pump fluence
    - Smoothing 1D and 2D datasets
    - Calculating the polar form of an ellipse

The public function names, argument order, default arguments, and return
structures are retained for compatibility with existing analysis notebooks.
"""


# ============================================================================
# Imports
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py

from scipy.special import gamma, gammainc, exp1, erf
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.fft import fft, ifft, fftshift, fftfreq
from scipy.ndimage import gaussian_filter

import lmfit
from lmfit import minimize, Parameters


# ============================================================================
# Dataset key generation
# ============================================================================

def keys(WL, label):
    """
    Generate sorted dictionary keys for a set of measurement conditions.

    Parameters
    ----------
    WL : array-like
        Numerical values identifying the datasets, such as pump frequency,
        wavelength, or photon energy.
    label : str
        Text appended to each numerical value.

    Returns
    -------
    key : list of str
        Sorted keys with the format ``"<value>_<label>"``.

    Examples
    --------
    >>> keys([20, 10, 15], 'THz')
    ['10_THz', '15_THz', '20_THz']
    """
    WL = np.array(WL)
    idx = np.argsort(WL)
    WL = WL[idx]

    key = []

    for i in range(len(WL)):
        key.append(str(WL[i]) + '_' + label)

    return key


# ============================================================================
# CSV pump-probe data loading
# ============================================================================

def get_data(path, m_name, j, k, PP, n_scans, n_pump):
    """
    Load one CSV pump-probe file and separate it into individual scans.

    Parameters
    ----------
    path : str
        Directory containing the data file.
    m_name : str
        Base measurement filename.
    j : int
        Angle index used in the filename.
    k : int
        Scan or acquisition-cycle index used in the filename.
    PP : array-like
        Legacy argument retained for compatibility. It is overwritten
        internally and is not otherwise used.
    n_scans : int
        Number of repeated scans stored in the file.
    n_pump : int
        Number of pump-delay points in each scan.

    Returns
    -------
    PPout : ndarray
        Pump-probe signal with shape ``(n_pump, n_scans)``.
    pump_delay : ndarray
        Pump-probe delay axis.

    Notes
    -----
    The function assumes:

    - column 5 contains the pump-delay coordinate
    - column 7 contains the measured signal
    - each scan contains exactly ``n_pump`` consecutive rows

    The delay axis is multiplied by -1 to preserve the experimental
    sign convention used in the original analysis.
    """
    PP = pd.read_csv(
        path + m_name + str(j) + '_scan' + str(k) + '.csv',
        skiprows=1,
        header=0
    )

    PPv = np.array(PP.values)

    pump_delay = np.linspace(np.min(PPv[:, 5]), np.max(PPv[:, 5]), n_pump)

    # Retained from the original acquisition code.
    pump_pos = np.linspace(np.min(PPv[:, 5]), np.max(PPv[:, 4]), n_pump)

    pump_delay = pump_delay * -1

    PPout = np.zeros((n_pump, n_scans))

    for i in range(n_scans):
        PPout[:, i] = PPv[n_pump * i:n_pump * (i + 1), 7]

    return PPout, pump_delay


def acquire_data(path, date, time, n_angles, n_cycles, I0, sample):
    """
    Load, average, normalize, and time-align one pump-probe measurement.

    Parameters
    ----------
    path : str
        Directory containing the measurement files.
    date : int or str
        Date identifier appearing in the filename.
    time : int or str
        Time identifier appearing in the filename.
    n_angles : int
        Number of measured angle settings.
    n_cycles : int
        Number of acquisition cycles.
    I0 : float
        Signal-normalization factor.
    sample : str
        Sample identifier appearing in the filename.

    Returns
    -------
    PPmean : ndarray
        Mean baseline-corrected transient for angle index 0.
    PPN : ndarray
        Normalized transient.
    pump_delay : ndarray
        Delay axis shifted so that the determined time zero is at ``t = 0``.

    Notes
    -----
    Files are assumed to follow the naming convention::

        <sample>_MIR-Idler_PP<date>_<time>_angle<angle>_scan<cycle>.csv

    The fifth point from the end of the averaged transient is used as the
    baseline reference, preserving the original analysis convention.
    """
    name = sample + '_MIR-Idler_PP' + str(date) + '_' + str(time) + '_angle'

    data_set = pd.read_csv(path + name + '0_scan0.csv', skiprows=1, header=0)

    n_pump = int(np.max(np.array(data_set.values)[:, 0])) + 1
    n_scans = int(np.max(np.array(data_set.values)[:, 1])) + 1

    PP = np.zeros((n_pump, n_angles, n_scans * n_cycles))

    for k in range(n_cycles):
        for i in range(n_angles):
            PP[:, i, k * n_scans:(k + 1) * n_scans], pump_delay = get_data(
                path, name, i, k, PP, n_scans, n_pump
            )

    PP /= -I0

    print(PP.shape)

    t0 = pump_delay[np.argmin(PP[:, :, :].mean(axis=-1))]
    pump_delay -= t0

    PPmean = PP[:, 0, :].mean(axis=-1) - PP[:, 0, :].mean(axis=-1)[-5]
    PPN = -PPmean / PPmean.min()

    return PPmean, PPN, pump_delay


def acquire_data_fluence(path, date, time, n_angles, n_cycles, I0, sample):
    """
    Load raw pump-probe scans for a fluence-dependent measurement.

    Unlike :func:`acquire_data`, this function does not shift the pump-delay
    axis and returns the full scan-resolved pump-probe array.

    Parameters
    ----------
    path : str
        Directory containing the measurement files.
    date : int or str
        Date identifier.
    time : int or str
        Time identifier.
    n_angles : int
        Number of measured angle settings.
    n_cycles : int
        Number of acquisition cycles.
    I0 : float
        Signal-normalization factor.
    sample : str
        Sample identifier.

    Returns
    -------
    PP : ndarray
        Full pump-probe dataset with shape
        ``(n_pump, n_angles, n_scans * n_cycles)``.
    pump_delay : ndarray
        Unshifted pump-probe delay axis.
    """
    name = sample + '_MIR-Idler_PP' + str(date) + '_' + str(time) + '_angle'

    data_set = pd.read_csv(path + name + '0_scan0.csv', skiprows=1, header=0)

    n_pump = int(np.max(np.array(data_set.values)[:, 0])) + 1
    n_scans = int(np.max(np.array(data_set.values)[:, 1])) + 1

    PP = np.zeros((n_pump, n_angles, n_scans * n_cycles))

    print(n_pump, n_angles, n_scans, n_cycles)

    for k in range(n_cycles):
        for i in range(n_angles):
            PP[:, i, k * n_scans:(k + 1) * n_scans], pump_delay = get_data(
                path, name, i, k, PP, n_scans, n_pump
            )

    PP /= -I0

    print(PP.shape)

    # Retained calculations from the original code.
    PPmean = PP[:, 0, :].mean(axis=-1) - PP[:, 0, :].mean(axis=-1)[-5]
    PPN = -PPmean / PPmean.min()

    return PP, pump_delay


def compile_data(root, date, time, n_cycles, n_angles, I0, WL, key, sample):
    """
    Compile several pump-probe measurements into dictionaries.

    Measurements and their metadata are sorted according to ``WL``.

    Parameters
    ----------
    root : str
        Root data directory.
    date : array-like
        Measurement dates.
    time : array-like
        Measurement times.
    n_cycles : array-like
        Number of acquisition cycles for each measurement.
    n_angles : array-like
        Number of angle settings for each measurement.
    I0 : array-like
        Normalization factors.
    WL : array-like
        Numerical variable used to sort the datasets.
    key : list of str
        Dataset dictionary keys.
    sample : str
        Sample identifier.

    Returns
    -------
    PPm : dict
        Mean baseline-corrected pump-probe signals.
    P_delay : dict
        Pump-probe delay axes.
    PPN : dict
        Normalized pump-probe signals.
    WL : ndarray
        Sorted measurement variable.
    """
    WL = np.array(WL)
    date = np.array(date)
    time = np.array(time)
    n_angles = np.array(n_angles)
    n_cycles = np.array(n_cycles)
    I0 = np.array(I0)

    idx = np.argsort(WL)

    WL = WL[idx]
    date = date[idx]
    time = time[idx]
    I0 = I0[idx]
    n_angles = n_angles[idx]
    n_cycles = n_cycles[idx]

    path = []

    for day in date:
        path.append(root + str(day) + '/')

    PPm = {}
    P_delay = {}
    PPN = {}

    for i in range(len(WL)):
        a = acquire_data(
            path[i], date[i], str(time[i]), n_angles[i],
            n_cycles[i], I0[i], sample
        )

        PPm[key[i]] = a[0]
        P_delay[key[i]] = a[2]
        PPN[key[i]] = a[1]

    return PPm, P_delay, PPN, WL


# ============================================================================
# Measurement uncertainty
# ============================================================================

def data_std(path, date, time, n_angles, n_cycles, I0, sample):
    """
    Calculate the scan-to-scan standard deviation of a pump-probe dataset.

    Parameters
    ----------
    path : str
        Directory containing the data.
    date : int or str
        Measurement date.
    time : int or str
        Measurement time.
    n_angles : int
        Number of angle settings.
    n_cycles : int
        Number of acquisition cycles.
    I0 : float
        Signal-normalization factor.
    sample : str
        Sample identifier.

    Returns
    -------
    PPstd : ndarray
        Standard deviation across scans for angle index 0.
    """
    name = sample + '_MIR-Idler_PP' + str(date) + '_' + str(time) + '_angle'

    data_set = pd.read_csv(path + name + '0_scan0.csv', skiprows=1, header=0)

    n_pump = int(np.max(np.array(data_set.values)[:, 0])) + 1
    n_scans = int(np.max(np.array(data_set.values)[:, 1])) + 1

    PP = np.zeros((n_pump, n_angles, n_scans * n_cycles))

    for k in range(n_cycles):
        for i in range(n_angles):
            PP[:, i, k * n_scans:(k + 1) * n_scans], pump_delay = get_data(
                path, name, i, k, PP, n_scans, n_pump
            )

    PP /= -I0

    print(PP.shape)

    t0 = pump_delay[np.argmin(PP[:, :, :].mean(axis=-1))]
    pump_delay -= t0

    PPmean = PP[:, 0, :].mean(axis=-1) - PP[:, 0, :].mean(axis=-1)[-5]
    PPstd = PP[:, 0, :].std(axis=-1)

    # Retained from the original routine even though it is not returned.
    PPN = -PPmean / PPmean.min()

    return PPstd


def compile_std(root, date, time, n_cycles, n_angles, I0, WL, key, sample):
    """
    Compile scan-to-scan standard deviations for multiple measurements.

    Returns
    -------
    std : dict
        Dictionary of standard-deviation traces indexed by ``key``.
    """
    WL = np.array(WL)
    date = np.array(date)
    time = np.array(time)
    n_angles = np.array(n_angles)
    n_cycles = np.array(n_cycles)
    I0 = np.array(I0)

    idx = np.argsort(WL)

    WL = WL[idx]
    date = date[idx]
    time = time[idx]
    I0 = I0[idx]
    n_angles = n_angles[idx]
    n_cycles = n_cycles[idx]

    path = []

    for day in date:
        path.append(root + str(day) + '/')

    std = {}

    for i in range(len(WL)):
        a = data_std(
            path[i], date[i], str(time[i]), n_angles[i],
            n_cycles[i], I0[i], sample
        )

        std[key[i]] = a

    return std


# ============================================================================
# Fluence-dependent data compilation
# ============================================================================

def compile_data_fluence(root, date, time, n_cycles, n_angles, I0, WL, key, sample):
    """
    Compile scan-resolved fluence-dependent pump-probe measurements.

    Parameters
    ----------
    root, date, time, n_cycles, n_angles, I0, WL, key, sample
        Same conventions as :func:`compile_data`.

    Returns
    -------
    PPm : dict
        Full scan-resolved pump-probe arrays.
    P_delay : dict
        Pump-probe delay axes.
    PPN : dict
        Normalized mean traces calculated from angle index 0.
    WL : ndarray
        Sorted measurement variable.

    Notes
    -----
    The original implementation attempted to access ``a[2]`` even though
    :func:`acquire_data_fluence` returns only two values. This implementation
    preserves the public return structure while correctly constructing PPN
    from the returned pump-probe array.
    """
    WL = np.array(WL)
    date = np.array(date)
    time = np.array(time)
    n_angles = np.array(n_angles)
    n_cycles = np.array(n_cycles)
    I0 = np.array(I0)

    idx = np.argsort(WL)

    WL = WL[idx]
    date = date[idx]
    time = time[idx]
    I0 = I0[idx]
    n_angles = n_angles[idx]
    n_cycles = n_cycles[idx]

    path = []

    for day in date:
        path.append(root + str(day) + '/')

    PPm = {}
    P_delay = {}
    PPN = {}

    for i in range(len(WL)):
        a = acquire_data_fluence(
            path[i], date[i], str(time[i]), n_angles[i],
            n_cycles[i], I0[i], sample
        )

        PP = a[0]
        pump_delay = a[1]

        PPmean = PP[:, 0, :].mean(axis=-1) - PP[:, 0, :].mean(axis=-1)[-5]
        normalized = -PPmean / PPmean.min()

        PPm[key[i]] = PP
        P_delay[key[i]] = pump_delay
        PPN[key[i]] = normalized

    return PPm, P_delay, PPN, WL


# ============================================================================
# Polarization-dependent HDF5 data
# ============================================================================

def read_datasets(filename):
    """
    Read every top-level dataset from an HDF5 file.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.

    Returns
    -------
    dict
        Dictionary mapping each dataset name to a NumPy array.
    """
    with h5py.File(filename, 'r') as f:
        return {name: f[name][:] for name in f.keys()}


def acquire_data_pol(filenames, R_sample):
    """
    Load and average polarization-dependent pump-probe HDF5 measurements.

    Parameters
    ----------
    filenames : list of str
        HDF5 files to average.
    R_sample : array-like
        Equilibrium reflectivity values corresponding to each angle.

    Returns
    -------
    time : ndarray
        Pump-probe delay axis in ps.
    angles : ndarray
        Polarization angles in degrees.
    PP_data_pol : ndarray
        Baseline-subtracted and reflectivity-normalized pump-probe data.

    Raises
    ------
    ValueError
        If the supplied HDF5 files do not contain the same datasets.
    """
    datasets_list = [read_datasets(filename) for filename in filenames]

    for i in range(len(filenames)):
        print(
            'datasets_list[' + str(i) + '].keys():',
            datasets_list[i].keys()
        )

    dataset_names = set(datasets_list[0].keys())

    for datasets in datasets_list:
        if set(datasets.keys()) != dataset_names:
            raise ValueError("All files must have the same datasets")

    averaged_datasets = {}

    for name in dataset_names:
        data_stack = np.array([datasets[name] for datasets in datasets_list])
        averaged_datasets[name] = np.mean(data_stack, axis=0)

    print(
        'shape of averaged_datasets lock-in data [0]:',
        averaged_datasets['lock-in_data'].shape
    )

    plt.figure()
    plt.title('averaged_datasets - lock-in data [0]')
    plt.plot(averaged_datasets['lock-in_data'][:, :, 0].T)
    plt.show()

    angles = averaged_datasets['angle (deg)']
    PP_data_pol = averaged_datasets['lock-in_data']

    print(PP_data_pol.shape)

    for i in range(len(angles)):
        PP_data_pol[i, :, 0] -= PP_data_pol[i, -15:, 0].mean(axis=0)
        PP_data_pol[i, :, 0] /= R_sample[i]

    plt.figure()
    plt.title('DR/R neg. times subtracted')
    plt.plot(PP_data_pol[:, :, 0].T)
    plt.show()

    t0 = np.argmin(PP_data_pol[:, :, 0].mean(axis=0))

    averaged_datasets['position (mm)'] -= averaged_datasets['position (mm)'][t0]

    time = -averaged_datasets['position (mm)'] * 6.66

    return time, angles, PP_data_pol


def compile_data_pol(data, angles, time):
    """
    Convert polarization-dependent data into dictionary form.

    Parameters
    ----------
    data : ndarray
        Pump-probe data indexed by polarization angle.
    angles : array-like
        Polarization angles.
    time : ndarray
        Pump-probe delay axis.

    Returns
    -------
    PPm : dict
        Raw pump-probe traces.
    P_delay : dict
        Time axes.
    PPN : dict
        Normalized traces.
    """
    PPm = {}
    P_delay = {}
    PPN = {}

    for i in range(len(angles)):
        PPm[str(angles[i])] = data[i, :]
        P_delay[str(angles[i])] = time
        PPN[str(angles[i])] = -data[i, :] / data[i, :].min()

    return PPm, P_delay, PPN


# ============================================================================
# Plotting utilities
# ============================================================================

def colorFader(num, c1, c2):
    """
    Generate a linear color gradient between two colors.

    Parameters
    ----------
    num : int
        Number of interpolation intervals.
    c1, c2 : Matplotlib color-like
        Initial and final colors.

    Returns
    -------
    colors : list of str
        ``num + 1`` hexadecimal colors.
    """
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))

    colors = []

    for x in range(num + 1):
        colors.append(
            mpl.colors.to_hex((1 - (x / num)) * c1 + (x / num) * c2)
        )

    return colors


# ============================================================================
# Pump-probe fitting models
# ============================================================================

def PP_model_1(params, t, signal=None):
    """
    Evaluate the non-oscillatory pump-probe response model.

    The model consists of a linear background and an error-function turn-on
    multiplying a persistent response and two exponential relaxation terms.

    Parameters
    ----------
    params : lmfit.Parameters
        Must contain ``m``, ``b``, ``td``, ``tau_0``, ``tau_1``, ``tau_2``,
        ``A0``, ``A1``, and ``A2``.
    t : array-like
        Pump-probe delay axis.
    signal : array-like, optional
        Experimental signal. If provided, the fitting residual
        ``model - signal`` is returned.

    Returns
    -------
    ndarray
        Model response or fitting residual.
    """
    m = params['m']
    b = params['b']
    td = params['td']

    tau_0 = params['tau_0']
    tau_1 = params['tau_1']
    tau_2 = params['tau_2']

    A0 = params['A0']
    A1 = params['A1']
    A2 = params['A2']

    line = m * t + b
    error_fn = 0.5 * (1 + erf((t - td) / tau_0))

    exp_1 = A1 * np.exp(-(t - td) / tau_1)
    exp_2 = A2 * np.exp(-(t - td) / tau_2)

    model = line + error_fn * (A0 + exp_1 + exp_2)

    if signal is None:
        return model

    return model - signal


def PP_model_1_parts(params, t):
    """
    Return the individual components of :func:`PP_model_1`.

    Returns
    -------
    line : ndarray
        Linear-background contribution.
    exp1 : ndarray
        First exponential multiplied by the turn-on function.
    exp2 : ndarray
        Second exponential multiplied by the turn-on function.
    step : ndarray
        Persistent step contribution.
    """
    m = params['m']
    b = params['b']
    td = params['td']

    tau_0 = params['tau_0']
    tau_1 = params['tau_1']
    tau_2 = params['tau_2']

    A0 = params['A0']
    A1 = params['A1']
    A2 = params['A2']

    line = m * t + b
    error_fn = 0.5 * (1 + erf((t - td) / tau_0))

    exp_1 = A1 * np.exp(-(t - td) / tau_1)
    exp_2 = A2 * np.exp(-(t - td) / tau_2)

    return line, error_fn * exp_1, error_fn * exp_2, A0 * error_fn


def PP_model_osc(params, t, signal=None):
    """
    Evaluate a pump-probe model containing a damped coherent oscillation.

    Parameters
    ----------
    params : lmfit.Parameters
        Must contain ``m``, ``b``, ``td``, ``tau_0``, ``tau_1``,
        ``tau_2``, ``tau_3``, ``A0``, ``A1``, ``A2``, ``A3``,
        ``freq``, and ``phase``.
    t : array-like
        Pump-probe delay axis, typically in ps.
    signal : array-like, optional
        Experimental signal.

    Returns
    -------
    ndarray
        Model response, or ``model - signal`` when ``signal`` is provided.

    Notes
    -----
    The oscillatory component is

    ``cos(2*pi*freq*t + phase)``

    multiplied by an exponential envelope with decay constant ``tau_3``.
    """
    m = params['m']
    b = params['b']
    td = params['td']

    tau_0 = params['tau_0']
    tau_1 = params['tau_1']
    tau_2 = params['tau_2']
    tau_3 = params['tau_3']

    A0 = params['A0']
    A1 = params['A1']
    A2 = params['A2']
    A3 = params['A3']

    w = params['freq']
    phi = params['phase']

    line = m * t + b
    error_fn = 0.5 * (1 + erf((t - td) / tau_0))

    exp_1 = A1 * np.exp(-(t - td) / tau_1)
    exp_2 = A2 * np.exp(-(t - td) / tau_2)
    exp_3 = A3 * np.exp(-(t - td) / tau_3)

    osc = np.cos(2 * np.pi * w * t + phi)

    model = line + error_fn * (A0 + exp_1 + exp_2 + exp_3 * osc)

    if signal is None:
        return model

    return model - signal


# ============================================================================
# Pump-probe fitting and Fourier analysis
# ============================================================================

def fit_PP(ts, tf, params, signal, t, key, pretty, guess_pl, fit_pl, osc_pl, FT_pl):
    """
    Fit one pump-probe transient and Fourier transform a selected residual window.

    Parameters
    ----------
    ts : int
        Starting array index of the residual used for Fourier analysis.
    tf : int
        Final array index of the residual used for Fourier analysis.
    params : lmfit.Parameters
        Initial fitting parameters.
    signal : array-like
        Experimental pump-probe signal.
    t : array-like
        Pump-probe delay axis.
    key : str
        Dataset identifier used in output and plot titles.
    pretty : bool
        Print fitted parameters when True.
    guess_pl : bool
        Plot the initial model guess when True.
    fit_pl : bool
        Plot the optimized fit when True.
    osc_pl : bool
        Plot the residual and FFT-window boundaries when True.
    FT_pl : bool
        Plot the Fourier magnitude when True.

    Returns
    -------
    out.params : lmfit.Parameters
        Optimized fit parameters.
    THz_axis : ndarray
        Fourier frequency axis.
    phonon_mag : ndarray
        Fourier magnitude.
    phonon_phase : ndarray
        Fourier phase in radians.
    t : ndarray
        Original pump-probe delay axis.
    diff : ndarray
        Difference ``fit - signal``.

    Notes
    -----
    The Fourier transform is performed only on::

        diff[ts:tf]

    The original frequency-axis convention

        t[0] - t[1]

    is intentionally preserved for compatibility with existing analysis.
    """
    guess = PP_model_1(params, t)

    out = minimize(
        PP_model_1,
        params,
        args=(t,),
        kws={'signal': signal},
        method='leastsq'
    )

    fit = PP_model_1(out.params, t)

    # Retained from the original implementation.
    t0 = np.argmin(signal)
    tFT = t0 + 1

    diff = fit.T - signal

    print(key)

    if pretty == True:
        out.params.pretty_print(
            oneline=False,
            colwidth=9,
            precision=4,
            fmt='g',
            columns=[
                'value', 'min', 'max', 'stderr',
                'vary', 'expr', 'brute_step'
            ]
        )

    phonon_phase = np.angle(np.fft.fft(diff[ts:tf], axis=0))
    phonon_mag = np.abs(np.fft.fft(diff[ts:tf], axis=0))

    THz_axis = np.fft.fftfreq(len(phonon_mag), t[0] - t[1])

    line, exp1, exp2, step = PP_model_1_parts(params, t)

    line_fit, exp1_fit, exp2_fit, step_fit = PP_model_1_parts(
        out.params, t
    )

    # ------------------------------------------------------------------------
    # Data + initial guess
    # ------------------------------------------------------------------------

    if guess_pl == True:
        plt.figure()
        plt.title(key)

        plt.plot(t, signal, '.', label='data')
        plt.plot(t, line, label='line')
        plt.plot(t, exp1, label='exp1')
        plt.plot(t, exp2, label='exp2')
        plt.plot(t, step, label='step')
        plt.plot(t, guess, label='guess', c='k')

        plt.xlabel('Pump-Probe Delay [ps]')
        plt.legend()
        plt.show()

    # ------------------------------------------------------------------------
    # Data + fitted components
    # ------------------------------------------------------------------------

    if fit_pl == True:
        plt.figure()
        plt.title(key)

        plt.tick_params(
            axis='both',
            direction='in',
            length=6,
            color='black'
        )

        plt.xlim(-2, 6.8)

        plt.plot(
            t, signal, '.',
            label='data',
            linewidth=3,
            c='mediumorchid'
        )

        plt.plot(
            t, fit,
            label='best fit',
            c='k',
            lw=1.5
        )

        plt.plot(
            t, step_fit,
            label='step_fit',
            c='darkgreen',
            lw=2
        )

        plt.plot(
            t, line_fit,
            label='line_fit',
            c='grey',
            lw=2
        )

        plt.plot(
            t, exp1_fit,
            label='fast',
            c='crimson',
            lw=2
        )

        plt.plot(
            t, exp2_fit,
            label='slow',
            c='cornflowerblue',
            lw=2
        )

        plt.xlabel('Pump-Probe Delay [ps]')
        plt.legend()
        plt.show()

    # ------------------------------------------------------------------------
    # Residual / oscillation plot
    # ------------------------------------------------------------------------

    if osc_pl == True:
        plt.figure()
        plt.title(key)

        plt.plot(t, diff, label='osc')
        plt.plot(t, signal / 50, label='signal/50')

        plt.vlines(t[tf], -0.01, 0.01, color='green')
        plt.vlines(t[ts], -0.01, 0.01, color='green')

        plt.legend()
        plt.show()

    # ------------------------------------------------------------------------
    # Fourier spectrum
    # ------------------------------------------------------------------------

    if FT_pl == True:
        plt.figure()
        plt.title(key)

        plt.xlabel('Frequency [THz]')
        plt.grid()

        plt.plot(
            THz_axis[0:50],
            phonon_mag[0:50],
            label='phonon_mag'
        )

        plt.legend()
        plt.show()

    return out.params, THz_axis, phonon_mag, phonon_phase, t, diff


def fit_compiler(
    ts, tf, params, signal, P_delay, WL, key, WL_except=[],
    pretty=False, guess_pl=False, fit_pl=False, osc_pl=False,
    FT_pl=False, comps=False
):
    """
    Fit a collection of pump-probe traces.

    Parameters
    ----------
    ts, tf : int
        Starting and ending indices of the residual FFT window.
    params : lmfit.Parameters
        Initial fit parameters.
    signal : dict
        Dictionary containing pump-probe signals.
    P_delay : dict
        Dictionary containing delay axes.
    WL : array-like
        Measurement conditions corresponding to ``key``.
    key : list of str
        Dataset identifiers.
    WL_except : list, optional
        Values of ``WL`` to exclude.
    pretty : bool, optional
        Print fitted parameters.
    guess_pl : bool, optional
        Plot initial guesses.
    fit_pl : bool, optional
        Plot optimized fits.
    osc_pl : bool, optional
        Plot residuals.
    FT_pl : bool, optional
        Plot Fourier spectra.
    comps : bool, optional
        Legacy argument retained for compatibility.

    Returns
    -------
    fit_par : dict
        Optimized fitting parameters.
    THz_ax : dict
        Fourier frequency axes.
    phonon_mag : dict
        Fourier magnitudes.
    phonon_phase : dict
        Fourier phases.
    t : dict
        Pump-probe delay axes.
    phonon : dict
        Fit residuals.
    """
    fit_par = {}
    THz_ax = {}
    phonon_mag = {}
    phonon_phase = {}
    t = {}
    phonon = {}

    for i in range(len(WL)):
        if WL[i] not in WL_except:

            (
                fit_par[key[i]],
                THz_ax[key[i]],
                phonon_mag[key[i]],
                phonon_phase[key[i]],
                t[key[i]],
                phonon[key[i]]
            ) = fit_PP(
                ts,
                tf,
                params,
                signal[key[i]],
                P_delay[key[i]],
                str(key[i]),
                pretty,
                guess_pl,
                fit_pl,
                osc_pl,
                FT_pl
            )

    return fit_par, THz_ax, phonon_mag, phonon_phase, t, phonon


def fit_comps(params, signal, t, THz, guess_comps=False, fit_comps=False):
    """
    Fit a transient and return the individual model components.

    Parameters
    ----------
    params : lmfit.Parameters
        Initial fitting parameters.
    signal : array-like
        Experimental signal.
    t : array-like
        Pump-probe delay axis.
    THz : str
        Pump-frequency identifier used in printed output.
    guess_comps : bool, optional
        Return initial-guess components.
    fit_comps : bool, optional
        Return optimized fit components.

    Returns
    -------
    Guess_Comps : pandas.DataFrame
        Returned when ``guess_comps=True``.
    Fit_Comps : pandas.DataFrame
        Returned when ``fit_comps=True``.
    """
    guess = PP_model_1(params, t)

    out = minimize(
        PP_model_1,
        params,
        args=(t,),
        kws={'signal': signal},
        method='leastsq'
    )

    fit = PP_model_1(out.params, t)
    diff = fit.T - signal

    print('MIR Pump = ' + THz)

    line, exp1, exp2, step = PP_model_1_parts(params, t)

    line_fit, exp1_fit, exp2_fit, step_fit = PP_model_1_parts(
        out.params, t
    )

    fc = [
        t,
        signal,
        fit,
        line_fit,
        exp1_fit,
        exp2_fit,
        step_fit,
        diff
    ]

    fit_keys = [
        't [ps]',
        'signal',
        'fit',
        'line_fit',
        'exp1_fit',
        'exp2_fit',
        'step_fit',
        'diff'
    ]

    Fit_Comps = pd.DataFrame(columns=fit_keys)

    for i in range(len(fit_keys)):
        Fit_Comps[fit_keys[i]] = fc[i]

    gc = [
        guess,
        line,
        exp1,
        exp2,
        step
    ]

    guess_keys = [
        'guess',
        'line',
        'exp1',
        'exp2',
        'step'
    ]

    Guess_Comps = pd.DataFrame(columns=guess_keys)

    for i in range(len(guess_keys)):
        Guess_Comps[guess_keys[i]] = gc[i]

    if guess_comps == True:
        return Guess_Comps

    if fit_comps == True:
        return Fit_Comps


# ============================================================================
# Phonon amplitude and phase maps
# ============================================================================

def phonon_maps(phonon_mag, phonon_phase, THz_ax, freq, key):
    """
    Construct Fourier-amplitude and relative-phase maps.

    Parameters
    ----------
    phonon_mag : dict
        Fourier-transform magnitudes.
    phonon_phase : dict
        Fourier phases.
    THz_ax : dict
        Fourier frequency axes in THz.
    freq : array-like
        Pump frequencies in THz.
    key : list of str
        Dataset identifiers.

    Returns
    -------
    eV : ndarray
        Pump-frequency values converted to eV.
    eV_ax : dict
        Fourier-frequency axes converted to eV.
    FT_map : ndarray
        Fourier-amplitude map.
    phase_map : ndarray
        Relative-phase map.

    Notes
    -----
    The conversion factor

        1 THz = 0.00414 eV

    is used.

    Each phase trace is referenced to the final trace in ``key``.
    """
    eV_ax = {}

    for i in range(len(freq)):
        eV_ax[key[i]] = THz_ax[key[i]] * 0.00414

    eV = freq * 0.00414

    length = []

    for i in range(len(freq)):
        length.append(phonon_mag[key[i]].shape[0])

    end = np.array(length).min()

    FT_map = np.zeros((end, len(freq)))

    for i in range(len(freq)):
        FT_map[:, i] = phonon_mag[key[i]][:end]

    phase_map = np.zeros((end, len(freq)))

    for i in range(len(freq)):
        phase_map[:, i] = np.unwrap(
            phonon_phase[key[i]][:end]
            - phonon_phase[key[-1]][:end]
        )

    return eV, eV_ax, FT_map, phase_map


# ============================================================================
# Fit parameter extraction
# ============================================================================

def extract_fit_params(fit_par, key):
    """
    Extract varying fit parameters and their lmfit uncertainties.

    Parameters
    ----------
    fit_par : dict
        Dictionary of ``lmfit.Parameters`` objects.
    key : list of str
        Ordered dataset keys.

    Returns
    -------
    parameter_lists : dict
        Fitted values for parameters whose ``vary`` property is True.
    stderr_lists : dict
        Corresponding lmfit standard errors.

    Notes
    -----
    Fixed parameters are intentionally excluded, preserving the behavior
    of the original function.
    """
    parameter_lists = {}

    for parameter in fit_par[key[0]].keys():
        if fit_par[key[0]][parameter].vary == True:

            temporary_list = []

            for par in fit_par.values():
                temporary_list.append(par[parameter].value)

            parameter_lists[parameter] = temporary_list

    stderr_lists = {}

    for parameter in fit_par[key[0]].keys():
        if fit_par[key[0]][parameter].vary == True:

            temporary_list = []

            for par in fit_par.values():
                temporary_list.append(par[parameter].stderr)

            stderr_lists[parameter] = temporary_list

    return parameter_lists, stderr_lists


# ============================================================================
# Fluence calculation
# ============================================================================

def get_fluence(Pow, Ppin, Pnopin, Pratio, rep_rate):
    """
    Calculate the peak pump fluence at the sample.

    Parameters
    ----------
    Pow : float or array-like
        Optical power measured at the reference position, in uW.
    Ppin : float
        Power measured through a 300 um pinhole, in uW.
    Pnopin : float
        Power measured without the pinhole, in uW.
    Pratio : float
        Ratio converting reference power to estimated sample power.
    rep_rate : float
        Laser repetition rate in Hz.

    Returns
    -------
    peak_fluence : float or ndarray
        Peak fluence in mJ/cm^2, rounded to two decimal places.

    Notes
    -----
    The beam is assumed to have a circular Gaussian intensity profile.

    ``w0`` is the 1/e^2 intensity radius in micrometers.
    """
    w0 = np.sqrt(
        (300**2 / np.log(Pnopin / (Pnopin - Ppin))) / 2
    )

    print('beam radius = ' + str(w0) + ' um')

    peak_fluence = np.around(
        (((Pow * Pratio * 10**(-3)) / rep_rate) / ((np.pi * w0**2) / 2))
        * 1e8,
        2
    )

    return peak_fluence


# ============================================================================
# Smoothing utilities
# ============================================================================

def moving_average_1d(data, window_size):
    """
    Smooth one-dimensional data using a simple moving average.

    Parameters
    ----------
    data : list or ndarray
        One-dimensional data to smooth.
    window_size : int
        Number of points in the averaging window.

    Returns
    -------
    smoothed_data : ndarray
        Moving-average result.

    Raises
    ------
    TypeError
        If ``data`` is not a list or NumPy array.
    ValueError
        If ``window_size`` is not a positive integer.

    Notes
    -----
    ``mode='valid'`` is used, so the returned array is shorter than the
    input by ``window_size - 1`` points.
    """
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("Data should be a list or numpy array.")

    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("Window size should be a positive integer.")

    data = np.asarray(data)

    smoothed_data = np.convolve(
        data,
        np.ones(window_size) / window_size,
        mode='valid'
    )

    return smoothed_data


def moving_average_2d(data, window_size, direction='horizontal'):
    """
    Apply a moving average along one direction of a 2D array.

    Parameters
    ----------
    data : ndarray
        Two-dimensional data array.
    window_size : int
        Number of points in the averaging window.
    direction : {'horizontal', 'vertical'}, optional
        Direction along which the moving average is applied.

    Returns
    -------
    smoothed_data : ndarray
        Smoothed 2D array.

    Raises
    ------
    ValueError
        If the data are not two-dimensional, the window size is invalid,
        or the direction is unsupported.

    Notes
    -----
    ``horizontal`` applies the filter along axis 1.

    ``vertical`` applies the filter along axis 0.

    Because ``mode='valid'`` is used, the smoothed dimension is shortened
    by ``window_size - 1`` points.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data should be a 2D numpy array.")

    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("Window size should be a positive integer.")

    if direction not in ['horizontal', 'vertical']:
        raise ValueError("Direction should be 'horizontal' or 'vertical'.")

    kernel = np.ones(window_size) / window_size

    if direction == 'horizontal':
        smoothed_data = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode='valid'),
            axis=1,
            arr=data
        )

    else:
        smoothed_data = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode='valid'),
            axis=0,
            arr=data
        )

    return smoothed_data


def smooth_2d(data, sigma):
    """
    Smooth two-dimensional data using a Gaussian filter.

    Parameters
    ----------
    data : ndarray
        Two-dimensional data array.
    sigma : float
        Standard deviation of the Gaussian kernel. Larger values produce
        stronger smoothing.

    Returns
    -------
    smoothed_data : ndarray
        Gaussian-smoothed array with the same shape as ``data``.

    Raises
    ------
    ValueError
        If ``data`` is not two-dimensional or ``sigma`` is not positive.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data should be a 2D numpy array.")

    if not isinstance(sigma, (int, float)) or sigma <= 0:
        raise ValueError("Sigma should be a positive number.")

    smoothed_data = gaussian_filter(data, sigma=sigma)

    return smoothed_data


# ============================================================================
# Geometry utilities
# ============================================================================

def ellipse_polar(a, b, theta):
    """
    Calculate the radial coordinate of an ellipse in polar coordinates.

    Parameters
    ----------
    a : float
        Semi-major axis length.
    b : float
        Semi-minor axis length.
    theta : float or array-like
        Polar angle in radians.

    Returns
    -------
    r : float or ndarray
        Radial distance from the ellipse center at each angle ``theta``.

    Notes
    -----
    The polar equation used is

        r(theta) = a*b / sqrt((b*cos(theta))^2 + (a*sin(theta))^2)
    """
    r = (a * b) / np.sqrt(
        (b * np.cos(theta))**2
        + (a * np.sin(theta))**2
    )

    return r