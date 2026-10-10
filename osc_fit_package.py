"""
Coherent-oscillation fitting utilities.

This section defines models for one or more damped coherent phonon modes and
provides a fitting routine using lmfit.

The public function names, argument order, defaults, and return values are
kept unchanged for compatibility with existing analysis notebooks.
"""

# ============================================================================
# Imports
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit import minimize, Parameters

# ============================================================================
# Oscillation models
# ============================================================================

def osc_model(p, t, num_phonon, signal=None):
    """
    Evaluate a sum of damped coherent phonon oscillations.

    Each phonon mode is modeled as

        a_i * (1 - exp(-t / tau_r_i))
            * exp(-t / tau_i)
            * cos(2*pi*w_i*t + phi_i)

    where the first exponential term describes the finite rise of the
    oscillation amplitude and the second exponential describes its decay.

    Parameters
    ----------
    p : lmfit.Parameters
        Fit parameters.

        For each phonon mode ``i`` from 1 to ``num_phonon``, the parameter
        set must contain:

        ``a<i>``
            Oscillation amplitude.

        ``tau_r<i>``
            Rise time of the oscillation amplitude.

        ``tau<i>``
            Decay time of the oscillation.

        ``w<i>``
            Oscillation frequency. If ``t`` is in ps, ``w`` is typically
            expressed in THz.

        ``phi<i>``
            Oscillation phase in radians.

        For example, for two phonon modes the parameter set should contain::

            a1, tau_r1, tau1, w1, phi1
            a2, tau_r2, tau2, w2, phi2

    t : array-like
        Time-delay axis, typically in ps.

    num_phonon : int
        Number of phonon modes included in the model.

    signal : array-like, optional
        Experimental oscillatory signal.

        If supplied, the function returns ``model - signal`` for use with
        ``lmfit.minimize``.

    Returns
    -------
    model : ndarray
        Sum of all phonon contributions when ``signal`` is None.

    residual : ndarray
        Difference ``model - signal`` when ``signal`` is supplied.
    """
    model = 0

    for i in range(1, num_phonon + 1):
        model += (
            p['a' + str(i)]
            * (1 - np.exp(-t / p['tau_r' + str(i)]))
            * np.exp(-t / p['tau' + str(i)])
            * np.cos(
                2 * np.pi * p['w' + str(i)] * t
                + p['phi' + str(i)]
            )
        )

    if signal is None:
        return model

    return model - signal


def single_phonon_model(p, t, num_phonon, signal=None):
    """
    Evaluate one phonon mode from the multi-phonon model.

    This function uses ``num_phonon`` as the index of the phonon mode to
    evaluate. For example,

        single_phonon_model(p, t, 2)

    evaluates the mode described by

        a2, tau_r2, tau2, w2, phi2.

    Parameters
    ----------
    p : lmfit.Parameters
        Parameter set containing the indexed phonon parameters.

    t : array-like
        Time-delay axis, typically in ps.

    num_phonon : int
        Index of the phonon mode to evaluate.

    signal : array-like, optional
        Experimental signal.

        If supplied, the function returns ``single_phn - signal``.

    Returns
    -------
    single_phn : ndarray
        Calculated contribution from the selected phonon mode.

    residual : ndarray
        Difference ``single_phn - signal`` when ``signal`` is supplied.
    """
    i = num_phonon

    single_phn = (
        p['a' + str(i)]
        * (1 - np.exp(-t / p['tau_r' + str(i)]))
        * np.exp(-t / p['tau' + str(i)])
        * np.cos(
            2 * np.pi * p['w' + str(i)] * t
            + p['phi' + str(i)]
        )
    )

    if signal is None:
        return single_phn

    return single_phn - signal


# ============================================================================
# Oscillation fitting
# ============================================================================

def fit_osc(ts, tf, p, signal, t, num_phonon, key, pretty, guess_pl, fit_pl):
    """
    Fit one or more damped coherent phonon modes to an oscillatory signal.

    The fit is performed only over the time interval selected by the array
    indices ``ts`` and ``tf``. The corresponding fitting window is

        t[ts:tf]

    and the data fitted are

        signal[ts:tf].

    Parameters
    ----------
    ts : int
        Starting array index of the fitting window.

    tf : int
        Ending array index of the fitting window.

        Because normal Python slicing is used, the element at ``tf`` is not
        included in the fit.

    p : lmfit.Parameters
        Initial parameters for the oscillation model.

        For each phonon ``i``, the parameter set should contain:

        ``a<i>``
            Oscillation amplitude.

        ``tau_r<i>``
            Oscillation rise time.

        ``tau<i>``
            Oscillation decay time.

        ``w<i>``
            Oscillation frequency.

        ``phi<i>``
            Oscillation phase.

    signal : array-like
        Experimental oscillatory signal.

    t : array-like
        Pump-probe delay axis, typically in ps.

    num_phonon : int
        Number of phonon modes included in the fit.

    key : str
        Dataset identifier used as the plot title.

    pretty : bool
        If True, print the optimized lmfit parameters.

    guess_pl : bool
        If True, plot the initial parameter guess together with the
        individual phonon components.

    fit_pl : bool
        If True, plot the optimized fit and the individual fitted phonon
        components.

    Returns
    -------
    out.params : lmfit.Parameters
        Optimized fitting parameters.

    Notes
    -----
    The total oscillation model is defined by :func:`osc_model`.

    Individual phonon contributions are generated with
    :func:`single_phonon_model`.

    Fitting is performed with ``lmfit.minimize`` using the least-squares
    method.
    """
    t = t[ts:tf]

    # ------------------------------------------------------------------------
    # Plot initial guess
    # ------------------------------------------------------------------------

    if guess_pl == True:
        plt.figure()
        plt.title(key)

        for i in range(1, num_phonon + 1):
            plt.plot(
                t,
                single_phonon_model(p, t, i, signal=None),
                '--',
                label=str(p['w' + str(i)].value) + str(i) + 'THz',
                lw=1,
                alpha=0.8
            )

        plt.plot(
            t,
            signal[ts:tf],
            '-',
            label='data',
            c='k',
            lw=1.5
        )

        plt.plot(
            t,
            osc_model(p, t, num_phonon, signal=None),
            alpha=0.8,
            label='guess'
        )

        plt.xlabel('Pump-Probe Delay [ps]')
        plt.ylabel('$\\Delta$R/R$_{osc}$')

        plt.legend()
        plt.show()

    # ------------------------------------------------------------------------
    # Fit oscillatory signal
    # ------------------------------------------------------------------------

    out = minimize(
        osc_model,
        p,
        args=(t, num_phonon, signal[ts:tf]),
        method='leastsq'
    )

    # ------------------------------------------------------------------------
    # Print optimized parameters
    # ------------------------------------------------------------------------

    if pretty == True:
        out.params.pretty_print(
            oneline=False,
            colwidth=9,
            precision=4,
            fmt='g',
            columns=[
                'value',
                'min',
                'max',
                'stderr',
                'vary',
                'expr',
                'brute_step'
            ]
        )

    # ------------------------------------------------------------------------
    # Plot optimized fit
    # ------------------------------------------------------------------------

    if fit_pl == True:
        plt.figure()
        plt.title(key)

        for i in range(1, num_phonon + 1):
            plt.plot(
                t,
                single_phonon_model(out.params, t, i, signal=None),
                '--',
                label=str(out.params['w' + str(i)].value) + str(i) + 'THz',
                lw=1,
                alpha=0.8
            )

        plt.plot(
            t,
            signal[ts:tf],
            '-',
            label='data',
            c='k',
            lw=1.5
        )

        plt.plot(
            t,
            osc_model(out.params, t, num_phonon, signal=None),
            alpha=0.8,
            label='fit',
            color='purple'
        )

        plt.ylabel('$\\Delta$R/R$_{osc}$')
        plt.xlabel('Pump-Probe Delay [ps]')

        plt.legend()
        plt.show()

        # Plot individual fitted phonon components separately.
        plt.figure()

        for i in range(1, num_phonon + 1):
            plt.plot(
                t,
                single_phonon_model(out.params, t, i, signal=None),
                '--',
                label=str(out.params['w' + str(i)].value) + str(i) + 'THz',
                lw=1,
                alpha=0.8
            )

        plt.legend()
        plt.show()

    return out.params