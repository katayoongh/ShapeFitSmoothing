"""
Authors: Katayoon Ghaemi, Nils Schöneberg
Last updated: 14th August 2025
"""


from __future__ import print_function
import classy
import os
import numpy as np
import math
import warnings
import re
import scipy.constants as const
import scipy.integrate
import matplotlib.pyplot as plt
import scipy.misc
import matplotlib as mpl

from copy import deepcopy

from scipy.interpolate import BSpline, CubicSpline, UnivariateSpline, interp1d, splrep, splev
from scipy.optimize import curve_fit, minimize
from scipy.integrate import simpson
from scipy.fft import fht,ifht
from scipy.fftpack import dst,idst
from scipy.signal import find_peaks,convolve, savgol_filter
from scipy.ndimage import median_filter


mpl.rcParams.update(mpl.rcParamsDefault)

class EHclass:
    @classmethod
    def EH98(self, cosmo: classy.Class, kvector: np.ndarray, redshift: float, scaling_factor: float, **add_args)->np.ndarray:
        """
        Computes the Eisenstein & Hu (1998) linear matter power spectrum P(k, z).

        Parameters
        ----------
        cosmo : object: classy.Class()
            Cosmology object providing cosmological parameters like Omega_m(), h(), etc.
        kvector : array_like: np.ndarray
            Wavenumber values in h/Mpc.
        redshift : float
            Redshift at which the power spectrum is evaluated.
        scaling_factor : float
            Multiplicative factor applied to some scales (e.g., for unit conversions).
        **add_args : dict
            Additional keyword arguments.

        Returns
        -------
        Pk : ndarray: np.ndarray
            Linear matter power spectrum at the given redshift in (Mpc/h)^3.
        """
        cdict = cosmo.get_current_derived_parameters(['z_d']) #z_d is the time of baryon drag
        h = cosmo.h() #h is the Hubble parameter given by h= H0/ 100 km /s /Mpc
        H_at_z = cosmo.Hubble(redshift) * const.c /1000. /(100.*h)
        Omm = cosmo.Omega_m()
        Omb = cosmo.Omega_b()
        Omc = cosmo.Omega0_cdm()
        Omm_at_z = Omm*(1.+redshift)**3./H_at_z**2.
        OmLambda_at_z = 1.-Omm_at_z
        ns = cosmo.n_s() #ns is the scalar tilt of primordial power spectrum
        rs = cosmo.rs_drag()*h/scaling_factor
        Omnu = Omm-Omb-Omc
        fnu = Omnu/Omm
        fb = Omb/Omm
        fnub = (Omb+Omnu)/Omm
        fc = Omc/Omm
        fcb = (Omc+Omb)/Omm
        pc = 1./4.*(5-np.sqrt(1+24*fc))
        pcb = 1./4.*(5-np.sqrt(1+24*fcb))
        Neff = cosmo.Neff() # the number of effective neutrino species
        # The neutrinos don't decouple instantasly
        Omg = cosmo.Omega_g()
        Omr = Omg * (1. + Neff * (7./8.)*(4./11.)**(4./3.))
        aeq = Omr/(Omb+Omc)/(1-fnu)
        zeq = 1./aeq -1. #redshift of equality
        Heq = cosmo.Hubble(zeq)/h #Hubble constant at the equality time
        keq = aeq*Heq*scaling_factor #scale of matter radiation equality
        zd = cdict['z_d'] #redshift at the baryon drag epoch
        yd = (1.+zeq)/(1.+zd)
        growth = cosmo.scale_independent_growth_factor(redshift)
        if (fnu<=0.0):
            fnu = 0.0
            Nnu = 0.
        else:
            Nnu = 1.
        alpha_nu = fc/fcb * (5 - 2*(pc+pcb))/(5-4*pcb) * (1-0.553*fnub + 0.126*fnub**3.) / (1 - 0.193*np.sqrt(fnu)*Nnu**0.2 + 0.169*fnu) \
                    *(1.+yd)**(pcb-pc) * (1+(pc-pcb)/2*(1+1./(3-4*pc)/(7-4*pcb))*(1.+yd)**(-1.))
        eff_shape = (np.sqrt(alpha_nu) + (1.-np.sqrt(alpha_nu))/(1+(0.43*kvector*rs)**4.))
        q0 = kvector/(keq/7.46e-2)/eff_shape
        betac = (1.-0.949*fnub)**(-1.)
        # transfer function is parametrized as:
        L0 = np.log(np.exp(1.)+1.84*np.sqrt(alpha_nu)*betac*q0)
        C0 = 14.4 + 325./(1+60.5*q0**1.08)
        T0 = L0/(L0+C0*q0**2.)
        #growth function
        D1 = 5*Omm_at_z/2./(Omm_at_z**(4./7.) - OmLambda_at_z + (1.+Omm_at_z/2.)*(1.+OmLambda_at_z/70.))
        if (fnu<=0.0):
            yfs=0.
            qnu=3.92*q0
            Dcbnu=D1
        else:
            yfs = 17.2*fnu*(1.+0.488*fnu**(-7./6.))*(Nnu*q0/fnu)**2.
            qnu = 3.92*q0*np.sqrt(Nnu/fnu)
            Dcbnu = (fcb**(0.7/pcb)+(D1/(1.+yfs))**0.7)**(pcb/0.7) * D1**(1.-pcb)


        Bk = 1. + 1.24*fnu**(0.64)*Nnu**(0.3+0.6*fnu)/(qnu**(-1.6)+qnu**0.8)
        Tcbnu = T0*Dcbnu/D1*Bk

        deltah = 1.94e-5 * Omm**(-0.785-0.05*np.log(Omm))*np.exp(-0.95*(ns-1)-0.169*(ns-1)**2.)
        Pk = 2*np.pi**2. * deltah**2. * (kvector)**(ns) * Tcbnu**2. * growth**2. /(cosmo.Hubble(0)/cosmo.h())**(3.+ns)
        return Pk

    @classmethod
    def EH98_fit(self, kvector: np.ndarray, pvec: np.ndarray)->np.ndarray:
        """
        Compute a simplified parametric Eisenstein & Hu power spectrum fit.

        This method uses a reduced set of parameters to reproduce the EH98
        transfer function shape without requiring full cosmological calculations.

        Parameters
        ----------
        kvector : array_like: np.ndarray
            Wavenumber values in h/Mpc.
        pvec : array_like: np.ndarray
            Parameters of the power spectrum.

        Returns
        -------
        Pk : ndarray: np.ndarray
            Power spectrum P(k) in arbitrary units (depends on pvec[5]).
        """
        keq = pvec[0] #7.61728119e-02,
        alpha_nu = pvec[1]
        eff_shape = np.sqrt(alpha_nu) + (1.-np.sqrt(alpha_nu))/(1+(pvec[2]*kvector)**4.)
        q0 = kvector/(keq/pvec[3])/eff_shape#7.61728119e-02/5.95675857e-01,
        L0 = np.log(np.exp(1.)+pvec[4]*np.sqrt(alpha_nu)*q0)
        C0 = 14.4 + 325./(1+60.5*q0**1.08)
        T0 = L0/(L0+C0*q0**2.)
        Pk = pvec[5] * (kvector)**(pvec[6]) * T0**2
        return Pk

    @classmethod
    def EH98_fit_modified(self, kvector: np.ndarray, pvec: np.ndarray)->np.ndarray:
        """
        Compute a modified parametric Eisenstein & Hu power spectrum fit.

        Same as EH98_fit, but allows modification of the exponent in the
        small-scale suppression term of the transfer function, making it
        more flexible for non-standard cosmologies.

        Parameters
        ----------
        kvector : array_like: np.ndarray
            Wavenumber values in h/Mpc.
        pvec : array_like: np.ndarray
            Parameters of the power spectrum.

        Returns
        -------
        Pk : np.ndarray
            Power spectrum P(k) in arbitrary units (depends on pvec[5]).
        """
        alpha_nu = pvec[0]
        eff_shape = np.sqrt(alpha_nu) + (1.-np.sqrt(alpha_nu))/(1+(pvec[1]*kvector)**4.)
        q0 = kvector/(pvec[2])/eff_shape #7.61728119e-02/5.95675857e-01,
        L0 = np.log(np.exp(1.)+pvec[3]*np.sqrt(alpha_nu)*q0)
        C0 = 14.4 + 325./(1+60.5*q0**1.08)
        T0 = L0/(L0+C0*q0**(2.+pvec[4]))
        Pk = pvec[5] * (kvector)**(pvec[6]) * T0**2#+pvec[7]*np.log(kvector)
        return Pk

class slope_maker:
    """
    This class contains different ways of post processing the power spectrum,
    and different derivative methods to compute the slope.

    Methods in this class:
      - smooth_ratio: Smooth a the power spectra ratio as a function of wavenumbers, k using the selected method.
      - slope_at_x: Compute the slope (derivative) of the ratio using a variety of numerical and analytical methods.
    """
    @classmethod
    def smooth_ratio(self, xvector: np.ndarray, yvector: np.ndarray,
                     smoothmethod: str ="none")->np.ndarray:
        """
        Smooth a the power spectra ratio as a function of wavenumbers, k using the selected method.

        Supported smoothing methods:
        - "none" : no smoothing, returns yvector unchanged
        - "mean" : Moving average smoothing over a fixed logarithmic window (~1.3 decades).
            Window size is determined relative to the log-scale extent of `xvector`.
            Boundary regions are handled with progressively smaller windows.

        - "savitzky" :
            Savitzky–Golay smoothing (polynomial least-squares fit in a moving window).
            Fits a polynomial of fixed order (default polyorder=3) over an odd-sized window.

        For the "mean" method:
        - Defines a smoothing window size based on a fixed `decade` scale (1.3 decades in log-space)
        - Applies an averaging filter with special handling for boundaries

        Parameters
        ----------
        xvector : array_like: np.ndarray
            k-values in h/Mpc
        yvector : array_like: np.ndarray
            Power spectrum ratio
        smoothmethod : str, optional
            Smoothing method to apply. One of {"none", "mean", "savitzky"}.
            Default is "none".

        Returns
        -------
        smoothed : ndarray: np.ndarray
            Smoothed power spectrum values, same shape as input.

        Raises
        ------
        Exception
            If an unrecognized smoothing method name is provided.
        """

        if smoothmethod == "none":
          return yvector
        elif smoothmethod == "mean":
          decade=1.3
          delta_ln_k = decade * np.log(10.0)

          k_max = np.max(xvector)
          k_min = np.min(xvector)

          # Determine window size for averaging and create an average
          window_size = int(len(xvector) * (delta_ln_k/(k_max-k_min)))
          avg_filter = np.ones(window_size)

          # Ensure the smoothing window is smaller than the data range
          assert((delta_ln_k/(k_max-k_min))<1)

          result = np.zeros_like(xvector)

          # Calculate half of the window size for boundary handling
          half_window = window_size//2+1

          # Apply the averaging filter to the main part of the data
          for i in range(half_window,len(xvector)-window_size+half_window):
            result[i] = np.sum(yvector[i-half_window:i-half_window+window_size]
                               * avg_filter)/window_size

          # Handle the left boundary of the data
          for i in range(half_window):
            size = min(window_size,(i*2+1))
            result[i] = np.sum(yvector[:size] * avg_filter[:size])/size

          # Handle the right boundary of the data
          for j in range(window_size-half_window):
            size = (j*2+1)
            i = len(xvector)-1-j
            result[i] = np.sum(yvector[-size:] * avg_filter[-size:])/size

          pk_mean = result
          return pk_mean

        elif smoothmethod == "savitzky":
          decade = 2.5
          delta_ln_k = decade * np.log(10.0)
          k_max = np.max(xvector)
          k_min = np.min(xvector)

          # Determine odd window size based on log-scale span
          window_size = int(len(xvector) * (delta_ln_k/(k_max-k_min)))

          # Apply Savitzky–Golay filter (3rd-order polynomial)
          pk_savgol = savgol_filter(yvector, window_length = window_size, polyorder=3)
          return pk_savgol

        else:
          raise Exception("Unrecognized smoothing method {}".format(smoothmethod))

    @classmethod
    def slope_at_x(self,xvector: np.ndarray,yvector: np.ndarray,
                   derivmethod: str ="sam", smoothmethod: str ="none",
                   inputinfo: dict = {}, return_approx: bool =False)->np.ndarray:
        """
        Compute the slope (derivative) of y(x) using a variety of numerical and analytical methods.

        Optionally applies smoothing before computing the derivative.

        Parameters
        ----------
        xvector : array_like: np.ndarray
            ln(k) or k values
        yvector : array_like: np.ndarray
            ln(P(k)) or P(k) values
        derivmethod : str, optional
            Method for computing the derivative. One of:
                "sam" :
                    Fit a cubic B-spline (k=5, s=3) to all data, take its derivative.
                "sam_res" :
                    Same as "sam", but fit only within a restricted log(k) range from `inputinfo`.
                "gradient" :
                    Use NumPy's `np.gradient` to compute finite differences.
                "poly" :
                    Fit a polynomial (default deg=2) within log(k) range from `inputinfo`, then take its derivative.
                "steps" :
                    Compute slope between two given x-values in `inputinfo["xvals"]` and return constant slope.
                "tanh" :
                    Fit an exponential-tanh function with fixed `a` and `kpiv` from `inputinfo`,
                    return derivative divided by function value.
                "tanh_fit" :
                    Similar to "tanh", but `a` and `kpiv` are also fitted, within bounds.

        smoothmethod : str, optional
            Smoothing method to apply before derivative calculation.
            Passed to `self.smooth_ratio()` (e.g., "none", "mean", "savitzky").
        inputinfo : dict, optional
            Additional parameters needed for some derivative methods:
                - kmin, kmax : bounds for restricted fits ("sam_res", "poly")
                - degree : polynomial degree for "poly"
                - xvals : [xlow, xhigh] for "steps"
                - a, kpiv : parameters for "tanh"
        return_approx : bool, optional
            If True, also return the smoothed/approximated y(x) used for the derivative.

        Returns
        -------
        diff : ndarray: np.ndarray
            Derivative dlnP(k)/dlnk.
        approx : ndarray, optional
            Only returned if `return_approx=True`. The approximated y(x) used for derivative.

        Raises
        ------
        Exception
            If `derivmethod` is not recognized.
        """

        # Apply smoothing if requested
        yvector = self.smooth_ratio(xvector,yvector, smoothmethod=smoothmethod)


        if derivmethod == "sam":
          # Create a B-spline representation of (xvector, yvector)
          f = splrep(xvector,yvector,k=5,s=3)

          # Evaluate the first derivative of the spline at all points in xvector
          diff = splev(xvector,f,der=1)

          # Optionally compute the spline approximation of yvector
          if return_approx:
              approx = splev(xvector, f)


        elif derivmethod == "sam_res":
          # Create a mask to select only x values within a specified log range
          mask = np.logical_and(
              xvector>np.log(inputinfo.get("kmin",1e-3)),
              xvector<np.log(inputinfo.get("kmax",1e-1)))

          # Fit a quintic spline to only the masked data
          f = splrep(xvector[mask],yvector[mask],k=5,s=0.3)

          # Evaluate first derivative of spline over entire xvector
          diff = splev(xvector,f,der=1)

          # Optionally compute spline approximation of yvector
          if return_approx:
              approx = splev(xvector, f)


        elif derivmethod == "gradient":
          # Use numpy's finite difference method to compute dy/dx
          diff = np.gradient(yvector,xvector)

          if return_approx:
              approx = yvector

        elif derivmethod == "poly":
          # Restrict data range using mask
          mask = np.logical_and(
              xvector>np.log(inputinfo.get("kmin",0.008)),
              xvector<np.log(inputinfo.get("kmin",1e-1)))

          # Fit polynomial of specified degree to masked data
          poly = np.polynomial.Polynomial.fit(
              xvector[mask],yvector[mask],
              deg=inputinfo.get("degree",2))

          # Compute first derivative of polynomial at all points
          diff = poly.deriv()(xvector)

          if return_approx:
              approx = poly(xvector)

        elif derivmethod == "steps":
          # Get log of low and high x values from inputinfo
          xlow = np.log(inputinfo["xvals"][0])
          xhigh = np.log(inputinfo["xvals"][1])

          # Find indices in xvector closest to xlow and xhigh
          idxlow = np.argmin((xvector-xlow)**2)
          idxhigh = np.argmin((xvector-xhigh)**2)

          # Get corresponding y values
          pklow = yvector[idxlow]
          pkhigh = yvector[idxhigh]

          # Compute slope between these two points
          slope = (pkhigh-pklow)/(xhigh-xlow)
          diff = slope*np.ones_like(xvector)

          if return_approx:
              approx = slope*(xvector-xhigh)+pkhigh

        elif derivmethod == "tanh":
          def func(lnk,c,d,deltans,a=inputinfo["a"],lnkpiv=np.log(inputinfo["kpiv"])):
            """
            Fitting the power spectrum
            """
            return np.exp(d+deltans*(lnk-lnkpiv)+c/a*np.tanh(a*(lnk-lnkpiv)))


          def dfunc_dlnk(lnk,c,d,deltans,a=inputinfo["a"],lnkpiv=np.log(inputinfo["kpiv"])):
            """
            Compute the derivative of the model with respect to lnk
            """
            return np.exp(d+deltans*(lnk-lnkpiv)+c/a*np.tanh(a*(lnk-lnkpiv)))* (c / np.cosh(a * (lnk-lnkpiv))**2 + deltans)

          # Mask: fit only within certain log(k) range
          mask = np.logical_and(xvector > np.log(1e-4),xvector < np.log(5))

          # Fit parameters of func to masked data (yvector is exponentiated)
          popt, pcov = curve_fit(func ,xvector[mask],np.exp(yvector[mask]),p0=[-0.011,0, 0])

          # Compute derivative
          diff = dfunc_dlnk(xvector, *popt)/func(xvector, *popt)

          if return_approx:
              approx = np.log(func(xvector, *popt))

        elif derivmethod == "tanh_fit":
          def func(lnk,c,d,a,deltans,lnkpiv):
            """
            Fitting the power spectrum
            """
            return np.exp(d+deltans*(lnk-lnkpiv)+c/a*np.tanh(a*(lnk-lnkpiv)))
          def dfunc_dlnk(lnk,c,d,a,deltans,lnkpiv):
            """
            Compute the derivative of the model with respect to lnk
            """
            return np.exp(d+deltans*(lnk-lnkpiv)+c/a*np.tanh(a*(lnk-lnkpiv))) *(c / np.cosh(a * (lnk-lnkpiv))**2 + deltans)

          # Mask range
          mask = np.logical_and(xvector > np.log(1e-4),xvector < np.log(5))

          # Fit with bounds on parameters
          popt, pcov = curve_fit(func ,xvector[mask],np.exp(yvector[mask]),
                                 p0=[-0.011,0, 0.5, 0, np.log(0.05)],
                                 bounds=([-10,-2, 0.2, -0.5, np.log(0.01)],[10, 2, 1.0, 0.5, np.log(0.1)]))

          # Compute relative slope
          diff = dfunc_dlnk(xvector, *popt)/func(xvector, *popt)

          if return_approx:
              approx = np.log(func(xvector, *popt))

        else:
            raise Exception(f"No such {derivmethod=}")
        if not return_approx:
            return diff
        else:
            return diff, approx

class smoother:
    """
    This class contains different smoothing algorithms for the power spectrum.
    The existing algorithms:
      - Numerical smoothing:
          -- smoothing_method_1509_1: Gaussian smoothing
      - Fitting smooth functions:
          -- smoothing_method_1604: Polynomial fit
          -- smoothing_method_1605_3 : Cubic Spline fit
          -- smoothing_method_1509_2 : B-Spline fit
          -- smoothing_method_1509_3 : Multiple B-Spline fit
          -- smoothing_method_univariate : Univariate Spline fit
          -- smoothing_method_1605_1 : EH fit
          -- smoothing_method_1605_2 : EH fit; second version
          -- smoothing_method_0907 : Cubic Spline fit
      - Inflections:
          -- MP, MP_v2 : Cubic inflections
      - Correlation function peak removal:
          -- smoothing_method_1301 : Hanckel transform
          -- smoothing_method_2004 : Fast sine transform
    """
    def MP(self, k_in: np.ndarray, pk_in: np.ndarray,
                   k_ref: np.ndarray = np.array([0.0022, 4.5e-1]))->np.ndarray:

        """
        Smoothing method based on ArXive paper 1210.7183, 1804.07261.

        Function computing the analytical solution for smoothing the power spectrum

        Parameters:
        ----------
        k_in: np.ndarray:
            set of wave numbers corresponding to the values of our power spectrum
        pk_in: np.ndarray:
            input power spectrum to be smoothed
        k_ref : np.ndarray:
            wave numbers corresponding to the BAO features

        Returns
        -------
        pk_smoothed : np.ndarray:
            smoothed power spectrum
        """

        # Minimum and maximum wave numbers that contain the BAO features
        k_ref_min = k_ref[0]
        k_ref_max = k_ref[1]

        # 1) Fit the power spectrum without BAO features using a univariate Spline
        # Spline all (log-log) points outside k_ref range:
        no_bao_features_idxs = np.where(np.logical_or(k_in <= k_ref_min, k_in >= k_ref_max))

        # Smoothing spline fit to a given set of data points
        fun_pk_smooth = UnivariateSpline( np.log(k_in[no_bao_features_idxs]), np.log(pk_in[no_bao_features_idxs]), k=3, s=0 )
        lambda_pk_smooth = lambda x: np.exp(fun_pk_smooth(np.log(x)))
        pk_smoothed = lambda_pk_smooth(k_in)

        # 2) Get a spline fitting the ratio between the input power spectrum and the smoothed one
        #    find its second derivative, and from it recover maxima and minima of the wiggles

        # UnivariateSpline is used to fit a spline pk_in=spl(k_in) of degree k to the provided k_in and pk_in/pk_smooth data
        fun_spline_wiggles = UnivariateSpline(k_in, pk_in / pk_smoothed, k=3, s=0)

        # Find all derivatives of the spline
        derivs = np.array([fun_spline_wiggles.derivatives(_k) for _k in k_in]).T

        # Fit a spline on the second derivative of the wiggles
        deriv_2 = UnivariateSpline(k_in, derivs[2], k=3, s=1.0)

        # Find maxima and minima of the gradient (zeros of 2nd deriv.), then put a
        # low-order spline through zeros to subtract smooth trend from wiggles fn.
        wiggle_zeros = deriv_2.roots()

        # Remove the zeros outside the range of interest (k<k_ref_min or k>k_ref_max)
        wiggle_zeros = wiggle_zeros[np.where(np.logical_and(wiggle_zeros >= k_ref_min, wiggle_zeros <= k_ref_max))]

        # Remove the first zero in the defined k range(k_ref_min,k_ref_max)
        wiggle_zeros = np.delete(wiggle_zeros, 0)

        # 3) Fit a spline through the maxima and minima of the fitted spline
        # Find the index of the maximum of the power spectrum to have the peak as a point for the interpolation
        k_pk_max = k_in[np.argmax(pk_in)]
        right_side = np.where(k_in > k_ref_max)
        right_side = np.squeeze(right_side)
        left_side = np.where(k_in < k_pk_max)#*0.6)
        left_side = np.squeeze(left_side)

        k_fin = np.concatenate((k_in[left_side], wiggle_zeros, k_in[right_side]))

        wiggle_spline_trend = UnivariateSpline(k_fin, fun_spline_wiggles(k_fin), k=3, s=0)

        # 4) Put together the results of step 1, 2 and 3
        # Construct smooth no-BAO:
        bao_features_idxs = np.where(np.logical_and(k_in > k_ref_min, k_in < k_ref_max))
        wave_numbers_bao_features = k_in[bao_features_idxs]

        # Update the smoothed values given the wiggle spline trend
        pk_smoothed[bao_features_idxs] *= wiggle_spline_trend(wave_numbers_bao_features)

        # Interpolate to get the final smoothed power spectrum
        ipk = interp1d(k_in, pk_smoothed, kind='linear', bounds_error=False, fill_value=0.)
        pk_smoothed = ipk(k_in)

        return pk_smoothed

    def MP_v2(self, k_in: np.ndarray, pk_in: np.ndarray, k_ref: np.ndarray =[8e-3, 4e-1], factor_left=1)->np.ndarray:
        """
        Smoothing method based on ArXive paper 1210.7183, 1804.07261

        Parameters:
        ----------
        k_in: np.ndarray:
            set of wave numbers corresponding to the values of our power spectrum
        pk_in: np.ndarray:
            input power spectrum to be smoothed
        k_ref : np.ndarray:
            wave numbers corresponding to the BAO features

        Returns
        -------
        pk_nobao : np.ndarray:
            smoothed power spectrum
        """

        # Spline all (log-log) points outside k_ref range:
        idxs = np.where(np.logical_or(k_in <= k_ref[0], k_in >= k_ref[1]))
        _pk_smooth = UnivariateSpline( np.log(k_in[idxs]),
                                      np.log(pk_in[idxs]), k=3, s=0 )
        pk_smooth = lambda x: np.exp(_pk_smooth(np.log(x)))

        # Find second derivative of each spline:
        fwiggle = UnivariateSpline(k_in, pk_in / pk_smooth(k_in), k=3, s=0)
        derivs = np.array([fwiggle.derivatives(_k) for _k in k_in]).T
        d2 = splrep(k_in, derivs[2], k=3, s=1.0)

        # Find maxima and minima of the gradient (zeros of 2nd deriv.), then put a
        # low-order spline through zeros to subtract smooth trend from wiggles fn.
        wzeros = scipy.interpolate.PPoly.from_spline(d2).roots(extrapolate=True)

        # Remove the zeros outside the range of interest (k<k_ref_min or k>k_ref_max)
        wzeros = wzeros[np.where(np.logical_and(wzeros >= k_ref[0], wzeros <= k_ref[1]))]
        wzeros = np.delete(wzeros, 0)

        # find the index of the maximum of the power spectrum to have the peak as a point for the interpolation
        k_pk_max = k_in[np.argmax(pk_in)]
        right_side = np.where(k_in > k_ref[1])
        right_side = np.squeeze(right_side)
        left_side = np.where(k_in < k_pk_max*factor_left)
        left_side = np.squeeze(left_side)

        k_fin = np.concatenate((k_in[left_side], wzeros, k_in[right_side]))

        wtrend = UnivariateSpline(k_fin, fwiggle(k_fin), k=3, s=0)

        # Construct smooth no-BAO:
        idxs = np.where(np.logical_and(k_in > k_ref[0]*factor_left, k_in < k_ref[1]))
        pk_nobao = pk_smooth(k_in)
        pk_nobao[idxs] *= wtrend(k_in[idxs])

        # Construct interpolating functions:
        ipk = interp1d( k_in, pk_nobao, kind='linear', bounds_error=False, fill_value=0. )

        pk_nobao = ipk(k_in)
        return pk_nobao

    def smoothing_method_1301(self, ks: np.ndarray, pks: np.ndarray, krepstart: float = 0.01,
                              krepwidth: float = 0.2, rfitlow: float = 50, rlow: float = 86,
                              rhigh: float = 150, rfithigh: float = 190)->np.ndarray:
        """
        Smoothing method based on ArXiv paper 1301.3456

        Parameters:
        ----------
        ks: np.ndarray:
            set of wave numbers, k corresponding to the values of our power spectrum.
        pks: np.ndarray:
            input power spectrum P(k) to be smoothed.
        k_repstart : float
            wavenumber where the replacement function starts to take effect.
        krepwidth : float
            transition width (in log-space) for smoothly blending smoothed and original spectra.
        rfitlow : float
            lower bound in configuration space for the fit region.
        rlow : float
            lower bound for the range of r where the correlation function will be replaced.
        rhigh : float
            upper bound for the range of r where the correlation function will be replaced.
        rfithigh : float
            upper bound in configuration space for the fit region.

        Returns
        -------
        smooth_pk : np.ndarray:
            smoothed no-wiggle power spectrum
        """

        def fitfunc(r, *pars):
            """
            Polynomial-like fitting function
            """
            return pars[0]*r + pars[1] + pars[2]/r + pars[3]/r**2 + pars[4]/r**3

        # 1) Transform P(k) → ξ(r) using the Fourier–Hankel transform
        xi_v = fht(pks*(ks**1.5), dln=np.diff(np.log(ks))[0], mu=0.5, offset=0, bias=-0.8)

        # Generate r-space array (distances corresponding to k-values)
        r = np.exp(-np.log(ks)[len(ks)//2] + (np.arange(len(ks))-len(ks)//2)*np.diff(np.log(ks))[0])

        # Normalize ξ(r) from the transformed ξ_v
        xi = xi_v/(2*np.pi*r)**1.5

        # 2) Identify fitting region in r-space
        # rmask selects points in the fitting range [rfitlow, rlow] U [rhigh, rfithigh]
        rmask = np.logical_or(np.logical_and(r>rfitlow, r<rlow) , np.logical_and(r>rhigh, r<rfithigh))

        xi_fit = xi[rmask] # Values to fit
        r_fit = r[rmask] # Corresponding r-values

        # 3) Fit correlation function in selected region
        popt, pcov = curve_fit(fitfunc, r_fit, xi_fit, p0 = [0, 0.01, 0, 0, 0])

        # 4) Replace ξ(r) in the target range (rlow, rhigt)
        rmask_replace = np.logical_and(r>rlow,r<rhigh)
        replace_xi = fitfunc(r[rmask_replace], *popt)
        xi[rmask_replace] = replace_xi

        # 5) Transform ξ(r) → smoothed P(k) using inverse Hankel transform
        smooth_pk = ifht(xi*(2*np.pi*r)**1.5, dln=np.diff(np.log(ks))[0], mu=0.5, offset=0, bias=-.8)/ks**1.5

        # Ensure no extremely small or negative values
        smooth_pk[smooth_pk<1e-20]=1e-20

        # 6) Blend smoothed and original spectrum for small k-values
        # Activation function controls transition in log(k)
        activation = 0.5-0.5*np.tanh((np.log(ks)-np.log(krepstart))/krepwidth)

        # Smooth further using a spline in log-log space
        smooth_pk = np.exp(UnivariateSpline(np.log(ks), np.log(smooth_pk), s=0.01)(np.log(ks)))

        # Combine original and smoothed P(k) with activation weighting
        smooth_pk = pks**activation*smooth_pk**(1-activation)
        return smooth_pk


    def smoothing_method_univariate(self, ks: np.ndarray, pks: np.ndarray, s: float = 0.01, krepstart: float = 0.005,
                                    krepend: float = 0.4, krepwidth: float = 0.4, kfitlow: float = 0.01,
                                    kfithigh: float = 0.3, w_suppression: float = 0.1)->np.ndarray:
        """
        Smooths a power spectrum using a Univariate Spline approach.

        The method fits a spline to the log-log representation of the power spectrum,
        optionally reducing the spline's weight in a specific k-range to
        avoid overfitting oscillatory features. The result is smoothly blended with
        the original spectrum in selected k-ranges.

        Parameters
        ----------
        ks : np.ndarray
            Array of wavenumbers (1/Mpc) corresponding to the power spectrum values.
        pks : np.ndarray
            Power spectrum values P(k) to be smoothed.
        s : float
            Smoothing factor for the Univariate Spline. Smaller values fit more closely.
        krepstart : float
            Wavenumber where blending from original to smoothed spectrum begins.
        krepend : float
            Wavenumber where blending back to original ends.
        krepwidth : float
            Width (in log-space) of the blending transition.
        kfitlow : float
            Lower bound of k-range where spline fit weights are suppressed.
        kfithigh : float
            Upper bound of k-range where spline fit weights are suppressed.
        w_suppression : float
            Multiplicative factor for weight suppression in the fit range.

        Returns
        -------
        pk_smooth: np.ndarray
            Smoothed power spectrum.
        """
        # 1) Create spline fit weights
        wts = np.ones_like(ks)
        # Reduce weight for k-values in [kfitlow, kfithigh] to suppress BAO wiggles
        wts[np.logical_and(ks>kfitlow, ks<kfithigh)]*=w_suppression

        # 2) Fit a spline to log-log data
        pk_smooth = np.exp(UnivariateSpline(np.log(ks),np.log(pks), s=s, w=wts)(np.log(ks)))

        # 3) Compute blending activation function
        # Transition from original (P(k)) to smoothed and back
        activation = 1-0.5*np.tanh((np.log(ks)-np.log(krepstart))/krepwidth)+0.5*np.tanh((np.log(ks)-np.log(krepend))/krepwidth)

        # 4) Blend smoothed and original spectra
        pk_smooth = pks**activation*pk_smooth**(1-activation)

        return pk_smooth

    def smoothing_method_1509_1(self, logkh: np.ndarray, pk_red: np.ndarray, kh: np.ndarray,
                              kstart: float, kend: float, krepstart: float = 0.005, krepend: float = 0.4,
                                krepwidth: float = 0.4, lambdav: float = 0.25)->np.ndarray:
        """
        Smoothing method based on ArXiv paper 1509.02120

        Smooths a reduced power spectrum using 1D Gaussian filtering in log(k) space.

        Parameters
        ----------
        logkh : np.ndarray
            Array of log(wavenumbers) corresponding to the power spectrum values.
        pk_red : np.ndarray
            Reduced power spectrum values P(k) to be smoothed.
        kh : np.ndarray
            Array of wavenumbers (1/Mpc) corresponding to the power spectrum values.
        kstart : float
            Lower bound of wavenumber range to smooth.
        kend : float
            Upper bound of wavenumber range to smooth.
        krepstart : float
            Wavenumber where blending from original to smoothed spectrum begins.
        krepend : float
            Wavenumber where blending back to original ends.
        krepwidth : float
            Width (in log-space) of the blending transition.
        lambdav : float
            Width of the Gaussian filter.

        Returns
        -------
        pk_smooth: np.ndarray
            Smoothed power spectrum.
        """

        # Copy input spectrum to avoid modifying the original
        pnw = pk_red.copy()

        # Apply Gaussian smoothing in log(k) space
        for i,k in enumerate(kh): #enumerate returns both the indexes and the values
            # Filter the values of logkh only if the value of logkh is between
            if logkh[i]-logkh[0]>4.*lambdav and logkh[-1]-logkh[i]>4.*lambdav:
              # Gaussian kernel in log(k) space
              weights = np.exp(-0.5 * (logkh - logkh[i])**2 / lambdav**2)
              # Integrate weighted spectrum using Simpson's rule
              pnw[i] = simpson(pk_red*weights,x=logkh)
              # Normalize by Gaussian kernel area
              pnw[i] *= 1./np.sqrt(2*np.pi*lambdav**2)

        pk_smooth = pnw

        return pk_smooth

    def smoothing_method_1509_2(self, logkh: np.ndarray, pk_red: np.ndarray, kh: np.ndarray,
                                kstart: float, kend: float, krepwidth: float = 0.4,
                                degree: int= 2, n: int = 10)->np.ndarray:
        """
        Smoothing method based on ArXiv paper 1509.02120

        Smooths a reduced power spectrum using a B-spline fitting method.

        Parameters
        ----------
        logkh : np.ndarray
            Logarithmic values of wavenumber.
        pk_red : np.ndarray
            Power spectrum values.
        kh : np.ndarray
            Linear k*h values corresponding to `logkh`.
        kstart : float
            Lower bound of k-range to fit and smooth.
        kend : float
            Upper bound of k-range to fit and smooth.
        krepwidth : float
            Width (in log-space) of the blending transition between smoothed and original.
        degree : int
            Degree of the B-spline used for fitting.
        n : int
            Number of knots for the B-spline.

        Returns
        -------
        pk_smooth: np.ndarray
            Smoothed power spectrum.
        """

        def minfun(coeff):
            """
            Computes sum of squared errors between the CubicSpline
            and a B-spline with the given coefficients over the fit range.
            """
            mse = np.sum((pk_red_t-BSpline(t,coeff,degree)(t_dense))**2)
            return mse

        # Copy spectrum to avoid modifying the input
        pnw = pk_red.copy()

        # Knot positions in log(k) space for B-spline fitting
        t = np.log(np.geomspace(kstart,kh[-1],num=n))

        # Mask for the fit region
        mask = np.logical_and(kh>kstart,kh<kend)
        t_dense = logkh[mask]

        # 1) Fit a cubic spline to the data
        pk_red_t = CubicSpline(logkh, pk_red)(t_dense)

        # 2) Get initial B-spline coefficients from fit range
        _, c, _ = splrep(t_dense, pk_red_t, s=0, k=degree)

        # 3) Optimize B-spline coefficients to minimize error vs. CubicSpline
        optimization_result = minimize(minfun,x0=c)

        # 4) Adjust transition region at the lower edge of the fit
        firstidx = np.argmax(kh>kstart)
        staridx = firstidx-50

        # Difference between B-spline and original at transition points
        dpnw = BSpline(t, optimization_result.x, degree)(logkh[firstidx])-pnw[firstidx]
        dpnwp1 = BSpline(t, optimization_result.x, degree)(logkh[firstidx+1])-pnw[firstidx+1]

        # Derivative estimate for transition matching
        y1p = (dpnwp1-dpnw)/(kh[firstidx+1]-kh[firstidx])*(kh[firstidx]-kh[staridx])
        y1 = dpnw

        # Coefficients for cubic polynomial interpolation in transition
        a = y1p-2*y1
        b = 3*y1-y1p
        x = (kh[staridx:firstidx]-kh[staridx])/(kh[firstidx]-kh[staridx])

        # Apply cubic interpolation correction
        pnw[staridx:firstidx] = pnw[staridx:firstidx] + (a*x*x*x+b*x*x)

        # Replace values in fit region with B-spline fit
        pnw[mask] = BSpline(t,optimization_result.x,degree)(t_dense)

        # 5) Blend original and smoothed spectrum with activation function
        activation = 1+0.5*np.tanh((logkh-np.log(kend)+3*krepwidth)/krepwidth)-0.5*np.tanh((logkh-np.log(kstart))/krepwidth)
        pk_smooth = pk_red*activation + pnw*(1-activation)

        return pk_smooth

    def smoothing_method_1509_3(self, logkh: np.ndarray, pk_red: np.ndarray, kh: np.ndarray,
                                kstart: float, kend: float, krepwidth: float = 0.4)->np.ndarray:
        """
        Smoothing method based on ArXiv paper 1509.02120

        Smooths a reduced power spectrum by averaging multiple B-spline fits
        with different degrees and knot counts.

        Parameters
        ----------
        logkh : np.ndarray
            Logarithmic values of wavenumber k.
        pk_red : np.ndarray
            Power spectrum values.
        kh : np.ndarray
            Linear k values corresponding to `logkh`.
        kstart : float
            Lower bound of k-range for smoothing.
        kend : float
            Upper bound of k-range for smoothing.
        krepwidth : float, optional
            Width (in log-space) of the transition between smoothed and original spectra.

        Returns
        -------
        pk_smooth: np.ndarray
            Smoothed power spectrum.

        """

        def minfun(coeff):
            """
            Computes sum of squared errors between the CubicSpline
            and a B-spline with given coefficients in the fit range.
            """
            mse = np.sum((pk_red_t-BSpline(t,coeff,pos['degree'])(t_dense))**2)
            return mse

        pnw = pk_red.copy()

        # Possible B-spline configurations to test
        possibilities = [{'degree':2, 'n':8}, {'degree':2, 'n':10},
                        {'degree':3, 'n':10}, {'degree':3, 'n':12},
                        {'degree':4, 'n':12},{'degree':4, 'n':14},
                        {'degree':5, 'n':14}, {'degree':5, 'n':16}]

        # Mask for fit range in k-space
        mask = np.logical_and(kh>kstart,kh<kend)
        t_dense = logkh[mask]

        nowiggles = np.empty((len(possibilities),np.count_nonzero(mask)))
        # 1) Fit CubicSpline to data in the fit range
        pk_red_t = CubicSpline(logkh, pk_red)(t_dense)

        # 2) Fit multiple B-splines with different settings
        for ipos,pos in enumerate(possibilities):
            deg = pos['degree']
            t = np.log(np.geomspace(kstart,kh[-1],num=pos['n']))

            # Initial guess for B-spline coefficients
            _, c, _ = splrep(t_dense, pk_red_t, s=0, k=deg)

            # Optimize coefficients to best match the CubicSpline
            optimization_result = minimize(minfun,x0=c)

            # Store smoothed values for this configuration
            nowiggles[ipos] = BSpline(t,optimization_result.x,deg)(t_dense)

        # 4) Average all "no-wiggle" fits
        nowiggle_tot = np.mean(nowiggles,axis=0)
        pnw = pk_red.copy()

        # Replace original spectrum values in mask range with averaged no-wiggle values
        pnw[mask] = nowiggle_tot

        # 5) Blend original and smoothed spectra with activation function
        activation = 1+0.5*np.tanh((logkh-np.log(kend))/krepwidth)-0.5*np.tanh((logkh-np.log(kstart))/krepwidth)
        pk_smooth = pk_red*activation + pnw*(1-activation)

        return pk_smooth

    def smoothing_method_1604(self, kh: np.ndarray, pk_red:np.ndarray, sigmasq: float = 1.0,
                              sigmasq_2: float = 5e-3, alpha: float = 0.5,
                              beta: float = 0.0, n: int = 13)->np.ndarray:
        """
        Smoothing method based on ArXiv paper 1604.01830

        Smooths a reduced power spectrum using a weighted polynomial fit in log(k) space

        Parameters
        ----------
        kh : np.ndarray
            Array of wavenumber k values.
        pk_red : np.ndarray
            Power spectrum values.
        sigmasq : float
            Variance of the first Gaussian weight suppression term.
        sigmasq_2 : float
            Variance of the second Gaussian weight adjustment term.
        alpha : float
            Amplitude of the first Gaussian suppression term.
        beta : float
            Amplitude of the second Gaussian adjustment term.
        n : int
            Degree of the polynomial fit.

        Returns
        -------
        pk_smooth: np.ndarray
            Smoothed power spectrum.
        """

        # 1) Convert k to log-space
        logkh = np.log(kh)

        # 2) Define polynomial fit with weighted suppression
        weight = 1. - alpha*np.exp(-kh*kh/(2.*sigmasq)) + beta*np.exp(-0.5*kh*kh/sigmasq_2)

        # 3) Fit polynomial in log(k) space
        poly = np.polynomial.polynomial.Polynomial.fit(logkh,pk_red,deg=n,w=weights)

        # 4) Evaluate polynomial to get smoothed P(k)
        pnw = poly(logkh)
        pk_smooth = pnw
        return pk_smooth

    def smoothing_method_1605_1(self, logkh: np.ndarray, pk_red: np.ndarray, kh: np.ndarray,
                                pk:np.ndarray, kstart: float, kend: float, krepstart: float = 0.01,
                                krepend: float = 0.2, krepwidth: float = 0.4)->np.ndarray:
        """
        First smoothing method based on ArXiv paper 1605.02149

        This approach fits the analytical Eisenstein & Hu 1998 (EH98) model
        to the power spectrum in a chosen k-range, then replaces the BAO
        wiggle region with the model prediction, blending smoothly between
        model and data.

        Parameters
        ----------
        logkh : np.ndarray
            log(k) values.
        pk_red : np.ndarray
            Power spectrum values.
        kh : np.ndarray
            k values.
        pk : np.ndarray
            Power spectrum.
        kstart, kend : float
            Fitting range for the EH98 model.
        krepstart, krepend : float
            Replacement range for blending model and data.
        krepwidth : float
            Width of the tanh activation transition.

        Returns
        -------
        pk_smooth: np.ndarray
            Smoothed reduced power spectrum.
        """
        def pk_model_eh(k,*par):
          """
          EH98 model wrapper for curve_fit
          """
          par_prime = np.array(par)
          return EHclass.EH98_fit(k,par_prime)

        pk_smooth = np.empty_like(pk_red)

        # Initial guess for EH98 parameters
        xini_eh = [1.,1.967,6.831,0.4043,55.53,1.5e7,0.96]

        # Fit EH98 model in fitting range
        mask = np.logical_and(kh<kend,kh>kstart)
        popt,pcov = curve_fit(pk_model_eh,kh[mask],pk[mask],p0=xini_eh,maxfev=2000)

        # Smooth transition between data and model
        activation = 1+0.5*np.tanh((logkh-np.log(krepend))/krepwidth)-0.5*np.tanh((logkh-np.log(krepstart))/krepwidth)

        # Construct the final smoothed power spectrum
        pk_smooth = pk_red*activation + np.log(pk_model_eh(kh,*popt))*(1-activation)

        return pk_smooth

    def smoothing_method_1605_2(self, kh: np.ndarray, pk: np.ndarray)->np.ndarray:
        """
        Second smoothing method based on ArXiv paper 1605.02149

        Parameters
        ----------
        kh : np.ndarray
            Wavenumbers k in units of h/Mpc.
        pk : np.ndarray
            Power spectrum values P(k).

        Returns
        -------
        pk_smooth: np.ndarray
            Smoothed power spectrum P(k).
        """
        def pk_model(x,*pars):
          """
          Parametric model for the smoothed matter power spectrum in log-space.

          Parameters
          ----------
          x : np.ndarray
              Wavenumbers (k) in units of h/Mpc.
          *pars : sequence of floats
              Model parameters in the order:
              [A, ns, c1, c2, c3, c4, c5, c6, c7]
              where:
                  A   : Amplitude scaling factor.
                  ns  : Spectral index (power-law index).
                  c1–c4 : Parameters controlling the effective wave number (keff)
                          and transfer function shape.
                  c5–c7 : Parameters controlling the broadband boost term Delta(k).

          Returns
          -------
          np.ndarray
              Natural log of the modeled power spectrum P(k) evaluated at x.
          """
          A = pars[0] # Overall amplitude of P(k)
          ns = pars[1] # Spectral index (power-law slope)
          c1,c2,c3,c4,c5,c6,c7 = pars[2:] # Shape parameters

          # Check: enforce non-negative shape parameters
          if c1 < 0 or c2 < 0 or c3 < 0 or c4<0 or c5<0 or c6<0 or c7<0:
            return 0.*x # return zeros if parameters are unphysical

          # Effective wave number scaling
          keff = c2 * x*1./(1.+c3/(1.+(c4*x)**4))

          # Logarithmic factor for transfer function numerator
          Lk = np.log(np.e+c1*keff)

          # Empirical denominator term for the transfer function
          Ck = 14.4+325./(1.+60.5*keff**1.08)

          # Transfer function T(k)
          Tk = Lk/(Lk+Ck*keff**2)

          # Smooth broadband "boost" term Delta(k)
          Delta = c5 * (np.tanh(np.log(c6*x)/c7)+1)

          return np.log(A*x**ns*Tk**2*(1+Delta))

        # Initial parameter guesses
        xini = [1.5e7,0.96,1.967,6.831,0.4043,55.53,0.00425,2.5,0.35]

        # Rescale amplitude A so that model matches pk[0] at the first k value
        xini[0] = pk[0]/np.exp(pk_model(kh[0],*xini))*xini[0]

        # Fit the parametric model to log(pk) using non-linear least squares
        popt,pcov = curve_fit(pk_model,kh,np.log(pk),p0=xini,maxfev=2000)

        # Evaluate the smoothed model using the best-fit parameters
        pnw = pk_model(kh,*popt)

        pk_smooth = pnw
        return pk_smooth

    def smoothing_method_1605_3(self, kh: np.ndarray, pk:np.ndarray, bao_left: float = 0.001,
                                bao_right: float = 0.8, n_count_wanted: int = 20,
                                kpiv: float = 0.03)->np.ndarray:
        """
        Third smoothing method based on ArXiv paper 1605.02149

        Parameters
        ----------
        kh : np.ndarray
            Wavenumbers k in units of h/Mpc.
        pk : np.ndarray
            Power spectrum values P(k).
        bao_left: float
            Left limit of BAO features.
        bao_right: float
            Right limit of BAO features.
        n_count_wanted: int
            Number of sampling points used for smoothing.
        kpiv: float
            Pivot scale.

        Returns
        -------
        pk_smooth : np.ndarray
            Smoothed power spectrum (log-space values).
        """

        # 1) Mask out the BAO region (only keep k < bao_left or k > bao_right)
        mask = np.logical_or(kh<bao_left,kh>bao_right)

        # 2) Determine downsampling step size based on target number of points
        step = (np.count_nonzero(mask)//n_count_wanted)

        # 3) Reduce the number of masked points by skipping most of them
        for i in range(len(mask)):
          if i%step==0:
            continue
          else:
            mask[i] = False

        # 4) Inclusion of the pivot point and the last point
        mask[np.argmin((kh-kpiv)**2)] = True
        mask[-1] = True

        # 5) Apply the mask to select subset of k and P(k)
        pk_mask = pk[mask]
        kh_mask = kh[mask]

        # 6) Fit cubic spline in log-log space and evaluate on full grid
        pnw = CubicSpline(np.log(kh_mask),np.log(pk_mask))(np.log(kh))
        pk_smooth = pnw

        return pk_smooth

    def smoothing_method_0907(self, kh: np.ndarray, pk:np.ndarray,bao_left: float = 0.001,
                              bao_right: float = 0.8) -> np.ndarray:
        """
        Third smoothing method based on ArXiv paper 0907.1659

        Parameters
        ----------
        kh : np.ndarray
            Wavenumbers k in units of h/Mpc.
        pk : np.ndarray
            Power spectrum values P(k).
        bao_left: float
            Left limit of BAO features.
        bao_right: float
            Right limit of BAO features.
        Returns
        -------
        pk_smooth : np.ndarray
            Smoothed power spectrum.
        """

        # 1) Mask points outside the BAO region
        mask = np.logical_or(kh<bao_left,kh>bao_right)

        # 2) Define key interpolation nodes
        nodes = np.concatenate([[7e-4], np.geomspace(0.0175, 0.262, num=8)])

        # 3) Ensure each node is represented in the mask
        for knod in nodes:
            mask[np.argmin((kh-knod)**2)] = True

        # 4) Extract masked k and P(k) values
        kh_mask = kh[mask]
        # Multiply by k^1.5 to reduce dynamic range before interpolation
        pk_mask = pk[mask]*kh_mask**1.5


        # 5) Interpolate in log-log space with a cubic spline
        # Then exponentiate and divide by k^1.5 to reverse the earlier weighting
        pnw = np.exp(CubicSpline(np.log(kh_mask),np.log(pk_mask))(np.log(kh)))/kh**1.5
        pk_smooth = pnw

        return pk_smooth

    def smoothing_method_2004(self, kh: np.ndarray, pk: np.ndarray, bao_idx_range: tuple, fit_bounds: tuple,
                              rep_bounds: tuple, power_even: float, power_odd: float, doplot: bool,
                              kmin: float = 1e-4, kmethod: float = 7e-5)->np.ndarray:
        """
        Smoothing method based on ArXiv paper 2004.10607
        The smoothing is done using a fast sine transform.

        Parameters
        ----------
        kh : np.ndarray
            Wavenumbers k in units of h/Mpc.
        pk : np.ndarray
            Power spectrum values P(k).
        bao_idx_range : tuple
            Index range corresponding to BAO features.
        fit_bounds : tuple
            Index bounds outside the BAO region used for fitting.
        rep_bounds : tuple
            Index bounds for which the DST components are replaced by fitted values.
        power_even : float
            Power used to scale even-indexed DST components before fitting.
        power_odd : float
            Power used to scale odd-indexed DST components before fitting.
        doplot : bool
            Whether to plot intermediate DST components for debugging.
        kmin : float
            Minimum wavenumber to include in final smoothed spectrum.
        kmethod : float
            Starting wavenumber for DST interpolation.

        Returns
        -------
        pk_smooth : np.ndarray
            Smoothed power spectrum.
        """
        def fitfunc(x,*pars):
          """
          Polynomial function for fitting DST components in log-space
          """
          sumres =  np.sum([pars[i]*x**(i+1.) for i in range(len(pars))],axis=0)
          return sumres

        # 1) Define high-resolution k-array for DST interpolation
        N = 65536
        k_method = np.linspace(kmethod,7.,num=N)

        # 2) Cubic spline interpolation of log(k*P(k))
        pk_method = CubicSpline(kh,np.log(kh*pk),extrapolate=False)(k_method)

        # 3) Compute DST (type 2) of interpolated log(k*P(k))
        dstres = dst(pk_method,type=2)
        idx_range = np.arange(N//2)

        # 4) Mask regions outside BAO features for fitting
        cut_mask = np.logical_or(
                      np.logical_and(idx_range>fit_bounds[0],idx_range<bao_idx_range[0]),
                      np.logical_and(idx_range<fit_bounds[1],idx_range>bao_idx_range[1]) )

        # 5) Initial guesses for polynomial fits
        p0 = [1e5,1e5,1e3,1e1,1e0,0.]

        # 6) Fit even and odd DST components in log-space
        popt_even, pcov_even = curve_fit(fitfunc,np.log(idx_range[cut_mask]),
                                         np.log((dstres[::2]*idx_range**power_even)[cut_mask]),p0=p0)

        popt_odd, pcov_odd = curve_fit(fitfunc,np.log(idx_range[cut_mask]),
                                       np.log((dstres[1::2]*idx_range**power_odd)[cut_mask]),p0=p0)

        # 7) Replace DST components in BAO-dominated region using fitted polynomials
        mask_replacement = np.logical_and(idx_range>rep_bounds[0],idx_range<rep_bounds[1])
        replace_even = (dstres[::2]*idx_range**power_even).copy()
        replace_even[mask_replacement] = np.exp(fitfunc(np.log(idx_range[mask_replacement]),*popt_even))
        replace_odd = (dstres[1::2]*idx_range**power_odd).copy()
        replace_odd[mask_replacement] = np.exp(fitfunc(np.log(idx_range[mask_replacement]),*popt_odd))

        # 8) Reconstruct full DST array with replaced values
        dstres_new = np.empty_like(dstres)
        dstres_new[2::2] = replace_even[1:]/idx_range[1:]**power_even
        dstres_new[3::2] = replace_odd[1:]/idx_range[1:]**power_odd
        dstres_new[0] = dstres[0]
        dstres_new[1] = dstres[1]

        # 9) Inverse DST to return to log(k*P(k)) space
        log_kPk_new = idst(dstres_new,type=2)/(2*N)

        # 10) Interpolate smoothed log(k*P(k)) back to original kh
        kh_mask = np.logical_and(kh>kmin, kh<7.)
        log_kPk_interpolated = CubicSpline(np.log(k_method),log_kPk_new,extrapolate=False)(np.log(kh[kh_mask]))

        # 11) Apply smoothed spectrum, keeping original values outside range
        Pk_smooth = np.empty_like(pk)
        Pk_smooth[kh_mask] =  np.exp(log_kPk_interpolated)/kh[kh_mask]
        Pk_smooth[~kh_mask] = pk[~kh_mask]

        return Pk_smooth

class Likelihood_with_EH:
    """
    General class implementing functions for smoothing the power spectrum

    smoothing_method_sam : smoothing method used in the original ShapeFit approach

    smooth_pkm:
        General function for smoothing the power spectrum using a given method.
    """



    def smoothing_method_sam(self, kh: np.ndarray, pk: np.ndarray, k_start: float,
                             power_reduction: str = "log", interpkind: str = "quadratic")->np.ndarray:
        """
        Method for smoothing the power spectrum based on ArXiv paper 2204.11868
        The smoothing can be done both in the linear and log domain

        Parameters:
        -----------
            kh: np.ndarray
                Wave numbers values.
            pk: np.ndarray
                Power spectrum values.
            k_start: float
                Initial wave number.
            power_reduction: str
                To determine whether to work in log or linear space.
            interpkind: str
                Kind of interpolation to perform.

        Returns:
        --------
            smoothpk (np.ndarray)
                smoothed power spectrum
        """

        logmode = (power_reduction == "log")
        logkh = np.log10(kh)
        if (logmode):
            logpk = np.log10(pk)
        else:
            logpk = pk

        # Compute derivative of P(k) in chosen domain
        gradient = np.gradient(logpk)

        # Index where k first exceeds k_start
        start = np.max(np.where(kh<k_start))

        # Locate local maxima/minima of the gradient
        locmax,extra = find_peaks(gradient)
        locmin,extra = find_peaks(-gradient)

        # Find global maximum index in selected region
        if (logmode):
            glomax = np.argmax(logpk[start:])+start #we ignore everything before start and want the index of the max
        else:
            glomax = start

        # Remove extrema before start cutoff
        locmin = np.delete(locmin,np.where(locmin<start))
        locmax = np.delete(locmax,np.where(locmax<start))

        # Index marking the end of oscillatory region
        commonend  = np.max(np.hstack([locmax,locmin])) + 1

        # Split into left and right segments depending on glomax position
        if (glomax<commonend):
            #"common left part"
            logkhl = logkh[:glomax]
            logpkl = logpk[:glomax]
            #"common right part"
            logkhr = logkh[commonend:]
            logpkr = logpk[commonend:]
        else:
            #"common left part"
            logkhl = logkh[:start]
            logpkl = logpk[:start]
            #"common right part"
            logkhr = logkh[commonend:]
            logpkr = logpk[commonend:]

        # Build curves for maxima and minima interpolation
        logkhma = np.hstack((logkhl,logkh[locmax],logkhr))
        logpkma = np.hstack((logpkl,logpk[locmax],logpkr))
        logkhmi = np.hstack((logkhl,logkh[locmin],logkhr))
        logpkmi = np.hstack((logpkl,logpk[locmin],logpkr))

        # Interpolators for maxima/minima
        maxfun =interp1d(logkhma,logpkma,kind=interpkind)
        minfun =interp1d(logkhmi,logpkmi,kind=interpkind)

        # Average interpolated maxima and minima curves
        if (logmode):
            smoothpk = (10** maxfun(logkh)+10** minfun(logkh)) *0.5
        else:
            smoothpk = (maxfun(logkh)+minfun(logkh))*0.5
        return smoothpk


    def smooth_pkm(self, kh: np.ndarray, pk: np.ndarray,
                   krepstart: float = 0.005,
                   krepend: float = 0.4,
                   krepwidth: float = 0.4,
                   kstart: float = 0.02,
                   kend: float = 0.4, power_reduction: str = "log",
                   interpkind: str = 'quadratic', method: str = "sam", options: dict = {},
                   bao_idx_range: tuple = [120,240], fit_bounds: tuple = [50,500],
                   rep_bounds: tuple = [120,240], power_even: float = 1.3,
                   power_odd: float = 1.5, doplot: bool = False)->np.ndarray:
      """
      High-level smoothing function for the power spectrum P(k).
      The default method is "sam".

      Parameters
      ----------
      kh : np.ndarray
          Wavenumbers k in units of h/Mpc.
      pk : np.ndarray
          Power spectrum values P(k).
      krepstart, krepend : float
          Wavenumber k range.
      krepwidth : float
          Width parameter for some spline smoothing methods.
      kstart, kend: float
          Range of wavenumbers including the BAO features.
      interpkind: str
          Interpolation kind.
      bao_idx_range : list
          BAO feature index range.
      fit_bounds : list
          Bounds for fittingt.
      rep_bounds : list
          Bounds for replacment.
      power_even, power_odd : float
          Parameters for smoothing_method_2004.

      Returns
      -------
      pk_smooth : np.ndarray
          Smoothed power spectrum.

      Raises
      ------
      ValueError
          If the input method is not recognized.
      """
      sm = smoother()
      logkh = np.log(kh)

      # 1) No smoothing
      if method == "none":
        return pk

      # 2) Monte-Python method
      if method=="MP":
        return sm.MP(kh, pk, np.array([kstart,kend]))
      if method=="MP_v2":
        return sm.MP_v2(kh,pk, np.array([kstart,kend]), factor_left=options.get('factor_left',1))

      # 3) ShapeFit's original smoothing method
      if method=="sam":
        if options["fiducial"]:
          pk = pk/options["EHfid"]
          pk = self.smoothing_method_sam(kh, pk, kstart, power_reduction, interpkind) * options["EHfid"]
          return pk
        else:
          pk = (pk / options["EH"])/(options["pk_temp"]/options["EHfid"])
          pk = self.smoothing_method_sam(kh, pk, kstart, power_reduction, interpkind)
          pk_sam = options["EH"]* pk /options["EHfid"]*options["pk_temp"]
          return pk_sam
      elif method=="sam_v2":
        # Mask for target smoothing range (scaled by hfid)
        mask_replace = np.logical_and(kh/options['hfid']>0.005, kh/options['hfid']<0.5)
        eh_cut_fid = options["EHfid"][mask_replace]
        eh_cut = options["EH"][mask_replace]
        if options["fiducial"]:
          pk_sam = pk[mask_replace]/eh_cut_fid
          pk_sam = self.smoothing_method_sam(kh[mask_replace]/options['hfid'], pk_sam, kstart, power_reduction, interpkind) * eh_cut_fid
          pk_ret = pk.copy()
          pk_ret[mask_replace] = pk_sam
          return pk_ret
        else:
          pk_sam = CubicSpline(kh,pk)(kh[mask_replace]*options["rdfac"])
          pk_sam = (pk_sam / eh_cut)/(options["pk_fid"][mask_replace]/eh_cut_fid)
          pk_sam = self.smoothing_method_sam(kh[mask_replace]/options['hfid'], pk_sam, kstart, power_reduction, interpkind)
          pk_sam = pk_sam * eh_cut /eh_cut_fid
          pk_ret = pk.copy()
          pk_ret[mask_replace] = CubicSpline(kh[mask_replace]/options['hfid'],pk_sam)(kh[mask_replace]*(1/options["rdfac"])/options['hfid']) *CubicSpline(kh,options['pk_temp'])(kh[mask_replace]*(1/options["rdfac"]))
          return pk_ret

      elif method=="sam_old":
        return self.smoothing_method_sam(kh, pk, kstart, interpkind)

      # 4) Algorithms implemented from the literature
      if power_reduction == "log":
        pk_red = np.log(pk)
      else:
        pk_red = pk

      if method=="0907.1659":
        # Cubic Spline fit
        pk_smooth = sm.smoothing_method_0907(kh, pk, **options)

      elif method=="1301.3456":
        # Correlation function Hankel transform
        pk_smooth = sm.smoothing_method_1301(kh, pk, krepstart=krepstart, krepwidth=krepwidth, **options)

      elif method=="univariate":
        # Fitting univariate spline
        pk_smooth = sm.smoothing_method_univariate(kh, pk, krepstart=krepstart, krepend=krepend, krepwidth=krepwidth, **options)

      elif method=="1509.02120":
        # Gaussian smoothing
        pk_smooth = sm.smoothing_method_1509_1(logkh, pk_red, kh, kstart, kend, **options)

      elif method=="1509.02120_2":
        #  B-spline
        pk_smooth = sm.smoothing_method_1509_2(logkh, pk_red, kh, kstart, kend, krepwidth=krepwidth, **options)

      elif method=="1509.02120_3":
        # multi B-Spline
        pk_smooth = sm.smoothing_method_1509_3(logkh, pk_red, kh, kstart, kend, krepwidth=krepwidth)

      elif method=="1604.01830":
        # Polynomial fit
        pk_smooth = sm.smoothing_method_1604(kh, pk_red, **options)

      elif method=="1605.02149":
        # EH fit
        pk_smooth = sm.smoothing_method_1605_1(logkh, pk_red, kh, pk, kstart, kend, krepstart=krepstart, krepend=krepend, krepwidth=krepwidth)

      elif method=="1605.02149_2":
        # EH fit v2
        pk_smooth = sm.smoothing_method_1605_2(kh, pk)

      elif method=="1605.02149_3":
        # Cubic Spline fit
        pk_smooth = sm.smoothing_method_1605_3(kh, pk, **options)

      elif method=="2004.10607":
        # Correlation function, Fast sine transform
        pk_smooth = sm.smoothing_method_2004(kh, pk, bao_idx_range, fit_bounds, rep_bounds, power_even, power_odd, doplot, kmin=kstart, kmethod=options['kmethod'])
      else:
          raise ValueError("Unexpected input: Method  '{}'".format(method))


      # Post-processing: revert domain transformation if needed
      if power_reduction == "log":
        return np.exp(pk_smooth)
      elif power_reduction == "EH":
        return pk_smooth*EHclass.EH98(cosmo, kh, 0., 1.)
      else:
        return pk_smooth

class cosmology_generator:
    """
    A class for generating and analyzing cosmological power spectra
    using various smoothing and derivative estimation methods.
    """

    def __init__(self, fiducial_cosmo: dict, cosmology: dict, kmin: float = 2e-5) -> np.ndarray:
        """
        Initial function including basic setup for computing the power spectrum using any cosmology.

        Parameters
        ----------
        fiducial_cosmo : dict
            Dictionary of fiducial cosmology parameters for comparison.
        cosmology : dict
            Dictionary of cosmology parameters for the target cosmology.
        kmin : float, optional
            Minimum wavenumber (in 1/Mpc) for the analysis.

        Attributes
        ----------
        cosmo : classy.Class or edeclassy.Class
            Computed cosmology object for target parameters.
        cosmo_fiducial : classy.Class
            Computed cosmology object for the fiducial parameters.
        ks, pks : ndarray
            Wavenumbers and power spectrum for the target cosmology.
        ks_out, ks_shift, ks_fid_shift : ndarray
            Output wavenumber arrays in various shifted/scaled forms.
        arguments : dict
            Dictionary of method configurations for smoothing algorithms.
        derivative_methods : list
            Available derivative computation methods.
        processing_methods : list
            Available power-spectrum post-processing methods to smooth the BAO residuals.
        """
        # --- Basic setup ---
        z_star = 0.61  # Reference redshift for power spectrum calculations
        k_max_class = 20 # Maximum k for class output
        omegancdmfid=0.00064 # Fiducial non-CDM density

        # --- Default ΛCDM cosmology parameters ---
        cosmology_default = {'output':'mPk',
                            'P_k_max_1/Mpc':k_max_class,
                            'YHe':0.24,
                            'recombination':'recfast',
                            'A_s':2.1e-9, # Scalar amplitude
                            'tau_reio':0.0952, # Optical depth
                            'N_ur':2.038, # Effective number of massless neutrinos
                            'N_ncdm':1,
                            'deg_ncdm':1,
                            'omega_ncdm':omegancdmfid,
                            'P_k_max_1/Mpc':k_max_class,
                            'k_per_decade_for_pk':40,
                            'z_max_pk':6}

        # --- EDE (Early Dark Energy) parameters ---
        ded_cosmo_dict = {'scf_potential' : 'axion',
                          'n_axion':3,
                          'log10_axion_ac': -3.531,
                          'scf_parameters':'2.72,0.0',
                          'scf_evolve_as_fluid':'no',
                          'scf_evolve_like_axionCAMB':'no',
                          'do_shooting':'yes',
                          'do_shooting_scf':'yes',
                          'scf_has_perturbations':'yes',
                          'attractor_ic_scf':'no',
                          'compute_phase_shift':'no',
                          'loop_over_background_for_closure_relation':'yes'}

        # --- Build cosmology object ---
        if "fraction_axion_ac" in cosmology:
          cosmo = edeclassy.Class()
          cosmo.set(cosmology_default)
          cosmo.set(ded_cosmo_dict)
          cosmo.set(cosmology)
          cosmo.compute()
          self.cosmo = cosmo
        else:
          cosmo = classy.Class()
          cosmo.set(cosmology_default)
          cosmo.set(cosmology)
          cosmo.compute()
          self.cosmo = cosmo

        # --- Fiducial cosmology ---
        cosmo_fiducial = classy.Class()
        cosmo_fiducial.set(cosmology_default)
        cosmo_fiducial.set(fiducial_cosmo)
        cosmo_fiducial.compute()

        self.cosmo_fiducial = cosmo_fiducial
        self.h_fid = fiducial_cosmo['h']

        # --- Wavenumber setup ---
        ks_h = np.geomspace(5/2.*kmin,15.,num=1001)

        # Radius of sound horizon at drag epoch
        rd_fid_in_Mpc = cosmo_fiducial.rs_drag()
        rd = cosmo.rs_drag()

        #for omegak we used 5e-4
        ks_noshift = np.geomspace(kmin,15.,num=1000) # in Mpc

        # --- Computing the power spectrum ---
        pks_noshift = cosmo.get_pk_all(ks_noshift,z=z_star) #get_pk_all will evaluate the power spectrum
        pks_fid_noshift = cosmo_fiducial.get_pk_all(ks_noshift,z=z_star)

        dict_sam = {'EH': EHclass.EH98(cosmo,ks_noshift*rd_fid_in_Mpc/cosmo.h()/rd,z_star,1.0),
                      'EHfid': EHclass.EH98(cosmo_fiducial,ks_noshift/h_fid,z_star,1.0),
                      'pk_fid':pks_fid_noshift,
                      'pk_temp':None,
                      'fiducial': True,
                      'hfid':h_fid,
                      'rdfac':rd_fid_in_Mpc/rd
                      }

        self.ks = ks_noshift
        self.pks = pks_noshift

        # --- Shifted k arrays ---
        ks = ks_h * h_fid * rd_fid_in_Mpc/rd
        ks_fid = ks_h * h_fid

        self.ks_out = ks_h # in h/Mpc
        self.ks_shift = ks # in Mpc
        self.ks_fid_shift = ks_fid # In Mpc
        self.rescale_factor = rd_fid_in_Mpc/rd
        self.pks_fid = pks_fid_noshift

        # --- BAO scale limits ---
        self.k_start_bao=0.01
        self.k_end_bao=4.5e-1
        self.k_start_bao_mod = self.k_start_bao*rd/rd_fid_in_Mpc
        self.k_end_bao_mod = self.k_end_bao*rd/rd_fid_in_Mpc
        self.kmpiv = 0.03 # Pivot scale

        # --- Method configurations ---
        self.arguments = {
            'none_v2':{'method':'none'},
            'sam_old':{'method':'sam_old'},
            'sam':{'method':'sam','kstart':0.02,'power_reduction':'none','options':dict_sam},
            'sam_v2':{'method':'sam_v2','kstart':0.02,'power_reduction':'none','options':dict_sam},
            '0907':{'method':'0907.1659','power_reduction':'none', 'options':{'bao_left':4e-3,'bao_right':2*self.k_end_bao}},
            '1301':{'method':'1301.3456','power_reduction':'none','options':{'rlow':100,'rhigh':200, 'rfitlow':50,'rfithigh':300,},
                                                    'krepstart':1e-2,
                                                    'krepwidth':0.2
                   },
            '1509':{'method':'1509.02120','options':{'lambdav':0.25}},
            '1509_2':{'method':'1509.02120_2','options':{'n':8,'degree':2},'kstart':0.015,'kend':1.0,'krepwidth':0.4},
            '1509_3':{'method':'1509.02120_3','kstart':0.005,'kend':1.0,'krepwidth':0.4},
            '1604':{'method':'1604.01830'},
            'univariate':{'method':'univariate','power_reduction':'none',
                          'options':{'s':0.05,'kfitlow':0.01,'kfithigh':0.3,'w_suppression':0.5},'krepwidth':0.4},
            #'1605':{'method':'1605.02149','kstart':0.001,'kend':0.8},
            '1605_2':{'method':'1605.02149_2','kstart':5e-5,'kend':0.8},
            '1605_3':{'method':'1605.02149_3','options':{'bao_left':self.k_start_bao,'bao_right':2*self.k_end_bao,'kpiv':self.kmpiv*self.h_fid}},
            '2004':{'method':'2004.10607','power_reduction':'none',
                    'bao_idx_range':[120,240], 'fit_bounds':[50,500], 'rep_bounds':[120,240],
                    'power_even':1.3, 'power_odd':1.5,
                    'kstart':1e-4,
                    'doplot':True,
                    'options':{'kmethod':7e-5}},
            #'MP':{'method':'MP','kstart':self.k_start_bao,'kend':self.k_end_bao},
            'MP_v2':{'method':'MP_v2','kstart':self.k_start_bao,'kend':self.k_end_bao, 'options':{'factor_left':0.01}}
        }


        self.derivative_methods = ["gradient", "sam", "sam_res", "poly", "steps", "tanh", "tanh_fit"]
        self.processing_methods = ["none", "mean", "savitzky"]#, "univariate"]

        inputinfos = {x:{} for x in self.derivative_methods}
        inputinfos["sam_res"] = {"kmin" : 0.01}
        inputinfos["steps"] = {"xvals" : [0.02, 0.052]}
        inputinfos["tanh"] = {"a" : 0.5, "kpiv":0.05}

        self.inputinfos = inputinfos


    def smooth_single(self, nowiggle_key: str, replace_options: dict = None, krepstart: float = None,
                      krepend: float = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Smooth the power spectrum for a given method.

        Parameters
        ----------
        nowiggle_key : str
            Key for the smoothing method in self.arguments.
        replace_options : dict
            Dictionary of options for the method.
        krepstart : float
            Starting k-value.
        krepend : float
            Ending k-value.

        Returns
        -------
        pk_smooth : np.ndarray
            Smoothed power spectrum.
        pk_smooth_fid : np.ndarray
            Smoothed fiducial power spectrum.
        """

        leh = Likelihood_with_EH()

        arg = deepcopy(self.arguments[nowiggle_key])
        if 'options' in arg and replace_options is not None:
            arg['options'].update(replace_options)
        if krepstart is not None:
            arg['krepstart'] = krepstart
        if krepend is not None:
            arg['krepend'] = krepend

        # Fiducial power spectrum smoothing
        pk_smooth_fid =  leh.smooth_pkm(self.ks, self.pks_fid, **arg)

        # Adjust parameters for power spectrum of the target cosmology
        arg_modified = deepcopy(arg)
        if nowiggle_key == 'sam_v2' or nowiggle_key=="sam":
           arg_modified['options']['fiducial'] = False
           arg_modified['options']['pk_temp'] = pk_smooth_fid.copy()
        elif nowiggle_key == 'MP' or nowiggle_key == 'MP_v2':
           arg_modified['kstart']=self.k_start_bao_mod
           arg_modified['kend']=self.k_end_bao_mod
        elif nowiggle_key == '1605_3':
           arg_modified['options']['bao_left']=arg_modified['options']['bao_left']/ self.rescale_factor#self.k_start_bao_mod
           arg_modified['options']['bao_right']=arg_modified['options']['bao_right']/ self.rescale_factor#2*self.k_end_bao_mod
        elif nowiggle_key == '0907':
           arg_modified['options']['bao_left']=self.k_start_bao_mod/0.01*4e-3
           arg_modified['options']['bao_right']=2*self.k_end_bao_mod

        pk_smooth =  leh.smooth_pkm(self.ks, self.pks, **arg_modified)

        return pk_smooth, pk_smooth_fid

    def smooth_and_shift(self,key: str, replace_options: dict = None, krepstart: float = None,
                         krepend: float = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Smooth the power spectrum for a given method and apply a rescaling/shift.

        Parameters
        ----------
        key : str
            Key identifying the smoothing method in `self.arguments`.
        replace_options : dict
            Dictionary of options to override in the method configuration.
        krepstart : float
            Starting k-value.
        krepend : float
            Ending k-value.

        Returns
        -------
        pk_smooth_h : ndarray
            Smoothed and shifted power spectrum for the target cosmology,
            in units of (h⁻³ Mpc³).
        pk_smooth_fid_h : ndarray
            Smoothed and shifted power spectrum for the fiducial cosmology,
            in units of (h⁻³ Mpc³).
        """

        # Compute smoothed spectra for both target and fiducial cosmologies
        pk_smooth, pk_smooth_fid = self.smooth_single(key, replace_options=replace_options, krepstart=krepstart, krepend=krepend)

        # Interpolate the smoothed spectra onto shifted k-grids
        pk_smooth_h = CubicSpline(self.ks, pk_smooth)(self.ks_shift) * (self.h_fid * self.rescale_factor) ** 3
        pk_smooth_fid_h = CubicSpline(self.ks, pk_smooth_fid)(self.ks_fid_shift) * (self.h_fid) ** 3

        return pk_smooth_h, pk_smooth_fid_h

    def smooth_all(self) -> dict:
        """
        Smooth the power spectrum for all methods.

        Returns
        -------
        pk_smooth_all : dict
            Dictionary of smoothed power spectra for all methods.
        pk_smooth_fid_all : dict
            Dictionary of smoothed fiducial power spectra for all methods.
        """
        pk_smooth_all = {}
        pk_smooth_fid_all = {}
        for key in self.arguments:
           pk_smooth, pk_smooth_fid = self.smooth_single(key)
           pk_smooth_all[key] = pk_smooth
           pk_smooth_fid_all[key] = pk_smooth_fid

        return pk_smooth_all, pk_smooth_fid_all

    def smooth_and_shift_all(self,key: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Smooth and shift the power spectrum for all methods.

        Parameters
        ----------
        key : str
            Key identifying the smoothing method in `self.arguments`.

        Returns
        -------
        pk_smooth_all : ndarray
            Smoothed and shifted power spectra for all methods.
        """
        pk_smooth_and_shift_all = {}
        pk_smooth_fid_and_shift_all = {}
        for key in self.arguments:
            pk_smooth, pk_smooth_fid = self.smooth_single(key)
            pk_smooth_and_shift_all = CubicSpline(self.ks, pk_smooth)(self.ks_shift) * (self.h_fid * self.rescale_factor) ** 3
            pk_smooth_fid_and_shift_all = CubicSpline(self.ks, pk_smooth_fid)(self.ks_fid_shift) * (self.h_fid) ** 3

        return pk_smooth_and_shift_all, pk_smooth_fid_and_shift_all

    def derivative_single(self, nowiggle_key: str, processing_key:
    				str, derivative_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the derivative for a given combination of
        smoothing, processing, and derivative methods.

        Parameters
        ----------
        nowiggle_key : str
            Key identifying the smoothing method in `self.arguments`.
        processing_key : str
            Key identifying the post-processing method in `self.processing_methods`.
        derivative_key : str
            Key identifying the derivative method in `self.derivative_methods`.

        Returns
        -------
        trafxs: np.ndarray
            Derivative of the power spectrum.
        info: np.ndarray
            Information about the derivative.
        mslope: np.ndarray
            Slope of the derivative.
        """

        # 1) Smooth and shift both target and fiducial power spectra
        pk_smooth_h , pk_smooth_fid_h = self.smooth_and_shift(nowiggle_key)

        # 2) Compute ratio of target to fiducial spectra
        ratio = pk_smooth_h / pk_smooth_fid_h

        # 3) Apply preprocessing to the ratio (optional)
        y_values = slope_maker.smooth_ratio(self.ks_out, ratio, smoothmethod = processing_key)

        # 4) Compute derivative of log(ratio) w.r.t. log(k)
        trafx, info = slope_maker.slope_at_x(np.log(self.ks_out), np.log(y_values),
                                             inputinfo = self.inputinfos[derivative_key],
                                             derivmethod=derivative_key)

        # Step 5: Calculate derivative at the pivot scale
        mslope = np.interp(self.kmpiv, self.ks_out, trafx)
        return trafx, info, mslope

    def derivative_all_for(self, nowiggle_key: str, processing_key: str,
    				replace_options: dict = None) -> tuple[np.ndarray, np.ndarray, dict, dict]:
        """
        Calculate derivatives for all derivative methods given a smoothing method
        and a post-processing filter.

        Parameters
        ----------
        nowiggle_key : str
            Key identifying the smoothing method in `self.arguments`.
        processing_key : str
            Key identifying the post-processing method in `self.processing_methods`.
        replace_options : dict
            Dictionary of options to override in the method configuration.

        Returns
        -------
        mslopes : np.ndarray
            Array of slope values for all derivative methods.
        y_values_h : dict
            Dictionary of smoothed and shifted power spectra for all methods.
        trafxs : dict
            Dictionary of derivative values for all methods.
        infos : dict
            Mapping from derivative method name to auxiliary output from
            `slope_maker.slope_at_x`.
        """
        y_values_h = {}
        trafxs = {}
        infos={}
        mslopes = np.empty(len(self.derivative_methods))

        # 1) Smooth and shift both target and fiducial spectra
        pk_smooth_h , pk_smooth_fid_h = self.smooth_and_shift(nowiggle_key, replace_options=replace_options)

        # 2) Compute the ratio and apply post-processing filter
        ratio = pk_smooth_h / pk_smooth_fid_h
        y_values = slope_maker.smooth_ratio(self.ks_out, ratio, smoothmethod = processing_key)

        # 3) Loop over all derivative methods and store results
        for ideriv, derivative_key in enumerate(self.derivative_methods):
           trafx, info = slope_maker.slope_at_x(np.log(self.ks_out), np.log(y_values), inputinfo = self.inputinfos[derivative_key], derivmethod=derivative_key)
           mslope = np.interp(self.kmpiv, self.ks_out, trafx)
           trafxs[derivative_key] = trafx
           infos[derivative_key] = info
           mslopes[ideriv] = mslope
        return mslopes, y_values, trafxs, infos


    def derivative_all(self):
        """
        Compute derivatives for all combinations of smoothing, processing,
        and derivative methods.

        Atributes:
        ----------
        ratios : dict
            Dictionary of smoothed and shifted power spectra for all methods.
        derivative_arrays : dict
            Dictionary of derivative values for all methods.
        derivative_infos : dict
            Mapping from derivative method name to auxiliary output from
            `slope_maker.slope_at_x`.
        mslopes : np.ndarray
            Array of slope values for all derivative methods.

        """
        y_values_h = {}
        trafxs = {}
        infos={}

        # Preallocate slope array: [derivative_method, processing_method, smoothing_method]
        mslopes = np.empty((len(self.derivative_methods),len(self.processing_methods),len(self.arguments)))

        # Loop over all smoothing methods
        for ix, nowiggle_key in enumerate(self.arguments):
          pk_smooth_h , pk_smooth_fid_h = self.smooth_and_shift(nowiggle_key)

          # Loop over all post-processing methods
          for ism, processing_key in enumerate(self.processing_methods):
             ratio = pk_smooth_h / pk_smooth_fid_h

              # Apply preprocessing to the ratio
             y_values = slope_maker.smooth_ratio(self.ks_out, ratio, smoothmethod = processing_key)
             y_values_h[(processing_key, nowiggle_key)] = y_values

              # Loop over all derivative methods
             for ideriv, derivative_key in enumerate(self.derivative_methods):
                trafx, info = slope_maker.slope_at_x(np.log(self.ks_out), np.log(y_values), inputinfo = self.inputinfos[derivative_key], derivmethod=derivative_key)
                mslope = np.interp(self.kmpiv, self.ks_out, trafx)
                trafxs[(processing_key, derivative_key, nowiggle_key)] = trafx
                infos[(processing_key, derivative_key, nowiggle_key)] = info
                mslopes[ideriv][ism][ix] = mslope

        # Save results to instance attributes
        self.ratios = y_values_h
        self.derivative_arrays = trafxs
        self.derivative_infos = infos
        self.mslopes = mslopes

    def derivative_for_method(self, processing_key: str, derivative_key: str, only_arguments: list = None,
                              add_info: dict = None) -> tuple[dict[str, float], dict[str, np.ndarray],
                                                              dict[str, np.ndarray], dict[str, any], dict[str, any]]:

        """
        Parameters
        ----------
        processing_key : str
            Identifier for the preprocessing method in `self.processing_methods`.
        derivative_key : str
            Identifier for the derivative method in `self.derivative_methods`.
        only_arguments : list[str]
            Subset of smoothing method keys from `self.arguments` to process.
            If None (default), all smoothing methods in `self.arguments` are used.
        add_info : dict, optional
            Additional options to merge into `self.inputinfos[derivative_key]`
            before calling `slope_maker.slope_at_x`.

        Returns
        -------
        mslopes : dict[str, float]
            Mapping from smoothing method key to slope value at the pivot scale `self.kmpiv`.
        y_values_h : dict[str, np.ndarray]
            Mapping from smoothing method key to processed ratio array.
        trafxs : dict[str, np.ndarray]
            Mapping from smoothing method key to derivative array
            (d ln P / d ln k) at each k in `self.ks_out`.
        infos : dict[str, any]
            Mapping from smoothing method key to auxiliary output from
            `slope_maker.slope_at_x`.
        approxs : dict[str, any]
            Mapping from smoothing method key to approximation object or data
            returned by `slope_maker.slope_at_x` when `return_approx=True`.
        """
        y_values_h = {}
        trafxs = {}
        infos={}
        mslopes = {}
        approxs = {}

        # Loop over selected smoothing methods
        for ix, nowiggle_key in enumerate(self.arguments if (only_arguments is None) else only_arguments):

          # 1) Smooth and shift
          pk_smooth_h , pk_smooth_fid_h = self.smooth_and_shift(nowiggle_key)

          # 2) Compute ratio and apply preprocessing
          ratio = pk_smooth_h / pk_smooth_fid_h
          y_values = slope_maker.smooth_ratio(self.ks_out, ratio, smoothmethod = processing_key)
          y_values_h[nowiggle_key] = y_values

          # 3) Prepare derivative method settings
          inputinfo  = self.inputinfos[derivative_key].copy()
          if add_info:
              inputinfo.update(add_info)

          # 4) Compute derivative, info, and approximation
          trafx, info, approx = slope_maker.slope_at_x(np.log(self.ks_out),
                                                       np.log(y_values),
                                                       inputinfo = inputinfo,
                                                       derivmethod=derivative_key,
                                                       return_approx=True)

          # 5) Calculate derivative at pivot scale
          mslope = np.interp(self.kmpiv, self.ks_out, trafx)

          # Store results
          trafxs[nowiggle_key] = trafx
          infos[nowiggle_key] = info
          mslopes[nowiggle_key] = mslope
          approxs[nowiggle_key] = approx
        return mslopes, y_values_h, trafxs, infos, approxs

# Fiducial cosmological parameters
h_fid = 0.676                   # Dimensionless Hubble parameter
Omegamfid = 0.31                # Total matter density parameter
Omegabfid = 0.0481425720388     # Baryon density parameter

# Fiducial cosmology dictionary
fiducial_cosmo = {'omega_m':Omegamfid*h_fid**2,   # Total matter density in h^2 units
                  'omega_b':Omegabfid*h_fid**2,   # Baryon density in h^2 units
                  'h':h_fid,                      # Hubble parameter
                  'n_s':0.97                      # Scalar spectral index
                  }

# Dictionary to store different cosmology parameter sets
cosmos = {}

def generate_cosmology(pars: dict, tag: str):
  """
  Generate a cosmology dictionary based on given parameters, update it
  with given values, and store it in the global `cosmos` dictionary.

  Parameters
  ----------
  pars : dict
      Dictionary of parameters for the cosmology.
  tag : str
      Identifier for the cosmology.
  """
  # Make a copy of the fiducial cosmology to avoid modifying the original
  cosmology = fiducial_cosmo.copy()

  # Update the cosmology with parameters provided in `pars`
  cosmology.update(pars)

  # If 'omega_cdm' is provided, remove 'omega_m' to avoid conflicts
  if 'omega_cdm' in pars:
    cosmology.pop('omega_m')

  # Store the updated cosmology in the global `cosmos` dictionary
  cosmos[tag] = cosmology

# Example: Create a new cosmology tagged as "test"
generate_cosmology({'omega_cdm':0.11,
                    'omega_b':0.018}, "test")

# Choose which cosmological model to work with
chosen_cosmo_key = "test"
chosen_cosmo = cosmos[chosen_cosmo_key]

# Create a cosmology generator object using fiducial and chosen cosmology
cog = cosmology_generator(fiducial_cosmo, chosen_cosmo)

