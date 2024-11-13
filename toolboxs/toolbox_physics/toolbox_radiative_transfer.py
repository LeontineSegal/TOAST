# %% load modules
from typing import overload, Union, Optional, List

import numpy as np

from numbers import Real
import numexpr as ne

from scipy.linalg import toeplitz

from toolboxs.toolbox_python.toolbox_python import bcolors, check_type, check_shape

# %% physical constants

PLANCK = 6.626070150 * 10**(-34)  # Planck constant [J/Hz]
BOLTZMANN = 1.380648520 * 10**(-23)  # Boltzmann constant [J/K]
T_BG = 2.73  # CMB temperature [K]

# %% FWHM and s_V

@overload
def from_FWHM_to_s_V(FWHM: float, 
                    DEBUG: Optional[bool] = True
) -> float:
    ...


@overload
def from_FWHM_to_s_V(FWHM: np.ndarray, 
                    DEBUG: Optional[bool] = True
) -> np.ndarray:
    ...


def from_FWHM_to_s_V(FWHM: Union[Real, np.ndarray], 
                     DEBUG: Optional[bool] = True
                     ) -> Union[Real, np.ndarray]:
    """Compute the linewidth sigma along velocity axis s_V (km/s) from FWHM (km/s).

    :param FWHM: (km/s) full width at half maximum
    :type FWHM: float | np.ndarray
    :return s_V: (km/s) linewidth sigma along velocity axis
    :rtype: float | np.ndarray
    """
    if DEBUG : 
        assert check_type(FWHM, type_of_reference=Real, from_function='from_FWHM_to_s_V', flag=False) or check_type(
        FWHM, type_of_reference=np.ndarray, from_function='from_FWHM_to_s_V', flag=False)
    return FWHM / (np.sqrt(8*np.log(2)))

@overload
def from_s_V_to_FWHM(s_V: float, 
                    DEBUG: Optional[bool] = True
                    ) -> float:
    ...


@overload
def from_s_V_to_FWHM(s_V: np.array, 
                    DEBUG: Optional[bool] = True
                    ) -> np.array:
    ...


def from_s_V_to_FWHM(s_V: Union[Real, np.array], 
                    DEBUG: Optional[bool] = True
) -> Union[Real, np.array]:
    """Compute the full width at half maximum FWHM (km/s) from the linewidth sigma along velocity axis s_V (km/s).

    :param s_V: (km/s) linewidth sigma along velocity axis
    :type s_V: float | np.array
    :return: (km/s) FWHM full width at half maximum
    :rtype: float | np.array
    """
    if DEBUG:
        assert check_type(s_V, type_of_reference=Real, from_function='from_s_V_to_FWHM', flag=False) or check_type(
        s_V, type_of_reference=np.ndarray, from_function='from_s_V_to_FWHM', flag=False)
    return s_V * (np.sqrt(8*np.log(2)))

# %% radiative transfer equation
# opacity line profile

def opacity_line_profile(
        tau: Union[Real, np.ndarray],
        velocity_axis: np.ndarray,
        C_V: Union[Real, np.ndarray],
        s_V: Union[Real, np.ndarray], 
        DEBUG: Optional[bool] = True
) -> np.ndarray:
    """Compute the line profile.

    :param tau: opacity
    :type tau: Real | np.ndarray
    :param velocity_axis: (km/s) velocity axis
    :type velocity_axis: np.ndarray
    :param C_V: (km/s) intensity-weighted velocity of the spectral line (velocity centroid)
    :type C_V: Real
    :param s_V: (km/s) linewidth sigma along velocity axis
    :type s_V: Real | np.ndarray
    :return: tau line profile
    :rtype: np.ndarray
    """

    # array inputs
    if not check_type(tau, s_V, C_V, type_of_reference=Real, flag=False):
        if DEBUG:
            assert check_type(tau, s_V, C_V, type_of_reference=np.ndarray)

        number_of_files = np.shape(tau)[0]
        if DEBUG:
            check_shape(tau, s_V, C_V, shape_of_reference=(number_of_files,))

        s_V = s_V[:, np.newaxis]
        C_V = C_V[:, np.newaxis]
        tau = tau[:, np.newaxis]

        res_1 = (- 1 / (2 * s_V**2))
        res_2 = (velocity_axis - C_V)**2
        res = res_1 * res_2

        res = res.astype(float) 
        res = ne.evaluate('exp(res)')
        res = tau*res

        return res

    # float inputs
    else:
        return tau*np.exp((- 1 / (2 * s_V**2) * (velocity_axis - C_V)**2))

# intensity
@overload
def intensity(T_ex: Real, freq: Real, DEBUG: Optional[bool] = True
) -> Real:
    ...


@overload
def intensity(T_ex: np.ndarray, freq: Real, DEBUG: Optional[bool] = True) -> np.ndarray:
    ...


def intensity(T_ex: Union[Real, np.ndarray], freq: Real, DEBUG: Optional[bool] = True) -> Union[Real, np.ndarray]:
    """Compute the emitted intensity from a gas layer.

    :param T_ex: (K) excitation temperature
    :type T_ex: float | np.ndarray of shape (number of layers or number of tests)
    :param freq: (GHz) rest frequency
    :type freq: float
    :return: (K) intensity 
    :rtype: float | np.ndarray
    """
    if DEBUG:
        assert check_type(freq, type_of_reference=Real)
    freq = freq * 10**9  # from GHz to Hz

    # array inputs
    if not check_type(T_ex, type_of_reference=Real, flag=False):
        if DEBUG:
            assert check_type(T_ex, type_of_reference=np.ndarray)

        res = ((PLANCK*freq) / (BOLTZMANN*T_ex))
        res = res.astype(float)

    # float inputs
    else:
        res = (PLANCK*freq) / (BOLTZMANN*T_ex)

    res_1 = ((PLANCK*freq) / BOLTZMANN)
    res_2 = ne.evaluate('exp(res)')
    tot_res = res_1 * 1 / (res_2 -1)

    return tot_res

def compute_radiative_tranfer_equation(
        T_ex: np.ndarray,
        tau: np.ndarray,
        freq: Real,
        velocity_axis: np.ndarray,
        velocity_resolution: Real,
        s_V: np.ndarray,
        C_V: np.ndarray,
        unique_layer_idx: Optional[List[int]] = None,
        decomposition: Optional[bool] = False,
        conserved_flux: Optional[bool] = True,
        reference_velocity_resolution: Optional[float] = 0.01,
        DEBUG: Optional[bool] = True, 
        number_of_C_V_components: Optional[int] = 1, 
        number_of_layers_per_clump: Optional[int] = 1, 
        peak_only: Optional[bool] = False,
        ) -> List[np.ndarray]:
    """Compute spectrum following the multi-layer model and decomposed it following each cloud layer.
    """
    if DEBUG:
        assert check_type(T_ex, tau, s_V, C_V, type_of_reference=np.ndarray)
        assert check_shape(T_ex, tau, s_V, C_V,
                        shape_of_reference=np.shape(T_ex))
        assert check_type(freq, velocity_resolution, type_of_reference=Real)
        assert check_shape(velocity_axis, shape_of_reference=(
            1, np.shape(velocity_axis)[1]))  # row vector
        if peak_only :
            assert np.shape(velocity_axis)[1] == number_of_C_V_components # each values is associated to a peak 
    
    number_of_files, number_of_layers = np.shape(T_ex)[0], np.shape(T_ex)[
        1]  # same convension for all inputs
    
    # by default, we consider that each layer is unique
    if np.shape(unique_layer_idx) == ():
        unique_layer_idx = np.arange(0, number_of_layers, 1)
    if DEBUG:
        assert velocity_resolution >= reference_velocity_resolution, f'{bcolors.WARNING}\n [error: compute_radiative_tranfer_equation] Wrong velocity sampling coefficient detected (must be higher than {reference_velocity_resolution * 10**2} m/s) \n{bcolors.ENDC}'
    
    # sub sampling coeff.
    k = round(velocity_resolution/reference_velocity_resolution)
    
    if k > 1 and conserved_flux:
        window = k

        # generate a high sampled signal
        edges = (window-1)/2

        if peak_only :
            for clump_idx in range(number_of_C_V_components) : 
                reference_velocity_axis_ = velocity_axis[0, clump_idx] + reference_velocity_resolution * \
                np.arange(-edges, k * (1-1) + (edges+1), 1) # one peak => np.shape(velocity_axis)[1] = 1
                reference_velocity_axis_ = reference_velocity_axis_.reshape(
                (1, len(reference_velocity_axis_)))
                if clump_idx == 0 :
                    reference_velocity_axis = reference_velocity_axis_
                    # useful to compute the mean below
                    velocity_axis_len_per_clump = reference_velocity_axis_.shape[-1]
                else : 
                    reference_velocity_axis = np.concatenate((reference_velocity_axis, reference_velocity_axis_), axis = None)
            reference_velocity_axis = reference_velocity_axis.reshape((1, reference_velocity_axis.shape[1]))
        else : 
            reference_velocity_axis = velocity_axis[0, 0] + reference_velocity_resolution * \
                np.arange(-edges, k * (np.shape(velocity_axis)
                        [1]-1) + (edges+1), 1)
            reference_velocity_axis = reference_velocity_axis.reshape(
                (1, len(reference_velocity_axis)))
        
        # length N + M - 1 from convolution
        first_column = np.zeros(
            np.shape(reference_velocity_axis)[1] + window - 1)
        first_column[0:window] = 1/window
        first_row = np.zeros(np.shape(reference_velocity_axis)[1])

        first_row[0] = 1/window
        mean_filter_matrix = toeplitz(first_column, first_row)
        mean_filter_matrix = mean_filter_matrix[k-1:-(k-1):k, :].T

    else:
        reference_velocity_axis = velocity_axis
        window = 1

    s = 0
    if decomposition:
        decomposed_s = np.zeros((number_of_layers + 2, number_of_files,
                                np.shape(velocity_axis)[1]))  # + total spectrum and CMB
    unique_intensity = []
    unique_exp = []

    cumulated_exp = np.ones(
    (number_of_files, np.shape(reference_velocity_axis)[1]))

    cumulated_exp = cumulated_exp.astype(float)

    for layer_idx in range(number_of_layers):

        try:
            # maybe the layer properties has already been computed
            intensity_layer_idx = unique_intensity[unique_layer_idx[layer_idx]]
            exp_layer_idx = unique_exp[unique_layer_idx[layer_idx]]

        except:
            # have to compute the layer properties
            intensity_layer_idx = intensity(
                T_ex[:, layer_idx].reshape((number_of_files, 1)), freq, DEBUG=DEBUG)
            unique_intensity.append(intensity_layer_idx)

            if peak_only :
                clump_idx = (layer_idx // number_of_layers_per_clump)
                mean_velocity = np.mean(reference_velocity_axis[0, clump_idx * velocity_axis_len_per_clump : (clump_idx + 1) * velocity_axis_len_per_clump])
                psi_layer_idx = opacity_line_profile(tau[:, layer_idx],
                                                     reference_velocity_axis - mean_velocity,
                                                     C_V[:, layer_idx] - C_V[:, clump_idx * number_of_layers_per_clump],
                                                     s_V[:, layer_idx], 
                                                     DEBUG=DEBUG)  # take into account the potential shift between layers
            else:
                psi_layer_idx = opacity_line_profile(tau[:, layer_idx],
                                                     reference_velocity_axis,
                                                     C_V[:, layer_idx],
                                                     s_V[:, layer_idx], 
                                                     DEBUG=DEBUG)
            res = - psi_layer_idx
            exp_layer_idx = ne.evaluate('exp(res)')
            unique_exp.append(exp_layer_idx)

        s_idx = np.multiply(
            intensity_layer_idx, (1-exp_layer_idx) * cumulated_exp)

        if window > 1 and conserved_flux:
            s_idx = np.tensordot(s_idx, mean_filter_matrix, axes=1)
        
        s += s_idx
        if decomposition:
            decomposed_s[layer_idx + 1, :, :] = s_idx[:, :]
        cumulated_exp *= exp_layer_idx

    CMB_intensity = intensity(T_BG, freq, DEBUG=DEBUG)

    CMB_s = CMB_intensity * (1-cumulated_exp)

    if window > 1 and conserved_flux:
        CMB_s = CMB_s @ mean_filter_matrix
        
    s -= CMB_s

    if decomposition:
        decomposed_s[-1, :, :] = -CMB_s[:, :]
        # by convention, the total emission is returned as first
        decomposed_s[0, :, :] = s

    if decomposition:
        return decomposed_s
    else:
        return [s]
