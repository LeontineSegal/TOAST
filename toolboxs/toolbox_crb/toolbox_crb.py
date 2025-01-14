# %% load modules
from typing import Optional, List, Union, Tuple

import numpy as np
from numbers import Real
from scipy.linalg import toeplitz

from toolboxs.toolbox_python import bcolors, check_type, check_shape, from_list_to_array
from toolboxs.toolbox_physics.toolbox_radiative_transfer import PLANCK, BOLTZMANN, T_BG
from toolboxs.toolbox_physics.toolbox_radiative_transfer import from_s_V_to_FWHM, compute_radiative_tranfer_equation, opacity_line_profile, intensity
from toolboxs.toolbox_radex.toolbox_radex import get_freq_bandwidth, from_OPR_to_densities, execute_radex

# %% noise


def snr_trick(
        ppv: np.array,
        sb: np.array,
        snr_threshold: float,
        mode: Optional[str] = 'PSNR', 
        DEBUG: Optional[bool] = True,
) -> np.array:
    """To adjust the additive noise dispersion. This allows to take into account faint (low SNR) but very informative emission lines, such as H13CO+.

    :param cube: data cube (PPV)
    :type cube: np.array[np.float64]
    :param s_b: (K) thermal noise dispersion 
    :type s_b: np.array[np.float64]
    :param SNR_threshold: SNR threshold
    :type SNR_threshold: float
    :param mode: saturating the peak SNR or the SNR. Among {'PSNR', 'SNR'}
    :type mode: str
    :return: adjusted thermal noise dispersion
    :rtype: np.array[np.float64]
    """
    if DEBUG: 
        assert check_type(ppv, sb, type_of_reference=np.ndarray)
        assert check_shape(ppv[:, :, 0], sb, shape_of_reference=np.shape(
        sb))

    if mode == 'PSNR':
        peak_intensity = np.nanmax(ppv, axis=-1)  # along the velocity axis
        adjusted_s_b = np.where(sb < np.divide(
            peak_intensity, snr_threshold), np.divide(peak_intensity, snr_threshold), sb)

    # deprecated version...
    elif mode == 'SNR':
        print(
            f'{bcolors.WARNING}\n [error: Using the depracated mode "SNR" to adjust the thermal noise dispersion.\n{bcolors.ENDC}')
        normalized_flux = np.nansum(ppv, axis=-1)/np.shape(ppv)[-1]
        adjusted_s_b = np.where(sb < np.divide(
            normalized_flux, snr_threshold), np.divide(normalized_flux, snr_threshold), sb)

    return adjusted_s_b

def get_s_c(name_line: str, 
            DEBUG: Optional[bool] = True,
) -> Real:
    """Get the calibration noise dispersion.

    :param name_line: name of the studied transition (from the .fits header)
    :type name_line: str
    :return: calibration noise dispersion
    :rtype: float
    """
    if DEBUG: 
        assert check_type(name_line, type_of_reference=str,
                      from_function='get_s_c')

    if name_line == '1-0':
        return 0.05  # 5%
    elif name_line == '2-1':
        return 0.1  # 10%
    else:
        print(
            f'{bcolors.WARNING}\n [error: get_s_c] No calibration noise is assigned to this transition line.\n{bcolors.ENDC}')
        return False

# %% Useful quantities to compute ds/d_theta

def derive_intensity_over_Tex(Tex: Real, freq: Real) -> float:
    """Derive intensity over excitation temperature.

    :param Tex: (K) excitation temperature
    :type Tex: Real
    :param freq: (GHz) rest frequency
    :type freq: Real
    :return: the derivative of the intensity over excitation temperature  
    :rtype: float
    """
    freq = freq * 10**9  # from GHz to Hz
    amplitude = (PLANCK * freq)/(BOLTZMANN * Tex)

    # numpy formalism python 3.11.2
    amplitude = amplitude.astype(float)

    exp_amplitude = np.exp(amplitude)
    intensity_over_T_ex = exp_amplitude / (exp_amplitude - 1)**2
    intensity_over_T_ex = amplitude**2 * intensity_over_T_ex

    return intensity_over_T_ex

def derive_opacity_profile_over_theta(
        velocity_axis: np.array,
        C_V: Real,
        s_V: Real,
        tau: Real,
        derivative_opacity_over_dtheta: Real,
        theta: str) -> np.array:
    """Derive the opacity line profile over a studied component of the vector "theta" of unknown parameters.

    :param velocity_axis: (km/s) velocity axis
    :type velocity_axis: np.array
    :param C_V: (km/s) intensity-weighted velocity of the spectral line (velocity centroid)
    :type C_V: Real
    :param s_V: (km/s) linewidth sigma along velocity axis
    :type s_V: Real
    :param tau: opacity
    :type opacity: Real
    :param derivative_opacity_over_dtheta: derivate of the opacity over the studied theta component
    :type derivative_opacity_over_dtheta: Real
    :param theta: name of the studied theta component among {'log10_T_kin', 'log10_nH2', 'log10_N', 'FWHM', 'C_V'}
    :type theta: str
    :return: the derivative of the opacity line profile over a studied theta component  
    :rtype: np.array
    """
    exp = np.exp(-((velocity_axis - C_V)**2) / (2 * s_V**2))
    if theta == 'FWHM':  # ds over dFWHM instead...
        derivative_opacityLineProfile_over_dtheta = (derivative_opacity_over_dtheta + tau / np.sqrt(8. * np.log(2)) * ((velocity_axis - C_V)**2 / (
            s_V**3))) * exp  # from derivative_opacity_over_s_V to derivative_opacityLineProfile_over_FWHM

    elif theta == 'C_V':
        derivative_opacityLineProfile_over_dtheta = tau * \
            ((velocity_axis - C_V) / (s_V**2)) * exp
    else:
        derivative_opacityLineProfile_over_dtheta = derivative_opacity_over_dtheta * exp

    return derivative_opacityLineProfile_over_dtheta

def derive_Tex_tau_over_theta(
        name_mol: str,
        file: str,
        freq_min: Real,
        freq_max: Real,
        T_kin: Real,
        colliders: list[str],
        nH2: Real,
        N: Real,
        FWHM: Real,
        geometry: str,
        Tex: Real,
        tau: Real,
        theta: Optional[list[str]] = ['log10_T_kin',
                                      'log10_nH2', 'log10_N', 'FWHM', 'C_V'],
        order: Optional[int] = 4,
        clean: Optional[bool] = True, 
        DEBUG: Optional[bool] = True,
) -> tuple[np.array, np.array]:
    """Compute the gradient of Tex and the opacity over a studied component of the vector "theta" of unknown parameters, for a single emission line.

    :param name_mol: name_mol name (ex : '12CO')
    :type name_mol: str
    :param file: file name without extension (ex: 'gradient_computation')
    :type file: str
    :param freq_min: (GHz) minimal frequency (Radex parameter)
    :type freq_min: Real
    :param freq_max: (GHz) maximal frequency (Radex parameter)
    :type freq_max: Real
    :param T_kin: kinetic temperature (K)
    :type T_kin: Real
    :param colliders: considered collider parterns ('H2', 'p-H2', 'o-H2', 'e', 'H', 'He', 'H+')
    :type colliders: list[str]
    :param nH2: (cm^-3) H2 density
    :type nH2: Real
    :param N: (cm^-2) column density
    :type N: Real
    :param FWHM: (km/s) full width at half maximum
    :type FWHM: Real
    :param geometry: used geometry among {'radexUnifSphere','radexExpSphere','radexParaSlab'}
    :type geometry: str
    :param Tex: (K) excitation temperature
    :type Tex: Real
    :param tau: opacity
    :type tau: Real
    :param theta: vector of unknown parameters, defaults to ['log10_T_kin', 'log10_nH2', 'log10_N', 'FWHM', 'C_V']
    :type theta: Optional[list[str]], optional
    :param order: number of used points for the interpolation, defaults to 4
    :type order: Optional[int], optional
    :param clean: remove the .inp and .out files, defaults to False
    :type clean: Optional[bool]
    :return: derivative_T_ex_over_theta, derivative_tau_over_theta
    :rtype: np.array, np.array
    """
    if DEBUG: 
        assert check_type(
        T_kin, nH2, N, FWHM, Tex, tau, type_of_reference=Real, from_function='derive_Tex_tau_over_theta')

    derivative_T_ex_over_theta = np.zeros(len(theta))
    derivative_tau_over_theta = np.zeros(len(theta))
    theta_steps = [T_kin/1024., 0.001, 0.001, FWHM/128]

    log10_nH2 = np.log10(nH2)
    log10_N = np.log10(N)

    for t_idx, t in enumerate(theta):

        if (str(t) == 'C_V'):
            derivative_T_ex_over_theta[t_idx] = 0
            derivative_tau_over_theta[t_idx] = 0

        else:
            # warning, here is still ds/dT_kin calculations rather than ds/d_log10_T_kin
            if (str(t) == 'log10_T_kin'):
                step = theta_steps[0]
                T_kin_neighbors = [
                    T_kin + step*i for i in range(-int(order/2), int(order/2)+1, 1)]
                T_kin_neighbors.remove(T_kin)
                T_kin_neighbors = from_list_to_array(T_kin_neighbors)
                log10_nH2_neighbors = log10_nH2 * np.ones(order)
                log10_N_neighbors = log10_N * np.ones(order)
                FWHM_neighbors = FWHM * np.ones(order)

            elif (str(t) == 'log10_nH2'):
                step = theta_steps[1]
                T_kin_neighbors = T_kin * np.ones(order)
                log10_nH2_neighbors = [
                    log10_nH2 + step*i for i in range(-int(order/2), int(order/2)+1, 1)]
                log10_nH2_neighbors.remove(log10_nH2)
                log10_nH2_neighbors = from_list_to_array(log10_nH2_neighbors)
                log10_N_neighbors = log10_N * np.ones(order)
                FWHM_neighbors = FWHM * np.ones(order)

            elif (str(t) == 'log10_N'):
                step = theta_steps[2]
                T_kin_neighbors = T_kin * np.ones(order)
                log10_nH2_neighbors = log10_nH2 * np.ones(order)
                log10_N_neighbors = [
                    log10_N + step*i for i in range(-int(order/2), int(order/2)+1, 1)]
                log10_N_neighbors.remove(log10_N)
                log10_N_neighbors = from_list_to_array(log10_N_neighbors)
                FWHM_neighbors = FWHM * np.ones(order)

            elif (str(t) == 'FWHM'):
                step = theta_steps[3]
                T_kin_neighbors = T_kin * np.ones(order)
                log10_nH2_neighbors = log10_nH2 * np.ones(order)
                log10_N_neighbors = log10_N * np.ones(order)
                FWHM_neighbors = [
                    FWHM + step*i for i in range(-int(order/2), int(order/2)+1, 1)]
                FWHM_neighbors.remove(FWHM)
                FWHM_neighbors = from_list_to_array(FWHM_neighbors)

            collider_densities = np.zeros(
                (len(colliders), len(T_kin_neighbors)))
            for c_idx, c in enumerate(from_OPR_to_densities(T_kin_neighbors, log10_nH2_neighbors, colliders=colliders)):
                collider_densities[c_idx, :] = 10**c

            N_neighbors = 10**(log10_N_neighbors)

            T_ex_neighbors, opacity_neighbors, T_r_neighbors = execute_radex(
                name_mol,
                f'derive_T_ex_opacity_over_theta_{file}',
                freq_min,
                freq_max,
                T_kin_neighbors,
                colliders,
                collider_densities,
                N_neighbors,
                FWHM_neighbors,
                geometry,
                clean=clean)

            # remove the continue composant
            T_ex_neighbors[:, 0] = T_ex_neighbors[:, 0] - Tex
            opacity_neighbors[:, 0] = opacity_neighbors[:, 0] - tau

            x = [theta_steps[t_idx] *
                 i for i in range(-int(order/2), int(order/2)+1, 1)]
            x.remove(0)
            x = from_list_to_array(x)

            # polyfit's convension
            T_ex_neighbors_polyfit = (T_ex_neighbors[:, 0]).astype(x.dtype)
            opacity_neighbors_polyfit = (
                opacity_neighbors[:, 0]).astype(x.dtype)

            derivative_T_ex_over_theta[t_idx] = np.polyfit(x, T_ex_neighbors_polyfit, 2)[
                1]  # first derivative selection
            derivative_tau_over_theta[t_idx] = np.polyfit(
                x, opacity_neighbors_polyfit, 2)[1]  # first derivative selection

            if (str(t) == 'log10_T_kin'):
                # from ds/dT_kin to ds/d_log10_T_kin
                derivative_T_ex_over_theta[t_idx] *= np.log(10) * T_kin
                derivative_tau_over_theta[t_idx] *= np.log(10) * T_kin

    return derivative_T_ex_over_theta, derivative_tau_over_theta

def derive_s_over_theta(
        Tex: np.array,
        tau: np.array,
        Tex_over_theta: np.array,
        tau_over_theta: np.array,
        freq: Real,
        velocity_axis: np.array,
        velocity_resolution: Real,
        s_V: np.array,
        C_V: np.array,
        unique_layer_idx: List[int],
        constraints_kinematics: List[str],
        constraints_geometry: List[str],
        layers: int, 
        number_of_unknowns: int,
        idxs_inner_layer: Optional[List[int]] = [],
        number_of_layers_per_clump:  Optional[int] = 0,
        number_of_clumps: Optional[int] = 0,
        conserved_flux: Optional[bool] = True,
        theta: Optional[List[str]] = ['log10_T_kin',
                                      'log10_nH2', 'log10_N', 'FWHM', 'C_V'],
        C_V_idx: Optional[int] = 4,
        reference_velocity_resolution: Optional[float] = 0.01,         
        DEBUG: Optional[bool] = True,
        peak_only: Optional[bool] = False
) -> np.array:
    if DEBUG : 
        assert check_type(Tex, tau, s_V, C_V, type_of_reference=np.ndarray)
        assert check_shape(Tex, tau, s_V, C_V,
                        shape_of_reference=(1, np.shape(Tex)[1]))
        assert check_type(freq, velocity_resolution, type_of_reference=Real)
        assert check_shape(velocity_axis, shape_of_reference=(
            1, np.shape(velocity_axis)[1]))  # row vector
        assert check_shape(Tex_over_theta, tau_over_theta,
                        shape_of_reference=(np.shape(Tex)[1], len(theta)))

        assert velocity_resolution >= reference_velocity_resolution, f'{bcolors.WARNING}\n [error: compute_radiative_tranfer_equation] Wrong velocity sampling coefficient detected (must be higher than {reference_velocity_resolution * 10**2} m/s) \n{bcolors.ENDC}'
        if peak_only :
            assert np.shape(velocity_axis)[1] == number_of_clumps # each values is associated to a peak 
    
    number_of_C_V_components = number_of_clumps

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

    s_over_thetas = []
    cumulated_opacity_profile, cumulated_intensity = 0, 0
    # the front medium is transparent to the observer
    opacity_profiles, intensities = [0], []
    intensity_over_T_exs = []

    for layer_idx in range(layers):

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

        intensity_layer_idx = intensity(
            Tex[:, layer_idx].reshape((len(Tex), 1)), freq).item()

        cumulated_opacity_profile += psi_layer_idx
        opacity_profiles.append(
            psi_layer_idx), intensities.append(intensity_layer_idx)
        intensity_over_T_exs.append(
            derive_intensity_over_Tex(Tex[:, layer_idx], freq))
    intensities.append(intensity(T_BG, freq))  # the CMB as the very last layer

    for layer_idx in range(layers, 0, -1):

        if peak_only :
            clump_idx = ((layer_idx-1) // number_of_layers_per_clump)
            mean_velocity = np.mean(reference_velocity_axis[0, clump_idx * velocity_axis_len_per_clump : (clump_idx + 1) * velocity_axis_len_per_clump])

        s_over_theta_layer_idx = np.zeros(
            (reference_velocity_axis.size, len(theta)))

        cumulated_intensity += (intensities[layer_idx-1] -
                                intensities[layer_idx]) * np.exp(-cumulated_opacity_profile)
        cumulated_opacity_profile -= opacity_profiles[layer_idx]

        A_1 = (1-np.exp(-opacity_profiles[layer_idx])
               ) * np.exp(-cumulated_opacity_profile)
        A_2 = cumulated_intensity

        for t_idx, t in enumerate(theta):

            if peak_only :
                if str(t) == 'C_V':
                    B_1 = 0
                    B_2 = derive_opacity_profile_over_theta(
                        reference_velocity_axis - mean_velocity,
                        C_V[:, layer_idx-1] - C_V[:, clump_idx * number_of_layers_per_clump],
                        s_V[:, layer_idx-1],
                        tau[:, layer_idx-1],
                        tau_over_theta[layer_idx-1, t_idx],
                        t)
                else:
                    B_1 = intensity_over_T_exs[layer_idx-1] * \
                        Tex_over_theta[layer_idx-1, t_idx]
                    B_2 = derive_opacity_profile_over_theta(
                        reference_velocity_axis - mean_velocity,
                        C_V[:, layer_idx-1] - C_V[:, clump_idx * number_of_layers_per_clump],
                        s_V[:, layer_idx-1],
                        tau[:, layer_idx-1],
                        tau_over_theta[layer_idx-1, t_idx],
                        t)
            else:
                if str(t) == 'C_V':
                    B_1 = 0.
                    B_2 = derive_opacity_profile_over_theta(
                        reference_velocity_axis,
                        C_V[:, layer_idx-1],
                        s_V[:, layer_idx-1],
                        tau[:, layer_idx-1],
                        tau_over_theta[layer_idx-1, t_idx],
                        t)
                else:
                    B_1 = intensity_over_T_exs[layer_idx-1] * \
                        Tex_over_theta[layer_idx-1, t_idx]
                    B_2 = derive_opacity_profile_over_theta(
                        reference_velocity_axis,
                        C_V[:, layer_idx-1],
                        s_V[:, layer_idx-1],
                        tau[:, layer_idx-1],
                        tau_over_theta[layer_idx-1, t_idx],
                        t)
                    
            s_over_theta_layer_idx[:, t_idx] = A_1*B_1 + A_2*B_2
        s_over_thetas.append(s_over_theta_layer_idx)
    s_over_thetas.reverse()  # from the front layer to the back one

    s_over_theta = np.zeros((velocity_axis.size, number_of_unknowns))

    for layer_idx in range(0, layers):

        # index where add it following redundancy in the sense of the estimator
        idx = unique_layer_idx[layer_idx]
        s_over_theta_layer_idx = s_over_thetas[layer_idx]

        # mean filtering then subsampling
        if window > 1 and conserved_flux:
            s_over_theta_layer_idx = np.tensordot(s_over_theta_layer_idx.T, mean_filter_matrix, axes=1)
            s_over_theta_layer_idx = s_over_theta_layer_idx.T

        if ('sandwich' in constraints_geometry) and ('mirror' in constraints_kinematics) :
            # in which sandwich are we ?
            idx_sandwich = (layer_idx//number_of_layers_per_clump) % number_of_clumps
            idx_inner_layer = idxs_inner_layer[idx_sandwich]
            if layer_idx <= idx_inner_layer : 
                s_over_theta[:, idx*len(theta):(idx+1)*len(theta)] += s_over_theta_layer_idx
            else : 
                s_over_theta[:, idx*len(theta):(idx+1)*len(theta)-1] += s_over_theta_layer_idx[:, :-1] # except C_V
                # ds/dC_V_out = ds/dC_V_out - ds/dC_V_opp
                s_over_theta[:, idx*len(theta)+C_V_idx] -= s_over_theta_layer_idx[:, C_V_idx]
                # ds/dC_V_in = ds/dC_V_in + 2 x ds/dC_V_opp
                s_over_theta[:, idx_inner_layer*len(theta)+C_V_idx] += 2 * s_over_theta_layer_idx[:, C_V_idx]
        else : 
            if idx == 0:  # first layer : from log10_T_kin to C_V, whatever the assumption on C_V is
                s_over_theta[:, idx*len(theta):(idx+1)*len(theta)
                            ] += s_over_theta_layer_idx
            else:
                if 'same_C_V_in_all_layers' in constraints_kinematics:  # do not repeat the C_V
                    s_over_theta[:, idx*len(theta):(idx+1)*len(theta)-1] += s_over_theta_layer_idx[:, :-1]
                    # but add the C_V contribution in the first layer
                    s_over_theta[:, C_V_idx] += s_over_theta_layer_idx[:, C_V_idx]

                else:  # the layer has its own C_V
                    s_over_theta[:, idx*len(theta):(idx+1)*len(theta)
                                ] += s_over_theta_layer_idx

    return s_over_theta

# %% Fisher information matrix (fim)

def compute_fim(
        name_mol: str,
        freq: Real,
        velocity_axis: np.array,
        velocity_resolution: Real,
        T_kin: np.array,
        nH2: np.array,
        N_molecule: np.array,
        FWHM: np.array,
        s_V: np.array,
        C_V: np.array,
        colliders: List[str],
        geometry: str,
        Tex: np.array,
        tau: np.array,
        c_0: float,
        s_c: float,
        s_b: float,
        unique_layer_idx: List[int],
        file_name: str,
        constraints_kinematics: List[str],
        constraints_geometry: List[str],
        layers: int,
        idxs_inner_layer: List[str],
        number_of_unknowns: int,
        number_of_clumps: Optional[int] = 0,
        number_of_layers_per_clump: Optional[int] = 0,
        peak_only: Optional[bool] = False,
        conserved_flux: Optional[bool] = True,
        reference_velocity_resolution: Optional[float] = 0.05,
        theta: Optional[List[str]] = ['log10_T_kin',
                                      'log10_nH2', 'log10_N', 'FWHM', 'C_V'],
        C_V_idx: Optional[int] = 4,
        NLL_g: Optional[bool] = False,
        x: Optional[np.array] = [], 
        DEBUG: Optional[bool] = True,
) -> Union[np.array, Tuple[np.array, np.array]]:
    """Compute the Fisher Information Matrix (fim), for a single emission line.

    :param name_mol: name_mol name (ex : '12CO')
    :type name_mol: str
    :param freq: (GHz) rest frequency
    :type freq: Real
    :param velocity_axis: _description_
    :type velocity_axis: np.array
    :param velocity_resolution: _description_
    :type velocity_resolution: Real
    :param T_kin: kinetic temperature (K)
    :type T_kin: np.array
    :param nH2: (cm^-3) H2 density
    :type nH2: np.array
    :param N_molecule: (cm^-2) column density
    :type N_molecule: np.array
    :param FWHM: (km/s) FWHM
    :type FWHM: np.array
    :param s_V: (km/s) linewidth sigma along velocity axis
    :type s_V: np.array
    :param C_V: (km/s) intensity-weighted velocity of the spectral line (velocity centroid)
    :type C_V: np.array
    :param colliders: considered collider parterns ('H2', 'p-H2', 'o-H2', 'e', 'H', 'He', 'H+')
    :type colliders: List[str]
    :param geometry: used geometry among {'radexUnifSphere','radexExpSphere','radexParaSlab'}
    :type geometry: str
    :param Tex: (K) excitation temperature
    :type Tex: np.array
    :param tau: tau
    :type tau: np.array
    :param c_0: _description_
    :type c_0: float
    :return: calibration noise dispersion
    :type s_c: float
    :param s_b: (K) thermal noise dispersion 
    :type s_b: float
    :param unique_layer_idx: for example, in the sandwich case : [0, 1, 0]
    :type unique_layer_idx: List[int]
    :param file_name: file
    :type file_name: str
    :param conserved_flux: _description_, defaults to True
    :type conserved_flux: Optional[bool], optional
    :param theta: vector of unknown parameters, defaults to ['log10_T_kin', 'log10_nH2', 'log10_N', 'FWHM', 'C_V']
    :type theta: Optional[List[str]], optional
    :return: the fim of the line of shape (number_of_layers * len(theta), number_of_layers * len(theta))
    :rtype: np.array
    """

    # number of layers of the cloud (may have redundancy in the sense of the estimator)
    number_of_layers = np.shape(Tex)[1]  # same convension for all inputs
    if DEBUG : 
        assert check_shape(T_kin, nH2, N_molecule, s_V, FWHM, C_V,
                       shape_of_reference=(1, number_of_layers))

    freq_min, freq_max = get_freq_bandwidth(
        freq)[0], get_freq_bandwidth(freq)[1]

    if ('sandwich' in constraints_geometry) and ('mirror' in constraints_kinematics):
        unique_layer_idx_ = np.arange(0, number_of_layers, 1)
    else:
        unique_layer_idx_ = unique_layer_idx

    # compute s
    s = compute_radiative_tranfer_equation(
        Tex,
        tau,
        freq,
        velocity_axis,
        velocity_resolution,
        s_V,
        C_V,
        unique_layer_idx=unique_layer_idx_,
        decomposition=False,
        conserved_flux=conserved_flux,
        reference_velocity_resolution=reference_velocity_resolution, 
        DEBUG=DEBUG, 
        number_of_C_V_components=number_of_clumps, 
        number_of_layers_per_clump = number_of_layers_per_clump,
        peak_only = peak_only
    )[0]  # spectrum of shape (1, len(velocity_axis))

    # compute the covariance matrix
    sTs = (s @ np.transpose(s)).item()  # scalar
    s_x = np.sqrt(s_b ** 2 + sTs * s_c ** 2)

    # compute s over theta (len(theta), velocity_axis)

    # compute Tex and tau over theta
    Tex_over_theta = np.zeros((number_of_layers, len(theta)))
    tau_over_theta = np.zeros((number_of_layers, len(theta)))

    for layer_idx in range(number_of_layers):
        T_kin_idx = T_kin[0, layer_idx]
        nH2_idx = nH2[0, layer_idx]
        N_molecule_idx = N_molecule[0, layer_idx]
        FWHM_layer_idx = FWHM[0, layer_idx]

        T_ex_line_layer_idx = Tex[0, layer_idx]
        opacity_line_layer_idx = tau[0, layer_idx]
        
        T_ex_over_theta_layer_idx, opacity_over_theta_layer_idx = derive_Tex_tau_over_theta(
            name_mol,
            f'{file_name}',
            freq_min,
            freq_max,
            T_kin_idx,
            colliders,
            nH2_idx,
            N_molecule_idx,
            FWHM_layer_idx,
            geometry,
            T_ex_line_layer_idx,
            opacity_line_layer_idx,
            theta,
            clean=(not DEBUG), 
            DEBUG=DEBUG)

        Tex_over_theta[layer_idx, :] = T_ex_over_theta_layer_idx
        tau_over_theta[layer_idx, :] = opacity_over_theta_layer_idx

    s_over_theta = derive_s_over_theta(
        Tex,
        tau,
        Tex_over_theta,
        tau_over_theta,
        freq,
        velocity_axis,
        velocity_resolution,
        s_V,
        C_V,
        unique_layer_idx = unique_layer_idx,
        constraints_kinematics = constraints_kinematics,
        constraints_geometry = constraints_geometry,
        layers = layers,
        number_of_unknowns = number_of_unknowns,
        idxs_inner_layer = idxs_inner_layer,
        number_of_layers_per_clump = number_of_layers_per_clump,
        number_of_clumps = number_of_clumps,
        conserved_flux = conserved_flux,
        theta = theta,
        C_V_idx = C_V_idx,
        reference_velocity_resolution = reference_velocity_resolution, 
        DEBUG = DEBUG, 
        peak_only = peak_only
    )

    fim_1 = ((c_0/s_b)**2 + (sTs * s_c**4) / (s_b * s_x)**2) * \
        (np.transpose(s_over_theta) @ s_over_theta)

    fim_2_1 = ((s_b * s_c**2)**2 - sTs * s_c**6) / (s_b * s_x**2)**2
    fim_2_2 = ((s_c * c_0) / (s_b * s_x))**2
    fim_2 = (fim_2_1 - fim_2_2) * ((np.transpose(s_over_theta)
                                    @ np.transpose(s)) @ (s @ s_over_theta))

    fim = fim_1 + fim_2

    if NLL_g:
        # compute the NLL gradient
        '''
        NLL_g = (1 / s_b**2) * (np.transpose(s_over_theta) @ np.transpose(s - x))
        '''
        b = x - c_0 * s
        bTs = (b @ np.transpose(s)).item()  # scalar
        bTs_over_theta = np.transpose(s_over_theta) @ np.transpose(b)
        sTs_over_theta = np.transpose(s_over_theta) @ np.transpose(s)

        NLL_g_1 = (s_c**2/s_x**2) * sTs
        NLL_g_2 = ( s_c**2 / (s_b**2 * s_x**2) ) * bTs * bTs_over_theta - ( s_c**4 / (s_b**2 * s_x**4))  * (bTs**2 * sTs_over_theta)
        NLL_g_3 = - (c_0 / s_b**2) * bTs_over_theta + (c_0 * s_c**2)/(s_b**2 * s_x**2) * (sTs_over_theta * bTs)

        NLL_g = NLL_g_1 - NLL_g_2 + NLL_g_3

        return fim, NLL_g

    else:
        return fim

def inverse_fim(
        fim: np.array,
        maximal_fim_cond: Optional[Real] = 10**9.
) -> np.array:

    try:
        fim_cond = np.linalg.cond(fim)

        if fim_cond > maximal_fim_cond:
            # fim is ill-conditioned
            u, s, v = np.linalg.svd(np.mat(fim))

            vh = v.T
            s_ = np.where(s > s[0]/maximal_fim_cond, 1, 0)

            r = sum(s_)
            vec_1 = np.ones(r)
            vec_2 = s[0:r]
            ss = np.diag(vec_1/vec_2)
            inversed_fim = np.dot(
                vh[:, 0:r], ss) * np.transpose(u[:, 0:r])

        else:
            inversed_fim = np.linalg.inv(fim)
    except:
        inversed_fim = np.zeros(np.shape(fim))

    return inversed_fim
