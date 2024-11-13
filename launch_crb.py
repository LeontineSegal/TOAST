'''
This routine allows to compute the Cramér-Rao lower Bound (CRB) on the estimation results, 
to give confidence intervals of reference. 
to launch it, type in your terminal: python launch_crb.py
'''

# %% load modules and required constants
import numpy as np
import os
import tqdm

from model import *

from toolboxs.toolbox_crb.toolbox_crb import get_s_c
from toolboxs.toolbox_python.toolbox_python import from_list_to_array, check_shape
from toolboxs.toolbox_physics.toolbox_radiative_transfer import from_FWHM_to_s_V
from toolboxs.toolbox_crb.toolbox_crb import compute_fim, inverse_fim

#%% settings 
CALIBRATION_NOISE_CRB = True # (DO NOT MODIFY)
FORMAT = 'npy' # among {'npy', 'fits'} (TO MODIFY)

if FORMAT == 'fits':
    from astropy.io import fits

#%% functions

def get_CRB_pixel(pixel, 
                  optimization_):
    '''Allows to print the estimations +/- sqrt(CRB) for one pixel.
    '''
    
    folder = 'accuracies'

    row_idx, column_idx = pixel[0], pixel[1]

    maps_theta = np.load(
        f'{path_save}/{folder_save}/maps_theta_{optimization_}.npy', allow_pickle=True)

    unknows = []
    for layer in range(number_of_different_layers):
        unknows.append(
            10**maps_theta[row_idx, column_idx, log10_T_kin_idx, layer])
        unknows.append(maps_theta[row_idx, column_idx, log10_nH2_idx, layer])
        unknows.append(maps_theta[row_idx, column_idx, log10_N_idx, layer])
        unknows.append(maps_theta[row_idx, column_idx, s_V_idx, layer])
        unknows.append(maps_theta[row_idx, column_idx, C_V_idx, layer])

    CRB = np.load(
        f'{path_save}/{folder_save}/{folder}/maps_crb_{optimization_}.npy', allow_pickle=True)

    for i_idx in range(CRB.shape[-2]):
        for j_idx in range(CRB.shape[-1]):
            if j_idx == i_idx:
                CRB_ = np.sqrt(CRB[row_idx, column_idx, i_idx, j_idx])

                print(f'\n{unknows[j_idx]} +/- {CRB_}')

#%% main 

if __name__ == '__main__':

    folder = 'accuracies'
    # create folder to save results
    os.system(f'mkdir -p {path_save}/{folder_save}/{folder}')
    # moving into the directory to avoid conflict between RADEX files 
    os.chdir(f'{path_save}/{folder_save}/{folder}')

    # load the dataset
    FoV = np.load(f'{path_inputs}/FoV.npy', allow_pickle=True)
    dataset_ppv = np.load(f'{path_inputs}/ppv.npy', allow_pickle=True)
    dataset_names_mol = np.load(
        f'{path_inputs}/names_mol.npy', allow_pickle=True)
    dataset_names_mol = list(dataset_names_mol)
    dataset_names_line = np.load(
        f'{path_inputs}/names_line.npy', allow_pickle=True)
    dataset_names_line = list(dataset_names_line)
    dataset_velocity_channels = np.load(
        f'{path_inputs}/velocity_channels.npy', allow_pickle=True)

    dataset_velocity_res = np.load(
        f'{path_inputs}/velocity_res.npy', allow_pickle=True)

    dataset_rest_frequencies = np.load(
        f'{path_inputs}/rest_frequencies.npy', allow_pickle=True)
    dataset_rest_frequencies = list(dataset_rest_frequencies)
    dataset_colden_ratios = np.load(
        f'{path_inputs}/colden_ratios.npy', allow_pickle=True)
    if SNR_TRICK:
        dataset_maps_s_b = np.load(
            f'{path_inputs}/maps_s_b_snr_trick.npy', allow_pickle=True)
    else:
        dataset_maps_s_b = np.load(
            f'{path_inputs}/maps_s_b.npy', allow_pickle=True)
    map_C_V = np.load(f'{path_inputs}/map_C_V.npy', allow_pickle=True)

    if len(map_C_V.shape) > 2:
        number_of_C_V_components = map_C_V.shape[-1]
        assert number_of_C_V_components == number_of_clumps, print(
            f'{bcolors.WARNING}\n [error] The number of velocity components should be the number of different clumps.\n{bcolors.ENDC}')
    else:
        number_of_C_V_components = 1

    # load the estimation results
    if 'gd' in optimization :
        optimization_ = 'gd'
    else : 
        optimization_ = 'rw'
    maps_Tex = np.load(f'{path_save}/{folder_save}/maps_Tex_{optimization_}.npy', allow_pickle=True)
    maps_tau = np.load(f'{path_save}/{folder_save}/maps_tau_{optimization_}.npy', allow_pickle=True)
    maps_theta = np.load(f'{path_save}/{folder_save}/maps_theta_{optimization_}.npy', allow_pickle=True)
    maps_log10_N = np.load(f'{path_save}/{folder_save}/maps_log10_N_{optimization_}.npy', allow_pickle=True)

    # formalizing the data to the model fitting optimization
    
    # keep only information about the considered species
    ppv = []
    velocity_channels = []
    velocity_res = []
    rest_frequencies = []
    maps_s_b = []
    maps_s_c = []
    colden_ratios = {}
    for name_mol_idx, name_mol in enumerate(names_mol):
        name_mol_idx_ = dataset_names_mol.index(name_mol)

        ppv.append([])
        velocity_channels.append([])
        velocity_res.append([])
        rest_frequencies.append([])
        maps_s_b.append([])
        maps_s_c.append([])
        colden_ratios[name_mol] = dataset_colden_ratios.item().get(name_mol)
        for name_line_idx, name_line in enumerate(names_line[name_mol_idx]):
            name_line_idx_ = dataset_names_line[name_mol_idx_].index(name_line)

            ppv[name_mol_idx].append(
                dataset_ppv[name_mol_idx_][name_line_idx_])
            velocity_channels[name_mol_idx].append(
                dataset_velocity_channels[name_mol_idx_][name_line_idx_])
            velocity_res[name_mol_idx].append(
                dataset_velocity_res[name_mol_idx_][name_line_idx_])
            rest_frequencies[name_mol_idx].append(
                dataset_rest_frequencies[name_mol_idx_][name_line_idx_])
            maps_s_b[name_mol_idx].append(
                dataset_maps_s_b[name_mol_idx_][name_line_idx_])

            if CALIBRATION_NOISE_CRB: # calibration noise is always taken into account for accuracy analysis
                maps_s_c[name_mol_idx].append(
                    get_s_c(name_line) * np.ones(FoV.shape))
            else: 
                maps_s_c[name_mol_idx].append(np.zeros(map_C_V.shape))

    if WINDOW_BANDWIDTH:
        if number_of_C_V_components > 1:
            map_closest_V_to_C_V_idx = []

            for name_mol_idx, name_mol in enumerate(names_mol):
                map_closest_V_to_C_V_idx.append([])

                for name_line_idx, name_line in enumerate(names_line[name_mol_idx]):

                    velocity_channels_ = velocity_channels[name_mol_idx][name_line_idx][0, :]
                    velocity_res_ = velocity_res[name_mol_idx][name_line_idx]

                    map_closest_V_to_C_V_idx_ = np.zeros(
                        (FoV.shape[0], FoV.shape[1], number_of_C_V_components), dtype=int)
                    for component_idx in range(number_of_C_V_components):
                        vector_C_V = np.ravel(map_C_V[:, :, component_idx])
                        map_velocity_channels_ = np.broadcast_to(
                            velocity_channels_, (vector_C_V.size, velocity_channels_.size)).T
                        map_closest_V_to_C_V = abs(
                            map_velocity_channels_ - vector_C_V)
                        map_closest_V_to_C_V_idx__ = np.argmin(
                            map_closest_V_to_C_V, axis=0)
                        map_closest_V_to_C_V_idx__ = map_closest_V_to_C_V_idx__.reshape(
                            map_C_V[:, :, component_idx].shape)
                        map_closest_V_to_C_V_idx_[
                            :, :, component_idx] = map_closest_V_to_C_V_idx__

                    map_closest_V_to_C_V_idx[name_mol_idx].append(
                        map_closest_V_to_C_V_idx_)
        else:
            map_closest_V_to_C_V_idx = []

            for name_mol_idx, name_mol in enumerate(names_mol):
                map_closest_V_to_C_V_idx.append([])

                for name_line_idx, name_line in enumerate(names_line[name_mol_idx]):

                    velocity_channels_ = velocity_channels[name_mol_idx][name_line_idx][0, :]
                    velocity_res_ = velocity_res[name_mol_idx][name_line_idx]

                    vector_C_V = np.ravel(map_C_V)
                    map_velocity_channels_ = np.broadcast_to(
                        velocity_channels_, (vector_C_V.size, velocity_channels_.size)).T
                    map_closest_V_to_C_V = abs(
                        map_velocity_channels_ - vector_C_V)
                    map_closest_V_to_C_V_idx_ = np.argmin(
                        map_closest_V_to_C_V, axis=0)
                    map_closest_V_to_C_V_idx_ = map_closest_V_to_C_V_idx_.reshape(
                        map_C_V.shape)
                    map_closest_V_to_C_V_idx[name_mol_idx].append(
                        map_closest_V_to_C_V_idx_)

    # total number of unknowns
    if 'same_C_V_in_all_layers' in constraints_kinematics:
        number_of_unknowns = (len(theta) - 1) * number_of_different_layers + 1
    else : 
        number_of_unknowns = len(theta) * number_of_different_layers

    coordinates = np.nonzero(np.where(np.isnan(FoV), 0, 1))
    coordinates_i, coordinates_j = coordinates[0], coordinates[1]
    if SINGLE_PIXEL_ANALYSIS : 
        # repet the pixel as much is needed
        coordinates_i = np.repeat(coordinates_i, number_of_analysis)
        coordinates_j = np.repeat(coordinates_j, number_of_analysis)
    LoS = list(zip(*tuple([coordinates_i, coordinates_j])))
    number_of_LoS = len(LoS)

    # initialize array to save results
    if not SINGLE_PIXEL_ANALYSIS : 
        maps_crb = np.zeros(
        (FoV.shape[0], FoV.shape[1], number_of_unknowns, number_of_unknowns))
    else : 
        maps_crb = np.zeros(
        (1, len(LoS), number_of_unknowns, number_of_unknowns))
    maps_crb.fill(np.nan)
    
    if VERBOSE:
        print(
            f"\n{bcolors.HEADER} Starting CRB computation {bcolors.ENDC}({number_of_LoS} pixels)")

    pbar_crb = tqdm.tqdm(total=number_of_LoS, position = 0, leave = True)
    for pixel_idx in range(number_of_LoS):

        pixel = (LoS[pixel_idx][0].item(), LoS[pixel_idx][1].item())
        row_idx, column_idx = pixel[0], pixel[1]

        LoS_C_V = map_C_V[row_idx, column_idx]
        LoS_s_b = [[maps_s_b[mol_idx][line_idx][row_idx, column_idx] for line_idx in range(len(names_line[mol_idx]))] for mol_idx in range(len(names_mol))]
        LoS_s_c = [[maps_s_c[mol_idx][line_idx][row_idx, column_idx] for line_idx in range(len(names_line[mol_idx]))] for mol_idx in range(len(names_mol))]

        LoS_velocity_channels = []

        # measures, useful of PEAK_ONLY
        LoS_ppv = [[ppv[mol_idx][line_idx][row_idx, column_idx, :].reshape(
        (1, ppv[mol_idx][line_idx][row_idx, column_idx, :].size)) for line_idx in range(len(names_line[mol_idx]))] for mol_idx in range(len(names_mol))]

        if WINDOW_BANDWIDTH:
            windowed_LoS_ppv = []

            if number_of_C_V_components > 1:

                for name_mol_idx in range(len(names_mol)):
                    LoS_velocity_channels.append([])
                    windowed_LoS_ppv.append([])

                    for name_line_idx in range(len(names_line[name_mol_idx])):

                        delta_V_ = delta_V[name_mol_idx][name_line_idx]
                        LoS_closest_V_to_C_V_idx = map_closest_V_to_C_V_idx[
                            name_mol_idx][name_line_idx][row_idx, column_idx, :]
                        velocity_res_ = velocity_res[name_mol_idx][name_line_idx]

                        min_bandwidth = np.min(LoS_closest_V_to_C_V_idx).item() - \
                            int(delta_V_/velocity_res_)
                        min_bandwidth = max(min_bandwidth, 0)

                        max_bandwidth = np.max(LoS_closest_V_to_C_V_idx).item() + \
                            int(delta_V_/velocity_res_) + 1
                        max_bandwidth = min(
                            max_bandwidth, velocity_channels[name_mol_idx][name_line_idx].size)

                        bandwidth = np.arange(min_bandwidth, max_bandwidth)

                        LoS_velocity_channels_ = velocity_channels[name_mol_idx][name_line_idx][0, bandwidth]
                        LoS_velocity_channels_ = LoS_velocity_channels_.reshape(
                            (1, bandwidth.size))
                        LoS_velocity_channels[name_mol_idx].append(
                            LoS_velocity_channels_)
                        
                        windowed_LoS_ppv_ = LoS_ppv[name_mol_idx][name_line_idx][0, bandwidth]
                        windowed_LoS_ppv_ = windowed_LoS_ppv_.reshape(
                        (1, bandwidth.size))
                        windowed_LoS_ppv[name_mol_idx].append(windowed_LoS_ppv_)

                LoS_ppv = windowed_LoS_ppv

            else:

                windowed_LoS_ppv = []

                for name_mol_idx in range(len(names_mol)):
                    LoS_velocity_channels.append([])
                    windowed_LoS_ppv.append([])
                    for name_line_idx in range(len(names_line[name_mol_idx])):

                        delta_V_ = delta_V[name_mol_idx][name_line_idx]

                        LoS_closest_V_to_C_V_idx = map_closest_V_to_C_V_idx[
                            name_mol_idx][name_line_idx][row_idx, column_idx]
                        velocity_res_ = velocity_res[name_mol_idx][name_line_idx]

                        min_bandwidth = LoS_closest_V_to_C_V_idx - \
                            int(delta_V_/velocity_res_)
                        min_bandwidth = max(min_bandwidth, 0)

                        max_bandwidth = LoS_closest_V_to_C_V_idx + \
                            int(delta_V_/velocity_res_) + 1
                        max_bandwidth = min(
                            max_bandwidth, velocity_channels[name_mol_idx][name_line_idx].size)

                        # symetrical bandwidth
                        size_window_left = len(
                            np.arange(min_bandwidth, LoS_closest_V_to_C_V_idx))
                        size_window_right = len(
                            np.arange(LoS_closest_V_to_C_V_idx+1, max_bandwidth))
                        min_window_size = min(size_window_left, size_window_right)

                        if size_window_left > min_window_size:
                            min_bandwidth = LoS_closest_V_to_C_V_idx - min_window_size
                        if size_window_right > min_window_size:
                            max_bandwidth = LoS_closest_V_to_C_V_idx + min_window_size

                        bandwidth = np.arange(min_bandwidth, max_bandwidth)

                        LoS_velocity_channels_ = velocity_channels[name_mol_idx][name_line_idx][0, bandwidth]
                        LoS_velocity_channels_ = LoS_velocity_channels_.reshape(
                            (1, bandwidth.size))
                        LoS_velocity_channels[name_mol_idx].append(
                            LoS_velocity_channels_)

                        windowed_LoS_ppv_ = LoS_ppv[name_mol_idx][name_line_idx][0, bandwidth]
                        windowed_LoS_ppv_ = windowed_LoS_ppv_.reshape(
                            (1, bandwidth.size))
                        windowed_LoS_ppv[name_mol_idx].append(windowed_LoS_ppv_)

                LoS_ppv = windowed_LoS_ppv
        else:
            LoS_velocity_channels = velocity_channels

        if PEAK_ONLY:

            for name_mol_idx, name_mol in enumerate(names_mol):
                for name_line_idx, name_line in enumerate(names_line[name_mol_idx]):
                    if f'{name_mol}({name_line})' in thick_lines:

                        LoS_ppv_ = LoS_ppv[name_mol_idx][name_line_idx]
                        LoS_velocity_channels_ = LoS_velocity_channels[name_mol_idx][name_line_idx]

                        if number_of_C_V_components == 1:  # no ambiguity on what the peak is...
                            peak_LoS_ppv = np.max(LoS_ppv_).reshape((1, 1))
                            peak_idx = np.unravel_index(
                                np.argmax(LoS_ppv_), LoS_ppv_.shape)

                            peak_LoS_velocity_channels = LoS_velocity_channels_[
                                0, peak_idx[1]].reshape((1, 1))

                        else:
                            peak_LoS_ppv = np.zeros((1, number_of_clumps))
                            peak_LoS_velocity_channels = np.zeros(
                                (1, number_of_clumps))
                            velocity_res_ = velocity_res[name_mol_idx][name_line_idx]

                            for clump_idx in range(number_of_clumps):
                                LoS_closest_V_to_C_V_idx = abs(
                                    LoS_velocity_channels_ - map_C_V[row_idx, column_idx, clump_idx])
                                LoS_closest_V_to_C_V_idx = np.argmin(
                                    LoS_closest_V_to_C_V_idx)

                                min_bandwidth = LoS_closest_V_to_C_V_idx - \
                                    int(delta_V_around_peak/velocity_res_)
                                min_bandwidth = max(min_bandwidth, 0)
                                max_bandwidth = LoS_closest_V_to_C_V_idx + \
                                    int(delta_V_around_peak/velocity_res_) + 1
                                max_bandwidth = min(
                                    max_bandwidth, LoS_velocity_channels_.size)
                                bandwidth = np.arange(min_bandwidth, max_bandwidth)

                                peak_LoS_ppv_clump = np.max(LoS_ppv_[0, bandwidth])

                                peak_idx_clump = np.unravel_index(
                                    np.argmax(LoS_ppv_[0, bandwidth]), LoS_ppv_[0, bandwidth].shape)[0]
                                peak_LoS_velocity_channels_clump = LoS_velocity_channels_[
                                    0, bandwidth[peak_idx_clump]]

                                peak_LoS_ppv[0, clump_idx] = peak_LoS_ppv_clump
                                peak_LoS_velocity_channels[0,
                                                        clump_idx] = peak_LoS_velocity_channels_clump

                        # update
                        LoS_ppv[name_mol_idx][name_line_idx] = peak_LoS_ppv
                        LoS_velocity_channels[name_mol_idx][name_line_idx] = peak_LoS_velocity_channels

        optimal_theta = maps_theta[row_idx, column_idx, :, :]
        log10_N = maps_log10_N[row_idx, column_idx, :, :]

        try:
            '''
            if VERBOSE:
                print(
                f'\t Processing the pixel (i, j) = {bcolors.BOLD}({row_idx}, {column_idx}){bcolors.ENDC}')
            '''
            crb_log10_T_kin = from_list_to_array(optimal_theta[log10_T_kin_idx, :])
            crb_T_kin = 10 ** crb_log10_T_kin
            crb_log10_nH2 = from_list_to_array(optimal_theta[log10_nH2_idx, :])
            crb_nH2 = 10 ** crb_log10_nH2
            crb_FWHM = from_list_to_array(optimal_theta[FWHM_idx, :])
            crb_s_V = from_FWHM_to_s_V(crb_FWHM)
            crb_C_V = from_list_to_array(optimal_theta[C_V_idx, :])

            # Fisher information matrix
            fim = np.zeros((number_of_unknowns, number_of_unknowns)) 
                    
            for molecule_idx, molecule in enumerate(names_mol):
                crb_N = 10**log10_N[molecule_idx, :]

                for line_idx, line in enumerate(names_line[molecule_idx]):
                    velocity_channels_line = LoS_velocity_channels[molecule_idx][line_idx]
                    s_b_line = LoS_s_b[molecule_idx][line_idx]
                    s_c_line = LoS_s_c[molecule_idx][line_idx]

                    freq_crb = rest_frequencies[molecule_idx][line_idx]
                    velocity_resolution = velocity_res[molecule_idx][line_idx]

                    T_ex_line, opacity_line = [], []
                    for layer in range(layers):
                        T_ex_line.append(
                            maps_Tex[molecule_idx][line_idx][layer][row_idx, column_idx])
                        opacity_line.append(
                            maps_tau[molecule_idx][line_idx][layer][row_idx, column_idx])
                    T_ex_line = from_list_to_array(T_ex_line)
                    opacity_line = from_list_to_array(opacity_line)

                    optimal_Tex_line = T_ex_line.reshape((len(T_ex_line), 1))
                    optimal_Tex_line = np.transpose(optimal_Tex_line)

                    optimal_tau_line = opacity_line.reshape((len(opacity_line), 1))
                    optimal_tau_line = np.transpose(optimal_tau_line)

                    if not check_shape(optimal_Tex_line, crb_s_V, crb_FWHM, shape_of_reference=optimal_Tex_line.shape, flag=False):
                        crb_s_V = crb_s_V.reshape(optimal_Tex_line.shape)
                        crb_FWHM = crb_FWHM.reshape(optimal_Tex_line.shape)
                        crb_C_V = crb_C_V.reshape(optimal_Tex_line.shape)
                    if not check_shape(crb_T_kin, crb_nH2, crb_N, crb_s_V, crb_FWHM, shape_of_reference=crb_s_V.shape, flag=False):
                        crb_T_kin = crb_T_kin.reshape(crb_s_V.shape)
                        crb_nH2 = crb_nH2.reshape(crb_s_V.shape)
                        crb_N = crb_N.reshape(crb_s_V.shape)

                    if velocity_resolution == 0.1 or velocity_resolution == 0.5: 
                        if optimization_ == 'gd' : 
                            reference_velocity_resolution_ = 0.01
                        else : 
                            reference_velocity_resolution_ = 0.1
                    elif velocity_resolution == 0.25 : 
                        if optimization_ == 'gd' : 
                            reference_velocity_resolution_ = 0.125
                        else : 
                            reference_velocity_resolution_ = 0.125

                    fim_ = compute_fim(
                                        molecule,
                                        freq_crb,
                                        velocity_channels_line,
                                        velocity_resolution,
                                        crb_T_kin,
                                        crb_nH2,
                                        crb_N,
                                        crb_FWHM,
                                        crb_s_V,
                                        crb_C_V,
                                        colliders,
                                        geometry,
                                        optimal_Tex_line,
                                        optimal_tau_line,
                                        1.,
                                        s_c_line,
                                        s_b_line,
                                        unique_layer_idx = unique_layer_idx,
                                        file_name = f'{row_idx}_{column_idx}',
                                        constraints_kinematics = constraints_kinematics,
                                        constraints_geometry=constraints_geometry,
                                        layers = layers,
                                        idxs_inner_layer = idxs_inner_layer,
                                        number_of_unknowns = number_of_unknowns, 
                                        number_of_clumps=number_of_clumps,
                                        number_of_layers_per_clump = number_of_layers_per_clump,
                                        peak_only = (PEAK_ONLY and f'{names_mol[molecule_idx]}({names_line[molecule_idx][line_idx]})' in thick_lines),
                                        conserved_flux=True,
                                        reference_velocity_resolution = reference_velocity_resolution_,
                                        theta = theta,
                                        C_V_idx=C_V_idx,
                                        DEBUG=DEBUG                                    
                                    )
                    fim += fim_
            
            crb = inverse_fim(fim)
            maps_crb[row_idx, column_idx, :, :] = crb
            
            if FORMAT == 'fits':
                for u_idx in range(len(crb)) : 
                    layer_idx = u_idx // len(theta)
                    t_idx = u_idx % len(theta)
                    t_map = maps_crb[:, :, u_idx, u_idx]
                    hdu = fits.PrimaryHDU(data=t_map)
                    hdu.writeto(f'map_crb_{theta[t_idx]}_{layer_idx+1}_{optimization_}.{FORMAT}', overwrite=True)

            else :
                np.save(f'maps_crb_{optimization_}.{FORMAT}', maps_crb)

        except : 
            crb_nan = np.zeros(
            (number_of_unknowns, number_of_unknowns))
            crb_nan.fill(np.nan)
            maps_crb[row_idx, column_idx, :, :] = crb_nan

            if FORMAT == 'fits':
                for u_idx in range(len(crb)) : 
                    layer_idx = u_idx // len(theta)
                    t_idx = u_idx % len(theta)
                    t_map = maps_crb[:, :, u_idx, u_idx]
                    hdu = fits.PrimaryHDU(data=t_map)
                    hdu.writeto(f'map_crb_{theta[t_idx]}_{layer_idx+1}_{optimization_}.{FORMAT}', overwrite=True)

            else :
                np.save(f'maps_crb_{optimization_}.{FORMAT}', maps_crb)

        os.system(f'rm -f *{row_idx}_{column_idx}.inp')
        os.system(f'rm -f *{row_idx}_{column_idx}.out')

        pbar_crb.update(1)

    os.system(f'rm -f radex.log')

    if VERBOSE:
        print(
            f"{bcolors.HEADER} CRB computation {bcolors.OKGREEN}done.{bcolors.ENDC}")

#get_CRB_pixel([16, 21], 'gd')
get_CRB_pixel([17, 40], 'gd')