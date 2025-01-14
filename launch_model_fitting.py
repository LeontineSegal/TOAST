'''
This file allows to launch the model fitting routine. 
to launch it, type in your terminal: python launch_model_fitting.py
'''

# %% load modules and required constants
import tqdm
import time
from typing import Optional

from model import *
from toolboxs.toolbox_crb.toolbox_crb import get_s_c
from toolboxs.toolbox_estimator.toolbox_estimator import random_walk, gradient_descent
from toolboxs.toolbox_python.toolbox_python import from_list_to_array
from toolboxs.toolbox_physics.toolbox_radiative_transfer import from_FWHM_to_s_V

if PLOT:
    import matplotlib.pyplot as plt
    from toolboxs.toolbox_plot.toolbox_plot import show_LoS, show_LoS_model
if VERBOSE:
    from toolboxs.toolbox_python.toolbox_python import bcolors
if WRITE_RESULTS_TXT_FILE:
    from toolboxs.toolbox_estimator.toolbox_estimator import write_idx_pixel_rw, write_result_file_rw_1, write_result_file_rw_2
    from toolboxs.toolbox_estimator.toolbox_estimator import write_idx_pixel_gd, write_result_file_gd_1, write_result_file_gd_2
if PARALLELISM:
    from multiprocessing import Pool, Manager, Process
from multiprocessing.queues import Queue

# %% functions 

def fit_pixel(
        pixel_idx: int,
        row_idx: int,
        column_idx: int,
        inputs: dict, 
        queue_rw: Optional[Queue] = None, 
        queue_gd: Optional[Queue] = None
) -> dict:

    results = {}
    results['pixel'] = (row_idx, column_idx) 

    if SINGLE_PIXEL_ANALYSIS : 
        results['pixel_idx'] = pixel_idx

    if VERBOSE:
        print(
            f'\tAnalyzing the pixel (i, j) = {bcolors.BOLD}({row_idx}, {column_idx}){bcolors.ENDC}')

    FPS = inputs['FPS']
    dimensions_FPS = inputs['dimensions_FPS'] 
    dimensions_FPS_without_colden = inputs['dimensions_FPS_without_colden'] 

    ppv = inputs['ppv']
    maps_s_b = inputs['maps_s_b']
    maps_s_c = inputs['maps_s_c']
    number_of_C_V_components = inputs['number_of_C_V_components']
    map_C_V = inputs['map_C_V']
    velocity_res = inputs['velocity_res']
    velocity_channels = inputs['velocity_channels'] 
    rest_frequencies = inputs['rest_frequencies']

    grids_Tex = inputs['grids_Tex']
    grids_tau = inputs['grids_tau']

    colden_ratios = inputs['colden_ratios']
    initial_idx_walkers_theta = inputs['initial_idx_walkers_theta']
    initial_idx_walkers_colden = inputs['initial_idx_walkers_colden']
    initial_walkers_colden_ratios_shifts = inputs['initial_walkers_colden_ratios_shifts']
    initial_s_V_walkers = inputs['initial_s_V_walkers']
    idx_min_max_colden = inputs['idx_min_max_colden']
    colden_res = inputs['colden_res']
    total_number_of_initial_walkers = inputs['total_number_of_initial_walkers']

    # measures
    LoS_ppv = [[ppv[mol_idx][line_idx][row_idx, column_idx, :].reshape(
        (1, ppv[mol_idx][line_idx][row_idx, column_idx, :].size)) for line_idx in range(len(names_line[mol_idx]))] for mol_idx in range(len(names_mol))]
    LoS_s_b = [[maps_s_b[mol_idx][line_idx][row_idx, column_idx] for line_idx in range(
        len(names_line[mol_idx]))] for mol_idx in range(len(names_mol))]
    LoS_s_c = [[maps_s_c[mol_idx][line_idx][row_idx, column_idx] for line_idx in range(
        len(names_line[mol_idx]))] for mol_idx in range(len(names_mol))]

    if number_of_C_V_components == 1:
        # initialize each layer velocity centroid on the first estimation velocity centroid
        initial_C_V_walkers = map_C_V[row_idx, column_idx] * \
            np.ones((total_number_of_initial_walkers, layers))
    else : 
        initial_C_V_walkers = np.ones(
            (total_number_of_initial_walkers, layers))
        
        for layer in range(layers) : 
            # in which clump are we ?
            idx_clump = (
                layer//number_of_layers_per_clump) % number_of_clumps
            # get the C_V of the clump
            C_V_clump = map_C_V[row_idx, column_idx, idx_clump]
            # normalizing [0, 1] values C_V between C_V_clump +/- sigma
            vmin = np.round(C_V_clump - 0.25, 3)
            vmax = np.round(C_V_clump + 0.25, 3)
            C_V_range = np.arange(vmin, vmax + C_V_res, C_V_res)

            if 'sandwich' in constraints_geometry : 
                # in which sandwich are we ?
                idx_inner_layer = idxs_inner_layer[idx_clump]
                if layer > idx_inner_layer : 
                    # find the opposite layer
                    shift = layer - idx_inner_layer
                    idx_opposed_layer = idx_inner_layer - shift
                    if 'mirror' in constraints_kinematics:
                        initial_C_V_walkers[:, layer] = 2 * initial_C_V_walkers[:,idx_inner_layer] - initial_C_V_walkers[:, idx_opposed_layer]
                    else : # basic sandwich model
                        initial_C_V_walkers[:, layer] = initial_C_V_walkers[:, idx_opposed_layer]
                else : 
                    initial_C_V_walkers[:, layer] = np.random.choice(C_V_range, size = total_number_of_initial_walkers)
            else : 
                initial_C_V_walkers[:, layer] = np.random.choice(C_V_range, size = total_number_of_initial_walkers)
    if PLOT:
        names_mol_line_latex = inputs['names_mol_line_latex']
   
        # measures
        LoS_ppv = [[ppv[mol_idx][line_idx][row_idx, column_idx, :].reshape(
            (1, ppv[mol_idx][line_idx][row_idx, column_idx, :].size)) for line_idx in range(len(names_line[mol_idx]))] for mol_idx in range(len(names_mol))]

        plot_LoS_ppv = []
        plot_LoS_velocity_channels = []
        plot_names_mol_line_latex = []
        plot_delta_V = []
        for name_mol_idx, name_mol in enumerate(names_mol):
            for name_line_idx, name_line in enumerate(names_line[name_mol_idx]):
                plot_LoS_ppv.append(
                    LoS_ppv[name_mol_idx][name_line_idx][0, :])
                plot_LoS_velocity_channels.append(
                    velocity_channels[name_mol_idx][name_line_idx][0, :])
                plot_names_mol_line_latex.append(
                    names_mol_line_latex[name_mol_idx][name_line_idx])
                plot_delta_V.append(delta_V[name_mol_idx][name_line_idx])

        plot_C_V_initial = []
        if number_of_C_V_components == 1:
            plot_C_V_initial.append(map_C_V[row_idx, column_idx])
        else:
            for n_o_c in range(number_of_C_V_components):
                plot_C_V_initial.append(
                    map_C_V[row_idx, column_idx, n_o_c])

        show_LoS(
            plot_LoS_ppv,
            plot_LoS_velocity_channels,
            plot_names_mol_line_latex,
            name_fig=f'figures/measures_{row_idx}_{column_idx}_{pixel_idx}',
            C_V_initial=plot_C_V_initial, 
            delta_V = plot_delta_V, 
            save = PLOT, 
            FORMAT_FILE=FORMAT_FILE
        )

    LoS_velocity_channels = []
    if WINDOW_BANDWIDTH:

        map_closest_V_to_C_V_idx = inputs['map_closest_V_to_C_V_idx']

        if number_of_C_V_components > 1:
            windowed_LoS_ppv = []

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

    # save all the measures, before potential only keep the peak line
    measured_LoS_ppv = [[LoS_ppv[mol_idx][line_idx] for line_idx in range(
        len(names_line[mol_idx]))] for mol_idx in range(len(names_mol))]
    measured_LoS_velocity_channels = [[LoS_velocity_channels[mol_idx][line_idx] for line_idx in range(
        len(names_line[mol_idx]))] for mol_idx in range(len(names_mol))]

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

    if 'rw' in optimization:

        tic_pixel_rw = time.perf_counter()

        # for progression bar 
        # in which chunk is the pixel ?
        number_of_LoS = inputs['number_of_LoS']
        
        if PARALLELISM : 
            POLL_SIZE = inputs['POLL_SIZE']
            if pixel_idx in [i for i in range(0, number_of_LoS, (number_of_LoS//POLL_SIZE))] : 
                show_prog_bar = True
            else : 
                show_prog_bar = False
        else : 
            show_prog_bar = False

        all_res = random_walk(
            FPS,
            dimensions_FPS,
            dimensions_FPS_without_colden,
            LoS_ppv,
            rest_frequencies,
            LoS_s_b,
            LoS_s_c,
            velocity_res,
            LoS_velocity_channels,
            grids_Tex,
            grids_tau,
            colden_ratios,
            initial_idx_walkers_theta,
            initial_idx_walkers_colden,
            initial_walkers_colden_ratios_shifts,
            initial_s_V_walkers,
            initial_C_V_walkers,
            idx_min_max_colden,
            colden_res,
            walkers_per_step=walkers_per_step,
            iterations_per_step=iterations_per_step,
            show_prog_bar=show_prog_bar
        )

        toc_pixel_rw = time.perf_counter()

        if PARALLELISM : 
            times_rw = f'{(toc_pixel_rw - tic_pixel_rw)/POLL_SIZE:0.2f}'
        else : 
            times_rw = f'{toc_pixel_rw - tic_pixel_rw:0.2f}'

        res_rw = all_res['results']
        optimal_theta_rw, optimal_Tex_rw, optimal_tau_rw, NLL_rw, count_NLL, optimal_colden_rw = res_rw

        results['results_rw'] = [optimal_theta_rw, optimal_Tex_rw,
                                 optimal_tau_rw, NLL_rw, count_NLL, optimal_colden_rw]

        if WRITE_RESULTS_TXT_FILE:
            params = all_res['params']
            results['params_rw'] = params

            params_rw = results['params_rw']
            params_convergence, params_results = params_rw[0], params_rw[1]

            inputs_rw = {}
            inputs_rw['pixel'] = (row_idx, column_idx)
            inputs_rw['pixel_idx'] = pixel_idx
            inputs_rw['params_convergence'] = params_convergence
            inputs_rw['params_results'] = params_results
            inputs_rw['results'] = results['results_rw']
            inputs_rw['time'] = times_rw

            if PARALLELISM:
                queue_rw.put(inputs_rw)
            else:
                write_results_txt_rw(inputs_rw)

    if 'gd' in optimization : 

        if 'rw' in optimization : 
            # get the results have just been computed 
            # have just to adapt the type of Tex and tau 
            optimal_Tex_rw_, optimal_tau_rw_ = [], []
            for mol_idx in range(len(names_mol)) : 
                optimal_Tex_rw_.append([])
                optimal_tau_rw_.append([])
                for line_idx in range(len(names_line[mol_idx])):
                    optimal_Tex_rw_[mol_idx].append([])
                    optimal_tau_rw_[mol_idx].append([])
                    for layer in range(layers) : 
                        optimal_Tex_rw_[mol_idx][line_idx].append(np.float64(optimal_Tex_rw[mol_idx][line_idx][layer]))
                        optimal_tau_rw_[mol_idx][line_idx].append(np.float64(optimal_tau_rw[mol_idx][line_idx][layer]))
            optimal_Tex_rw = optimal_Tex_rw_
            optimal_tau_rw = optimal_tau_rw_
        
        else : 

            maps_theta_rw = inputs['maps_theta_rw']
            maps_Tex_rw = inputs['maps_Tex_rw']
            maps_tau_rw = inputs['maps_tau_rw']
            map_NLL_rw = inputs['map_NLL_rw']
            maps_log10_N_rw = inputs['maps_log10_N_rw']

            # load pre computed results from rw
            optimal_theta_rw = maps_theta_rw[row_idx, column_idx, :, :]
            NLL_rw = map_NLL_rw[row_idx, column_idx]
            optimal_colden_rw = maps_log10_N_rw[row_idx, column_idx, :, :]
            optimal_Tex_rw, optimal_tau_rw = [], []
            for mol_idx in range(len(names_mol)) : 
                optimal_Tex_rw.append([])
                optimal_tau_rw.append([])
                for line_idx in range(len(names_line[mol_idx])):
                    optimal_Tex_rw[mol_idx].append([])
                    optimal_tau_rw[mol_idx].append([])
                    for layer in range(layers) : 
                        optimal_Tex_rw[mol_idx][line_idx].append(maps_Tex_rw[mol_idx][line_idx][layer][row_idx, column_idx])
                        optimal_tau_rw[mol_idx][line_idx].append(maps_tau_rw[mol_idx][line_idx][layer][row_idx, column_idx])
        
        tic_pixel_gd = time.perf_counter()

        # for progression bar 
        # in which chunk is the pixel ?
        number_of_LoS = inputs['number_of_LoS']
        
        if PARALLELISM : 
            POLL_SIZE = inputs['POLL_SIZE']
            if pixel_idx in [i for i in range(0, number_of_LoS, (number_of_LoS//POLL_SIZE))] : 
                show_prog_bar = True
            else : 
                show_prog_bar = False
        else : 
            show_prog_bar = False

        all_res = gradient_descent(
            optimal_theta_rw, 
            optimal_Tex_rw, 
            optimal_tau_rw, 
            NLL_rw, 
            optimal_colden_rw, 
            LoS_ppv, 
            rest_frequencies, 
            LoS_s_b,
            LoS_s_c,
            velocity_res, 
            LoS_velocity_channels, 
            pixel = [row_idx, column_idx]
        )
        toc_pixel_gd = time.perf_counter()

        if PARALLELISM : 
            times_gd = f'{(toc_pixel_gd - tic_pixel_gd)/POLL_SIZE:0.2f}'
        else : 
            times_gd = f'{toc_pixel_gd - tic_pixel_gd:0.2f}'

        res_gd = all_res['results']
        optimal_theta_gd, optimal_Tex_gd, optimal_tau_gd, NLL_gd, iter, optimal_colden_gd = res_gd

        results['results_gd'] = [optimal_theta_gd, optimal_Tex_gd,
                              optimal_tau_gd, NLL_gd, iter, optimal_colden_gd]

        if WRITE_RESULTS_TXT_FILE:
            inputs_gd = {}
            inputs_gd['pixel'] = (row_idx, column_idx)
            inputs_gd['results'] = results['results_gd']
            inputs_gd['time'] = times_gd
            inputs_gd['pixel_idx'] = pixel_idx

            if PARALLELISM:
                queue_gd.put(inputs_gd)
            else:
                write_results_txt_gd(inputs_gd)

    if PLOT:
        if 'gd' in optimization:
            # results from gd
            optimal_Tex = optimal_Tex_gd
            optimal_tau = optimal_tau_gd
            optimal_theta = optimal_theta_gd

        else:  # results from rw
            optimal_Tex = optimal_Tex_rw
            optimal_tau = optimal_tau_rw
            optimal_theta = optimal_theta_rw

        plot_LoS_velocity_channels = []
        plot_LoS_ppv = []
        measured_plot_LoS_velocity_channels = []
        measured_plot_LoS_ppv = []
        plot_velocity_resolution = []
        plot_Tex = []
        plot_tau = []
        plot_freqs = []
        peak_only_plot = []
        for name_mol_idx, name_mol in enumerate(names_mol):
            for name_line_idx, name_line in enumerate(names_line[name_mol_idx]):

                plot_LoS_velocity_channels.append(
                    #    velocity_channels[name_mol_idx][name_line_idx][0, :])
                    LoS_velocity_channels[name_mol_idx][name_line_idx][0, :])
                plot_LoS_ppv.append(LoS_ppv[name_mol_idx][name_line_idx][0, :])

                # measures
                measured_plot_LoS_velocity_channels.append(
                    measured_LoS_velocity_channels[name_mol_idx][name_line_idx][0, :])
                measured_plot_LoS_ppv.append(
                    measured_LoS_ppv[name_mol_idx][name_line_idx][0, :])

                plot_velocity_resolution.append(
                    velocity_res[name_mol_idx][name_line_idx])
                plot_freqs.append(
                    rest_frequencies[name_mol_idx][name_line_idx])
                optimal_Tex_line = optimal_Tex[name_mol_idx][name_line_idx]
                optimal_tau_line = optimal_tau[name_mol_idx][name_line_idx]
                plot_Tex.append(optimal_Tex_line)
                plot_tau.append(optimal_tau_line)
                peak_only_plot.append(
                    PEAK_ONLY and f'{names_mol[name_mol_idx]}({names_line[name_mol_idx][name_line_idx]})' in thick_lines)

        plot_s_V = [np.round(from_FWHM_to_s_V(optimal_theta[s_V_idx, layer].item(
        )), decimals=2).item() for layer in range(layers)]
        plot_C_V = [optimal_theta[C_V_idx, layer].item()
                    for layer in range(layers)]

        show_LoS_model(
            measured_plot_LoS_ppv,
            measured_plot_LoS_velocity_channels,
            plot_LoS_ppv,
            plot_LoS_velocity_channels,
            plot_names_mol_line_latex,
            plot_s_V,
            plot_C_V,
            plot_velocity_resolution,
            plot_Tex,
            plot_tau,
            plot_freqs,
            name_fig=f'figures/model-fit-{row_idx}-{column_idx}-{pixel_idx}',
            peak_only=peak_only_plot,
            number_of_C_V_components=number_of_C_V_components,
            number_of_layers_per_clump=number_of_layers_per_clump, 
            save = PLOT, 
            FORMAT_FILE=FORMAT_FILE
        )

    if DEBUG and ('gd' in optimization):
        # clean pixel without bug, at the very end of the model fitting
        os.system('rm -f ' + f'*_{row_idx}_{column_idx}*.inp')
        os.system('rm -f ' + f'*_{row_idx}_{column_idx}*.out')
        os.system('rm -f radex.log')

    return results

def fit_pixel_wrapped(arg):
    return fit_pixel(*arg)  # unpacks args

# write results...

# ... with basic processes
def write_results_txt_rw(
    inputs: dict
):

    pixel = inputs['pixel']
    pixel_idx = inputs['pixel_idx']
    params_convergence = inputs['params_convergence']
    params_results = inputs['params_results']
    results = inputs['results']
    time = inputs['time']

    number_of_steps = len(params_convergence)
    optimal_theta_rw, optimal_Tex_rw, optimal_tau_rw, NLL_rw, count_NLL, optimal_colden_rw = results

    # write the pixel
    write_idx_pixel_rw(f'results_rw', 
                       pixel,
                       pixel_idx
                       )

    for step_idx in range(number_of_steps):

        parameters = []
        parameters.append(step_idx+1)

        iter_over_max_iter_per_steps_ = params_convergence[step_idx][1]
        parameters.append(iter_over_max_iter_per_steps_)
        parameters.append(
            f'{params_convergence[step_idx][3]}/{params_convergence[step_idx][2]}')
        NLL_per_step_ =  params_convergence[step_idx][4]
        parameters.append(NLL_per_step_)
        Time_per_step_ = params_convergence[step_idx][5]
        parameters.append(Time_per_step_)

        if step_idx == number_of_steps - 1:
            write_result_file_rw_1(f'results_rw',
                                   parameters,
                                   pixel,
                                   pixel_idx,
                                   time,
                                   write_total_time=True
                                   )
        else:
            write_result_file_rw_1(f'results_rw',
                                   parameters,
                                   pixel,
                                   pixel_idx,
                                   '',
                                   write_total_time=False
                                   )

    write_theta_proportions = params_results
    params = [optimal_theta_rw,
              optimal_colden_rw,
              write_theta_proportions,
              optimal_Tex_rw,
              optimal_tau_rw]
    
    write_result_file_rw_2(f'results_rw',
                           params,
                           pixel, 
                           pixel_idx)

def write_results_txt_gd(
    inputs: dict
    ):

    pixel = inputs['pixel']
    pixel_idx = inputs['pixel_idx']
    time = inputs['time']
    results = inputs['results']

    optimal_theta_gd, optimal_Tex_gd, optimal_tau_gd, NLL_gd, iterations_NLL_gd, optimal_colden_gd = results

    # write the pixel
    write_idx_pixel_gd(f'results_gd', 
                       pixel, 
                       pixel_idx)

    # write convergence info
    parameters = [np.round(NLL_gd, 4), 
                f'{iterations_NLL_gd}/{max_iterations}', 
                time]
    
    write_result_file_gd_1(
        f'results_gd',
        parameters,
        pixel, 
        pixel_idx)
    
    params = [optimal_theta_gd,
                optimal_colden_gd, 
                optimal_Tex_gd, 
                optimal_tau_gd]
    
    write_result_file_gd_2(
        f'results_gd',
        params,
        pixel, 
        pixel_idx)
    
# ... or with multiprocessing

def write_results_txt_parallelism_rw(queue):
    with open(f'results_rw', "a") as result_file_rw:
        while True:
            inputs = queue.get()
            if inputs['pixel'] is None:
                break
            # else...
            write_results_txt_rw(inputs)

def write_results_txt_parallelism_gd(queue):
    with open(f'results_gd', "a") as result_file_gd:
        while True:
            inputs = queue.get()
            if inputs['pixel'] is None:
                break
            # else...
            write_results_txt_gd(inputs)


# %% main 

if __name__ == '__main__':

    # create folder to save results
    os.system(f'mkdir -p {path_save}/{folder_save}')
    # moving into the directory to avoid conflict between RADEX files 
    os.chdir(f'{path_save}')

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

    if PLOT:
        os.system(f'mkdir -p figures')

        dataset_names_mol_line_latex = np.load(
            f'{path_inputs}/names_mol_line_latex.npy', allow_pickle=True)
        dataset_names_mol_line_latex = list(dataset_names_mol_line_latex)
        names_mol_line_latex = []
        for name_mol_idx, name_mol in enumerate(names_mol):
            names_mol_line_latex.append([])
            name_mol_idx_ = dataset_names_mol.index(name_mol)
            for name_line_idx, name_line in enumerate(names_line[name_mol_idx]):
                name_line_idx_ = dataset_names_line[name_mol_idx_].index(
                    name_line)
                names_mol_line_latex[name_mol_idx].append(
                    dataset_names_mol_line_latex[name_mol_idx_][name_line_idx_])

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

            if CALIBRATION_NOISE:
                maps_s_c[name_mol_idx].append(
                    get_s_c(name_line) * np.ones(FoV.shape))
            else:
                maps_s_c[name_mol_idx].append(np.zeros(FoV.shape))

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

    grids_Tex = []
    grids_tau = []
    for name_mol_idx, name_mol in enumerate(names_mol):
        grids_Tex.append([]), grids_tau.append([])
        name_mol_idx_ = dataset_names_mol.index(name_mol)
        for name_line_idx, name_line in enumerate(names_line[name_mol_idx]):
            name_line_idx_ = dataset_names_line[name_mol_idx_].index(name_line)

            grid_Tex = np.load(
                f'{path_grids_Tex_tau}/{geometry}/Tex_{name_mol}.npy')
            grid_tau = np.load(
                f'{path_grids_Tex_tau}/{geometry}/tau_{name_mol}.npy')
            grid_Tex_line = grid_Tex[..., name_line_idx_]
            grid_tau_line = grid_tau[..., name_line_idx_]

            grids_Tex[name_mol_idx].append(grid_Tex_line)
            grids_tau[name_mol_idx].append(grid_tau_line)

    # cut the parameters space, following assumptions on abundance ratios
    FPS = np.load(f'{path_grids_Tex_tau}/{geometry}/FPS.npy',
                  allow_pickle=True)
    FPS = list(FPS)
    FPS_dimensions = len(FPS)
    FPS_dimensions_without_colden = len(FPS) - 1

    # number of dimensions to explore for each cloud layer (without take into account centroid velocities)
    dimensions_FPS = tuple([len(FPS[idx])
                            for idx in range(FPS_dimensions)])
    dimensions_FPS_without_colden = np.array([len(FPS[idx])
                                              for idx in [0, 1, 3]])
    space_length = np.prod(dimensions_FPS)

    # column density resolution
    colden_res = np.round(FPS[log10_N_idx][1] -
                          FPS[log10_N_idx][0], decimals=2)

    idx_min_max_colden = np.zeros((number_of_different_layers, 2), dtype=int)

    for layer_ in range(number_of_different_layers):
        # find the index
        layer = unique_layer_idx.index(layer_)

        if ('same_in_all_layers' in constraints_abundances) and (layer >= 1):
            idx_min_max_colden[layer_, 0] = idx_min_max_colden[0, 0]
            idx_min_max_colden[layer_, 1] = idx_min_max_colden[0, 1]
        else:
            idx_min_all_mol = 0
            idx_max_all_mol = 0
            for mol_idx, mol in enumerate(names_mol):
                if mol == 'h13cop' and '12co-13co' in constraints_abundances:
                    # 12co
                    min_idx_12co = np.round(
                        np.min(colden_ratios_shifts['12co'][layer])/colden_res, 1).astype(int)
                    max_idx_12co = np.round(
                        np.max(colden_ratios_shifts['12co'][layer])/colden_res, 1).astype(int)

                    # hcop
                    min_idx_hcop = np.round(
                        np.min(colden_ratios_shifts['hcop'][layer])/colden_res, 1).astype(int)
                    max_idx_hcop = np.round(
                        np.max(colden_ratios_shifts['hcop'][layer])/colden_res, 1).astype(int)

                    min_idx_molecule = min_idx_hcop - max_idx_12co
                    max_idx_molecule = max_idx_hcop - min_idx_12co
                else:
                    min_idx_molecule = np.round(
                        np.min(colden_ratios_shifts[mol][layer])/colden_res, 1).astype(int)
                    max_idx_molecule = np.round(
                        np.max(colden_ratios_shifts[mol][layer])/colden_res, 1).astype(int)

                idx_min_colden = min_idx_molecule
                idx_max_colden = max_idx_molecule

                idx_min_all_mol = min(idx_min_all_mol, idx_min_colden)
                idx_max_all_mol = max(idx_max_all_mol, idx_max_colden)

            # the idx_min_all_mol first values of the Tex, tau cubes yield to outliers
            idx_min_max_colden[layer_, 0] = abs(idx_min_all_mol)
            idx_min_max_colden[layer_, 1] = dimensions_FPS[log10_N_idx] - \
                1 - abs(idx_max_all_mol)

    # spread walkers in the FPS if it needed
    # same spreading for all theta component, except for centroid velocities

    # without Cv nor colden
    total_dim = (FPS_dimensions) * number_of_different_layers

    total_number_of_initial_walkers = walkers_per_step[0] * \
        iterations_per_step[0]

    initial_idx_walkers = np.zeros((total_dim, total_number_of_initial_walkers))
    initial_idx_walkers.fill(np.nan)

    # indexes of the initial theta without Cv or log10_N
    initial_idx_walkers_theta = np.zeros(
        (number_of_different_layers, FPS_dimensions - 1, total_number_of_initial_walkers), dtype=int)
    initial_idx_walkers_colden = np.zeros((number_of_different_layers, len(
        names_mol), total_number_of_initial_walkers), dtype=int)

    idx = [0, 1, 3]  # log10_T_kin, log10_nH2, FWHM
    for layer in range(number_of_different_layers):
        for unknown_idx in range(FPS_dimensions_without_colden):
            initial_idx_walkers_theta__ = np.random.randint(
                                                    low = 0, 
                                                    high = len(FPS[idx[unknown_idx]]),
                                                    size = total_number_of_initial_walkers)
            initial_idx_walkers_theta[layer, unknown_idx,
                                    :] = initial_idx_walkers_theta__

    # commun to all species
    initial_FWHM_walkers, initial_s_V_walkers = np.zeros(
        (total_number_of_initial_walkers, layers)), np.zeros((total_number_of_initial_walkers, layers))
    for layer in range(layers):
        if ('sandwich' in constraints_geometry):
            # in which sandwich are we ?
            idx_sandwich = (
                layer//number_of_layers_per_clump) % number_of_clumps
            idx_inner_layer = idxs_inner_layer[idx_sandwich]

            if layer > idx_inner_layer:
                # find the opposite layer
                shift = layer - idx_inner_layer
                idx_opposed_layer = idx_inner_layer - shift
                initial_FWHM_walkers[:,
                                    layer] = initial_FWHM_walkers[:, idx_opposed_layer]
                initial_s_V_walkers[:,
                                    layer] = initial_s_V_walkers[:, idx_opposed_layer]
            else:
                initial_FWHM_walkers[:, layer] = FPS[FWHM_idx][np.array(
                    initial_idx_walkers_theta[unique_layer_idx[layer], 2, :])]
                initial_s_V_walkers[:, layer] = from_FWHM_to_s_V(
                    initial_FWHM_walkers[:, layer])
        else:
            initial_FWHM_walkers[:, layer] = FPS[FWHM_idx][np.array(
                initial_idx_walkers_theta[unique_layer_idx[layer], 2, :])]
            initial_s_V_walkers[:, layer] = from_FWHM_to_s_V(
                initial_FWHM_walkers[:, layer])

    # colden
    # begin by the colden of reference
    idx_name_mol_ref = names_mol.index(name_mol_ref_)
    for layer in range(number_of_different_layers):
        initial_idx_walkers_colden__ = np.random.randint(
                                    low = idx_min_max_colden[layer, 0], 
                                    high = idx_min_max_colden[layer, 1] + 1,
                                    size = total_number_of_initial_walkers)
        initial_idx_walkers_colden[layer, idx_name_mol_ref,
                                :] = initial_idx_walkers_colden__

    # then, all the species. Derive the colden indexes from a fixed abundance shift
    initial_walkers_colden_ratios_shifts = np.zeros((number_of_different_layers, len(
        names_mol), total_number_of_initial_walkers), dtype=int)

    for layer in range(number_of_different_layers):
        # get the indexes of the molecular species of reference, already normalized
        initial_idx_walkers_colden_ref = initial_idx_walkers_colden[layer,
                                                                    idx_name_mol_ref, :]

        if 'same_in_all_layers' in constraints_abundances and (layer >= 1):
            # abundance shifts have been already chosen
            for mol_idx, mol in enumerate(names_mol):
                initial_idx_walkers_colden_ = initial_idx_walkers_colden_ref + \
                    initial_walkers_colden_ratios_shifts[0, mol_idx, :]
                initial_idx_walkers_colden[layer, mol_idx,
                                        :] = initial_idx_walkers_colden_
                initial_walkers_colden_ratios_shifts[layer, mol_idx,
                                                    :] = initial_walkers_colden_ratios_shifts[0, mol_idx, :]
        else:
            for mol_idx, mol in enumerate(names_mol):

                if mol == 'h13cop' and '12co-13co' in constraints_abundances:
                    # induce by the abundance of 12co, 13co and hcop
                    # 12co
                    initial_colden_ratios_shifts_12co = initial_walkers_colden_ratios_shifts[
                        layer, idx_12co, :]
                    # hcop
                    initial_colden_ratios_shifts_hcop = initial_walkers_colden_ratios_shifts[
                        layer, idx_hcop, :]
                    # h13cop
                    initial_colden_ratios_shifts_ = - initial_colden_ratios_shifts_12co + \
                        initial_colden_ratios_shifts_hcop
                    initial_walkers_colden_ratios_shifts[layer,
                                                        mol_idx, :] = initial_colden_ratios_shifts_

                else:
                    layer_ = unique_layer_idx.index(layer)

                    # unknown colden to estimate. Randomly peaking a shift
                    if len(colden_ratios_shifts[mol][layer_]) > 1:
                        initial_colden_ratios_shifts_ = np.random.choice(
                            colden_ratios_shifts[mol][layer_], size=(total_number_of_initial_walkers))
                        initial_colden_ratios_shifts_ = np.round(
                            initial_colden_ratios_shifts_/colden_res, 1).astype(int)
                        # save it in case of 'same_in_all_layer' in constraints_abundances
                        initial_walkers_colden_ratios_shifts[layer,
                                                            mol_idx, :] = initial_colden_ratios_shifts_

                    else:  # single value
                        initial_colden_ratios_shifts_ = np.round(
                            colden_ratios_shifts[mol][layer_]/colden_res, 1).astype(int) * np.ones(total_number_of_initial_walkers)
                        # save it in case of 'same_in_all_layer' in constraints_abundances
                        initial_walkers_colden_ratios_shifts[layer,
                                                            mol_idx, :] = initial_colden_ratios_shifts_

                initial_idx_walkers_colden_ = initial_idx_walkers_colden_ref + \
                    initial_colden_ratios_shifts_
                initial_idx_walkers_colden[layer, mol_idx,
                                        :] = initial_idx_walkers_colden_

    coordinates = np.nonzero(np.where(np.isnan(FoV), 0, 1))
    coordinates_i, coordinates_j = coordinates[0], coordinates[1]
    if SINGLE_PIXEL_ANALYSIS : 
        # repet the pixel as much is needed
        coordinates_i = np.repeat(coordinates_i, number_of_analysis)
        coordinates_j = np.repeat(coordinates_j, number_of_analysis)
    LoS = list(zip(*tuple([coordinates_i, coordinates_j])))
    number_of_LoS = len(LoS)

    if VERBOSE:
        print(
            f"\n{bcolors.HEADER} Starting {DATASET}'s analysis {bcolors.ENDC}({number_of_LoS} pixels)")

    # initialize array to save results
    if 'rw' in optimization:
        # maps of estimation results

        if not SINGLE_PIXEL_ANALYSIS : 
            # vector of unknowns, for each layer of the cloud
            maps_theta_rw = np.zeros(
                (np.shape(FoV)[0], np.shape(FoV)[1], len(theta), layers))
            maps_theta_rw.fill(np.nan)
            # column densities
            maps_log10_N_rw = np.empty(
                (np.shape(FoV)[0], np.shape(FoV)[1], len(names_mol), layers))
            # Tex and tau
            maps_Tex_rw = [[[np.empty(np.shape(FoV)) for l in range(layers)] for idx_line in range(
                len(names_line[idx_mol]))] for idx_mol in range(len(names_mol))]
            maps_tau_rw = [[[np.empty(np.shape(FoV)) for l in range(layers)] for idx_line in range(
                len(names_line[idx_mol]))] for idx_mol in range(len(names_mol))]

            # maps of information about the optimization

            # neg-likelihood NLL
            map_NLL_rw = np.empty(np.shape(FoV))
            map_NLL_rw.fill(np.nan)
            # number of walkers at the optimal NLL (useful to check on convergency)
            '''
            map_walkers_rw = np.empty(np.shape(FoV))
            map_walkers_rw.fill(np.nan)
            '''
        else : 
            # vector of unknowns, for each layer of the cloud
            maps_theta_rw = np.zeros(
                (1, len(LoS), len(theta), layers))
            maps_theta_rw.fill(np.nan)
            # column densities
            maps_log10_N_rw = np.empty(
                (1, len(LoS), len(names_mol), layers))
            # Tex and tau
            maps_Tex_rw = [[[np.empty((1, len(LoS))) for l in range(layers)] for idx_line in range(
                len(names_line[idx_mol]))] for idx_mol in range(len(names_mol))]
            maps_tau_rw = [[[np.empty((1, len(LoS))) for l in range(layers)] for idx_line in range(
                len(names_line[idx_mol]))] for idx_mol in range(len(names_mol))]

            # maps of information about the optimization

            # neg-likelihood NLL
            map_NLL_rw = np.empty((1, len(LoS)))
            map_NLL_rw.fill(np.nan)
            # number of walkers at the optimal NLL (useful to check on convergency)
            '''
            map_walkers_rw = np.empty((1, len(LoS)))
            map_walkers_rw.fill(np.nan)
            '''

    else:
        # maps of estimation results from the rw required to initialize the gradient
        maps_theta_rw = np.load(f'{folder_save}/maps_theta_rw.npy', allow_pickle=True)
        maps_Tex_rw = np.load(f'{folder_save}/maps_Tex_rw.npy', allow_pickle=True)
        maps_tau_rw = np.load(f'{folder_save}/maps_tau_rw.npy', allow_pickle=True)
        map_NLL_rw = np.load(f'{folder_save}/map_NLL_rw.npy', allow_pickle=True)
        maps_log10_N_rw = np.load(f'{folder_save}/maps_log10_N_rw.npy', allow_pickle=True)

    if 'gd' in optimization:

        if SINGLE_PIXEL_ANALYSIS : 
            # maps of estimation results

            # vector of unknowns, for each layer of the cloud
            maps_theta_gd = np.zeros(
                (1, len(LoS), len(theta), layers))
            maps_theta_gd.fill(np.nan)
            # column densities
            maps_log10_N_gd = np.empty(
                (1, len(LoS), len(names_mol), layers))
            # Tex and tau
            maps_Tex_gd = [[[np.empty((1, len(LoS))) for l in range(layers)] for idx_line in range(
                len(names_line[idx_mol]))] for idx_mol in range(len(names_mol))]
            maps_tau_gd = [[[np.empty((1, len(LoS))) for l in range(layers)] for idx_line in range(
                len(names_line[idx_mol]))] for idx_mol in range(len(names_mol))]

            # maps of information about the optimization

            # neg-likelihood NLL
            map_NLL_gd = np.empty((1, len(LoS)))
            map_NLL_gd.fill(np.nan)
            # number of iterations when the gd have stopped
            '''
            map_iterations_gd = np.empty((1, len(LoS)))
            map_iterations_gd.fill(np.nan)
            '''

        else : 
            # maps of estimation results

            # vector of unknowns, for each layer of the cloud
            maps_theta_gd = np.zeros(
                (np.shape(FoV)[0], np.shape(FoV)[1], len(theta), layers))
            maps_theta_gd.fill(np.nan)
            # column densities
            maps_log10_N_gd = np.empty(
                (np.shape(FoV)[0], np.shape(FoV)[1], len(names_mol), layers))
            # Tex and tau
            maps_Tex_gd = [[[np.empty(np.shape(FoV)) for l in range(layers)] for idx_line in range(
                len(names_line[idx_mol]))] for idx_mol in range(len(names_mol))]
            maps_tau_gd = [[[np.empty(np.shape(FoV)) for l in range(layers)] for idx_line in range(
                len(names_line[idx_mol]))] for idx_mol in range(len(names_mol))]

            # maps of information about the optimization

            # neg-likelihood NLL
            map_NLL_gd = np.empty(np.shape(FoV))
            map_NLL_gd.fill(np.nan)
            # number of iterations when the gd have stopped
            '''
            map_iterations_gd = np.empty(np.shape(FoV))
            map_iterations_gd.fill(np.nan)
            '''

    if WRITE_RESULTS_TXT_FILE:
        if 'rw' in optimization:

            with open(f'results_rw', "w") as result_file_rw:
                result_file_rw.write('')
                result_file_rw.close()

        if 'gd' in optimization:
            with open(f'results_gd', "w") as result_file_gd:
                result_file_gd.write('')
                result_file_gd.close()

    # model fitting processing
    inputs = {}

    inputs['FPS'] = FPS
    inputs['dimensions_FPS'] = dimensions_FPS
    inputs['dimensions_FPS_without_colden'] = dimensions_FPS_without_colden

    inputs['ppv'] = ppv
    inputs['maps_s_b'] = maps_s_b
    inputs['maps_s_c'] = maps_s_c
    inputs['number_of_C_V_components'] = number_of_C_V_components
    inputs['map_C_V'] = map_C_V
    if WINDOW_BANDWIDTH : 
        inputs['map_closest_V_to_C_V_idx'] = map_closest_V_to_C_V_idx
    inputs['velocity_res'] = velocity_res
    inputs['velocity_channels'] = velocity_channels
    inputs['rest_frequencies'] = rest_frequencies

    inputs['grids_Tex'] = grids_Tex
    inputs['grids_tau'] = grids_tau

    inputs['colden_ratios'] = colden_ratios
    inputs['initial_idx_walkers_theta'] = initial_idx_walkers_theta
    inputs['initial_idx_walkers_colden'] = initial_idx_walkers_colden
    inputs['initial_walkers_colden_ratios_shifts'] = initial_walkers_colden_ratios_shifts
    inputs['initial_s_V_walkers'] = initial_s_V_walkers
    inputs['idx_min_max_colden'] = idx_min_max_colden
    inputs['colden_res'] = colden_res
    inputs['total_number_of_initial_walkers'] = total_number_of_initial_walkers

    if optimization == 'gd' : 
        inputs['maps_theta_rw'] = maps_theta_rw
        inputs['maps_Tex_rw'] = maps_Tex_rw
        inputs['maps_tau_rw'] = maps_tau_rw
        inputs['map_NLL_rw'] = map_NLL_rw
        inputs['maps_log10_N_rw'] = maps_log10_N_rw

    if PLOT:
        inputs['names_mol_line_latex'] = names_mol_line_latex

    inputs['number_of_LoS'] = number_of_LoS # for progression bar

    if PARALLELISM:

        if WRITE_RESULTS_TXT_FILE:
            manager = Manager()
            if 'rw' in optimization:
                POLL_SIZE -= 1  # have to allocate one process to writing
                assert POLL_SIZE > 1, print(
                    f'{bcolors.FAIL}\n [error] Not enough processes to write results in .txt file...{bcolors.ENDC}')
                queue_rw = manager.Queue()
                p_rw = Process(target=write_results_txt_parallelism_rw, args=(queue_rw,))
                p_rw.start()

            if 'gd' in optimization:
                POLL_SIZE -= 1  # have to allocate one process to writing
                assert POLL_SIZE > 1, print(
                    f'{bcolors.FAIL}\n [error] Not enough processes to write results in .txt file...{bcolors.ENDC}')
                queue_gd = manager.Queue()
                p_gd = Process(target=write_results_txt_parallelism_gd, args=(queue_gd,))
                p_gd.start()

        if 'rw' in optimization:
            if 'gd' in optimization:
                zip_args = [
                    (
                        pixel_idx,
                        LoS[pixel_idx][0].item(),
                        LoS[pixel_idx][1].item(),
                        inputs,
                        queue_rw,
                        queue_gd
                    )
                    for pixel_idx in range(number_of_LoS)]
            else:
                zip_args = [
                    (
                        pixel_idx,
                        LoS[pixel_idx][0].item(),
                        LoS[pixel_idx][1].item(),
                        inputs,
                        queue_rw,
                    )
                    for pixel_idx in range(number_of_LoS)]
        else:
            zip_args = [
                (
                    pixel_idx,
                    LoS[pixel_idx][0].item(),
                    LoS[pixel_idx][1].item(),
                    inputs,
                    None,
                    queue_gd
                )
                for pixel_idx in range(number_of_LoS)]

        inputs['POLL_SIZE'] = POLL_SIZE # for progression bar

        with tqdm.tqdm(total=number_of_LoS, desc='pixels', leave = True, position = 0) as pbar_main:

            with Pool(processes=POLL_SIZE) as pool:

                for res_idx, results_pixel in enumerate(pool.imap_unordered(fit_pixel_wrapped, zip_args)):
                    
                    if not SINGLE_PIXEL_ANALYSIS : 
                        pixel = results_pixel['pixel']
                        row_idx, column_idx = pixel[0], pixel[1]
                    else : 
                        pixel_idx = results_pixel['pixel_idx']
                        row_idx, column_idx = 0, pixel_idx

                    if 'rw' in optimization:
                        res_rw = results_pixel['results_rw']
                        optimal_theta_rw, optimal_Tex_rw, optimal_tau_rw, NLL_rw, count_NLL, optimal_colden_rw = res_rw

                        # update estimation maps
                        map_NLL_rw[row_idx, column_idx] = NLL_rw
                        np.save(f'{folder_save}/map_NLL_rw.npy', map_NLL_rw)

                        # update estimation maps
                        '''
                        map_walkers_rw[row_idx, column_idx] = count_NLL
                        np.save(
                            f'{folder_save}/map_walkers_rw.npy', map_walkers_rw)
                        '''

                        maps_theta_rw[row_idx, column_idx,
                                    :, :] = optimal_theta_rw[:, :]
                        np.save(f'{folder_save}/maps_theta_rw.npy',
                                from_list_to_array(maps_theta_rw))

                        # update TAU and T_EX maps
                        for molecule_idx in range(len(names_mol)):
                            for line_idx in range(len(names_line[molecule_idx])):
                                for layer in range(layers):
                                    maps_Tex_rw[molecule_idx][line_idx][layer][row_idx,
                                                                            column_idx] = optimal_Tex_rw[molecule_idx][line_idx][layer]
                                    maps_tau_rw[molecule_idx][line_idx][layer][row_idx,
                                                                            column_idx] = optimal_tau_rw[molecule_idx][line_idx][layer]
                                    maps_log10_N_rw[row_idx, column_idx, molecule_idx,
                                                    layer] = optimal_colden_rw[molecule_idx, layer]

                        np.save(f'{folder_save}/maps_Tex_rw.npy',
                                from_list_to_array(maps_Tex_rw, inhomogeneous=True))
                        np.save(f'{folder_save}/maps_tau_rw.npy',
                                from_list_to_array(maps_tau_rw, inhomogeneous=True))
                        np.save(
                            f'{folder_save}/maps_log10_N_rw.npy', maps_log10_N_rw)

                    if 'gd' in optimization:
                        res_gd = results_pixel['results_gd']
                        optimal_theta_gd, optimal_Tex_gd, optimal_tau_gd, NLL_gd, iterations_NLL_gd, optimal_colden_gd = res_gd

                        # update estimation maps
                        map_NLL_gd[row_idx, column_idx] = NLL_gd
                        np.save(f'{folder_save}/map_NLL_gd.npy', map_NLL_gd)

                        '''
                        map_iterations_gd[row_idx, column_idx] = iterations_NLL_gd
                        np.save(f'{folder_save}/map_iterations_gd.npy', map_iterations_gd)
                        '''

                        maps_theta_gd[row_idx, column_idx,
                                    :, :] = optimal_theta_gd[:, :]
                        np.save(f'{folder_save}/maps_theta_gd.npy',
                                from_list_to_array(maps_theta_gd))

                        # update TAU and T_EX maps
                        for molecule_idx in range(len(names_mol)):
                            for line_idx in range(len(names_line[molecule_idx])):
                                for layer in range(layers):
                                    maps_Tex_gd[molecule_idx][line_idx][layer][row_idx,
                                                                            column_idx] = optimal_Tex_gd[molecule_idx][line_idx][layer]
                                    maps_tau_gd[molecule_idx][line_idx][layer][row_idx,
                                                                            column_idx] = optimal_tau_gd[molecule_idx][line_idx][layer]
                                    maps_log10_N_gd[row_idx, column_idx, molecule_idx,
                                                    layer] = optimal_colden_gd[molecule_idx, layer]

                        np.save(f'{folder_save}/maps_Tex_gd.npy',
                                from_list_to_array(maps_Tex_gd, inhomogeneous=True))
                        np.save(f'{folder_save}/maps_tau_gd.npy',
                                from_list_to_array(maps_tau_gd, inhomogeneous=True))
                        np.save(f'{folder_save}/maps_log10_N_gd.npy', maps_log10_N_gd)

                    pbar_main.update(1)
            
        if 'rw' in optimization:
            end = {}
            end['pixel'] = None
            queue_rw.put(end)
            p_rw.join()
            p_rw.close()

        if 'gd' in optimization:
            end = {}
            end['pixel'] = None
            queue_gd.put(end)
            p_gd.join()
            p_gd.close()

    else:

        with tqdm.tqdm(total=number_of_LoS, desc='pixels', leave = True, position = 0) as pbar_main:
            for pixel_idx in range(number_of_LoS):

                pixel = (LoS[pixel_idx][0].item(), LoS[pixel_idx][1].item())
                row_idx, column_idx = pixel[0], pixel[1]

                results_pixel = fit_pixel(
                    pixel_idx,
                    row_idx,
                    column_idx,
                    inputs=inputs)

                if SINGLE_PIXEL_ANALYSIS : 
                    row_idx, column_idx = 0, pixel_idx

                if 'rw' in optimization:
                    res_rw = results_pixel['results_rw']
                    optimal_theta_rw, optimal_Tex_rw, optimal_tau_rw, NLL_rw, count_NLL, optimal_colden_rw = res_rw

                    # update estimation maps
                    map_NLL_rw[row_idx, column_idx] = NLL_rw
                    np.save(f'{folder_save}/map_NLL_rw.npy', map_NLL_rw)

                    # update estimation maps
                    '''
                    map_walkers_rw[row_idx, column_idx] = count_NLL
                    np.save(
                        f'map_walkers_rw.npy', map_walkers_rw)
                    '''
                    maps_theta_rw[row_idx, column_idx,
                                    :, :] = optimal_theta_rw[:, :]
                    np.save(f'{folder_save}/maps_theta_rw.npy',
                            from_list_to_array(maps_theta_rw))

                    # update TAU and T_EX maps
                    for molecule_idx in range(len(names_mol)):
                        for line_idx in range(len(names_line[molecule_idx])):
                            for layer in range(layers):
                                maps_Tex_rw[molecule_idx][line_idx][layer][row_idx,
                                                                            column_idx] = optimal_Tex_rw[molecule_idx][line_idx][layer]
                                maps_tau_rw[molecule_idx][line_idx][layer][row_idx,
                                                                            column_idx] = optimal_tau_rw[molecule_idx][line_idx][layer]
                                maps_log10_N_rw[row_idx, column_idx, molecule_idx,
                                                layer] = optimal_colden_rw[molecule_idx, layer]

                    np.save(f'{folder_save}/maps_Tex_rw.npy',
                            from_list_to_array(maps_Tex_rw, inhomogeneous=True))
                    np.save(f'{folder_save}/maps_tau_rw.npy',
                            from_list_to_array(maps_tau_rw, inhomogeneous=True))
                    np.save(
                        f'{folder_save}/maps_log10_N_rw.npy', maps_log10_N_rw)

                if 'gd' in optimization:
                    res_gd = results_pixel['results_gd']
                    optimal_theta_gd, optimal_Tex_gd, optimal_tau_gd, NLL_gd, iterations_NLL_gd, optimal_colden_gd = res_gd

                    # update estimation maps
                    map_NLL_gd[row_idx, column_idx] = NLL_gd
                    np.save(f'{folder_save}/map_NLL_gd.npy', map_NLL_gd)
                    '''
                    map_iterations_gd[row_idx, column_idx] = iterations_NLL_gd
                    np.save(f'map_iterations_gd.npy', map_iterations_gd)
                    '''
                    maps_theta_gd[row_idx, column_idx,
                                    :, :] = optimal_theta_gd[:, :]
                    np.save(f'{folder_save}/maps_theta_gd.npy',
                            from_list_to_array(maps_theta_gd))

                    # update TAU and T_EX maps
                    for molecule_idx in range(len(names_mol)):
                        for line_idx in range(len(names_line[molecule_idx])):
                            for layer in range(layers):
                                maps_Tex_gd[molecule_idx][line_idx][layer][row_idx,
                                                                            column_idx] = optimal_Tex_gd[molecule_idx][line_idx][layer]
                                maps_tau_gd[molecule_idx][line_idx][layer][row_idx,
                                                                            column_idx] = optimal_tau_gd[molecule_idx][line_idx][layer]
                                maps_log10_N_gd[row_idx, column_idx, molecule_idx,
                                                layer] = optimal_colden_gd[molecule_idx, layer]

                    np.save(f'{folder_save}/maps_Tex_gd.npy',
                            from_list_to_array(maps_Tex_gd, inhomogeneous=True))
                    np.save(f'{folder_save}/maps_tau_gd.npy',
                            from_list_to_array(maps_tau_gd, inhomogeneous=True))
                    np.save(f'{folder_save}/maps_log10_N_gd.npy', maps_log10_N_gd)
        
                pbar_main.update(1)

    if VERBOSE:
        print(
            f"{bcolors.HEADER} {DATASET}'s analysis {bcolors.OKGREEN}done.{bcolors.ENDC}")
