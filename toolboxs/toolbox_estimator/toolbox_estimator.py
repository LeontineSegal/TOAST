# %% load modules
from toolboxs.toolbox_python.toolbox_python import check_type, check_length, check_shape, from_float_to_string, from_list_to_array
from toolboxs.toolbox_physics.toolbox_radiative_transfer import from_FWHM_to_s_V, compute_radiative_tranfer_equation
from toolboxs.toolbox_crb.toolbox_crb import compute_fim, inverse_fim
from toolboxs.toolbox_radex.toolbox_radex import get_freq_bandwidth, from_OPR_to_densities, execute_radex

from typing import Optional, Union, List, Tuple
from numbers import Real

import numpy as np 

import tqdm
import time

from model import *
idx_name_mol_ref = names_mol.index(name_mol_ref_)

FPS = np.load(f'{path_grids_Tex_tau}/{geometry}/FPS.npy', allow_pickle=True)
min_log10_T_kin, max_log10_T_kin = np.min(FPS[log10_T_kin_idx]).item(), np.max(FPS[log10_T_kin_idx]).item()
min_log10_nH2, max_log10_nH2 = np.min(FPS[log10_nH2_idx]).item(), np.max(FPS[log10_nH2_idx]).item()
min_log10_N_ref, max_log10_N_ref = np.min(FPS[log10_N_idx]).item(), np.max(FPS[log10_N_idx]).item()
min_FWHM, max_FWHM = np.min(FPS[FWHM_idx]).item(), np.max(FPS[FWHM_idx]).item()

# %% when WRITE_RESULTS_TXT_FILE

def write_idx_pixel_rw(
        file: str,
        i_j: tuple,
        pixel_idx: int,
        space: Optional[int] = 30):

    columnNames = ("Step", "Iter./Max Iter.",
                   "Walkers min(NLL)/tot.", "Min(NLL)", "Time")
    columnSizes = [len(e) for e in columnNames]
    separatorRow = "%s+%s+%s+%s+%s" % ("-" * (columnSizes[0] + 2),
                                       "-" * (columnSizes[1] + 2),
                                       "-" * (columnSizes[2] + 2),
                                       "-" * (columnSizes[3] + 2),
                                       "-" * (columnSizes[4] + 2)
                                       )
    lineFormat = " %%-%ds | %%-%ds | %%-%ds | %%-%ds | %%%ds " % \
        (columnSizes[0], columnSizes[1],
            columnSizes[2], columnSizes[3],
            columnSizes[4]
         )

    with open(f'{file}', "a") as result_file:
        result_file.write(f'{separatorRow}\n')
        result_file.write(f'{separatorRow}\n')
        result_file.write(f'Test {pixel_idx} : (i, j) = {i_j}\n')
        result_file.write(f'{separatorRow}\n')
        result_file.write(f'{lineFormat % columnNames}\n')
        result_file.write(f'{separatorRow}\n')

        spaces = '\n' * space
        result_file.write(f'{spaces}')

    return 1

def write_result_file_rw_1(
        file: str,
        parameters: List[str],
        pixel: tuple,
        pixel_idx: int,
        total_time: str,
        write_total_time: Optional[bool] = False
):
    """Write a file of results
    """
    # find the studied pixel
    read_file = open(file, 'r')
    lines = read_file.readlines()

    i_j = np.where(
        [f'Test {pixel_idx} : (i, j) = ({pixel[0]}, {pixel[1]})' in l for l in lines])[0].tolist()[0]
    blanck_lines = np.where([f'\n' == l[0]
                            for l in lines[i_j::]])[0].tolist()[0]
    parameters = tuple(parameters)
    columnNames = ("Step", "Iter./Max Iter.",
                   "Walkers min(NLL)/tot.", "Min(NLL)", "Time")
    columnSizes = [len(e) for e in columnNames]
    lineFormat = " %%-%ds | %%-%ds | %%-%ds | %%-%ds | %%%ds " % \
        (columnSizes[0], columnSizes[1],
            columnSizes[2], columnSizes[3],
            columnSizes[4]
         )

    new_line = f'{lineFormat % parameters}\n'
    lines[i_j + blanck_lines] = new_line

    if write_total_time:
        # the last row : total execution time per pixel
        parameters = tuple(['All', '', '', '', total_time])
        new_line = f'{lineFormat % parameters}\n'
        lines[i_j + blanck_lines + 1] = new_line

    read_file.close()

    with open(file, 'w') as result_file:
        result_file.writelines(lines)

    return 1

def write_result_file_rw_2(file: str,
                        parameters: List[str],
                        pixel: tuple,
                        pixel_idx: int
                        ):
    """Write a file of results
    """
    theta, theta_proportions = parameters[0], parameters[2]
    Tex, tau = parameters[3], parameters[4]
    params, layers = theta.shape[0], theta.shape[1]

    columnNames = ["param / Layer"]
    columnNames += [f'{layer+1}' for layer in range(layers)]
    columnNames = tuple(columnNames)
    columnSizes = [18 for e in columnNames]

    lineFormat = " %%-%ds "  # "param / Layer"
    separatorRow = "%s"  # "param / Layer"
    for layer in range(layers - 1):
        lineFormat += "| %%-%ds "
        separatorRow += "+%s"
    lineFormat += "| %%%ds "
    separatorRow += "+%s"

    lineFormat = lineFormat % tuple(
        [columnSizes[i] for i in range(len(columnSizes))])
    separatorRow = separatorRow % tuple(
        ["-" * (columnSizes[i] + 2) for i in range(len(columnSizes))])

    # find the studied pixel
    read_file = open(file, 'r')
    lines = read_file.readlines()
    read_file.close()

    i_j = np.where(
        [f'Test {pixel_idx} : (i, j) = ({pixel[0]}, {pixel[1]})' in l for l in lines])[0].tolist()[0]
    blanck_lines = np.where([f'\n' == l[0]
                            for l in lines[i_j::]])[0].tolist()[0]

    lines[i_j + blanck_lines] = f'{separatorRow}\n'
    lines[i_j + blanck_lines + 1] = f'{lineFormat % columnNames}\n'
    lines[i_j + blanck_lines + 2] = f'{separatorRow}\n'

    idx = i_j + blanck_lines + 3

    # theta without colden nor Tex, tau
    LineNames = ("log(Tkin / K)", "log(nH2 /cm-3)",
                 f"", "FWHM [km/s]", "CV [km/s]")

    for t_idx in range(params):

        if t_idx == 0:  # log(Tkin)
            theta_layer = [LineNames[t_idx]]
            for layer in range(layers):
                theta_ = f'log({from_float_to_string(10**theta[t_idx, layer], "float", 3, 3)})' + \
                    f' ({from_float_to_string(theta_proportions[t_idx, layer], "float", 2, 0)} %)'
                theta_layer.append(theta_)

            theta_layers = tuple(theta_layer)
            lines[idx] = f'{lineFormat % theta_layers}\n'
            idx += 1

            # Tex and tau
            lines[idx] = f" Tex [K], tau\n"
            idx += 1

            for mol_idx, mol in enumerate(names_mol):

                for l_idx, l in enumerate(names_line[mol_idx]):
                    if l_idx == 0:
                        theta_layer = [f'{mol} ({l})']
                        len_theta_layer = len(f'{mol}')

                    else:
                        theta_layer = [' '*len_theta_layer + f' ({l})']

                    # Tex and tau
                    for layer in range(layers):
                        Tex_ = Tex[mol_idx][l_idx][layer]
                        Tex_ = from_float_to_string(Tex_, 'float', 2, 2)
                        tau_ = tau[mol_idx][l_idx][layer]
                        tau_ = from_float_to_string(tau_, 'float', 2, 2)
                        theta_ = f'{Tex_}, {tau_}'
                        theta_layer.append(theta_)

                    theta_layers = tuple(theta_layer)
                    lines[idx] = f'{lineFormat % theta_layers}\n'
                    idx += 1

        elif t_idx == log10_N_idx:  # column density
            lines[idx] = f" log(N /cm-2)\n"
            idx += 1

            for mol_idx, mol in enumerate(names_mol):
                theta_layer = [f'{mol}']

                for layer in range(layers):
                    theta_ = from_float_to_string(
                        parameters[1][mol_idx, layer], 'float', 2, 2) + f' ({from_float_to_string(theta_proportions[t_idx, layer], "float", 2, 0)} %)'
                    theta_layer.append(theta_)

                theta_layers = tuple(theta_layer)
                lines[idx] = f'{lineFormat % theta_layers}\n'
                idx += 1

        else:
            theta_layer = [LineNames[t_idx]]
            for layer in range(layers):
                theta_ = from_float_to_string(
                    theta[t_idx, layer], 'float', 3, 3) + f' ({from_float_to_string(theta_proportions[t_idx, layer], "float", 2, 0)} %)'
                theta_layer.append(theta_)

            theta_layers = tuple(theta_layer)
            lines[idx] = f'{lineFormat % theta_layers}\n'
            idx += 1

    with open(file, 'w') as result_file:
        result_file.writelines(lines)

    return 1

def write_idx_pixel_gd(
        file: str,
        i_j: tuple,
        pixel_idx: int,
        space: Optional[int] = 30):

    columnNames = ("Min(NLL)", "Iter./Max Iter.", "Time")
    columnSizes = [len(e) for e in columnNames]
    separatorRow = "%s+%s+%s" % ("-" * (columnSizes[0] + 2),
                                       "-" * (columnSizes[1] + 2),
                                       "-" * (columnSizes[2] + 2)
                                       )
    lineFormat = " %%-%ds | %%-%ds | %%%ds " % \
        (columnSizes[0], columnSizes[1],
            columnSizes[2]
         )

    with open(f'{file}', "a") as result_file:
        result_file.write(f'{separatorRow}\n')
        result_file.write(f'{separatorRow}\n')
        result_file.write(f'Test {pixel_idx} : (i, j) = {i_j}\n')
        result_file.write(f'{separatorRow}\n')
        result_file.write(f'{lineFormat % columnNames}\n')
        result_file.write(f'{separatorRow}\n')

        spaces = '\n' * space
        result_file.write(f'{spaces}')

    return 1

def write_result_file_gd_1(
        file: str,
        parameters: List[str],
        pixel: tuple,
        pixel_idx: int
):
    """Write a file of results
    """
    # find the studied pixel
    read_file = open(file, 'r')
    lines = read_file.readlines()

    i_j = np.where(
        [f'Test {pixel_idx} : (i, j) = ({pixel[0]}, {pixel[1]})' in l for l in lines])[0].tolist()[0]
    blanck_lines = np.where([f'\n' == l[0]
                            for l in lines[i_j::]])[0].tolist()[0]
    parameters = tuple(parameters)
    columnNames = ("Min(NLL)", "Iter./Max Iter.", "Time")
    columnSizes = [len(e) for e in columnNames]
    lineFormat = " %%-%ds | %%-%ds | %%%ds " % \
        (columnSizes[0], columnSizes[1],
            columnSizes[2]
         )

    new_line = f'{lineFormat % parameters}\n'
    lines[i_j + blanck_lines] = new_line

    read_file.close()

    with open(file, 'w') as result_file:
        result_file.writelines(lines)

    return 1

def write_result_file_gd_2(file: str,
                        parameters: List[str],
                        pixel: tuple,
                        pixel_idx: int
                        ):
    """Write a file of results
    """
    theta = parameters[0]
    Tex, tau = parameters[2], parameters[3]
    params, layers = theta.shape[0], theta.shape[1]

    columnNames = ["param / Layer"]
    columnNames += [f'{layer+1}' for layer in range(layers)]
    columnNames = tuple(columnNames)
    columnSizes = [18 for e in columnNames]

    lineFormat = " %%-%ds "  # "param / Layer"
    separatorRow = "%s"  # "param / Layer"
    for layer in range(layers - 1):
        lineFormat += "| %%-%ds "
        separatorRow += "+%s"
    lineFormat += "| %%%ds "
    separatorRow += "+%s"

    lineFormat = lineFormat % tuple(
        [columnSizes[i] for i in range(len(columnSizes))])
    separatorRow = separatorRow % tuple(
        ["-" * (columnSizes[i] + 2) for i in range(len(columnSizes))])

    # find the studied pixel
    read_file = open(file, 'r')
    lines = read_file.readlines()
    read_file.close()

    i_j = np.where(
        [f'Test {pixel_idx} : (i, j) = ({pixel[0]}, {pixel[1]})' in l for l in lines])[0].tolist()[0]
    blanck_lines = np.where([f'\n' == l[0]
                            for l in lines[i_j::]])[0].tolist()[0]

    lines[i_j + blanck_lines] = f'{separatorRow}\n'
    lines[i_j + blanck_lines + 1] = f'{lineFormat % columnNames}\n'
    lines[i_j + blanck_lines + 2] = f'{separatorRow}\n'

    idx = i_j + blanck_lines + 3

    # theta without colden nor Tex, tau
    LineNames = ("log(Tkin / K)", "log(nH2 /cm-3)",
                 f"", "FWHM [km/s]", "CV [km/s]")

    for t_idx in range(params):

        if t_idx == 0:  # log(Tkin)
            theta_layer = [LineNames[t_idx]]
            for layer in range(layers):
                theta_ = f'log({from_float_to_string(10**theta[t_idx, layer], "float", 3, 3)})'
                theta_layer.append(theta_)

            theta_layers = tuple(theta_layer)
            lines[idx] = f'{lineFormat % theta_layers}\n'
            idx += 1

            # Tex and tau
            lines[idx] = f" Tex [K], tau\n"
            idx += 1

            for mol_idx, mol in enumerate(names_mol):

                for l_idx, l in enumerate(names_line[mol_idx]):
                    if l_idx == 0:
                        theta_layer = [f'{mol} ({l})']
                        len_theta_layer = len(f'{mol}')

                    else:
                        theta_layer = [' '*len_theta_layer + f' ({l})']

                    # Tex and tau
                    for layer in range(layers):
                        Tex_ = Tex[mol_idx][l_idx][layer]
                        Tex_ = from_float_to_string(Tex_, 'float', 2, 2)
                        tau_ = tau[mol_idx][l_idx][layer]
                        tau_ = from_float_to_string(tau_, 'float', 2, 2)
                        theta_ = f'{Tex_}, {tau_}'
                        theta_layer.append(theta_)

                    theta_layers = tuple(theta_layer)
                    lines[idx] = f'{lineFormat % theta_layers}\n'
                    idx += 1

        elif t_idx == log10_N_idx:  # column density
            lines[idx] = f" log(N /cm-2)\n"
            idx += 1

            for mol_idx, mol in enumerate(names_mol):
                theta_layer = [f'{mol}']

                for layer in range(layers):
                    theta_ = from_float_to_string(
                        parameters[1][mol_idx, layer], 'float', 2, 2) 
                    theta_layer.append(theta_)

                theta_layers = tuple(theta_layer)
                lines[idx] = f'{lineFormat % theta_layers}\n'
                idx += 1

        else:
            theta_layer = [LineNames[t_idx]]
            for layer in range(layers):
                theta_ = from_float_to_string(
                    theta[t_idx, layer], 'float', 3, 3)
                theta_layer.append(theta_)

            theta_layers = tuple(theta_layer)
            lines[idx] = f'{lineFormat % theta_layers}\n'
            idx += 1

    with open(file, 'w') as result_file:
        result_file.writelines(lines)

# %% optimization 

def compute_criterion(
        x: np.ndarray,
        s: np.ndarray,
        s_b: float,
        s_c: Optional[Real] = 0.,
        c_0: Optional[Real] = 1.
) -> Union[Real, np.ndarray[np.float64]]:
    """Compute the criterion, the Negative Log-Likelihood (NLL) of a single emission line.
    The column axis of all spectra corresponds to velocity channels. The row axis corresponds to realizations.
    For instance, the shape of the observed spectrum x is (1, 121). 
    The shape of the simulated spectra s can be (10, 121), where 10 corresponds to 10 realizations (useful in Monte-Carlo experiments and in the random walks).

    :param x: observed spectrum  
    :type x: np.ndarray
    :param s: estimated spectrum, reconstructed with the radiative transfer equation
    :type s: np.ndarray
    :param s_b: (K) thermal noise dispersion 
    :type s_b: float
    :param s_c: calibration noise dispersion, defaults to 0
    :type s_c: Optional[Real], optional
    :param c_0: calibration noise mean, defaults to 1
    :type c_0: Optional[Real], optional
    :return: Negative Log-Likelihood (NLL)
    :rtype: Union[float, np.ndarray]
    """
    if DEBUG:
        assert check_type(x, type_of_reference=np.ndarray,
                          from_function='compute_criterion')
        assert check_type(s, type_of_reference=np.ndarray,
                          from_function='compute_criterion')
        assert check_type(s_b, type_of_reference=Real,
                          from_function='compute_criterion')
        assert check_type(s_c, type_of_reference=Real,
                          from_function='compute_criterion')

        assert check_length(x, length_of_reference=1,
                            from_function='compute_criterion')
        assert check_shape(s, shape_of_reference=(np.shape(s)[0], x.size),
                           from_function='compute_criterion')

    N = x.size  # number of velocity channels

    # speed version
    s_c_2 = s_c**2
    s_b_2 = s_b**2
    s_2 = s**2
    x_2 = x**2
    sum_s_2 = np.sum(s_2, axis=-1)
    sum_x_2 = np.sum(x_2, axis=-1) 
    x_s_T = x @ s.T

    s_cb_2 = s_c_2 * sum_s_2 + s_b_2

    criterion_1 = 1/s_b_2 * sum_x_2
    criterion_2 = np.squeeze((c_0 ** 2)/s_cb_2 * sum_s_2)
    criterion_3 = np.squeeze((s_c_2 / (s_b_2 * s_cb_2)) * (x_s_T**2))
    criterion_4 = np.squeeze(((2 * c_0) / s_cb_2) * (x_s_T))

    NLL_1 = np.log(s_cb_2)
    NLL_2 = N * np.log(2*np.pi)
    NLL_3 = (N - 1) * np.log(s_b_2)
    NLL_4 = (criterion_1 + criterion_2 - criterion_3 - criterion_4)

    NLL = 0.5 * (NLL_1 + NLL_2 + NLL_3 + NLL_4)

    return NLL

# %% random walk

def random_walk(
        FPS: List[np.ndarray],
        dimensions_FPS: tuple[int],
        dimensions_FPS_without_colden: np.ndarray[int],
        x: List[List[np.ndarray]],  # measures
        freqs: List[List[np.ndarray]],
        s_b: List[List[Real]],
        s_c: List[List[Real]],
        velocity_res: List[List[np.ndarray]],
        velocity_channels: List[List[np.ndarray]],
        grids_Tex: List[List[np.ndarray]],
        grids_tau: List[List[np.ndarray]], 
        colden_ratios: dict,
        initial_idx_walkers_theta: np.ndarray, # shape (number_of_different_layers, FPS_dimension -1, walkers)
        initial_idx_walkers_colden: np.ndarray, # shape (number_of_different_layers, len(names_mol), walkers)
        initial_walkers_colden_ratios_shifts: np.ndarray, # shape (number_of_different_layers, len(names_mol), walkers)
        initial_walkers_s_V: np.ndarray,  # shape (walkers, layers)
        initial_walkers_C_V: np.ndarray, # initial centroid velocity. shape (walkers, layers)
        idx_min_max_colden: np.ndarray, # shape (number_of_different_layers, 2)
        colden_res: Real,
        walkers_per_step: Optional[List[int]] = walkers_per_step,
        iterations_per_step: Optional[List[int]] = iterations_per_step,
        reference_velocity_resolution: Optional[Real] = 0.05, #0.1
) -> tuple[np.ndarray, List, List, Real]:

    if VERBOSE:
        print(
            f'\t[is running] Grid search')

    if WRITE_RESULTS_TXT_FILE:
        params_convergence = []

    # number of dimensions to explore for each cloud layer (without take into account centroid velocities nor column densities)
    len_FPS_without_colden = len(dimensions_FPS) - 1

    ##################
    # initialization #
    ##################
    tic = time.perf_counter()

    step = 0
    walkers = walkers_per_step[step]
    iterations = iterations_per_step[step]

    # compute the initial NLL
    initial_NLL = np.zeros((iterations, walkers), dtype=Real)

    # to show the progression bar
    total_iterations = sum(iterations_per_step)
    if not PARALLELISM : 
        pbar_rw = tqdm.tqdm(total = total_iterations, position=0, leave=True)
    for iteration in range(iterations):

        idx_start, idx_end = iteration * walkers, (iteration + 1) * walkers

        for mol_idx in range(len(names_mol)):
            for line_idx in range(len(names_line[mol_idx])):
                freq = freqs[mol_idx][line_idx]
                if DEBUG:
                    assert check_type(freq, type_of_reference=float)
                velocity_resolution = velocity_res[mol_idx][line_idx]
                velocity_channels_line = velocity_channels[mol_idx][line_idx]

                grid_Tex, grid_tau = grids_Tex[mol_idx][
                    line_idx], grids_tau[mol_idx][line_idx]

                initial_Tex_walkers, initial_tau_walkers = np.zeros(
                    (walkers, layers)), np.zeros((walkers, layers))
                
                for layer in range(layers):
                    if ('sandwich' in constraints_geometry): 
                        # in which sandwich are we ?
                        idx_sandwich = (layer//number_of_layers_per_clump) % number_of_clumps
                        idx_inner_layer = idxs_inner_layer[idx_sandwich]

                        if layer > idx_inner_layer : 
                            # find the opposite layer
                            shift = layer - idx_inner_layer
                            idx_opposed_layer = idx_inner_layer - shift
                            initial_Tex_walkers[:,
                                                layer] = initial_Tex_walkers[:, idx_opposed_layer]
                            initial_tau_walkers[:,
                                                layer] = initial_tau_walkers[:, idx_opposed_layer]
                        else : 
                            initial_idx_walkers_log10_T_kin = np.array(
                            initial_idx_walkers_theta[unique_layer_idx[layer], 0, idx_start:idx_end])
                            initial_idx_walkers_log10_nH2 = np.array(
                                initial_idx_walkers_theta[unique_layer_idx[layer], 1, idx_start: idx_end])
                            initial_idx_walkers_FWHM = np.array(
                                initial_idx_walkers_theta[unique_layer_idx[layer], 2, idx_start: idx_end])

                            initial_idx_walkers_log10_N = np.array(
                                initial_idx_walkers_colden[unique_layer_idx[layer], mol_idx, idx_start:idx_end])

                            initial_Tex_walkers[:, layer] = grid_Tex[initial_idx_walkers_log10_T_kin,
                                                                    initial_idx_walkers_log10_nH2,
                                                                    initial_idx_walkers_log10_N,
                                                                    initial_idx_walkers_FWHM]
                            initial_tau_walkers[:, layer] = grid_tau[initial_idx_walkers_log10_T_kin,
                                                                    initial_idx_walkers_log10_nH2,
                                                                    initial_idx_walkers_log10_N,
                                                                    initial_idx_walkers_FWHM]

                    else:
                        initial_idx_walkers_log10_T_kin = np.array(
                        initial_idx_walkers_theta[unique_layer_idx[layer], 0, idx_start:idx_end])
                        initial_idx_walkers_log10_nH2 = np.array(
                            initial_idx_walkers_theta[unique_layer_idx[layer], 1, idx_start:idx_end])
                        initial_idx_walkers_FWHM = np.array(
                            initial_idx_walkers_theta[unique_layer_idx[layer], 2, idx_start:idx_end])

                        initial_idx_walkers_log10_N = np.array(
                            initial_idx_walkers_colden[unique_layer_idx[layer], mol_idx, idx_start:idx_end])

                        initial_Tex_walkers[:, layer] = grid_Tex[initial_idx_walkers_log10_T_kin,
                                                                initial_idx_walkers_log10_nH2,
                                                                initial_idx_walkers_log10_N,
                                                                initial_idx_walkers_FWHM]
                        initial_tau_walkers[:, layer] = grid_tau[initial_idx_walkers_log10_T_kin,
                                                                initial_idx_walkers_log10_nH2,
                                                                initial_idx_walkers_log10_N,
                                                                initial_idx_walkers_FWHM]

                initial_walkers_s_V_ = initial_walkers_s_V[idx_start:idx_end, :]
                initial_walkers_s_V_ = initial_walkers_s_V_.reshape((walkers, layers))

                initial_walkers_C_V_ = initial_walkers_C_V[idx_start:idx_end, :]
                initial_walkers_C_V_ = initial_walkers_C_V_.reshape(
                    (walkers, layers))

                if ('sandwich' in constraints_geometry) and ('mirror' in constraints_kinematics):
                    unique_layer_idx_ = np.arange(0, layers, 1)
                else:
                    unique_layer_idx_ = unique_layer_idx
            
                if velocity_resolution == 0.1 or velocity_resolution == 0.5: 
                    reference_velocity_resolution_ = 0.1
                elif velocity_resolution == 0.25 : 
                    reference_velocity_resolution_ = 0.125

                # simulated spectrum (initial solution)
                s_line = compute_radiative_tranfer_equation(
                    initial_Tex_walkers,
                    initial_tau_walkers,
                    freq,
                    velocity_channels_line,
                    velocity_resolution,
                    initial_walkers_s_V_,
                    initial_walkers_C_V_,
                    unique_layer_idx=unique_layer_idx_,
                    decomposition=False,
                    conserved_flux = True,
                    reference_velocity_resolution=reference_velocity_resolution_,
                    DEBUG=DEBUG, 
                    number_of_C_V_components = number_of_clumps, 
                    number_of_layers_per_clump = number_of_layers_per_clump,
                    peak_only = (PEAK_ONLY and f'{names_mol[mol_idx]}({names_line[mol_idx][line_idx]})' in thick_lines) 
                    )[0]
                
                # measure
                x_line = x[mol_idx][line_idx]
                # noise
                s_b_line = s_b[mol_idx][line_idx]
                s_c_line = s_c[mol_idx][line_idx]
                # criterion
                initial_NLL_ = compute_criterion(
                    x_line, s_line, s_b_line, s_c_line)
                initial_NLL[iteration, :] += initial_NLL_
        if not PARALLELISM:
            pbar_rw.update(1)

    # sort walkers in ascending order
    NLL_sorted, idx_NLL_sorted, count_NLL_sorted = np.unique(
        initial_NLL, return_index=True, return_counts=True)  # flattened array of size iterations x walkers

    walkers_next_step = len(NLL_sorted)  # redondance may decrease the size
    walkers_next_step = min(walkers_per_step[step+1], walkers_next_step)

    initial_NLL, idx_initial_NLL, count_initial_NLL = np.copy(NLL_sorted[:walkers_next_step]), np.copy(
        idx_NLL_sorted[:walkers_next_step]), np.copy(count_NLL_sorted[:walkers_next_step])

    # peak the best walkers
    idx_walkers_theta_ = np.zeros(
        (number_of_different_layers, len_FPS_without_colden, walkers_next_step), dtype=int)
    idx_walkers_colden_ = np.zeros((number_of_different_layers, len(
        names_mol), walkers_next_step), dtype=int)
    walkers_colden_ratios_shifts_ = np.zeros((number_of_different_layers, len(
        names_mol), walkers_next_step), dtype=int)
    walkers_C_V_ = np.zeros((walkers_next_step, layers))

    for layer in range(layers):
        # theta without colden
        for dim_idx in range(len_FPS_without_colden):
            ravel_idx_walkers_theta_layer = np.ravel(
                initial_idx_walkers_theta[unique_layer_idx[layer], dim_idx, :])[idx_initial_NLL]
            idx_walkers_theta_[unique_layer_idx[layer], dim_idx, :] = ravel_idx_walkers_theta_layer[:]
        # colden
        for mol_idx in range(len(names_mol)):
            ravel_idx_walkers_colden_layer = np.ravel(
                initial_idx_walkers_colden[unique_layer_idx[layer], mol_idx, :])[idx_initial_NLL]
            idx_walkers_colden_[unique_layer_idx[layer], mol_idx, :] = ravel_idx_walkers_colden_layer[:]
            # colden shifts
            ravel_walkers_colden_ratios_shifts = np.ravel(
                initial_walkers_colden_ratios_shifts[unique_layer_idx[layer], mol_idx, :])[idx_initial_NLL]
            walkers_colden_ratios_shifts_[unique_layer_idx[layer], mol_idx, :] = ravel_walkers_colden_ratios_shifts[:]
        # centroid velocity
        ravel_walkers_C_V_layer = np.ravel(initial_walkers_C_V[:, layer])[
            idx_initial_NLL]
        walkers_C_V_[:, layer] = ravel_walkers_C_V_layer[:]

    ### ### ### ### ### ### ### ####
    # let free memory numpy arrays #
    ### ### ### ### ### ### ### ####
    del ravel_idx_walkers_theta_layer
    del ravel_idx_walkers_colden_layer
    del ravel_walkers_colden_ratios_shifts
    del ravel_walkers_C_V_layer
    ### ### ### ### ### ### ### ####
    ### ### ### ### ### ### ### ####

    initial_idx_walkers_theta = idx_walkers_theta_
    initial_idx_walkers_colden = idx_walkers_colden_
    initial_walkers_colden_ratios_shifts = walkers_colden_ratios_shifts_
    initial_walkers_C_V = walkers_C_V_

    toc = time.perf_counter()

    if WRITE_RESULTS_TXT_FILE:
        params_convergence.append([step, f'{iteration + 1}/{iterations}', walkers, count_initial_NLL[0],
                                   from_float_to_string(initial_NLL[0], width_precision=2, digit_precision=2), f'{toc - tic:0.4f}'])

    ### ### ### ### ### ### ### ####
    # let free memory numpy arrays #
    ### ### ### ### ### ### ### ####
    del idx_walkers_theta_
    del idx_walkers_colden_
    del walkers_colden_ratios_shifts_
    del walkers_C_V_
    ### ### ### ### ### ### ### ####
    ### ### ### ### ### ### ### ####

    # before exploration
    NLL, idx_NLL = initial_NLL, idx_initial_NLL
    NLL, idx_NLL = np.unique(NLL, return_index=True)  # to get the proper index

    idx_walkers_theta = initial_idx_walkers_theta
    idx_walkers_colden = initial_idx_walkers_colden
    walkers_colden_ratios_shifts = initial_walkers_colden_ratios_shifts
    walkers_C_V = initial_walkers_C_V

    ########################
    # steps of exploration #
    ########################

    for step in range(1, len(walkers_per_step)):

        tic = time.perf_counter()

        walkers = min(walkers_per_step[step], walkers_next_step)
        idx_walkers = np.arange(walkers, dtype=int)
        iterations = iterations_per_step[step]

        # keep only the lowest NLL values of the previous step
        NLL, idx_NLL = NLL[:walkers], idx_NLL[:walkers]

        idx_walkers_theta = idx_walkers_theta[:, :, idx_NLL]
        idx_walkers_colden = idx_walkers_colden[:, :, idx_NLL]
        walkers_C_V = walkers_C_V[idx_NLL, :]
        walkers_colden_ratios_shifts = walkers_colden_ratios_shifts[:, :, idx_NLL]
        
        for explo_idx in range(1, iterations + 1):

            ############################################
            # theta without coldens: (Tkin, nH2, FWHM) #
            ############################################
            idx_walkers_theta_ = np.copy(idx_walkers_theta)

            # which theta to change ? (only one among all unknowns, among all layers)
            idx_walkers_theta_explo = np.random.choice(np.arange(
                len_FPS_without_colden * number_of_different_layers,  dtype=int), size=walkers)
            idx_walkers_theta_explo_ur = np.unravel_index(idx_walkers_theta_explo, shape=(
                number_of_different_layers, len_FPS_without_colden))
            idx_layers, idx_theta = idx_walkers_theta_explo_ur[0], idx_walkers_theta_explo_ur[1]
            # decrease (-1) | not moving (0) | increase (+1)
            direction = np.random.choice([-1, 0, 1], size=walkers)
            idx_walkers_theta_[idx_layers, idx_theta, idx_walkers] += direction

            # dealing with outliers (Radex grids)...
            positive_outliers = idx_walkers_theta_[
                idx_layers, idx_theta, idx_walkers] >= dimensions_FPS_without_colden[idx_theta]
            negative_outliers = idx_walkers_theta_[
                idx_layers, idx_theta, idx_walkers] < 0
            idx_outliers_walkers = np.nonzero(
                np.logical_or(positive_outliers, negative_outliers))
            # ...by moving in the opposite direction
            idx_walkers_theta_[idx_layers[idx_outliers_walkers], idx_theta[idx_outliers_walkers],
                            idx_walkers[idx_outliers_walkers]] -= 2 * direction[idx_outliers_walkers]

            #####################
            # centroid velocity #
            #####################
            walkers_C_V_ = np.copy(walkers_C_V)

            # which C_V to change ? (only one among all unknown C_V)
            idx_layers = np.random.choice(
                np.arange(number_of_unknown_C_V,  dtype=int), size=walkers)
            # decrease (-1) | not moving (0) | increase (+1)
            direction = np.random.choice([-1, 0, 1], size=walkers)
            walkers_C_V_[idx_walkers, idx_layers] = np.round(
                (walkers_C_V_[idx_walkers, idx_layers] + C_V_res * direction), 2)

            if 'same_C_V_in_all_layers' in constraints_kinematics:
                # copy for all the layers the updated values of C_V
                for layer in range(layers):
                    walkers_C_V_[:, layer] = walkers_C_V_[:, 0]
            else : 
                if 'sandwich' in constraints_geometry : 
                    for idx_sandwich in range(number_of_clumps):
                        idx_inner_layer = idxs_inner_layer[idx_sandwich]
                        for layer in range(idx_inner_layer + 1, number_of_layers_per_clump):
                            # find the opposite layer
                            shift = layer - idx_inner_layer
                            idx_opposed_layer = idx_inner_layer - shift
                            if 'mirror' in constraints_kinematics:
                                walkers_C_V_[:, layer] = 2 * walkers_C_V_[:,
                                                                    idx_inner_layer] - walkers_C_V_[:, idx_opposed_layer]
                            else:  # basic sandwich model
                                walkers_C_V_[:, layer] = walkers_C_V_[
                                    :, idx_opposed_layer]

            #######################################
            # colden of the molecule of reference #
            #######################################
            idx_walkers_colden_ = np.copy(idx_walkers_colden)

            # change the colden of the molecule of reference (only one among all layers)
            idx_layers = np.random.choice(
                np.arange(number_of_different_layers,  dtype=int), size=walkers)
            # decrease (-1) | not moving (0) | increase (+1)
            direction = np.random.choice([-1, 0, 1], size=walkers)
            idx_walkers_colden_[idx_layers,
                                idx_name_mol_ref, idx_walkers] += direction

            # dealing with outliers (Radex grids)...
            positive_outliers = idx_walkers_colden_[
                idx_layers, idx_name_mol_ref, idx_walkers] >= idx_min_max_colden[idx_layers, 1]
            negative_outliers = idx_walkers_colden_[
                idx_layers, idx_name_mol_ref, idx_walkers] < idx_min_max_colden[idx_layers, 0]
            idx_outliers = np.nonzero(np.logical_or(
                positive_outliers, negative_outliers))[0]
            # ...by moving in the opposite direction
            idx_walkers_colden_[idx_layers[idx_outliers], idx_name_mol_ref,
                                idx_walkers[idx_outliers]] -= 2 * direction[idx_outliers]

            ####################
            # abundance ratios #
            ####################
            walkers_colden_ratios_shifts_ = np.copy(walkers_colden_ratios_shifts) # shape (number_of_different_layers, len(names_mol), walkers)

            # estime abundance ratios...
            if len(idx_unknown_mol_layer) > 0:
                # which species and layer to change ?
                idx_walkers_ab_explo = np.random.choice(
                    idx_unknown_mol_layer, size=walkers)
                idx_walkers_ab_explo_ur = np.unravel_index(
                    idx_walkers_ab_explo, shape=(number_of_different_layers, len(names_mol)))
                idx_layers, idx_mol = idx_walkers_ab_explo_ur[0], idx_walkers_ab_explo_ur[1]

                idx_mol_unique = np.unique(idx_mol)
                for mol_idx in idx_mol_unique:
                    idx = np.argwhere(idx_mol == mol_idx)
                    idx_layers_unique = np.unique(idx_layers[idx])
                    for layer in idx_layers_unique:
                        # get the indexes of walkers to change
                        idx = np.argwhere(np.logical_and(
                            idx_mol == mol_idx, idx_layers == layer))  # of size <= walkers
                        
                        # get the range of abondance shifts to explore
                        layer_ = unique_layer_idx.index(layer)
                        #colden_ratios_shifts_mol_layer = colden_ratios_shifts[names_mol[mol_idx]
                        #                                            ][layer]
                        colden_ratios_shifts_mol_layer = colden_ratios_shifts[names_mol[mol_idx]
                                                                    ][layer_]

                        # convert it to indexes
                        colden_ratios_shifts_mol_layer = np.round(colden_ratios_shifts_mol_layer/colden_res, 1).astype(int)

                        # get the current colden idx for all walkers 
                        walkers_colden_ratios_shifts_mol_layer = walkers_colden_ratios_shifts_[layer, mol_idx, idx]
                        # find the corresponding shift for all walkers
                        idx_walkers_ab_shift = np.searchsorted(colden_ratios_shifts_mol_layer, walkers_colden_ratios_shifts_mol_layer)
                        
                        # decrease (-1) | not moving (0) | increase (+1)
                        direction = np.random.choice([-1, 0, 1], size=idx_walkers_ab_shift.shape)
                        idx_walkers_ab_shift += direction
                        
                        # dealing with outliers...
                        positive_outliers = idx_walkers_ab_shift >= len(colden_ratios_shifts_mol_layer)
                        negative_outliers = idx_walkers_ab_shift < 0
                        idx_outliers = np.nonzero(np.logical_or(
                            positive_outliers, negative_outliers))[0]
                        # ...by moving in the opposite direction
                        idx_walkers_ab_shift[idx_outliers, 0] -= 2 * direction[idx_outliers, 0]
                        
                        # update the colden ratio shifts...
                        walkers_colden_ratios_shifts_[layer, mol_idx, idx] = colden_ratios_shifts_mol_layer[idx_walkers_ab_shift]

                    if ('same_in_all_layers' in constraints_abundances) : 
                        for layer in range(number_of_different_layers):
                            walkers_colden_ratios_shifts_[layer, mol_idx, :] = walkers_colden_ratios_shifts_[0, mol_idx, :]
            else : 
                idx_mol = []

            # ...then derive the coldens from that one of the molecule of reference
            counter = 0
            # save it in case of 'same_in_all_layers' in constraints_abundances
            colden_ratios_shifts_buffer = np.zeros((number_of_different_layers, len(names_mol), walkers))

            for layer_ in range(number_of_different_layers):

                # find the index 
                layer = unique_layer_idx.index(layer_)
                # get the colden indexes of the molecular species of reference
                idx_walkers_colden_ref = idx_walkers_colden_[
                    layer_, idx_name_mol_ref, :] # true layer idx
                
                for mol_idx, mol in enumerate(names_mol) :
                    if (mol == 'h13cop') and ('12co-13co' in constraints_abundances):
                        # induce by the abundance of 12co, 13co and hcop
                        # 12co
                        colden_ratios_shifts_12co = walkers_colden_ratios_shifts_[
                            layer_, idx_12co, :]
                        # hcop
                        colden_ratios_shifts_hcop = walkers_colden_ratios_shifts_[
                            layer_, idx_hcop, :]
                        # h13cop
                        colden_ratios_shifts__ = colden_ratios_shifts_hcop - colden_ratios_shifts_12co
                    elif counter in idx_unknown_mol_layer : 
                        # already dealt
                        colden_ratios_shifts__ = walkers_colden_ratios_shifts_[layer_, mol_idx, :]   
                    else : # fixed value
                        colden_ratios_shifts__ = np.round(
                                colden_ratios_shifts[mol][layer]/colden_res, 1).astype(int) * np.ones(walkers)
                    
                    if ('same_in_all_layers' in constraints_abundances) and (layer_ >= 1):
                        colden_ratios_shifts__ = colden_ratios_shifts_buffer[0, mol_idx, :]

                    idx_walkers_colden___ = idx_walkers_colden_ref + colden_ratios_shifts__
                    idx_walkers_colden_[layer_, mol_idx, :] = idx_walkers_colden___

                    # save it in case of 'same_in_all_layers' in constraints_abundances
                    colden_ratios_shifts_buffer[layer_, mol_idx, :] = colden_ratios_shifts__

                    counter += 1

            #######################
            # compute the new NLL #
            #######################
            walkers_s_V_ = np.zeros((walkers, layers))
            for layer in range(layers):
                if ('sandwich' in constraints_geometry):
                # in which sandwich are we ?
                    idx_sandwich = (layer//number_of_layers_per_clump) % number_of_clumps
                    idx_inner_layer = idxs_inner_layer[idx_sandwich]

                    if layer > idx_inner_layer : 
                        # find the opposite layer
                        shift = layer - idx_inner_layer
                        idx_opposed_layer = idx_inner_layer - shift
                        walkers_s_V_[:, layer] = walkers_s_V_[:, idx_opposed_layer]
                    else :  
                        walkers_FWHM_ = FPS[FWHM_idx][np.array(idx_walkers_theta_[unique_layer_idx[layer], 2, :])]
                        walkers_s_V_[:, layer] = from_FWHM_to_s_V(walkers_FWHM_)
                else:
                    walkers_FWHM_ = FPS[FWHM_idx][np.array(idx_walkers_theta_[unique_layer_idx[layer], 2, :])]
                    walkers_s_V_[:, layer] = from_FWHM_to_s_V(walkers_FWHM_)

            NLL_ = np.zeros(np.shape(NLL))

            for mol_idx in range(len(names_mol)):
                for line_idx in range(len(names_line[mol_idx])):

                    freq = freqs[mol_idx][line_idx]
                    if DEBUG:
                        assert check_type(freq, type_of_reference=float)
                    velocity_resolution = velocity_res[mol_idx][line_idx]
                    velocity_channels_line = velocity_channels[mol_idx][line_idx]

                    grid_Tex, grid_tau = grids_Tex[mol_idx][line_idx], grids_tau[mol_idx][line_idx]

                    Tex_walkers, tau_walkers = np.zeros(
                        (walkers, layers)), np.zeros((walkers, layers))

                    for layer in range(layers):
                        if ('sandwich' in constraints_geometry): 
                            # in which sandwich are we ?
                            idx_sandwich = (layer//number_of_layers_per_clump) % number_of_clumps
                            idx_inner_layer = idxs_inner_layer[idx_sandwich]

                            if layer > idx_inner_layer : 
                                # find the opposite layer
                                shift = layer - idx_inner_layer
                                idx_opposed_layer = idx_inner_layer - shift
                                Tex_walkers[:, layer] = Tex_walkers[:,
                                                                idx_opposed_layer]
                                tau_walkers[:, layer] = tau_walkers[:,
                                                                idx_opposed_layer]
                            else : 
                                idx_walkers_log10_T_kin = np.array(idx_walkers_theta_[unique_layer_idx[layer], 0, :])
                                idx_walkers_log10_nH2 = np.array(idx_walkers_theta_[unique_layer_idx[layer], 1, :])
                                idx_walkers_FWHM = np.array(idx_walkers_theta_[unique_layer_idx[layer], 2, :])

                                idx_walkers_log10_N = np.array(idx_walkers_colden_[unique_layer_idx[layer], mol_idx, :])

                                Tex_walkers[:, layer] = grid_Tex[idx_walkers_log10_T_kin,
                                                            idx_walkers_log10_nH2,
                                                            idx_walkers_log10_N,
                                                            idx_walkers_FWHM]
                                tau_walkers[:, layer] = grid_tau[idx_walkers_log10_T_kin,
                                                            idx_walkers_log10_nH2,
                                                            idx_walkers_log10_N,
                                                            idx_walkers_FWHM]
                        else:
                            idx_walkers_log10_T_kin = np.array(
                                idx_walkers_theta_[unique_layer_idx[layer], 0, :])
                            idx_walkers_log10_nH2 = np.array(
                                idx_walkers_theta_[unique_layer_idx[layer], 1, :])
                            idx_walkers_FWHM = np.array(
                                idx_walkers_theta_[unique_layer_idx[layer], 2, :])

                            idx_walkers_log10_N = np.array(
                                idx_walkers_colden_[unique_layer_idx[layer], mol_idx, :])

                            Tex_walkers[:, layer] = grid_Tex[idx_walkers_log10_T_kin,
                                                            idx_walkers_log10_nH2,
                                                            idx_walkers_log10_N,
                                                            idx_walkers_FWHM]
                            tau_walkers[:, layer] = grid_tau[idx_walkers_log10_T_kin,
                                                            idx_walkers_log10_nH2,
                                                            idx_walkers_log10_N,
                                                            idx_walkers_FWHM]

                    if ('sandwich' in constraints_geometry) and ('mirror' in constraints_kinematics):
                        unique_layer_idx_ = np.arange(0, layers, 1)
                    else:
                        unique_layer_idx_ = unique_layer_idx

                    if velocity_resolution == 0.1 or velocity_resolution == 0.5: 
                        reference_velocity_resolution_ = 0.1
                    elif velocity_resolution == 0.25 : 
                        reference_velocity_resolution_ = 0.125

                    # simulated spectrum
                    s_line = compute_radiative_tranfer_equation(
                        Tex_walkers,
                        tau_walkers,
                        freq,
                        velocity_channels_line,
                        velocity_resolution,
                        walkers_s_V_,
                        walkers_C_V_,
                        unique_layer_idx=unique_layer_idx_,
                        decomposition=False,
                        conserved_flux=True,
                        reference_velocity_resolution=reference_velocity_resolution_,
                        DEBUG=DEBUG, 
                        number_of_C_V_components=number_of_clumps, 
                        number_of_layers_per_clump=number_of_layers_per_clump, 
                        peak_only=(PEAK_ONLY and f'{names_mol[mol_idx]}({names_line[mol_idx][line_idx]})' in thick_lines))[0]
                    # measure
                    x_line = x[mol_idx][line_idx]
                    # noise
                    s_b_line = s_b[mol_idx][line_idx]
                    s_c_line = s_c[mol_idx][line_idx]
                    # criterion
                    NLL__ = compute_criterion(
                        x_line, s_line, s_b_line, s_c_line)

                    NLL_ += NLL__

            # update indexes of walkers that allow to decrease NLL
            idx_decreasing_NLL = np.nonzero(NLL_ < NLL)

            for layer in range(layers):
                # update C_V
                walkers_C_V[idx_decreasing_NLL, layer] = walkers_C_V_[
                    idx_decreasing_NLL, layer]
                # update theta (without colden) indexes
                idx_walkers_theta[unique_layer_idx[layer], :, idx_decreasing_NLL] = idx_walkers_theta_[unique_layer_idx[layer], :, idx_decreasing_NLL]
                # update the colden indexes
                idx_walkers_colden[unique_layer_idx[layer], :, idx_decreasing_NLL] = idx_walkers_colden_[unique_layer_idx[layer], :, idx_decreasing_NLL]
                # update the colden shift 
                walkers_colden_ratios_shifts[unique_layer_idx[layer], :, idx_decreasing_NLL] = walkers_colden_ratios_shifts_[unique_layer_idx[layer], :, idx_decreasing_NLL]
            
            # update the NLL
            NLL[idx_decreasing_NLL] = NLL_[idx_decreasing_NLL]

            if not PARALLELISM:
                pbar_rw.update(1)

        # sort the NLL and select the best walkers for the next step
        # flattened array of size walkers
        NLL, idx_NLL, count_NLL = np.unique(
            NLL, return_index=True, return_counts=True)
        walkers_next_step = len(NLL)
        
        toc = time.perf_counter()

        if WRITE_RESULTS_TXT_FILE:
            # convergence results
            params_convergence.append([step, f'{explo_idx}/{iterations}', walkers, count_NLL[0],
                                    from_float_to_string(NLL[0], width_precision=2, digit_precision=2), f'{toc - tic:0.4f}'])
        
    # get and save the best solution
    if WRITE_RESULTS_TXT_FILE:
        # for each estimates, compute the proportion of walkers at the optimum
        theta_proportions = np.zeros((len(theta), layers))
        # get the walker that corresponds to the optimum value of the NLL
        idx_optimal_NLL = idx_NLL[:1].item()

        for layer in range(layers):
            if ('sandwich' in constraints_geometry): 
                # in which sandwich are we ?
                idx_sandwich = (layer//number_of_layers_per_clump) % number_of_clumps
                idx_inner_layer = idxs_inner_layer[idx_sandwich]

                if layer > idx_inner_layer : 
                    # find the opposite layer
                    shift = layer - idx_inner_layer
                    idx_opposed_layer = idx_inner_layer - shift
                    theta_proportions[:, layer] = theta_proportions[:, idx_opposed_layer]

                else : 
                    count = 0  # without colden
                    for t_idx, t in enumerate(theta):
                        if str(t) == 'C_V':
                            # update C_V
                            t_layer = walkers_C_V[idx_optimal_NLL, layer]
                            walkers_t_layer = walkers_C_V[:, layer]
                        elif str(t) == 'log10_N':
                            t_layer = FPS[log10_N_idx][np.array(
                                idx_walkers_colden[unique_layer_idx[layer], idx_name_mol_ref, idx_optimal_NLL])]
                            walkers_t_layer = FPS[log10_N_idx][np.array(
                                idx_walkers_colden[unique_layer_idx[layer], idx_name_mol_ref, :])]
                        else:
                            t_layer = FPS[t_idx][np.array(
                                idx_walkers_theta[unique_layer_idx[layer], count, idx_optimal_NLL])]
                            walkers_t_layer = FPS[t_idx][np.array(
                                idx_walkers_theta[unique_layer_idx[layer], count, :])]
                            count += 1  # without colden
                        theta_proportions[t_idx, layer] = (np.count_nonzero(
                            walkers_t_layer == t_layer) * 100)/walkers
                        
            else:
                count = 0  # without colden
                for t_idx, t in enumerate(theta):
                    if str(t) == 'C_V':
                        # update C_V
                        t_layer = walkers_C_V[idx_optimal_NLL, layer]
                        walkers_t_layer = walkers_C_V[:, layer]
                    elif str(t) == 'log10_N':
                        t_layer = FPS[log10_N_idx][np.array(
                            idx_walkers_colden[unique_layer_idx[layer], idx_name_mol_ref, idx_optimal_NLL])]
                        walkers_t_layer = FPS[log10_N_idx][np.array(
                            idx_walkers_colden[unique_layer_idx[layer], idx_name_mol_ref, :])]
                    else:
                        t_layer = FPS[t_idx][np.array(
                            idx_walkers_theta[unique_layer_idx[layer], count, idx_optimal_NLL])]
                        walkers_t_layer = FPS[t_idx][np.array(
                            idx_walkers_theta[unique_layer_idx[layer], count, :])]
                        count += 1  # without colden
                    theta_proportions[t_idx, layer] = (np.count_nonzero(
                        walkers_t_layer == t_layer) * 100)/walkers
            params_results = theta_proportions

    walkers = 1

    # get the walker that corresponds to the optimum value of the NLL
    optimal_theta = np.zeros((len(theta), layers)) #, dtype=Real)
    optimal_colden = np.zeros((len(names_mol), layers))
    # to get the optimal indexes in Radex grids
    optimal_idx_walkers_theta = np.zeros(
        (number_of_different_layers, len_FPS_without_colden, 1), dtype=int)
    optimal_idx_walkers_colden = np.zeros(
        (number_of_different_layers, len(names_mol), 1), dtype=int)

    NLL, idx_NLL, count_NLL = NLL[:walkers].item(
    ), idx_NLL[:walkers].item(), count_NLL[:walkers].item()

    for layer in range(layers):         
        if ('sandwich' in constraints_geometry): 
            # in which sandwich are we ?
            idx_sandwich = (layer//number_of_layers_per_clump) % number_of_clumps
            idx_inner_layer = idxs_inner_layer[idx_sandwich]

            if layer > idx_inner_layer : 
                # find the opposite layer
                shift = layer - idx_inner_layer
                idx_opposed_layer = idx_inner_layer - shift
                
                optimal_theta[:, layer] = optimal_theta[:, idx_opposed_layer]
                if 'mirror' in constraints_kinematics:
                    # except for the C_V
                    optimal_theta[C_V_idx, layer] = walkers_C_V[idx_NLL, layer]
                optimal_colden[:, layer] = optimal_colden[:, idx_opposed_layer]

            else : 
                count = 0  # without colden
                for t_idx, t in enumerate(theta):
                    if str(t) == 'C_V':
                        # update velocity shift
                        t_layer = walkers_C_V[idx_NLL, layer]
                    elif str(t) == 'log10_N':
                        # begin by the specie of reference
                        optimal_idx_walkers_colden_ = idx_walkers_colden[unique_layer_idx[layer], idx_name_mol_ref, idx_NLL]
                        optimal_idx_walkers_colden[unique_layer_idx[layer], idx_name_mol_ref, 0] = optimal_idx_walkers_colden_
                        t_layer = FPS[log10_N_idx][np.array(
                            optimal_idx_walkers_colden_)]
                        for mol_idx, mol in enumerate(names_mol):
                            optimal_colden[mol_idx, layer] = FPS[log10_N_idx][np.array(
                                #idx_walkers_colden[unique_layer_idx[layer], mol_idx, idx_NLL])] + colden_ratios.item().get(mol)
                                idx_walkers_colden[unique_layer_idx[layer], mol_idx, idx_NLL])] + colden_ratios.get(mol)
                            optimal_idx_walkers_colden[unique_layer_idx[layer], mol_idx,
                                                    0] = idx_walkers_colden[unique_layer_idx[layer], mol_idx, idx_NLL]
                    else:
                        optimal_idx_walkers_theta_ = idx_walkers_theta[unique_layer_idx[layer], count, idx_NLL]
                        optimal_idx_walkers_theta[unique_layer_idx[layer],
                                                count, 0] = optimal_idx_walkers_theta_
                        t_layer = FPS[t_idx][np.array(optimal_idx_walkers_theta_)]
                        count += 1

                    optimal_theta[t_idx, layer] = t_layer

        else:
            count = 0  # without colden
            for t_idx, t in enumerate(theta):
                if str(t) == 'C_V':
                    # update velocity shift
                    t_layer = walkers_C_V[idx_NLL, layer]
                elif str(t) == 'log10_N':
                    # begin by the specie of reference
                    optimal_idx_walkers_colden_ = idx_walkers_colden[unique_layer_idx[layer], idx_name_mol_ref, idx_NLL]
                    optimal_idx_walkers_colden[unique_layer_idx[layer], idx_name_mol_ref, 0] = optimal_idx_walkers_colden_
                    t_layer = FPS[log10_N_idx][np.array(
                        optimal_idx_walkers_colden_)]
                    for mol_idx, mol in enumerate(names_mol):
                        optimal_colden[mol_idx, layer] = FPS[log10_N_idx][np.array(
                            #idx_walkers_colden[unique_layer_idx[layer], mol_idx, idx_NLL])] + colden_ratios.item().get(mol)
                            idx_walkers_colden[unique_layer_idx[layer], mol_idx, idx_NLL])] + colden_ratios.get(mol)
                        optimal_idx_walkers_colden[unique_layer_idx[layer], mol_idx,
                                                   0] = idx_walkers_colden[unique_layer_idx[layer], mol_idx, idx_NLL]
                else:
                    optimal_idx_walkers_theta_ = idx_walkers_theta[unique_layer_idx[layer], count, idx_NLL]
                    optimal_idx_walkers_theta[unique_layer_idx[layer], count, 0] = optimal_idx_walkers_theta_
                    t_layer = FPS[t_idx][np.array(optimal_idx_walkers_theta_)]
                    count += 1

                optimal_theta[t_idx, layer] = t_layer

    # ...and the Tex and tau of each species
    optimal_Tex, optimal_tau = [], []

    for mol_idx, mol in enumerate(names_mol):
        optimal_Tex.append([]), optimal_tau.append([])
        for line_idx in range(len(names_line[mol_idx])):
            optimal_Tex[mol_idx].append(
                []), optimal_tau[mol_idx].append([])

            grid_Tex, grid_tau = grids_Tex[mol_idx][line_idx], grids_tau[mol_idx][line_idx]

            for layer in range(layers):
                if ('sandwich' in constraints_geometry): 
                    # in which sandwich are we ?
                    idx_sandwich = (layer//number_of_layers_per_clump) % number_of_clumps
                    idx_inner_layer = idxs_inner_layer[idx_sandwich]

                    if layer > idx_inner_layer : 
                        # find the opposite layer
                        shift = layer - idx_inner_layer
                        idx_opposed_layer = idx_inner_layer - shift
                        optimal_Tex[mol_idx][line_idx].append(optimal_Tex[mol_idx][line_idx][idx_opposed_layer])
                        optimal_tau[mol_idx][line_idx].append(optimal_tau[mol_idx][line_idx][idx_opposed_layer])
                    else : 
                        optimal_idx_log10_T_kin = np.array(
                        optimal_idx_walkers_theta[unique_layer_idx[layer], 0, :])
                        optimal_idx_log10_nH2 = np.array(
                            optimal_idx_walkers_theta[unique_layer_idx[layer], 1, :])
                        optimal_idx_FWHM = np.array(
                            optimal_idx_walkers_theta[unique_layer_idx[layer], 2, :])

                        optimal_idx_log10_N = np.array(
                            optimal_idx_walkers_colden[unique_layer_idx[layer], mol_idx, :])

                        optimal_Tex[mol_idx][line_idx].append(
                            grid_Tex[
                                optimal_idx_log10_T_kin,
                                optimal_idx_log10_nH2,
                                optimal_idx_log10_N,
                                optimal_idx_FWHM].item())
                        optimal_tau[mol_idx][line_idx].append(
                            grid_tau[
                                optimal_idx_log10_T_kin,
                                optimal_idx_log10_nH2,
                                optimal_idx_log10_N,
                                optimal_idx_FWHM].item())
                else:
                    optimal_idx_log10_T_kin = np.array(
                        optimal_idx_walkers_theta[unique_layer_idx[layer], 0, :])
                    optimal_idx_log10_nH2 = np.array(
                        optimal_idx_walkers_theta[unique_layer_idx[layer], 1, :])
                    optimal_idx_FWHM = np.array(
                        optimal_idx_walkers_theta[unique_layer_idx[layer], 2, :])

                    optimal_idx_log10_N = np.array(
                        optimal_idx_walkers_colden[unique_layer_idx[layer], mol_idx, :])

                    optimal_Tex[mol_idx][line_idx].append(
                        grid_Tex[
                            optimal_idx_log10_T_kin,
                            optimal_idx_log10_nH2,
                            optimal_idx_log10_N,
                            optimal_idx_FWHM].item())
                    optimal_tau[mol_idx][line_idx].append(
                        grid_tau[
                            optimal_idx_log10_T_kin,
                            optimal_idx_log10_nH2,
                            optimal_idx_log10_N,
                            optimal_idx_FWHM].item())

    res = {}
    res['results'] = [optimal_theta, optimal_Tex,
                      optimal_tau, NLL, count_NLL, optimal_colden]

    if WRITE_RESULTS_TXT_FILE:
        params = [params_convergence, params_results]
        res['params'] = params

    if VERBOSE:
        print(f'')

    return res

# %% gradient descent 

def find_optimal_step(
        initial_updated_theta_ur: np.ndarray, 
        fim: np.ndarray, 
        NLL_g: np.ndarray, 
        inversed_fim_time_NLL_g: np.ndarray,
        x: List[np.ndarray],
        velocity_channels: List[List[np.ndarray]],
        velocity_resolutions: List[List[Real]],
        freqs: List[List[np.ndarray]],
        s_b: List[List[Real]],
        s_c: List[List[Real]],
        log10_abundances: np.ndarray,
        reference_velocity_resolution: Optional[float] = 0.01,
        steps: Optional[List] = [0.1, 0.5, 1],
        max_iters: Optional[int] = 20, 
        pixel: Optional[List[int]] = [0, 0]
) -> float:

    if 'same_C_V_in_all_layers' in constraints_kinematics : 
        C_V_idxs = [C_V_idx + idx * len(theta) for idx in range(1, layers)]

    NLL = []

    for step in steps:
        
        updated_theta_ur = initial_updated_theta_ur - step * inversed_fim_time_NLL_g

        ##############################################################
        # check if the updated theta remains in an acceptable ranges #
        ##############################################################
        updated_FWHM = np.zeros(number_of_different_layers)
        updated_C_V = np.zeros(number_of_different_layers)
        updated_log10_T_kin = np.zeros(number_of_different_layers)
        updated_log10_nH2 = np.zeros(number_of_different_layers)
        updated_log10_N = np.zeros(number_of_different_layers)
        for layer in range(number_of_different_layers):
            updated_FWHM[layer] = updated_theta_ur[FWHM_idx + layer * len(theta)]
            updated_log10_T_kin[layer] = updated_theta_ur[log10_T_kin_idx + layer * len(theta)]
            updated_log10_nH2[layer] = updated_theta_ur[log10_nH2_idx + layer * len(theta)]
            updated_log10_N[layer] = updated_theta_ur[log10_N_idx + layer * len(theta)]
            if ('same_C_V_in_all_layers' in constraints_kinematics) and (layer > 0) : 
                updated_C_V[layer] = updated_theta_ur[C_V_idx]
            else : 
                updated_C_V[layer] = updated_theta_ur[C_V_idx + layer * len(theta)]
        
        res = limit_theta(updated_log10_T_kin, 
                          updated_log10_nH2, 
                          updated_log10_N, 
                          updated_FWHM, 
                          updated_C_V
                          )
        updated_log10_T_kin, updated_log10_nH2, updated_log10_N, updated_FWHM, updated_C_V, idx_unknowns, still_in_ranges = res

        if 'same_C_V_in_all_layers' in constraints_kinematics : 
            idx_unknowns_ = np.zeros(number_of_unknowns)
            counter = 0
            for t_idx in range(len(idx_unknowns)):
                if t_idx not in C_V_idxs : 
                    idx_unknowns_[counter] =  idx_unknowns[counter]
                    counter += 1
            idx_unknowns = idx_unknowns_

        iter_check = 0 
        while (not still_in_ranges) and (iter_check < max_iterations): 

            # [1] update the parameters that have been modified (limited by the allowed values)
            updated_theta_ur_ = np.zeros(np.shape(updated_theta_ur))

            for layer in range(number_of_different_layers):
                updated_theta_ur_[log10_T_kin_idx + layer * len(theta)] = updated_log10_T_kin[layer]
                updated_theta_ur_[log10_nH2_idx + layer * len(theta)] = updated_log10_nH2[layer]
                updated_theta_ur_[log10_N_idx + layer * len(theta)] = updated_log10_N[layer]
                updated_theta_ur_[FWHM_idx + layer * len(theta)] = updated_FWHM[layer]
                if 'same_C_V_in_all_layers' and layer > 0 : 
                    updated_theta_ur_[C_V_idx + layer * len(theta)] = updated_C_V[0]
                else : 
                    updated_theta_ur_[C_V_idx + layer * len(theta)] = updated_C_V[layer]
            updated_theta_ur = updated_theta_ur_

            # [2] update the fim and the NLL_g 
            nan_idx_unknowns = np.where(
                idx_unknowns == 0, np.nan, idx_unknowns)
            idx_fixed_param = np.argwhere(np.isnan(nan_idx_unknowns))
            idx_idx_unknowns = np.nonzero(idx_unknowns)

            # remove the column and the row corresponding to the dimension of fixed theta components
            fim_ = np.delete(fim, idx_fixed_param, axis=0)  # remove the row
            fim_ = np.delete(fim_, idx_fixed_param, axis=1)  # remove the column
            inversed_fim_ = inverse_fim(fim_, maximal_fim_cond=maximal_fim_cond)
            NLL_g_ = np.delete(NLL_g, idx_fixed_param, axis=0)  # remove the row
            inversed_fim_time_NLL_g_ = inversed_fim_ @ NLL_g_

            # then update the vector of parameters
            updated_theta_ur[idx_idx_unknowns] = initial_updated_theta_ur[idx_idx_unknowns] - step * inversed_fim_time_NLL_g_[:]

            ##############################################################
            # check if the updated theta remains in an acceptable ranges #
            ##############################################################
            updated_FWHM = np.zeros(number_of_different_layers)
            updated_C_V = np.zeros(number_of_different_layers)
            updated_log10_T_kin = np.zeros(number_of_different_layers)
            updated_log10_nH2 = np.zeros(number_of_different_layers)
            updated_log10_N = np.zeros(number_of_different_layers)
            for layer in range(number_of_different_layers):
                updated_FWHM[layer] = updated_theta_ur[FWHM_idx + layer * len(theta)]
                updated_log10_T_kin[layer] = updated_theta_ur[log10_T_kin_idx + layer * len(theta)]
                updated_log10_nH2[layer] = updated_theta_ur[log10_nH2_idx + layer * len(theta)]
                updated_log10_N[layer] = updated_theta_ur[log10_N_idx + layer * len(theta)]
                if 'same_C_V_in_all_layers' and layer > 0 : 
                    updated_C_V[layer] = updated_theta_ur[C_V_idx]
                else : 
                    updated_C_V[layer] = updated_theta_ur[C_V_idx + layer * len(theta)]
            res = limit_theta(updated_log10_T_kin, 
                            updated_log10_nH2, 
                            updated_log10_N, 
                            updated_FWHM, 
                            updated_C_V
                            )
            updated_log10_T_kin, updated_log10_nH2, updated_log10_N, updated_FWHM, updated_C_V, idx_unknowns, still_in_ranges = res
            
            if 'same_C_V_in_all_layers' in constraints_kinematics : 
                idx_unknowns_ = np.zeros(number_of_unknowns)
                counter = 0
                for t_idx in range(len(idx_unknowns)):
                    if t_idx not in C_V_idxs : 
                        idx_unknowns_[counter] =  idx_unknowns[counter]
                        counter += 1
                idx_unknowns = idx_unknowns_

            iter_check += 1

        if iter_check == max_iters : 
            NLL.append(10**6)
        
        else : 
            ########################################
            # compute the NLL at the updated point #
            ########################################
            updated_NLL = 0

            # update estimations for all layers (redondance for the estimator but required for NLL computation)
            updated_theta = np.zeros((len(theta), layers))
            for layer in range(layers) : 
                updated_theta[log10_T_kin_idx, layer] = updated_log10_T_kin[unique_layer_idx[layer]]
                updated_theta[log10_nH2_idx, layer] = updated_log10_nH2[unique_layer_idx[layer]]
                updated_theta[log10_N_idx, layer] = updated_log10_N[unique_layer_idx[layer]]
                updated_theta[FWHM_idx, layer] = updated_FWHM[unique_layer_idx[layer]]
                if ('sandwich' in constraints_geometry) and ('mirror' in constraints_kinematics): 
                    # in which sandwich are we ?
                    idx_sandwich = (layer//number_of_layers_per_clump) % number_of_clumps
                    idx_inner_layer = idxs_inner_layer[idx_sandwich]

                    if layer > idx_inner_layer : 
                        # find the opposite layer
                        shift = layer - idx_inner_layer
                        idx_opposed_layer = idx_inner_layer - shift 
                        updated_theta[C_V_idx, layer] = 2 * updated_C_V[idx_inner_layer] - updated_C_V[idx_opposed_layer]
                    else : 
                        updated_theta[C_V_idx, layer] = updated_C_V[unique_layer_idx[layer]]
                else : 
                    updated_theta[C_V_idx, layer] = updated_C_V[unique_layer_idx[layer]]

            log10_T_kin = updated_theta[log10_T_kin_idx, :]
            T_kin = 10 ** log10_T_kin
            log10_nH2 = updated_theta[log10_nH2_idx, :]
            log10_N = updated_theta[log10_N_idx, :]
            FWHM = updated_theta[FWHM_idx, :]
            s_V = from_FWHM_to_s_V(FWHM)
            C_V = updated_theta[C_V_idx, :]

            collider_densities = np.zeros(
                (len(colliders), T_kin.size))
            for c_idx, c in enumerate(from_OPR_to_densities(T_kin, log10_nH2, colliders=colliders)):
                collider_densities[c_idx, :] = 10**c

            updated_Tex = []
            updated_tau = []
            for mol_idx, mol in enumerate(names_mol):
                updated_Tex.append([])
                updated_tau.append([])
                log10_N_mol = log10_N + log10_abundances[mol_idx, :]
                N_mol = 10**log10_N_mol
                for line_idx in range(len(names_line[mol_idx])):
                    velocity_channels_line = velocity_channels[mol_idx][line_idx]
                    s_b_line = s_b[mol_idx][line_idx]
                    s_c_line = s_c[mol_idx][line_idx]

                    x_line = x[mol_idx][line_idx]
                    freq = freqs[mol_idx][line_idx]
                    velocity_resolution = velocity_resolutions[mol_idx][line_idx]

                    # compute the Tex and tau 
                    freq_min, freq_max = get_freq_bandwidth(freq)[0], get_freq_bandwidth(freq)[1]

                    if not check_shape(FWHM, shape_of_reference=T_kin.shape, flag=False):
                        # FWHM has been changed from (layers,) -> (1, layers)
                        FWHM = FWHM[0, :]
                        
                    Tex_line, tau_line, Tr_line = execute_radex(
                    mol,
                    f'find_optimal_step_{pixel[0]}_{pixel[1]}_{step}',
                    freq_min,
                    freq_max,
                    T_kin,
                    colliders,
                    collider_densities,
                    N_mol,
                    FWHM,
                    geometry,
                    clean= (not DEBUG))
                
                    Tex_line = Tex_line.T
                    tau_line = tau_line.T

                    updated_Tex[mol_idx].append(Tex_line)
                    updated_tau[mol_idx].append(tau_line)

                    if not check_shape(Tex_line, s_V, shape_of_reference=Tex_line.shape, flag=False):
                        s_V = s_V.reshape(Tex_line.shape)
                        FWHM = FWHM.reshape(Tex_line.shape)
                        C_V =  C_V.reshape(Tex_line.shape)

                    if ('sandwich' in constraints_geometry) and ('mirror' in constraints_kinematics):
                        unique_layer_idx_ = np.arange(0, layers, 1)
                    else:
                        unique_layer_idx_ = unique_layer_idx

                    if velocity_resolution == 0.1 or velocity_resolution == 0.5: 
                        reference_velocity_resolution_ = 0.01
                    elif velocity_resolution == 0.25 : 
                        reference_velocity_resolution_ = 0.125

                    s_line = compute_radiative_tranfer_equation(
                        Tex_line,
                        tau_line,
                        freq,
                        velocity_channels_line,
                        velocity_resolution,
                        s_V,
                        C_V,
                        unique_layer_idx=unique_layer_idx_,
                        decomposition=False,
                        conserved_flux=True,
                        reference_velocity_resolution=reference_velocity_resolution_,
                        DEBUG=DEBUG,
                        number_of_C_V_components=number_of_clumps, 
                        number_of_layers_per_clump=number_of_layers_per_clump, 
                        peak_only = (PEAK_ONLY and f'{names_mol[mol_idx]}({names_line[mol_idx][line_idx]})' in thick_lines))[0]
                    
                    # criterion
                    updated_NLL_ = compute_criterion(
                        x_line, s_line, s_b_line, s_c_line)

                    updated_NLL += updated_NLL_.item()

            NLL.append(updated_NLL)

    # find the step that minimizes the tested NLL
    optimal_step_idx = NLL.index(min(NLL))
    optimal_step = steps[optimal_step_idx]

    if optimal_step != steps[0] and optimal_step != steps[-1]:  # intermediate step
        steps_ = steps.copy()

        # compute NLL values between the previous ones
        for step_idx in range(len(steps)-1):
            intermediate_step = (
                steps[step_idx] + steps[step_idx + 1])/2
            steps_.insert(2*step_idx+1, intermediate_step)

            updated_theta_ur = initial_updated_theta_ur - \
                intermediate_step * inversed_fim_time_NLL_g

            ##############################################################
            # check if the updated theta remains in an acceptable ranges #
            ##############################################################
            updated_FWHM = np.zeros(number_of_different_layers)
            updated_C_V = np.zeros(number_of_different_layers)
            updated_log10_T_kin = np.zeros(number_of_different_layers)
            updated_log10_nH2 = np.zeros(number_of_different_layers)
            updated_log10_N = np.zeros(number_of_different_layers)
            for layer in range(number_of_different_layers):
                updated_FWHM[layer] = updated_theta_ur[FWHM_idx + layer * len(theta)]
                updated_log10_T_kin[layer] = updated_theta_ur[log10_T_kin_idx + layer * len(theta)]
                updated_log10_nH2[layer] = updated_theta_ur[log10_nH2_idx + layer * len(theta)]
                updated_log10_N[layer] = updated_theta_ur[log10_N_idx + layer * len(theta)]
                if ('same_C_V_in_all_layers' in constraints_kinematics) and (layer > 0) : 
                    updated_C_V[layer] = updated_theta_ur[C_V_idx]
                else : 
                    updated_C_V[layer] = updated_theta_ur[C_V_idx + layer * len(theta)]
            
            res = limit_theta(updated_log10_T_kin, 
                            updated_log10_nH2, 
                            updated_log10_N, 
                            updated_FWHM, 
                            updated_C_V
                            )
            updated_log10_T_kin, updated_log10_nH2, updated_log10_N, updated_FWHM, updated_C_V, idx_unknowns, still_in_ranges = res

            if 'same_C_V_in_all_layers' in constraints_kinematics : 
                idx_unknowns_ = np.zeros(number_of_unknowns)
                counter = 0
                for t_idx in range(len(idx_unknowns)):
                    if t_idx not in C_V_idxs : 
                        idx_unknowns_[counter] =  idx_unknowns[counter]
                        counter += 1
                idx_unknowns = idx_unknowns_

            iter_check = 0 
            while (not still_in_ranges) and (iter_check < max_iterations):  

                # [1] update the parameters that have been modified (limited by the allowed values)
                updated_theta_ur_ = np.zeros(np.shape(updated_theta_ur))

                for layer in range(number_of_different_layers):
                    updated_theta_ur_[log10_T_kin_idx + layer * len(theta)] = updated_log10_T_kin[layer]
                    updated_theta_ur_[log10_nH2_idx + layer * len(theta)] = updated_log10_nH2[layer]
                    updated_theta_ur_[log10_N_idx + layer * len(theta)] = updated_log10_N[layer]
                    updated_theta_ur_[FWHM_idx + layer * len(theta)] = updated_FWHM[layer]
                    if 'same_C_V_in_all_layers' and layer > 0 : 
                        updated_theta_ur_[C_V_idx + layer * len(theta)] = updated_C_V[0]
                    else : 
                        updated_theta_ur_[C_V_idx + layer * len(theta)] = updated_C_V[layer]
                updated_theta_ur = updated_theta_ur_

                # [2] update the fim and the NLL_g 
                nan_idx_unknowns = np.where(
                    idx_unknowns == 0, np.nan, idx_unknowns)
                idx_fixed_param = np.argwhere(np.isnan(nan_idx_unknowns))
                idx_idx_unknowns = np.nonzero(idx_unknowns)

                # remove the column and the row corresponding to the dimension of fixed theta components
                fim_ = np.delete(fim, idx_fixed_param, axis=0)  # remove the row
                fim_ = np.delete(fim_, idx_fixed_param, axis=1)  # remove the column
                inversed_fim_ = inverse_fim(fim_, maximal_fim_cond = maximal_fim_cond)
                NLL_g_ = np.delete(NLL_g, idx_fixed_param, axis=0)  # remove the row
                inversed_fim_time_NLL_g_ = inversed_fim_ @ NLL_g_

                # then update the vector of parameters
                updated_theta_ur[idx_idx_unknowns] = initial_updated_theta_ur[idx_idx_unknowns] - intermediate_step * inversed_fim_time_NLL_g_[:]

                ##############################################################
                # check if the updated theta remains in an acceptable ranges #
                ##############################################################
                updated_FWHM = np.zeros(number_of_different_layers)
                updated_C_V = np.zeros(number_of_different_layers)
                updated_log10_T_kin = np.zeros(number_of_different_layers)
                updated_log10_nH2 = np.zeros(number_of_different_layers)
                updated_log10_N = np.zeros(number_of_different_layers)
                for layer in range(number_of_different_layers):
                    updated_FWHM[layer] = updated_theta_ur[FWHM_idx + layer * len(theta)]
                    updated_log10_T_kin[layer] = updated_theta_ur[log10_T_kin_idx + layer * len(theta)]
                    updated_log10_nH2[layer] = updated_theta_ur[log10_nH2_idx + layer * len(theta)]
                    updated_log10_N[layer] = updated_theta_ur[log10_N_idx + layer * len(theta)]
                    if 'same_C_V_in_all_layers' and layer > 0 : 
                        updated_C_V[layer] = updated_theta_ur[C_V_idx]
                    else : 
                        updated_C_V[layer] = updated_theta_ur[C_V_idx + layer * len(theta)]
                res = limit_theta(updated_log10_T_kin, 
                                updated_log10_nH2, 
                                updated_log10_N, 
                                updated_FWHM, 
                                updated_C_V
                                )
                updated_log10_T_kin, updated_log10_nH2, updated_log10_N, updated_FWHM, updated_C_V, idx_unknowns, still_in_ranges = res

                if 'same_C_V_in_all_layers' in constraints_kinematics : 
                    idx_unknowns_ = np.zeros(number_of_unknowns)
                    counter = 0
                    for t_idx in range(len(idx_unknowns)):
                        if t_idx not in C_V_idxs : 
                            idx_unknowns_[counter] =  idx_unknowns[counter]
                            counter += 1
                    idx_unknowns = idx_unknowns_

                iter_check += 1
        
            if iter_check == max_iters : 
                steps_.remove(intermediate_step)

            else : 
                ########################################
                # compute the NLL at the updated point #
                ########################################
                updated_NLL = 0

                # update estimations for all layers (redondance for the estimator but required for NLL computation)
                updated_theta = np.zeros((len(theta), layers))
                for layer in range(layers) : 
                    updated_theta[log10_T_kin_idx, layer] = updated_log10_T_kin[unique_layer_idx[layer]]
                    updated_theta[log10_nH2_idx, layer] = updated_log10_nH2[unique_layer_idx[layer]]
                    updated_theta[log10_N_idx, layer] = updated_log10_N[unique_layer_idx[layer]]
                    updated_theta[FWHM_idx, layer] = updated_FWHM[unique_layer_idx[layer]]
                    if ('sandwich' in constraints_geometry) and ('mirror' in constraints_kinematics): 
                        # in which sandwich are we ?
                        idx_sandwich = (layer//number_of_layers_per_clump) % number_of_clumps
                        idx_inner_layer = idxs_inner_layer[idx_sandwich]

                        if layer > idx_inner_layer : 
                            # find the opposite layer
                            shift = layer - idx_inner_layer
                            idx_opposed_layer = idx_inner_layer - shift 
                            updated_theta[C_V_idx, layer] = 2 * updated_C_V[idx_inner_layer] - updated_C_V[idx_opposed_layer]
                        else : 
                            updated_theta[C_V_idx, layer] = updated_C_V[unique_layer_idx[layer]]
                    else : 
                        updated_theta[C_V_idx, layer] = updated_C_V[unique_layer_idx[layer]]

                log10_T_kin = updated_theta[log10_T_kin_idx, :]
                T_kin = 10 ** log10_T_kin
                log10_nH2 = updated_theta[log10_nH2_idx, :]
                log10_N = updated_theta[log10_N_idx, :]
                FWHM = updated_theta[FWHM_idx, :]
                s_V = from_FWHM_to_s_V(FWHM)
                C_V = updated_theta[C_V_idx, :]

                collider_densities = np.zeros(
                    (len(colliders), T_kin.size))
                for c_idx, c in enumerate(from_OPR_to_densities(T_kin, log10_nH2, colliders=colliders)):
                    collider_densities[c_idx, :] = 10**c

                updated_Tex = []
                updated_tau = []
                for mol_idx, mol in enumerate(names_mol):
                    updated_Tex.append([])
                    updated_tau.append([])
                    log10_N_mol = log10_N + log10_abundances[mol_idx, :]
                    N_mol = 10**log10_N_mol
                    for line_idx in range(len(names_line[mol_idx])):
                        velocity_channels_line = velocity_channels[mol_idx][line_idx]
                        s_b_line = s_b[mol_idx][line_idx]
                        s_c_line = s_c[mol_idx][line_idx]

                        x_line = x[mol_idx][line_idx]
                        freq = freqs[mol_idx][line_idx]
                        velocity_resolution = velocity_resolutions[mol_idx][line_idx]

                        # compute the Tex and tau 
                        freq_min, freq_max = get_freq_bandwidth(freq)[0], get_freq_bandwidth(freq)[1]

                        if not check_shape(FWHM, shape_of_reference=T_kin.shape, flag=False):
                            # FWHM has been changed from (layers,) -> (1, layers)
                            FWHM = FWHM[0, :]
                            
                        Tex_line, tau_line, Tr_line = execute_radex(
                        mol,
                        f'find_optimal_step_{pixel[0]}_{pixel[1]}_{intermediate_step}',
                        freq_min,
                        freq_max,
                        T_kin,
                        colliders,
                        collider_densities,
                        N_mol,
                        FWHM,
                        geometry,
                        clean=(not DEBUG))
                    
                        Tex_line = Tex_line.T
                        tau_line = tau_line.T

                        updated_Tex[mol_idx].append(Tex_line)
                        updated_tau[mol_idx].append(tau_line)

                        if not check_shape(Tex_line, s_V, shape_of_reference=Tex_line.shape, flag=False):
                            s_V = s_V.reshape(Tex_line.shape)
                            FWHM = FWHM.reshape(Tex_line.shape)
                            C_V =  C_V.reshape(Tex_line.shape)

                        if ('sandwich' in constraints_geometry) and ('mirror' in constraints_kinematics):
                            unique_layer_idx_ = np.arange(0, layers, 1)
                        else:
                            unique_layer_idx_ = unique_layer_idx

                        if velocity_resolution == 0.1 or velocity_resolution == 0.5: 
                            reference_velocity_resolution_ = 0.01
                        elif velocity_resolution == 0.25 : 
                            reference_velocity_resolution_ = 0.125

                        s_line = compute_radiative_tranfer_equation(
                            Tex_line,
                            tau_line,
                            freq,
                            velocity_channels_line,
                            velocity_resolution,
                            s_V,
                            C_V,
                            unique_layer_idx=unique_layer_idx_,
                            decomposition=False,
                            reference_velocity_resolution=reference_velocity_resolution_,
                            DEBUG=DEBUG,
                            number_of_C_V_components=number_of_clumps,
                            number_of_layers_per_clump=number_of_layers_per_clump, 
                            peak_only = (PEAK_ONLY and f'{names_mol[mol_idx]}({names_line[mol_idx][line_idx]})' in thick_lines))[0]
                        
                        # criterion
                        updated_NLL_ = compute_criterion(
                            x_line, s_line, s_b_line, s_c_line)

                        updated_NLL += updated_NLL_.item()

                NLL.insert(2*step_idx+1, updated_NLL)

        z = np.poly1d(np.polyfit(from_list_to_array(steps_), from_list_to_array(NLL), 2))
        xz = np.linspace(steps_[0], steps_[-1], 50)
        optimal_step_idx = list(z(xz)).index(min(list(z(xz))))
        optimal_step = xz[optimal_step_idx]

    return optimal_step
 
def limit_theta(
        log10_T_kin: np.ndarray,
        log10_nH2: np.ndarray,
        log10_N_ref: np.ndarray,
        FWHM: np.ndarray,
        C_V: np.ndarray, 
        log10_T_kin_min_max: Optional[List] = [min_log10_T_kin, max_log10_T_kin],
        log10_nH2_min_max: Optional[List] = [min_log10_nH2, max_log10_nH2],
        log10_N_ref_min_max: Optional[List] = [min_log10_N_ref, max_log10_N_ref],
        FWHM_min_max: Optional[List] = [min_FWHM, max_FWHM],
        C_V_min_max: Optional[List] = [0, 20] #[7.5, 13.5]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:

    if DEBUG : 
        assert np.shape(log10_T_kin)[0] == number_of_different_layers

    number_of_components = len(theta)
    idx_unknowns = np.ones(
        number_of_different_layers * number_of_components)  # to get index where component are fixed

    # Whether at least component is out of boundaries, still_in_ranges = False
    still_in_ranges = True

    # which component is out of limit before apply the correction ?
    test = np.any(log10_T_kin < log10_T_kin_min_max[0]) or np.any(
        log10_T_kin > log10_T_kin_min_max[1])  # KO test
    still_in_ranges = still_in_ranges and not test
    # correction
    log10_T_kin = np.where(
        log10_T_kin < log10_T_kin_min_max[0], log10_T_kin_min_max[0], log10_T_kin)
    log10_T_kin = np.where(
        log10_T_kin > log10_T_kin_min_max[1], log10_T_kin_min_max[1], log10_T_kin)
    # which component is now "fixed" ?
    idx = np.argwhere(np.logical_or(
        log10_T_kin == log10_T_kin_min_max[0], log10_T_kin == log10_T_kin_min_max[1]))
    idx_unknowns[log10_T_kin_idx + idx * number_of_components] = 0

    # which component is out of limit before apply the correction ?
    test = np.any(log10_nH2 < log10_nH2_min_max[0]) or np.any(
        log10_nH2 > log10_nH2_min_max[1])
    still_in_ranges = still_in_ranges and not test
    # correction
    log10_nH2 = np.where(
        log10_nH2 < log10_nH2_min_max[0], log10_nH2_min_max[0], log10_nH2)
    log10_nH2 = np.where(
        log10_nH2 > log10_nH2_min_max[1], log10_nH2_min_max[1], log10_nH2)
    # which component is now "fixed" ?
    idx = np.argwhere(np.logical_or(
        log10_nH2 == log10_nH2_min_max[0], log10_nH2 == log10_nH2_min_max[1]))
    idx_unknowns[log10_nH2_idx + idx * number_of_components] = 0

    # which component is out of limit before apply the correction ?
    test = np.any(log10_N_ref < log10_N_ref_min_max[0]) or np.any(
        log10_N_ref > log10_N_ref_min_max[1])
    still_in_ranges = still_in_ranges and not test
    # correction
    log10_N_ref = np.where(
        log10_N_ref < log10_N_ref_min_max[0], log10_N_ref_min_max[0], log10_N_ref)
    log10_N_ref = np.where(
        log10_N_ref > log10_N_ref_min_max[1], log10_N_ref_min_max[1], log10_N_ref)
    # which component is now "fixed" ?
    idx = np.argwhere(np.logical_or(
        log10_N_ref == log10_N_ref_min_max[0], log10_N_ref == log10_N_ref_min_max[1]))
    idx_unknowns[log10_N_idx + idx * number_of_components] = 0

    # which component is out of limit before apply the correction ?
    test = np.any(FWHM < FWHM_min_max[0]) or np.any(FWHM > FWHM_min_max[1])
    still_in_ranges = still_in_ranges and not test
    # correction
    FWHM = np.where(FWHM < FWHM_min_max[0], FWHM_min_max[0], FWHM)
    FWHM = np.where(FWHM > FWHM_min_max[1], FWHM_min_max[1], FWHM)
    # which component is now "fixed" ?
    idx = np.argwhere(np.logical_or(
        FWHM == FWHM_min_max[0], FWHM == FWHM_min_max[1]))
    idx_unknowns[FWHM_idx + idx * number_of_components] = 0

    # which component is out of limit before apply the correction ?
    test = np.any(C_V < C_V_min_max[0]) or np.any(C_V > C_V_min_max[1])
    still_in_ranges = still_in_ranges and not test
    # correction
    C_V = np.where(C_V < C_V_min_max[0], C_V_min_max[0], C_V)
    C_V = np.where(C_V > C_V_min_max[1], C_V_min_max[1], C_V)
    # which component is now "fixed" ?
    idx = np.argwhere(np.logical_or(
        C_V == C_V_min_max[0], C_V == C_V_min_max[1]))
    idx_unknowns[C_V_idx + idx * number_of_components] = 0

    return log10_T_kin, log10_nH2, log10_N_ref, FWHM, C_V, idx_unknowns, still_in_ranges

def gradient_descent(
        initial_theta: np.ndarray, # shape (len(theta), layers) 
        initial_Tex, 
        initial_tau, 
        initial_NLL: float, # NLL corresponding to the initial solution
        initial_colden: np.ndarray, # shape (len(names_mol), layers)
        x: List[List[np.ndarray]],  # measures
        freqs: List[List[np.ndarray]],
        s_b: List[List[Real]],
        s_c: List[List[Real]],
        velocity_res: List[List[np.ndarray]],
        velocity_channels: List[List[np.ndarray]],
        reference_velocity_resolution: Optional[Real] = 0.01, 
        pixel: Optional[List[int]] = [0, 0]
    ):
    
    if VERBOSE:
        print(
            f'\t[is running] Gradient descent', end = '', flush = True)

    stop = False

    # get the estimated abundance ratios from the rw
    log10_abundances = np.zeros((len(names_mol), layers))
    for layer in range(layers) : 
        for mol_idx in range(len(names_mol)) : 
            log10_abundances[mol_idx, layer] = initial_colden[mol_idx, layer] - initial_colden[idx_name_mol_ref, layer]

    # initial_theta's shape is (len(theta), layers) so far => redundance from the estimator point of view,
    # so it required to only keep dimension of unknowns
    initial_theta_ur = np.zeros((number_of_unknowns, 1)) # vector of unknowns unraveled
    unknown_idx = 0
    
    for layer in range(number_of_different_layers):
        # find the index 
        layer_ = unique_layer_idx.index(layer)
        for theta_idx in range(len(theta)):
            if theta_idx == C_V_idx: 
                if ('same_C_V_in_all_layers' in constraints_kinematics) and (layer_ > 0):
                    # redondance from the estimator point of view
                    pass 
                else :
                    initial_theta_ur[unknown_idx, 0] = initial_theta[theta_idx, layer_]
                    unknown_idx += 1
            else: 
                initial_theta_ur[unknown_idx, 0] = initial_theta[theta_idx, layer_]
                unknown_idx += 1

    if 'same_C_V_in_all_layers' in constraints_kinematics : 
        C_V_idxs = [C_V_idx + idx * len(theta) for idx in range(1, layers)]

    # the theta of the iteration (i-1)
    current_optimal_theta = np.copy(initial_theta)
    current_optimal_theta_ur = initial_theta_ur
    current_optimal_Tex = [[from_list_to_array(initial_Tex[mol_idx][line_idx]) for line_idx in range(
        len(names_line[mol_idx]))] for mol_idx in range(len(names_mol))]
    current_optimal_tau = [[from_list_to_array(initial_tau[mol_idx][line_idx]) for line_idx in range(
        len(names_line[mol_idx]))] for mol_idx in range(len(names_mol))]
    current_optimal_NLL = initial_NLL
    optimal_NLL = initial_NLL

    # the solution
    optimal_theta = np.zeros(np.shape(initial_theta))
    optimal_theta[:] = initial_theta[:]
    optimal_Tex = [[from_list_to_array(initial_Tex[mol_idx][line_idx]).reshape((1, layers)) for line_idx in range(
        len(names_line[mol_idx]))] for mol_idx in range(len(names_mol))]
    optimal_tau = [[from_list_to_array(initial_tau[mol_idx][line_idx]).reshape((1, layers)) for line_idx in range(
        len(names_line[mol_idx]))] for mol_idx in range(len(names_mol))]
    optimal_NLL = initial_NLL

    iter = 1

    while not stop and iter < max_iterations:
        
        #############################################################
        # compute the fim and the NLL gradient at the current point #
        #############################################################
        log10_T_kin = current_optimal_theta[log10_T_kin_idx, :]
        T_kin = 10 ** log10_T_kin
        log10_nH2 = current_optimal_theta[log10_nH2_idx, :]
        nH2 = 10 ** log10_nH2
        log10_N = current_optimal_theta[log10_N_idx, :]
        FWHM = current_optimal_theta[FWHM_idx, :]
        s_V = from_FWHM_to_s_V(FWHM)
        C_V = current_optimal_theta[C_V_idx, :]

        fim = np.zeros((number_of_unknowns, number_of_unknowns))
        NLL_g = np.zeros((number_of_unknowns, 1))

        for mol_idx, mol in enumerate(names_mol):
            log10_N_mol = log10_N + log10_abundances[mol_idx, :]
            N_mol = 10**log10_N_mol
            for line_idx in range(len(names_line[mol_idx])):
                velocity_channels_line = velocity_channels[mol_idx][line_idx]
                s_b_line = s_b[mol_idx][line_idx]
                s_c_line = s_c[mol_idx][line_idx]

                x_line = x[mol_idx][line_idx]
                freq = freqs[mol_idx][line_idx]
                velocity_resolution = velocity_res[mol_idx][line_idx]

                Tex_line = current_optimal_Tex[mol_idx][line_idx]
                Tex_line = Tex_line.reshape((Tex_line.size, 1)).T

                tau_line = current_optimal_tau[mol_idx][line_idx]
                tau_line = tau_line.reshape((tau_line.size, 1)).T

                if not check_shape(Tex_line, s_V, C_V, shape_of_reference=Tex_line.shape, flag=False):
                    s_V = s_V.reshape(Tex_line.shape)
                    FWHM = FWHM.reshape(Tex_line.shape)
                    C_V =  C_V.reshape(Tex_line.shape)

                if not check_shape(T_kin, nH2, N_mol, s_V, FWHM, shape_of_reference=s_V.shape, flag=False):
                    T_kin = T_kin.reshape(s_V.shape)
                    nH2 = nH2.reshape(s_V.shape)
                    N_mol = N_mol.reshape(s_V.shape)

                if velocity_resolution == 0.1 or velocity_resolution == 0.5: 
                    reference_velocity_resolution_ = 0.01
                elif velocity_resolution == 0.25 : 
                    reference_velocity_resolution_ = 0.125

                fim_, NLL_g_ = compute_fim(
                                    mol,
                                    freq,
                                    velocity_channels_line,
                                    velocity_resolution,
                                    T_kin,
                                    nH2,
                                    N_mol,
                                    FWHM,
                                    s_V,
                                    C_V,
                                    colliders,
                                    geometry,
                                    Tex_line,
                                    tau_line,
                                    1.,
                                    s_c_line,
                                    s_b_line,
                                    unique_layer_idx,
                                    f'fim_gd_{pixel[0]}_{pixel[1]}',
                                    constraints_kinematics,
                                    constraints_geometry,
                                    layers = layers,
                                    idxs_inner_layer = idxs_inner_layer,
                                    number_of_unknowns = number_of_unknowns,
                                    number_of_clumps = number_of_clumps,
                                    number_of_layers_per_clump = number_of_layers_per_clump,
                                    peak_only = (PEAK_ONLY and f'{names_mol[mol_idx]}({names_line[mol_idx][line_idx]})' in thick_lines),
                                    conserved_flux=True,
                                    reference_velocity_resolution = reference_velocity_resolution_,
                                    theta = theta,
                                    C_V_idx=C_V_idx,
                                    NLL_g=True, 
                                    x = x_line,
                                    DEBUG=DEBUG
                                )
                fim += fim_
                NLL_g += NLL_g_
        
        #########################
        # find the optimal step #
        #########################
        inversed_fim = inverse_fim(fim, maximal_fim_cond = maximal_fim_cond)
        inversed_fim_time_NLL_g = inversed_fim @ NLL_g

        alpha = find_optimal_step(
            current_optimal_theta_ur, 
            fim, 
            NLL_g, 
            inversed_fim_time_NLL_g, 
            x, 
            velocity_channels, 
            velocity_res, 
            freqs, 
            s_b, 
            s_c, 
            log10_abundances, 
            reference_velocity_resolution=reference_velocity_resolution, 
            steps = [0.1, 0.5, 1], 
            max_iters = 20, 
            pixel = [pixel[0], pixel[1]]
            )
        
        updated_theta_ur = current_optimal_theta_ur - alpha * inversed_fim_time_NLL_g

        ##############################################################
        # check if the updated theta remains in an acceptable ranges #
        ##############################################################
        updated_FWHM = np.zeros(number_of_different_layers)
        updated_C_V = np.zeros(number_of_different_layers)
        updated_log10_T_kin = np.zeros(number_of_different_layers)
        updated_log10_nH2 = np.zeros(number_of_different_layers)
        updated_log10_N = np.zeros(number_of_different_layers)
        for layer in range(number_of_different_layers):
            updated_FWHM[layer] = updated_theta_ur[FWHM_idx + layer * len(theta)]
            updated_log10_T_kin[layer] = updated_theta_ur[log10_T_kin_idx + layer * len(theta)]
            updated_log10_nH2[layer] = updated_theta_ur[log10_nH2_idx + layer * len(theta)]
            updated_log10_N[layer] = updated_theta_ur[log10_N_idx + layer * len(theta)]
            if ('same_C_V_in_all_layers' in constraints_kinematics) and (layer > 0) : 
                updated_C_V[layer] = updated_theta_ur[C_V_idx]
            else : 
                updated_C_V[layer] = updated_theta_ur[C_V_idx + layer * len(theta)]
        
        res = limit_theta(updated_log10_T_kin, 
                          updated_log10_nH2, 
                          updated_log10_N, 
                          updated_FWHM, 
                          updated_C_V
                          )
        updated_log10_T_kin, updated_log10_nH2, updated_log10_N, updated_FWHM, updated_C_V, idx_unknowns, still_in_ranges = res

        if 'same_C_V_in_all_layers' in constraints_kinematics : 
            idx_unknowns_ = np.zeros(number_of_unknowns)
            counter = 0
            for t_idx in range(len(idx_unknowns)):
                if t_idx not in C_V_idxs : 
                    idx_unknowns_[counter] =  idx_unknowns[counter]
                    counter += 1
            idx_unknowns = idx_unknowns_

        if not still_in_ranges : 
            # some parameters have been fixed to allowed boundaries => have to remove the corresponding dimension in the fim and NLL_g
            iter_check = 0 
            while not still_in_ranges and iter_check < max_iterations:  # if some component are not interpretable

                # [1] update the parameters that have been modified (limited by the allowed values)
                updated_theta_ur_ = np.zeros(np.shape(updated_theta_ur))

                for layer in range(number_of_different_layers):
                    updated_theta_ur_[log10_T_kin_idx + layer * len(theta)] = updated_log10_T_kin[layer]
                    updated_theta_ur_[log10_nH2_idx + layer * len(theta)] = updated_log10_nH2[layer]
                    updated_theta_ur_[log10_N_idx + layer * len(theta)] = updated_log10_N[layer]
                    updated_theta_ur_[FWHM_idx + layer * len(theta)] = updated_FWHM[layer]
                    if 'same_C_V_in_all_layers' and layer > 0 : 
                        updated_theta_ur_[C_V_idx + layer * len(theta)] = updated_C_V[layer]
                    else : 
                        updated_theta_ur_[C_V_idx + layer * len(theta)] = updated_C_V[layer]
                updated_theta_ur = updated_theta_ur_

                # [2] update the fim and the NLL_g 
                nan_idx_unknowns = np.where(
                    idx_unknowns == 0, np.nan, idx_unknowns)
                idx_fixed_param = np.argwhere(np.isnan(nan_idx_unknowns))
                idx_idx_unknowns = np.nonzero(idx_unknowns)

                # remove the column and the row corresponding to the dimension of fixed theta components
                fim_ = np.delete(fim, idx_fixed_param, axis=0)  # remove the row
                fim_ = np.delete(fim_, idx_fixed_param, axis=1)  # remove the column
                inversed_fim_ = inverse_fim(fim_, maximal_fim_cond = maximal_fim_cond)
                NLL_g_ = np.delete(NLL_g, idx_fixed_param, axis=0)  # remove the row
                inversed_fim_time_NLL_g_ = inversed_fim_ @ NLL_g_

                # then update the vector of parameters
                updated_theta_ur[idx_idx_unknowns] = current_optimal_theta_ur[idx_idx_unknowns] - alpha * inversed_fim_time_NLL_g_[:]

                ##############################################################
                # check if the updated theta remains in an acceptable ranges #
                ##############################################################
                updated_FWHM = np.zeros(number_of_different_layers)
                updated_C_V = np.zeros(number_of_different_layers)
                updated_log10_T_kin = np.zeros(number_of_different_layers)
                updated_log10_nH2 = np.zeros(number_of_different_layers)
                updated_log10_N = np.zeros(number_of_different_layers)
                for layer in range(number_of_different_layers):
                    updated_FWHM[layer] = updated_theta_ur[FWHM_idx + layer * len(theta)]
                    updated_log10_T_kin[layer] = updated_theta_ur[log10_T_kin_idx + layer * len(theta)]
                    updated_log10_nH2[layer] = updated_theta_ur[log10_nH2_idx + layer * len(theta)]
                    updated_log10_N[layer] = updated_theta_ur[log10_N_idx + layer * len(theta)]
                    if 'same_C_V_in_all_layers' and layer > 0 : 
                        updated_C_V[layer] = updated_theta_ur[C_V_idx]
                    else : 
                        updated_C_V[layer] = updated_theta_ur[C_V_idx + layer * len(theta)]
                res = limit_theta(updated_log10_T_kin, 
                                updated_log10_nH2, 
                                updated_log10_N, 
                                updated_FWHM, 
                                updated_C_V
                                )
                updated_log10_T_kin, updated_log10_nH2, updated_log10_N, updated_FWHM, updated_C_V, idx_unknowns, still_in_ranges = res

                if 'same_C_V_in_all_layers' in constraints_kinematics : 
                    idx_unknowns_ = np.zeros(number_of_unknowns)
                    counter = 0
                    for t_idx in range(len(idx_unknowns)):
                        if t_idx not in C_V_idxs : 
                            idx_unknowns_[counter] =  idx_unknowns[counter]
                            counter += 1
                    idx_unknowns = idx_unknowns_

                iter_check += 1

        ########################################
        # compute the NLL at the updated point #
        ########################################
        updated_NLL = 0

        # update estimations for all layers (redondance for the estimator but required for NLL computation)
        updated_theta = np.zeros((len(theta), layers))
        for layer in range(layers) : 
            updated_theta[log10_T_kin_idx, layer] = updated_log10_T_kin[unique_layer_idx[layer]]
            updated_theta[log10_nH2_idx, layer] = updated_log10_nH2[unique_layer_idx[layer]]
            updated_theta[log10_N_idx, layer] = updated_log10_N[unique_layer_idx[layer]]
            updated_theta[FWHM_idx, layer] = updated_FWHM[unique_layer_idx[layer]]
            if ('sandwich' in constraints_geometry) and ('mirror' in constraints_kinematics): 
                # in which sandwich are we ?
                idx_sandwich = (layer//number_of_layers_per_clump) % number_of_clumps
                idx_inner_layer = idxs_inner_layer[idx_sandwich]

                if layer > idx_inner_layer : 
                    # find the opposite layer
                    shift = layer - idx_inner_layer
                    idx_opposed_layer = idx_inner_layer - shift
                    updated_theta[C_V_idx, layer] = 2 * updated_C_V[idx_inner_layer] - updated_C_V[idx_opposed_layer]
                else : 
                    updated_theta[C_V_idx, layer] = updated_C_V[unique_layer_idx[layer]]
            else : 
                updated_theta[C_V_idx, layer] = updated_C_V[unique_layer_idx[layer]]

        log10_T_kin = updated_theta[log10_T_kin_idx, :]
        T_kin = 10 ** log10_T_kin
        log10_nH2 = updated_theta[log10_nH2_idx, :]
        nH2 = 10 ** log10_nH2
        log10_N = updated_theta[log10_N_idx, :]
        FWHM = updated_theta[FWHM_idx, :]
        s_V = from_FWHM_to_s_V(FWHM)
        C_V = updated_theta[C_V_idx, :]

        collider_densities = np.zeros(
            (len(colliders), T_kin.size))
        for c_idx, c in enumerate(from_OPR_to_densities(T_kin, log10_nH2, colliders=colliders)):
            collider_densities[c_idx, :] = 10**c

        updated_Tex = []
        updated_tau = []
        for mol_idx, mol in enumerate(names_mol):
            updated_Tex.append([])
            updated_tau.append([])
            log10_N_mol = log10_N + log10_abundances[mol_idx, :]
            N_mol = 10**log10_N_mol

            for line_idx in range(len(names_line[mol_idx])):
                velocity_channels_line = velocity_channels[mol_idx][line_idx]
                s_b_line = s_b[mol_idx][line_idx]
                s_c_line = s_c[mol_idx][line_idx]

                x_line = x[mol_idx][line_idx]
                freq = freqs[mol_idx][line_idx]
                velocity_resolution = velocity_res[mol_idx][line_idx]

                # compute the Tex and tau 
                freq_min, freq_max = get_freq_bandwidth(freq)[0], get_freq_bandwidth(freq)[1]

                if not check_shape(FWHM, shape_of_reference=T_kin.shape, flag=False):
                    # FWHM has been changed from (layers,) -> (1, layers)
                    FWHM = FWHM[0, :]
                    
                Tex_line, tau_line, Tr_line = execute_radex(
                mol,
                f'updated_Tex_tau_{pixel[0]}_{pixel[1]}',
                freq_min,
                freq_max,
                T_kin,
                colliders,
                collider_densities,
                N_mol,
                FWHM,
                geometry,
                clean=(not DEBUG))
            
                Tex_line = Tex_line.T
                tau_line = tau_line.T

                updated_Tex[mol_idx].append(Tex_line)
                updated_tau[mol_idx].append(tau_line)

                if not check_shape(Tex_line, s_V, shape_of_reference=Tex_line.shape, flag=False):
                    s_V = s_V.reshape(Tex_line.shape)
                    FWHM = FWHM.reshape(Tex_line.shape)
                    C_V =  C_V.reshape(Tex_line.shape)

                if ('sandwich' in constraints_geometry) and ('mirror' in constraints_kinematics):
                    unique_layer_idx_ = np.arange(0, layers, 1)
                else:
                    unique_layer_idx_ = unique_layer_idx

                if velocity_resolution == 0.1 or velocity_resolution == 0.5: 
                    reference_velocity_resolution_ = 0.01
                elif velocity_resolution == 0.25 : 
                    reference_velocity_resolution_ = 0.125

                s_line = compute_radiative_tranfer_equation(
                    Tex_line,
                    tau_line,
                    freq,
                    velocity_channels_line,
                    velocity_resolution,
                    s_V,
                    C_V,
                    unique_layer_idx=unique_layer_idx_,
                    decomposition=False,
                    conserved_flux=True,
                    reference_velocity_resolution=reference_velocity_resolution_,
                    DEBUG=DEBUG,
                    number_of_C_V_components=number_of_clumps,
                    number_of_layers_per_clump=number_of_layers_per_clump, 
                    peak_only = (PEAK_ONLY and f'{names_mol[mol_idx]}({names_line[mol_idx][line_idx]})' in thick_lines))[0]
                
                # criterion
                updated_NLL_ = compute_criterion(
                    x_line, s_line, s_b_line, s_c_line)

                updated_NLL += updated_NLL_.item()

        if optimal_NLL > updated_NLL:
            optimal_NLL = updated_NLL
            optimal_theta = updated_theta
            optimal_Tex = updated_Tex
            optimal_tau = updated_tau

        if iter == max_iterations:
            stop = True

        # test for convergence
        if abs(updated_NLL - current_optimal_NLL) < convergence_threshold:
            stop = True

        # updating the optimal theta (even if the NLL may be not minimized)
        current_optimal_theta_ur = updated_theta_ur
        current_optimal_theta = updated_theta
        current_optimal_NLL = updated_NLL
        current_optimal_Tex = updated_Tex
        current_optimal_tau = updated_tau

        iter += 1
        
    # update the colden 
    optimal_colden = np.zeros((len(names_mol), layers), dtype = np.single)
    optimal_log10_N = optimal_theta[log10_N_idx, :]
    for mol_idx in range(len(names_mol)) : 
        optimal_colden[mol_idx, :] = optimal_log10_N + log10_abundances[mol_idx, :]

    # from np.array to list 
    optimal_Tex_, optimal_tau_ = [], []
    for mol_idx in range(len(names_mol)) : 
        optimal_Tex_.append([]), optimal_tau_.append([])
        for line_idx in range(len(names_line[mol_idx])) : 
            optimal_Tex_[mol_idx].append([]), optimal_tau_[mol_idx].append([])
            for layer in range(layers) : 
                optimal_Tex_[mol_idx][line_idx].append(optimal_Tex[mol_idx][line_idx][0, layer].item())
                optimal_tau_[mol_idx][line_idx].append(optimal_tau[mol_idx][line_idx][0, layer].item())
    optimal_Tex, optimal_tau = optimal_Tex_, optimal_tau_

    res = {}
    res['results'] = [optimal_theta, optimal_Tex,
                      optimal_tau, optimal_NLL, iter, optimal_colden]

    if VERBOSE:
        print(
            f'{bcolors.OKGREEN} Done.{bcolors.ENDC}')

    return res 