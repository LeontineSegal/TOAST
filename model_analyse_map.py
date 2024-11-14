'''
File name: model_analyze_one_pixel.py
Customize the model of the sight-line.
All informations contain here are then loeaded when the model fitting procecure starts.
'''

#%% load modules (DO NOT MODIFY)
import os
import numpy as np
from toolboxs.toolbox_python.toolbox_python import bcolors

import warnings
warnings.filterwarnings("ignore")

# %% paths
DATASET = 'horsehead' # name it as you want, according to the different datasets you are studying
path_base = '***/TOAST/'

path_inputs = f'{path_base}/examples/data-{DATASET}/inputs'  # all information about the dataset
path_grids_Tex_tau = f'{path_base}/examples/data-{DATASET}/grids-Tex-tau'
path_save = f'{path_base}/examples/analyze_map/outputs' 
folder_save = 'estimations'

# %% verbose and plot setting
VERBOSE = False
PLOT =  False
FORMAT_FILE = '.png'  # for plot, among {'.png', '.pdf'}
WRITE_RESULTS_TXT_FILE = True
PARALLELISM, POLL_SIZE = True, 20
SINGLE_PIXEL_ANALYSIS, number_of_analysis = False, 1 # if True, allows to repeate the analysis of a single pixel, e.g., to study the robustness of the estimation results (Monte-Carlo)

# useful when coding/debugging only 
DEBUG = False

# %% molecular lines
name_mol_ref_ = '13co'

names_mol = [
    '12co',
    '13co',
    'c18o',
    'hcop',
    'h13cop'
]
names_line = [
    ['1-0'],
    ['1-0', '2-1'],
    ['1-0', '2-1'],
    ['1-0'],
    ['1-0']
]

# %% layers and geometry

layers = 3 # total number of layers composing the line-of-sight
constraints_geometry = ['sandwich']  # among {none, 'sandwich'}
number_of_layers_per_clump = layers # number of layer per clump
number_of_clumps = layers // number_of_layers_per_clump

#%% chemistry

model_chemistry = 'dense-core'

# load the basic values of column density ratios
colden_ratios_radex = np.load(f'{path_inputs}/colden_ratios.npy', allow_pickle=True)
# for each layer the values [min, ..., max] of log[alpha] below, where log[N(x)] = log[alpha] + log[N(13co)]
colden_ratios_ranges = {}
# convert them to shift in respect to basic values
colden_ratios_shifts = {}

if model_chemistry == 'dense-core':
    constraints_abundances = [
        '12co-13co'
    ]

    for key, key_d in colden_ratios_radex.item().items():

        if key == '12co':
            # N(12CO)/N(13CO) = 10^(1.8) x 10^[-0.5, ..., 0.1] =~ [20, 80]
            colden_ratios_ranges[key] = [
                # layer 1 : outer
                colden_ratios_radex.item()[key] + np.round(np.arange(-0.5, 0.15, 0.05), 2),
                # layer 2 : inner (basic value)
                np.array([colden_ratios_radex.item()[key]]),
                # layer 3 : background, with the 'sandwich' model, the value does't matter since it forced to be the same than the layer 1
                colden_ratios_radex.item()[key] + np.round(np.arange(-0.5, 0.15, 0.05), 2),
            ]
            # for layer 1 and 2
            colden_ratios_shifts[key] = [np.round(colden_ratios_ranges[key][layer_idx] - key_d, 3)
                                    for layer_idx in range(layers - 1)]
            # layer 3 -> induce by layer 1
            colden_ratios_shifts[key].append(np.array([np.nan]))

        elif key == '13co':
            # 0. for the species of reference
            colden_ratio_range = np.array([key_d])
            colden_ratios_ranges[key] = [colden_ratio_range for layer_idx in range(layers)] 
            colden_ratios_shifts[key] = [np.round(colden_ratios_ranges[key][layer_idx] - key_d, 3) for layer_idx in range(layers)]

        elif key == 'c18o':
            # N(C18O)/N(13CO) = 10^(-0.9) x 10^[-0.5, ..., 0.15] = 10^[-1.4, -0.75]
            colden_ratios_ranges[key] = [
                colden_ratios_radex.item()[key] + np.round(np.arange(-0.5, 0.16, 0.05), 2),
                np.array([colden_ratios_radex.item()[key]]),
                colden_ratios_radex.item()[key] + np.round(np.arange(-0.5, 0.16, 0.05), 2),
            ]
            # for layer 1 and 2
            colden_ratios_shifts[key] = [np.round(colden_ratios_ranges[key][layer_idx] - key_d, 3)
                                    for layer_idx in range(layers - 1)]
            # layer 3 -> induce by layer 1
            colden_ratios_shifts[key].append(np.array([np.nan]))

        elif key == 'hcop':
            # N(HCO+)/N(13CO) = 10^(-2.88) x 10^[-0.9, ..., 0.] = 10^[-3.78, -2.88]
            colden_ratios_ranges[key] = [
                colden_ratios_radex.item()[key] + np.round(np.arange(-0.9, 0.05, 0.05), 2),
                colden_ratios_radex.item()[key] + np.round(np.arange(-0.9, 0.05, 0.05), 2),
                colden_ratios_radex.item()[key] + np.round(np.arange(-0.9, 0.05, 0.05), 2),
            ]
            # for layer 1 and 2
            colden_ratios_shifts[key] = [np.round(colden_ratios_ranges[key][layer_idx] - key_d, 3)
                                    for layer_idx in range(layers - 1)]
            # layer 3 -> induce by layer 1
            colden_ratios_shifts[key].append(np.array([np.nan]))

        elif key == 'h13cop': 
            # induce by '12co-13co'
            colden_ratio_range = np.array([np.nan])
            colden_ratios_ranges[key] = [colden_ratio_range
                                        for layer_idx in range(layers)]
            colden_ratios_shifts[key] = [np.array([np.nan]) for layer_idx in range(layers)]

#%% kinematics
constraints_kinematics = []  # among {'', 'same_C_V_in_all_layers', 'mirror'}

# %% model fitting settings

# optically thick lines treatment
'''
if True (recommended for the Horsehead nebula dataset), 
only the peak of the line of 12co(1-0) will be fitted.
'''
PEAK_ONLY = True
delta_V_around_peak = 1. # km/s
thick_lines = ['12co(1-0)']

# noise model
SNR_TRICK = True # (True recommended, see reference [1])
CALIBRATION_NOISE = False # (False recommended, see reference [1])

# the optimization process 
'''
choose among :
'rw' (random walk, grid search)
'gd' (gradient descent). Results from a previous 'rw' are required to initialized a gd
'rw-gd' (both successively)
'''
optimization = 'rw'

# for rw optimization
walkers_per_step = [10000, 10000, 1000]
iterations_per_step = [10, 1000, 5000]
C_V_res = 0.01  # velocity resolution when estimating the centroid velocity (km/s)

# for gd optimization
max_iterations = 30
convergence_threshold = 10**(-6)
maximal_fim_cond = 10 ** 9 # condition number for the Fisher information matrix

# to speed up the process...
'''
if True (recommended),
reduce the studied bandwidth to WINDOW_BANDWIDTH = [C_V - DELTA_V, C_V + DELTA_V]
where C_V is the systemic centroid velocity C_V (estimated in prepare-dataset.py).
'''
WINDOW_BANDWIDTH = True
delta_V = [
    [1.5], 
    [1.5, 1.5], 
    [1.5, 1.5], 
    [1.5], 
    [1.5]
] # km/s

# vector of unknowns (DO NOT MODIFY)
theta = ['log10_T_kin', 'log10_nH2', 'log10_N', 'FWHM', 'C_V']
theta_latex = [r'log(T$_{kin}$ / K)', r'log($n_{H_2}$ / cm$^{-3}$)',
                r'log(N / cm$^{-2}$)', r'FWHM / km/s', r'$C_V$ / km/s']

log10_T_kin_idx = theta.index('log10_T_kin')
log10_nH2_idx = theta.index('log10_nH2')
log10_N_idx = theta.index('log10_N')
FWHM_idx = theta.index('FWHM')
s_V_idx = FWHM_idx
C_V_idx = theta.index('C_V')

# %% Radex settings (DO NOT MODIFY)
geometry = 'radexUnifSphere'  # for escape probability approximation
colliders = ['p-H2', 'o-H2', 'e']

# %% check the model's consistence (DO NOT MODIFY)

# layers and geometry
if 'sandwich' in constraints_geometry:
    assert number_of_layers_per_clump%2!=0, print(
        f'{bcolors.WARNING}\n [error] The sandwich model is only implemented for odd number of layers.\n{bcolors.ENDC}')

# chemistry 
if model_chemistry == '13co-c18o':
    assert ('13co' in names_mol) and ('c18o' in names_mol), print(
        f'{bcolors.WARNING}\n [error] The 13co-c18o model is not suited for the selected molecular lines.\n{bcolors.ENDC}')

if model_chemistry == 'dense-core':
    assert ('sandwich' in constraints_geometry), print(
        f'{bcolors.WARNING}\n [error] Assumptions on abundance ratios are suited to the sandwich geometry. Please, add "sandwich" in constraints_geometry.\n{bcolors.ENDC}')

if '12co-13co' in constraints_abundances:
    assert ('12co' in names_mol) and ('13co' in names_mol) and ('hcop' in names_mol) and ('h13cop' in names_mol), print(
        f'{bcolors.WARNING}\n [error] The constraint 12co-13co that imposes N(H13CO+) = N(HCO+) x N(13CO)/N(12CO) seems not required here. Please remove it or add the missing molecular species.\n{bcolors.ENDC}')

    assert (names_mol.index('h13cop') > names_mol.index('12co')) and (names_mol.index('h13cop') > names_mol.index('13co')) and (names_mol.index('h13cop') > names_mol.index('hcop')), print(
    f'{bcolors.WARNING}\n [error] Please, order the studied species and lines such as h13cop is treated after 12co, 13co and hcop.\n{bcolors.ENDC}')

    assert names_mol.index(name_mol_ref_) == names_mol.index('13co'), print(
        f'{bcolors.WARNING}\n [error] Please, assign 13co as the molecule of reference.\n{bcolors.ENDC}')

for layer in range(layers):
    assert len(colden_ratios_shifts[name_mol_ref_][layer]) == 1, print(
        f'{bcolors.WARNING}\n [error] colden_ratios_shifts should be a fixed single value for the species of reference.\n{bcolors.ENDC}')
    assert colden_ratios_shifts[name_mol_ref_][layer] == 0, print(
        f'{bcolors.WARNING}\n [error] colden_ratios_shifts of the reference species should be 0.\n{bcolors.ENDC}')

# kinematics
if 'mirror' in constraints_kinematics : 
    assert 'sandwich' in constraints_geometry, print(
        f'{bcolors.WARNING}\n [error] The "mirror" constraint for centroid velocities is only implemented for sandwich geometry, so far.\n{bcolors.ENDC}')

# the optimization process
walkers_per_step_ = walkers_per_step[0]
for idx_step in range(1, len(walkers_per_step)) : 
    assert walkers_per_step[idx_step] <= walkers_per_step_, print(
        f'{bcolors.WARNING}\n [error] The number of walkers have to decrease as the step increases.\n{bcolors.ENDC}')
    walkers_per_step_ = walkers_per_step[idx_step]

assert len(walkers_per_step) == len(iterations_per_step), print(
        f'{bcolors.WARNING}\n [error] Please, indicate for each step, how many walkers and iterations when exploring the solution space.\n{bcolors.ENDC}')
assert len(walkers_per_step) > 1, print(
        f'{bcolors.WARNING}\n [error] Please, set at least two steps. The first one is dedicated to a simple draw of the walkers, while the exploration start from the second step.\n{bcolors.ENDC}')

#%% derive useful quantities from the settings (DO NOT MODIFY)

# layer and geometry 
if 'sandwich' in constraints_geometry : 
    unique_layer_idx = []
    idx_sandwich = 0 # first sandwich
    cursor = 0 
    for idx_sandwich in range(number_of_clumps) : 
        unique_layer_idx += [cursor * idx_sandwich + i for i in range(0, number_of_layers_per_clump // 2 + 1)]
        unique_layer_idx += [cursor * idx_sandwich + i for i in range(number_of_layers_per_clump // 2 - 1, - 1, -1)]
        cursor += len(range(number_of_layers_per_clump//2 + 1))
    del idx_sandwich
else:
    unique_layer_idx = [i for i in range(0, layers)]
number_of_different_layers = len(np.unique(unique_layer_idx))

# kinematics 
if ('sandwich' in constraints_geometry):
    idx_first_inner_layer = number_of_layers_per_clump // 2 
    idxs_inner_layer = [idx_first_inner_layer + i * number_of_layers_per_clump for i in range(number_of_clumps)]
else : 
    idxs_inner_layer = []
if 'same_C_V_in_all_layers' in constraints_kinematics: 
    number_of_unknown_C_V = 1
else : 
    # for the Fisher information matrix
    if ('sandwich' in constraints_geometry) and ('mirror' in constraints_kinematics) : 
        number_of_unknown_C_V = len(idxs_inner_layer) + (number_of_layers_per_clump // 2) * number_of_clumps
    else :
        number_of_unknown_C_V = number_of_different_layers

# column density ratios, i.e., abundances
# to get the indexes of species x layers whose the abundance ratios are estimated 
idx_unknown_mol_layer = [] 
i = 0 
for layer_ in range(number_of_different_layers) : 
    layer = unique_layer_idx.index(layer_)

    for mol_idx, mol in enumerate(names_mol) : 
        if mol == name_mol_ref_ : 
            pass # do not accounting the colden of the molecule of reference as it will be estimated besides
        else : 
            if colden_ratios_shifts[mol][layer].size > 1 : 
                idx_unknown_mol_layer.append(i)
        i += 1
idx_unknown_mol_layer = np.array(idx_unknown_mol_layer)

if '12co-13co' in constraints_abundances : 
    idx_12co = names_mol.index('12co')
    idx_hcop = names_mol.index('hcop')

# the optimization process
if 'gd' in optimization : 
    # total number of unknowns
    if 'same_C_V_in_all_layers' in constraints_kinematics:
        number_of_unknowns = (len(theta) - 1) * number_of_different_layers + 1
    else : 
        number_of_unknowns = len(theta) * number_of_different_layers
