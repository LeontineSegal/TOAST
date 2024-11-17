'''
This routine allows to create excitation temperature (Tex) and opacity (tau) grids with RADEX.
to launch it, type in your terminal: python launch_radex_grids.py
'''

# %% load modules
import numpy as np
import tqdm
import os

from toolboxs.toolbox_radex.toolbox_radex import get_freq_bandwidth, from_OPR_to_densities, execute_radex
from toolboxs.toolbox_python import from_list_to_array

# %% paths & settings (TO MODIFY)
OVERWRITE = True # to save the grids 
RUN = True # to run the radex execution  
DATASET = 'horsehead' # name it as you want, according to the different datasets you are studying

# explicit path to avoid conflict when reload the model.py as a module 
path_base = '***/TOAST'
path_inputs = f'{path_base}/data-{DATASET}/inputs'  # all information about the dataset
path_save = f'{path_base}/data-{DATASET}/grids-Tex-tau' 

# radex settings
geometry = 'radexUnifSphere'  # escape probability approximation, chose among {'radexUnifSphere', 'radexExpSphere', 'radexParaSlab'}
path_save += f'/{geometry}'
os.system(f'mkdir -p {path_save}')

colliders = ['p-H2', 'o-H2', 'e']

# all the dataset
dataset_names_mol = np.load(f'{path_inputs}/names_mol.npy', allow_pickle=True)
dataset_names_mol = list(dataset_names_mol)
dataset_names_line = np.load(f'{path_inputs}/names_line.npy', allow_pickle=True)
dataset_names_line = list(dataset_names_line)


# choose the molecular lines (TO MODIFY)

# all the studied dataset directly...
names_mol = dataset_names_mol
names_line = dataset_names_line

# ... some additionnal lines, for instance : 
"""
names_mol = [
            '12co', 
            '13co'
             ]

names_line = [
        ['2-1'], 
        ['3-2', '4-3']
]
"""

# physical space to explore (TO MODIFY)

# kinetic temperature 
min_log10_T_kin, max_log10_T_kin = np.log10(4), np.log10(113) # log(K)
# H2 volume density 
min_log10_nH2, max_log10_nH2 = 2., 6.5 # log(cm-3)
# column density  of the molecular species of reference (13co, here)
name_mol_ref = '13co'
min_log10_N_ref, max_log10_N_ref  = 13., 18. # log(cm-2)
# column density ratios (relative abundances) in respect to the molecular species of reference
# indicate log[alpha], where log[N(x)] = log[alpha] + log[N(name_mol_ref)]
if OVERWRITE:
    colden_ratios = {
                    '12co': 1.8, 
                    '13co': 0.,                 
                    'c18o': -0.9,
                    'hcop': - 2.88,
                    'h13cop': - 4.68, 
                    }
    np.save(f'{path_inputs}/colden_ratios', colden_ratios)
# FWHM 
min_FWHM, max_FWHM  = 0.25, 2. # [km/s]

# define the space sampling method between 
# 1) constrain the sample step
# 2) constrain the number of samples per dimension

sampling_method = 1

if sampling_method == 1 : 
        
        step_log10_T_kin = 0.05
        step_log10_nH2 = 0.1 
        step_log10_N_ref = 0.05
        step_FWHM = 0.05

        axis_log10_T_kin = np.arange(
                min_log10_T_kin, max_log10_T_kin, step_log10_T_kin) # start, stop (excluded), sample step
        axis_log10_nH2 = np.arange(
                min_log10_nH2, max_log10_nH2 + step_log10_nH2, step_log10_nH2)
        axis_log10_N_ref = np.arange(
                min_log10_N_ref, max_log10_N_ref + step_log10_N_ref, step_log10_N_ref)
        #print(axis_log10_N_ref) # check wether the min max are the values expected. Here the axis reaches 18.05 instead of 18.
        axis_log10_N_ref = axis_log10_N_ref[:-1] # remove the last sample to reach 18.
        axis_FWHM = np.arange(
                min_FWHM, max_FWHM + step_FWHM, step_FWHM) 

        # round 
        '''
        axis_log10_T_kin = np.round(axis_log10_T_kin, 2)
        axis_log10_nH2 = np.round(axis_log10_nH2, 2)
        axis_log10_N_ref = np.round(axis_log10_N_ref, 2)
        axis_FWHM = np.round(axis_FWHM, 2)
        '''

elif sampling_method == 2 :
         
        axis_log10_T_kin = np.linspace(
                min_log10_T_kin, max_log10_T_kin, num = 3) # start, stop (excluded), samples
        axis_log10_nH2 = np.linspace(
                min_log10_nH2, max_log10_nH2, num = 3)
        axis_log10_N_ref = np.linspace(
                min_log10_N_ref, max_log10_N_ref, num = 3)
        axis_FWHM = np.linspace(
                min_FWHM, max_FWHM, num = 3) 
        
FPS = from_list_to_array([axis_log10_T_kin, axis_log10_nH2, axis_log10_N_ref, axis_FWHM], inhomogeneous=True) 

if OVERWRITE:
    np.save(f'{path_save}/FPS.npy', FPS)

print(f'\n log(Tkin) : {min(axis_log10_T_kin)} -> {max(axis_log10_T_kin)}, {len(axis_log10_T_kin)} samples')
print(f'\n log(nH2) : {min(axis_log10_nH2)} -> {max(axis_log10_nH2)}, {len(axis_log10_nH2)} samples')
print(f'\n log(N_ref) : {min(axis_log10_N_ref)} -> {max(axis_log10_N_ref)}, {len(axis_log10_N_ref)} samples')
print(f'\n FWHM : {min(axis_FWHM)} -> {max(axis_FWHM)}, {len(axis_FWHM)} samples')

#%% run the radex execution 

# load information 
rest_frequencies = np.load(f'{path_inputs}/rest_frequencies.npy', allow_pickle=True)
samples = [len(axis_log10_T_kin), len(axis_log10_nH2), len(axis_log10_N_ref), len(axis_FWHM)]

if RUN and OVERWRITE:
    
    import warnings
    warnings.filterwarnings("ignore")

    for name_mol_idx, name_mol  in enumerate(names_mol) : 

        with tqdm.tqdm(total=len(names_line[name_mol_idx])*samples[0]*samples[1]*samples[2]) as pbar:

            name_mol_idx_ = dataset_names_mol.index(name_mol)

            grid_T_ex = np.zeros(
            (samples[0], samples[1], samples[2], samples[3], len(names_line[name_mol_idx])))
            grid_tau = np.zeros(
            (samples[0], samples[1], samples[2], samples[3], len(names_line[name_mol_idx])))

            for name_line_idx, name_line in enumerate(names_line[name_mol_idx]) : 
                name_line_idx_ = dataset_names_line[name_mol_idx_].index(name_line)

                freq = rest_frequencies[name_mol_idx_][name_line_idx_]
                min_freq, max_freq = get_freq_bandwidth(
                freq)[0], get_freq_bandwidth(freq)[1]

                for log10_T_kin_idx, log10_T_kin in enumerate(axis_log10_T_kin):
                    for log10_nH2_idx, log10_nH2 in enumerate(axis_log10_nH2):
                        for log10_N_ref_idx, log10_N_ref in enumerate(axis_log10_N_ref):

                            # column density of the considered species
                            log10_N = log10_N_ref + colden_ratios[name_mol]
                            
                            # avoid enumerate FWHM axis to save a little of time 
                            vector_log10_T_kin = log10_T_kin * np.ones((len(axis_FWHM)))
                            vector_T_kin = 10 ** vector_log10_T_kin
                            vector_log10_nH2 = log10_nH2 * \
                            np.ones((len(axis_FWHM)))
                            vector_log10_N = log10_N * np.ones((len(axis_FWHM)))

                            collider_densities = np.zeros(
                            (len(colliders), vector_log10_T_kin.size))
                            for c_idx, c in enumerate(from_OPR_to_densities(vector_log10_T_kin, vector_log10_nH2, colliders=colliders)):
                                    collider_densities[c_idx, :] = 10**c

                            T_ex, tau, T_r = execute_radex(name_mol,
                                    f'grid_{name_mol}_{name_line}_{geometry}',
                                    min_freq,
                                    max_freq,
                                    10**vector_log10_T_kin,
                                    colliders,
                                    collider_densities,
                                    10**vector_log10_N,
                                    axis_FWHM,
                                    geometry,
                                    clean=False)
                            
                            grid_T_ex[log10_T_kin_idx, log10_nH2_idx,
                                    log10_N_ref_idx, :, name_line_idx] = T_ex[:, 0]
                            grid_tau[log10_T_kin_idx, log10_nH2_idx,
                                    log10_N_ref_idx, :, name_line_idx] = tau[:, 0]

                            np.save(f'{path_save}/Tex_{name_mol}.npy',
                                    grid_T_ex)
                            np.save(f'{path_save}/tau_{name_mol}.npy',
                                    grid_tau)

                            pbar.update(1)

                # clean workspace 
                os.system(f'rm -f grid_{name_mol}_{name_line}_{geometry}.*')
                os.system('rm -f radex.log')
