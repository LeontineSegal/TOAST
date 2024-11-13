'''
This file allows one to transform the studied dataset to the format expected by the routine code. 
to launch it, type in your terminal: python prepare_dataset.py
'''

# %% load modules
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '.', '..'))

# to deal with .fits files
import astropy.units as u
from astropy.io import fits
from spectral_cube import SpectralCube

import numpy as np

# TOAST toolboxs
from toolboxs.toolbox_fits import convert_name_mol_line
from toolboxs.toolbox_python import bcolors, from_list_to_array
from toolboxs.toolbox_plot import plot_percentile_mol_line, plot_maps
from toolboxs.toolbox_crb.toolbox_crb import snr_trick

# to clean terminal
import warnings
warnings.filterwarnings("ignore")

# %% paths & settings (TO MODIFY)

#path_base = os.getcwd()
path_base = '***/TOAST/examples'

VERBOSE = True
OVERWRITE = True
FORMAT_FILE = '.pdf' # for plot 
DATASET = 'horsehead' # name it as you want, according to the different datasets you are studying

# assumptions on noise
snr_threshold = 10 # for S/N trick (10 recommended)

# %% import the dataset of ppv cubes

path_inputs = f'{path_base}/data-{DATASET}/inputs'
path_raw_ppv = f'{path_base}/data-{DATASET}/raw-ppv' # original dataset
path_ppv = f'{path_base}/data-{DATASET}/ppv' # modified dataset (e.g., if you are focusing on a smaller frequency bandwidth)

#%% process the dataset 

if DATASET == 'horsehead' : # for examples

    fits_raw_ppv = [
        f'{path_raw_ppv}/12co10.fits',
        f'{path_raw_ppv}/13co10.fits',
        f'{path_raw_ppv}/13co21.fits',
        f'{path_raw_ppv}/c18o10.fits',
        f'{path_raw_ppv}/c18o21.fits',
        f'{path_raw_ppv}/hcop10.fits',
        f'{path_raw_ppv}/h13cop10.fits'
    ]

    # get useful information about the data
    spectralcube_raw_ppv = [SpectralCube.read(file).with_spectral_unit(
    u.km/u.s) for file in fits_raw_ppv]  # from [m/s] to [km/s]
    raw_ppv = [np.swapaxes(fits.open(file)[0].data, 0, 1) for file in fits_raw_ppv]
    raw_ppv = [np.swapaxes(cube, 1, 2) for cube in raw_ppv]

    names_mol_line_fits = [fits.open(file)[0].header['Line']
                       for file in fits_raw_ppv]  # molecular line names in fits
    names_mol_line_radex = [convert_name_mol_line(
    mol_line, from_file='fits', to_file='radex', dataset = DATASET) for mol_line in names_mol_line_fits]  # in radex
    names_mol_line_latex = [convert_name_mol_line(
    mol_line, from_file='fits', to_file='latex', dataset = DATASET) for mol_line in names_mol_line_fits]  # in latex
    rest_frequencies = [round(fits.open(file)[0].header['RESTFREQ'] * 10**-9, 6)
                    for file in fits_raw_ppv]  # rest frequencies from [Hz] to [GHz]

    # increase the digits for the (2-1) transitions
    rest_frequencies[names_mol_line_radex.index('13co(2-1)')] = 220.3986842
    rest_frequencies[names_mol_line_radex.index('c18o(2-1)')] = 219.5603541

    # rest velocities from [m/s] to [km/s]
    rest_velocities = [round(fits.open(file)[0].header['VELO-LSR']
                   * 10**-3, 3) for file in fits_raw_ppv]
    # observations
    velocity_res = [round(fits.open(file)[0].header['CDELT3'], 3) *
                    10**-3 for file in fits_raw_ppv]  # velocity resolutions [km/s]
    
    velocity_channels = [np.array(np.around(cube.world[:, 0, 0][0], 2))
                     for cube in spectralcube_raw_ppv]  # velocity channels from [m/s] to [km/s]

    # check
    if VERBOSE:
        print(f'\n{bcolors.OKCYAN}RAW DATASET{bcolors.ENDC}\n')
        for line_idx, line in enumerate(names_mol_line_radex):
            print(
            f'{bcolors.OKCYAN}{repr(line)}{bcolors.ENDC}',
            f'shape of the ppv cube: {repr(np.shape(raw_ppv[line_idx]))}',
            f'rest frequency: {repr(rest_frequencies[line_idx])} GHz',
            f'rest velocity: {repr(rest_velocities[line_idx])} km/s',
            f'velocity channels: [{repr(float(velocity_channels[line_idx][0]))}, {repr(float(velocity_channels[line_idx][-1]))}] km/s',
            f'velocity resolution: {repr(velocity_res[line_idx])} km/s -> {len(velocity_channels[line_idx])} channels\n', 
            sep = '\n'
            )

    # post-process the dataset

    # look at the percentiles of the line profiles over the field of view
    percentile_spectra = []

    for mol_line_idx, cube in enumerate(raw_ppv):
        quantile_spectra_mol_line = np.zeros((4, np.shape(cube)[-1]))
        flatten_cube = cube.reshape(
            (np.shape(cube)[0]*np.shape(cube)[1], np.shape(cube)[2]))
        percentile_spectra.append(np.nanpercentile(
            flatten_cube, [5, 50, 95], axis=0))

    # plot
    C_V=[10.5 for i in range(len(percentile_spectra))] # km/s
    s_V=[1.5 for i in range(len(percentile_spectra))] # km/s

    name_fig = 'percentile_line_profiles'
    fig = plot_percentile_mol_line(percentile_spectra,
                                velocity_channels,
                                names_mol_line_latex,
                                C_V=C_V,  # km/s
                                s_V=s_V  # km/s
                                )
    fig.savefig(f'{path_raw_ppv}/{name_fig}{FORMAT_FILE}')

    # reduce the bandwidth to keep only channels where there is signal 
    # and thus reduce the number of velocity channels
    vmin, vmax = 7, 16 # km/s

    windowed_ppv = []
    windowed_spectralcube_ppv = []

    for cube_idx, cube in enumerate(spectralcube_raw_ppv):
        windowed_spectralcube_ppv.append(cube.spectral_slab(vmin * u.km / u.s, vmax * u.km / u.s)) 
        numpy_cube = np.array(windowed_spectralcube_ppv[cube_idx][:])
        numpy_cube = np.swapaxes(numpy_cube, 0, 1)
        numpy_cube = np.swapaxes(numpy_cube, 1, 2)
        windowed_ppv.append(numpy_cube)

    # updating the velocity channels of each ppv cube
    windowed_velocity_channels = [np.array(np.around(cube.world[:, 0, 0][0], 2)) for cube in windowed_spectralcube_ppv] # velocity channels from [m/s] to [km/s]

    # plot
    percentile_spectra = []
    for mol_line_idx, cube in enumerate(windowed_ppv):
        quantile_spectra_mol_line = np.zeros((4, np.shape(cube)[-1]))
        flatten_cube = cube.reshape((np.shape(cube)[0]*np.shape(cube)[1], np.shape(cube)[2]))
        percentile_spectra.append(np.nanpercentile(flatten_cube, [5, 50, 95], axis=0))

    name_fig = 'percentile_line_profiles'
    fig = plot_percentile_mol_line(percentile_spectra, 
                                windowed_velocity_channels, 
                                names_mol_line_latex, 
                                C_V = [10.5 for i in range(len(percentile_spectra))], # km/s
                                s_V = [1.5 for i in range(len(percentile_spectra))] # km/s
                                    )
    try : 
        fig.savefig(f'{path_ppv}/{name_fig}{FORMAT_FILE}')
    except : 
        os.system(f'mkdir {path_ppv}')
        fig.savefig(f'{path_ppv}/{name_fig}{FORMAT_FILE}')

    # save the post processed ppv cubes
    if OVERWRITE:
        for cube_idx, sub_cube in enumerate(windowed_spectralcube_ppv):
            file = fits_raw_ppv[cube_idx].replace('raw-ppv', 'ppv')
            file = file.replace('.fits', '')
            sub_cube.write(f'{file}.fits', overwrite=True) 

    # formatting the post processed ppv cubes
    fits_ppv = [ 
        f'{path_ppv}/12co10.fits',
        f'{path_ppv}/13co10.fits',
        f'{path_ppv}/13co21.fits',
        f'{path_ppv}/c18o10.fits',
        f'{path_ppv}/c18o21.fits',
        f'{path_ppv}/hcop10.fits',
        f'{path_ppv}/h13cop10.fits'
    ]

    # get useful information about the data
    spectralcube_ppv = [SpectralCube.read(file).with_spectral_unit(
        u.km/u.s) for file in fits_ppv]  # from [m/s] to [km/s]
    ppv = [np.swapaxes(fits.open(file)[0].data, 0, 1) for file in fits_ppv]
    ppv = [np.swapaxes(cube, 1, 2) for cube in ppv]

    names_mol_line_fits = [fits.open(file)[0].header['Line']
                        for file in fits_ppv]  # molecular line names in fits
    names_mol_line_radex = [convert_name_mol_line(
        mol_line, from_file='fits', to_file='radex', dataset = DATASET) for mol_line in names_mol_line_fits]  # in radex
    names_mol_line_latex = [convert_name_mol_line(
        mol_line, from_file='fits', to_file='latex', dataset = DATASET) for mol_line in names_mol_line_fits]  # in latex
    rest_frequencies = [round(fits.open(file)[0].header['RESTFREQ'] * 10**-9, 6)
                        for file in fits_ppv]  # rest frequencies from [Hz] to [GHz]

    # increase the digits for the (2-1) transitions
    rest_frequencies[names_mol_line_radex.index('13co(2-1)')] = 220.3986842
    rest_frequencies[names_mol_line_radex.index('c18o(2-1)')] = 219.5603541

    # rest velocities from [m/s] to [km/s]
    rest_velocities = [round(fits.open(file)[0].header['VELO-LSR']
                    * 10**-3, 3) for file in fits_ppv]
    velocity_res = [round(fits.open(file)[0].header['CDELT3'], 3) for file in fits_ppv]  # velocity resolutions [km/s]
    velocity_channels = [np.array(np.around(cube.world[:, 0, 0][0], 2))
                        for cube in spectralcube_ppv]  # velocity channels from [m/s] to [km/s]

    # thermal noise 
    # get from CUBE 
    maps_s_b = [
        f'{path_raw_ppv}/noise/12co10-noise.fits',
        f'{path_raw_ppv}/noise/13co10-noise.fits',
        f'{path_raw_ppv}/noise/13co21-noise.fits',
        f'{path_raw_ppv}/noise/c18o10-noise.fits',
        f'{path_raw_ppv}/noise/c18o21-noise.fits',
        f'{path_raw_ppv}/noise/hcop10-noise.fits',
        f'{path_raw_ppv}/noise/h13cop10-noise.fits'
    ]

    maps_s_b = [np.swapaxes(fits.open(file)[0].data, 0, 1) for file in maps_s_b]
    maps_s_b = [np.swapaxes(map, 1, 2) for map in maps_s_b]
    maps_s_b = [map[:, :, 0] for map in maps_s_b]

    # plot 
    name_fig = 'maps_s_b'
    fig = plot_maps(
        maps_s_b, 
        names_mol_line_latex
        )
    fig.savefig(f'{path_raw_ppv}/noise/{name_fig}{FORMAT_FILE}')
    (f'{path_ppv}/{name_fig}{FORMAT_FILE}')

    # thermal noise with snr trick 
    maps_sb_snr_trick = []
    for ppv_idx, ppv_ in enumerate(ppv) :
        maps_sb_snr_trick.append(snr_trick(ppv_, 
                                            maps_s_b[ppv_idx], 
                                            snr_threshold
                                            ))
    # plot
    name_fig = 'maps_sb_snr_trick'
    fig = plot_maps(
        maps_sb_snr_trick, 
        names_mol_line_latex)
    fig.savefig(f'{path_ppv}/{name_fig}{FORMAT_FILE}')

    # check
    if VERBOSE:
        print(f'\n{bcolors.OKCYAN}POST PROCESSED DATASET{bcolors.ENDC}\n')
        for line_idx, line in enumerate(names_mol_line_radex):
            print(
                f'{bcolors.OKCYAN}{repr(line)}{bcolors.ENDC}',
                f'shape of the (p,p,v) cube: {repr(np.shape(ppv[line_idx]))}',
                f'rest frequency: {repr(rest_frequencies[line_idx])} GHz',
                f'rest velocity: {repr(rest_velocities[line_idx])} km/s',
                f'velocity channels: [{repr(float(velocity_channels[line_idx][0]))}, {repr(float(velocity_channels[line_idx][-1]))}] km/s',
                f'velocity resolution: {repr(velocity_res[line_idx])} km/s -> {len(velocity_channels[line_idx])} channels', 
                f'median thermal noise dispersion = 'f': {repr(round(float(np.nanmedian(maps_s_b[line_idx])), 5))}',
                f'median thermal noise dispersion (snr trick) = 'f': {repr(round(float(np.nanmedian(maps_sb_snr_trick[line_idx])), 5))}\n',
                sep = '\n'
            )

    # formating the data to TOAST
    names_mol, names_line = [], []
    f_names_mol_line_latex = []
    f_rest_frequencies = [] 
    f_rest_velocities = []
    f_velocity_res = [] 
    f_velocity_channels = []
    f_ppv = []
    f_maps_sb = []
    f_maps_sb_snr_trick = []

    for mol_line_idx, name_mol_line in enumerate(names_mol_line_radex):

        name_mol_line_latex = names_mol_line_latex[mol_line_idx]
        rest_frequency = rest_frequencies[mol_line_idx]
        rest_velocity = rest_velocities[mol_line_idx]
        velocity_res_ = velocity_res[mol_line_idx]
        velocity_channels_ = velocity_channels[mol_line_idx]
        velocity_channels_ = velocity_channels_.reshape(1, len(velocity_channels_))

        ppv_ = ppv[mol_line_idx]
        map_sb = maps_s_b[mol_line_idx]
        map_sb_snr_trick = maps_sb_snr_trick[mol_line_idx]

        name_mol, name_line = name_mol_line.split('(')[0], name_mol_line.split('(')[1]
        name_line = name_line.split(')')[0]

        if name_mol not in names_mol:  # adding the molecule to the list of studied molecules
            names_mol.append(name_mol)
                
            names_line.append([])
            names_line[-1].append(name_line)

            f_names_mol_line_latex.append([])
            f_names_mol_line_latex[-1].append(name_mol_line_latex)

            f_rest_frequencies.append([])
            f_rest_frequencies[-1].append(rest_frequency)

            f_rest_velocities.append([])
            f_rest_velocities[-1].append(rest_velocity)

            f_velocity_res.append([])
            f_velocity_res[-1].append(velocity_res_)

            f_velocity_channels.append([])
            f_velocity_channels[-1].append(velocity_channels_)

            f_ppv.append([])
            f_ppv[-1].append(ppv_)

            f_maps_sb.append([])
            f_maps_sb[-1].append(map_sb)

            f_maps_sb_snr_trick.append([])
            f_maps_sb_snr_trick[-1].append(map_sb_snr_trick)
        
        else:
            name_mol_idx = names_mol.index(name_mol)
            names_line[name_mol_idx].append(name_line)

            f_names_mol_line_latex[name_mol_idx].append(name_mol_line_latex)

            f_rest_frequencies[name_mol_idx].append(rest_frequency)

            f_rest_velocities[name_mol_idx].append(rest_velocity)

            f_velocity_res[name_mol_idx].append(velocity_res_)

            f_velocity_channels[name_mol_idx].append(velocity_channels_)

            f_ppv[name_mol_idx].append(ppv_)

            f_maps_sb[name_mol_idx].append(map_sb)

            f_maps_sb_snr_trick[name_mol_idx].append(map_sb_snr_trick)

else : # your own dataset
    # customize the process following your .fits by inspiring you from the above code. 
    pass

# whatever the dataset is, save the data as the following format 
if OVERWRITE:
    try : 
        np.save(f'{path_inputs}/names_mol', names_mol)
    except : 
        os.system(f'mkdir {path_inputs}')
    np.save(f'{path_inputs}/names_mol', names_mol)
    np.save(f'{path_inputs}/names_line', from_list_to_array(names_line, inhomogeneous=True))
    np.save(f'{path_inputs}/names_mol_line_latex', from_list_to_array(f_names_mol_line_latex, inhomogeneous=True))
    np.save(f'{path_inputs}/rest_frequencies', from_list_to_array(f_rest_frequencies, inhomogeneous=True))
    np.save(f'{path_inputs}/rest_velocities', from_list_to_array(f_rest_velocities, inhomogeneous=True))
    np.save(f'{path_inputs}/velocity_res', from_list_to_array(f_velocity_res, inhomogeneous=True))
    np.save(f'{path_inputs}/velocity_channels', from_list_to_array(f_velocity_channels, inhomogeneous=True))
    np.save(f'{path_inputs}/ppv', from_list_to_array(f_ppv, inhomogeneous=True))
    np.save(f'{path_inputs}/maps_s_b', from_list_to_array(f_maps_sb, inhomogeneous=True))
    np.save(f'{path_inputs}/maps_s_b_snr_trick', from_list_to_array(f_maps_sb_snr_trick, inhomogeneous=True))
