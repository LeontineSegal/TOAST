# %% load modules
from typing import Optional, List
from numbers import Real

import numpy as np
import os

from toolboxs.toolbox_python.toolbox_python import check_type, check_shape, check_length
from toolboxs.toolbox_python.toolbox_python import from_list_to_array, from_float_to_string

# constants
FREQ_MARGIN = 0.01
from toolboxs.toolbox_physics.toolbox_radiative_transfer import T_BG

#%% physics

def get_freq_bandwidth(freq: float, 
                       margin: Optional[float] = FREQ_MARGIN) -> tuple[float, float]:
    min_freq = freq*(1 - margin)
    max_freq = freq*(1 + margin)
    return min_freq, max_freq

def from_OPR_to_densities(T_kin: np.ndarray, 
                          log10_nH2: np.ndarray, 
                          colliders: Optional[List] = ['p-H2', 'o-H2', 'e']) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the ortho and para ratios.

    :param T_kin: (K) kinetic temperature 
    :type T_kin: np.ndarray
    :param log10_nH2: H2 density
    :type log10_nH2: np.ndarray
    :param colliders: considered colliders among 'p-H2', 'o-H2', 'e'. All by default.
    :return: log10_pH2, log10_oH2, log10_e
    :rtype: List[np.ndarray, np.ndarray, np.ndarray]
    """
    assert check_type(T_kin, 
                      log10_nH2, 
                      type_of_reference=np.ndarray,
                      from_function='from_OPR_to_densities')
    assert check_shape(T_kin, 
                       log10_nH2, 
                       from_function='from_OPR_to_densities')

    grid_length = len(T_kin)
    log10_pH2 = np.zeros(grid_length)
    log10_oH2 = np.zeros(grid_length)
    log10_e = np.zeros(grid_length)
    for i in range(grid_length):
        OPR = 3*(2. + 1) * np.exp(- 170.502/T_kin[i]) + 3*(
            2.*3. + 1) * np.exp(- 1015.153/T_kin[i]) + 3*(2.*5. + 1) * np.exp(- 2503.870/T_kin[i])
        OPR = OPR / (1. + (2.*2. + 1)*np.exp(- 509.850 /
                                             T_kin[i]) + (2.*4. + 1)*np.exp(- 1681.678/T_kin[i]))
        OPR = max(10**(-3), min(3., OPR))  # from Javier R. Goicoechea

        log10_pH2_i = log10_nH2[i] - np.log10(1 + OPR)
        log10_oH2_i = log10_nH2[i] - np.log10(1 + 1./OPR)
        log10_e_i = np.log10(2) - 3 + 0.5*log10_nH2[i]
        llog10_e_i = -8 + log10_nH2[i]
        log10_e_i = max(log10_e_i, llog10_e_i)

        log10_pH2[i] = log10_pH2_i
        log10_oH2[i] = log10_oH2_i
        log10_e[i] = log10_e_i

    result = []
    if 'p-H2' in colliders:
        result.append(log10_pH2)
    if 'o-H2' in colliders:
        result.append(log10_oH2)
    if 'e' in colliders:
        result.append(log10_e)

    return result

#%%
def add_inp_extension(file: str) -> str:
    """Add the '.inp' extension to a file name. 

    :param str file: file name without extension (ex : 'gradient_computation')
    :type file: str
    :return str: 'file'.inp
    :rtype: str
    """
    return f'{file}.inp'


def add_out_extension(file: str) -> str:
    """Add the '.out' extension to a file name. 

    :param str file: file name without extension (ex : 'gradient_computation')
    :type file: str
    :return str: 'file'.out
    :rtype: str
    """
    return f'{file}.out'


def get_dat_file(molecule: str) -> str:
    """Return the .dat file of the corresponding molecule. 

    :param molecule: molecule name (ex : '12CO')
    :type molecule: str
    :return: .dat file name (ex : '12co.dat')
    :rtype: str
    """
    molecules = ['12co', 
                 '13co', 
                 'c18o', 
                 'hcop', 
                 'h13cop', 
                 ]
    dat_files = ['12co.dat', 
                 '13co.dat',
                 'c18o.dat', 
                 'hcop-h2-e.dat', 
                 'h13cop-h2-e.dat', 
                 ]
    idx = molecules.index(molecule)
    return dat_files[idx]

#%% radex execution
def write_inp_file(
        molecule: str,
        file: str,
        min_freq: float,
        max_freq: float,
        T_kin: np.ndarray,
        colliders: list[str],
        collider_densities: np.ndarray,
        N: np.ndarray,
        FWHM: np.ndarray):
    """Write a Radex input file (format .inp).

    :param molecule: molecule name (ex : '12co')
    :type molecule: str
    :param file: file name without extension (ex: 'gradient_computation')
    :type file: str
    :param min_freq: (GHz) minimal frequency (Radex parameter)
    :type min_freq: float
    :param max_freq: (GHz) maximal frequency (Radex parameter)
    :type max_freq: float
    :param T_kin: kinetic temperature (K)
    :type T_kin: np.ndarray
    :param colliders: considered collider parterns ('H2', 'p-H2', 'o-H2', 'e', 'H', 'He', 'H+')
    :type colliders: list[str]
    :param collider_densities: (cm^-3) densities of the considered collider parterns
    :type collider_densities: np.ndarray of shape (len(colliders) x grid_length)
    :param N: (cm^-2) column density
    :type N: np.ndarray
    :param FWHM: (km/s) full width at half maximum
    :type FWHM: np.ndarray
    """
    assert check_type(T_kin, collider_densities, N, FWHM,
                      type_of_reference=np.ndarray, from_function='write_inp_file')
    for c_idx in range(len(collider_densities)):
        assert check_length(
            T_kin, collider_densities[c_idx, :], N, FWHM, from_function='write_inp_file')
    assert check_shape(collider_densities, shape_of_reference=(
        len(collider_densities), len(T_kin)), from_function='write_inp_file')

    parameters = []
    grid_length = len(T_kin)
    dat_file = get_dat_file(molecule)
    out_file = add_out_extension(file)

    for i in range(grid_length):
        parameters.append(dat_file)
        parameters.append(out_file)
        parameters.append(from_float_to_string(min_freq) +
                          ' '+from_float_to_string(max_freq))
        parameters.append(from_float_to_string(T_kin[i]))
        parameters.append(len(colliders))
        for idx, c in enumerate(colliders):
            colliders_i = collider_densities[:, i]
            collider_idx = np.where(~np.isnan(colliders_i))[0]
            parameters.append(c)
            parameters.append(from_float_to_string(
                colliders_i[collider_idx[idx]], format='exp'))
        parameters.append(T_BG)
        parameters.append(from_float_to_string(N[i], format='exp'))
        parameters.append(from_float_to_string(FWHM[i]))
        if i == (grid_length-1):
            parameters.append(0)
        else:
            parameters.append(1)

    with open(add_inp_extension(file), 'w') as inp_file:
        for param in parameters:
            inp_file.write(f"{param}\n")

    inp_file.close()
    return 1


def run_radex(geometry: str, file: str):
    """Run the shell command radex.

    :param geometry: used geometry among {'radexUnifSphere','radexExpSphere','radexParaSlab'}
    :type geometry: str
    :param file: file name without extension (ex: 'gradient_computation')
    :type file: str
    """
    os.system(str(geometry)+' < ' + add_inp_extension(file)+' > /dev/null')
    return 1


def read_out_file(file: str, colliders: list[str]) -> tuple[np.ndarray,  np.ndarray,  np.ndarray]:
    """Write a radex output file (format .out)

    :param file: file name without extension (ex: 'gradient_computation')
    :type file: str
    :param colliders: names of the considered collider parterns ('H2', 'p-H2', 'o-H2', 'e', 'H', 'He', 'H+')
    :type colliders: list[str]
    :return: (K) excitation temperature, opacity, (K) Rayleigh–Jeans or brightness temperature
    :rtype: np.ndarray, np.ndarray, np.ndarray
    """
    QN_ul = []
    E_UP = []
    freq = []
    WAVEL = []
    T_ex = []
    opacity = []
    T_r = []

    out_file = open(add_out_extension(file), 'r')
    lines = out_file.readlines()
    radex_version_idx = np.where(
        [r'Radex version' in l for l in lines])[0].tolist()
    number_of_files = np.size(radex_version_idx)
    radex_version_idx.append(len(lines))  # ending line index

    number_of_colliders = 1  # H2 whatever
    if np.logical_or(np.logical_and(r'p-H2' in colliders, r'o-H2' not in colliders), np.logical_and(r'o-H2' in colliders, r'p-H2' not in colliders)):
        number_of_colliders += 1  # p-H2 or o-H2 is dominant
    elif np.logical_and(r'p-H2' in colliders, r'o-H2' in colliders):
        number_of_colliders += 2
    if r'e' in colliders:
        number_of_colliders += 1

    for i in range(number_of_files):
        # * radex, * Geometry, * Molecular data file, * T(kin), * H2 (whaterever)
        # * T(background), * Column density, * Line width, iterations, Dims, Units
        # results_idx = radex_version_idx[i]+8+number_of_colliders+6
        results_idx = radex_version_idx[i] + 4 + number_of_colliders + 6
        results = lines[results_idx:radex_version_idx[i+1]]
        QN_ul.append([])
        E_UP.append([])
        freq.append([])
        WAVEL.append([])
        T_ex.append([])
        opacity.append([])
        T_r.append([])
        for l in results:
            QN_ul[i].append(''.join(l.split()[0:3]))
            E_UP[i].append(float(l.split()[3]))
            freq[i].append(float(l.split()[4]))
            WAVEL[i].append(float(l.split()[5]))
            T_ex[i].append(float(l.split()[6]))
            opacity[i].append(float(l.split()[7]))
            T_r[i].append(float(l.split()[8]))
    out_file.close()
    return from_list_to_array(T_ex), from_list_to_array(opacity), from_list_to_array(T_r)


def execute_radex(
        molecule: str,
        file: str,
        min_freq: Real,
        max_freq: Real,
        T_kin: np.ndarray,
        colliders: list[str],
        collider_densities: np.ndarray,
        N: np.ndarray,
        FWHM: np.ndarray,
        geometry: str,
        clean: Optional[bool] = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute radex command

    :param molecule: molecule name (ex : '12CO')
    :type molecule: str
    :param file: file name without extension (ex: 'gradient_computation')
    :type file: str
    :param min_freq: (GHz) minimal frequency (Radex parameter)
    :type min_freq: float
    :param max_freq: (GHz) maximal frequency (Radex parameter)
    :type max_freq: float
    :param T_kin: kinetic temperature (K)
    :type T_kin: np.ndarray
    :param colliders: names of the considered collider parterns ('H2', 'p-H2', 'o-H2', 'e', 'H', 'He', 'H+')
    :type colliders: list[str]
    :param collider_densities: (cm^-3) densities of the considered collider parterns
    :type collider_densities: np.ndarray of shape (len(colliders) x gridLength)
    :param N: (cm^-2) column density
    :type N: np.ndarray
    :param FWHM: (km/s) full width at half maximum
    :type FWHM: np.ndarray
    :param geometry: used geometry among {'radexUnifSphere','radexExpSphere','radexParaSlab'}
    :type geometry: str
    :param clean: remove the .inp and .out files, defaults to False
    :type clean: bool, optional
    :return: (K) excitation temperature, opacity, (K) Rayleigh–Jeans or brightness temperature
    :rtype: np.ndarray, np.ndarray, np.ndarray
    """
    write_inp_file(molecule, file, min_freq, max_freq, T_kin,
                   colliders, collider_densities, N, FWHM)
    run_radex(geometry, file)
    T_ex, opacity, T_r = read_out_file(file, colliders)

    if clean:  # very slow !
        os.system('rm -f ' + str(add_inp_extension(file)))
        os.system('rm -f ' + str(add_out_extension(file)))
        os.system('rm -f radex.log')
    return T_ex, opacity, T_r
