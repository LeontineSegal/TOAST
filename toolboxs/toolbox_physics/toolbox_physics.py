# %% load modules
from typing import List, Optional

import numpy as np 

from numbers import Real


# %% moments

def compute_moments(
        order: int,
        ppv: np.ndarray,
        velocity_channels: np.ndarray,
        velocity_res: Real,
        C_V: np.ndarray, 
        s_V: Optional[Real] = 1.5, # km/s
        mask: Optional[bool] = True
        ) -> List[np.ndarray]:
    """Compoute all computed moments to reach the expected one i.e. w{order}.

    :param order: order of the moment
    :type order: int
    :param ppv: data cube (PPV)
    :type ppv: np.ndarray
    :param velocity_channels: (km/s) velocity axis
    :type velocity_channels: np.ndarray
    :param velocity_res: (km/s) velocity resolution
    :type velocity_res: Real
    :return: List of maps of moments
    :rtype: List[np.ndarray]
    """
    if mask:
        map_mask = np.where(np.isnan(C_V), np.nan, 1)

    # integrated intensity
    w_0_map = np.empty((np.shape(ppv)[0], np.shape(ppv)[1]))
    w_0_map.fill(np.nan)

    if order >= 1:
        # velocity centroid
        w_1_map = np.empty((np.shape(ppv)[0], np.shape(ppv)[1]))
        w_1_map.fill(np.nan)

    if order >= 2:
        # linewidth^2
        w_2_map = np.empty((np.shape(ppv)[0], np.shape(ppv)[1]))
        w_2_map.fill(np.nan)

    if order >= 3:
        # skewness
        w_3_map = np.empty((np.shape(ppv)[0], np.shape(ppv)[1]))
        w_3_map.fill(np.nan)

    if order >= 4:
        # kurtosis
        w_4_map = np.empty((np.shape(ppv)[0], np.shape(ppv)[1]))
        w_4_map.fill(np.nan)

    for x in range(np.shape(ppv)[0]):
        for y in range(np.shape(ppv)[1]):

            s = ppv[x, y, :]

            if np.any(np.isnan(s)):
                pass

            else:
                s = s.reshape((1, len(s)))

                if C_V is not None : 
                    ### windowing spectra around C_V aver a bandwidth of 3 km/s ###
                    C_V_velocity_channel = np.argmin(
                        abs(C_V[x, y] - velocity_channels))

                    min_bandwidth = max(0, C_V_velocity_channel -
                                        int(s_V/velocity_res))
                    max_bandwidth = min(C_V_velocity_channel + int(s_V /
                                        velocity_res) + 1, velocity_channels.size)
                    
                    # symetrical bandwidth
                    if velocity_channels.size > 1:
                        left_window_size = len(
                            np.arange(min_bandwidth, C_V_velocity_channel))
                        right_window_size = len(
                            np.arange(C_V_velocity_channel + 1, max_bandwidth))
                        minimal_window_size = min(
                            left_window_size, right_window_size)

                        if left_window_size > minimal_window_size:
                            min_bandwidth = C_V_velocity_channel - minimal_window_size
                        if right_window_size > minimal_window_size:
                            max_bandwidth = C_V_velocity_channel + minimal_window_size

                    bandwidth = np.arange(min_bandwidth, max_bandwidth)

                    windowed_velocity_axis = velocity_channels[0, bandwidth]
                    windowed_velocity_axis = windowed_velocity_axis.reshape(
                        (1, len(bandwidth)))

                    # find the velocity window around the peak of the line
                    # number of sample to take into account
                    s = s[0, bandwidth]
                    s = s.reshape((1, len(bandwidth)))

                else : 
                    windowed_velocity_axis = velocity_channels

                normalization = s/np.sum(s)
                # normalization = s

                w_0_map[x, y] = velocity_res * np.sum(s)  # K.km/s

                if order >= 1:
                    w_1_map[x, y] = np.sum(
                        windowed_velocity_axis * normalization)

                if order >= 2:
                    w_2_map[x, y] = np.sum(
                        np.power(windowed_velocity_axis - w_1_map[x, y], 2) * normalization)

                if order >= 3:
                    w_3_map[x, y] = np.sum(
                        np.power(windowed_velocity_axis - w_1_map[x, y], 3) * normalization) / np.sqrt(w_2_map[x, y])**3

                if order >= 4:
                    w_4_map[x, y] = np.sum(
                        np.power(windowed_velocity_axis - w_1_map[x, y], 4) * normalization) / np.sqrt(w_2_map[x, y])**4

    if mask:
        w_0_map = np.where(np.isnan(map_mask), np.nan, w_0_map)
    moments = [w_0_map]
    if order >= 1:
        if mask:
            w_1_map = np.where(np.isnan(mask), np.nan, w_1_map)
        moments.append(w_1_map)
    if order >= 2:
        if mask:
            w_2_map = np.where(np.isnan(mask), np.nan, w_2_map)
        moments.append(w_2_map)
    if order >= 3:
        if mask:
            w_3_map = np.where(np.isnan(mask), np.nan, w_3_map)
        moments.append(w_3_map)
    if order >= 4:
        if mask:
            w_4_map = np.where(np.isnan(mask), np.nan, w_4_map)
        moments.append(w_4_map)

    return moments

def estime_C_V_from_multi_mol_lines(
        x: List[np.ndarray],
        velocity_channels: List[np.ndarray]
        ) -> tuple[np.ndarray]:
    """Estime the centroid velocity by computing the mean of centroid velocities of under study tracer cubes.
    Useful to initialize velocity centroids of all layers of the cloud model to then compute the FPS exploration. 
    :param x: observed cubes
    :type x: List[np.ndarray]
    :param velocity_channels: velocity channels
    :type velocity_channels: List[np.ndarray]
    :return: velocity centroid 
    :rtype: np.ndarray
    """

    C_V = np.zeros((np.shape(x[0])[0], np.shape(x[0])[1])) 
    C_V.fill(np.nan)
    mask = np.ones(C_V.shape)
    velocity_channels_reference = velocity_channels[0]
    sum_x = np.zeros(np.shape(x[0]))
    for mol_line_idx in range(len(x)):
        sum_x += x[mol_line_idx]
        mask = np.where(np.logical_or(np.isnan(mask), np.isnan(x[mol_line_idx][:, :, 0])), np.nan, 1)

    C_V = velocity_channels_reference[0, np.argmax(sum_x, axis=-1)]
    C_V = np.round(C_V, decimals=2)
    C_V = np.where(np.isnan(mask), np.nan, C_V)

    return C_V
