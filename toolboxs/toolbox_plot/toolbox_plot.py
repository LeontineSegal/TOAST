# %% load modules
import numpy as np

from typing import Optional, List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcol
import matplotlib.cm as cm

from toolboxs.toolbox_python.toolbox_python import from_float_to_string

from toolboxs.toolbox_physics.toolbox_radiative_transfer import compute_radiative_tranfer_equation

#%% 

def fmt(x, pox):
    s = f'{x:.2f}'
    if s[-1] == '0':
        if s[-2] == '0':
            return f'{x:.0f}'
        else:
            return f'{x:.1f}'
    else:
        return f'{x:.2f}'
    
# %% spectra

def plot_percentile_mol_line(
        pc_spectra: List[np.array],
        velocity_axis: List[float],
        suplot_titles: List[str],
        C_V: Optional[List[float]] = [],
        s_V: Optional[List[float]] = [],
        pc: Optional[List[float]] = [5, 50, 95]
) -> plt.Figure:

    pc_labels = [f'{pc[i]} %' for i in range(len(pc))]
    pc_colors = ['r', 'g', 'b']
    alpha = 1

    total_subplots = len(pc_spectra)
    rows = max(1, total_subplots // 2)
    columns = total_subplots // rows
    columns += total_subplots % rows
    subplot_position = range(1, total_subplots + 1)

    fig = plt.figure(figsize=(3.8 * columns, 3. * rows))
    axe_idx = 0

    for spectrum_idx, spectrum in enumerate(pc_spectra):
        axe = fig.add_subplot(
            rows, columns, subplot_position[axe_idx])
        for q in range(len(spectrum)):
            axe.step(
                velocity_axis[spectrum_idx],
                spectrum[q, :],
                color=pc_colors[q],
                alpha=alpha,
                where='mid',
                label=f'{pc_labels[q]}')

        if len(C_V) != 0:
            axe.axvline(x=C_V[spectrum_idx],
                        linewidth=1,
                        color='k',
                        linestyle='dashed',
                        alpha=1,
                        label=r'$C_V$ = 'f'{np.round(C_V[spectrum_idx], 2)} km/s')
            if len(s_V) != 0:
                axe.axvline(x=C_V[spectrum_idx] + s_V[spectrum_idx],
                            linewidth=0.5,
                            linestyle='dotted',
                            color='k',
                            alpha=1,
                            label=r'$s_V = \pm$'f'{np.round(s_V[spectrum_idx], 2)} km/s')
                axe.axvline(x=C_V[spectrum_idx] - s_V[spectrum_idx],
                            linewidth=0.5,
                            linestyle='dotted',
                            color='k',
                            alpha=1)

        axe.axhline(y=0,
                    linestyle='dashed',
                    color='g',
                    linewidth=1,
                    alpha=1,
                    label='baseline')

        axe.set_title(f'{suplot_titles[axe_idx]}')

        if axe_idx == 0:
            l = axe.legend(loc='upper left', frameon=False)

        axe_idx += 1

    axs = fig.axes
    for c in range(1, columns+1):
        axs[-c].set_xlabel(r'Velocity [km/s]')

    for r in range(0, total_subplots, columns):
        axs[r].set_ylabel(r'Intensity [K]')
    fig.tight_layout()

    return fig

def show_LoS(
        spectra: List[np.array],
        velocity_axis: List[float],
        suplot_titles: List[str],
        name_fig: str, 
        C_V_initial: Optional[List[float]] = [False], 
        delta_V: Optional[List[List]] = [False], 
        save: Optional[bool] = False, 
        FORMAT_FILE: Optional[str] = '.pdf'
) -> plt.Figure:

    total_subplots = len(spectra)
    rows = max(1, total_subplots // 2)
    columns = total_subplots // rows
    columns += total_subplots % rows
    subplot_position = range(1, total_subplots + 1)

    fig = plt.figure(figsize=(3.8 * columns, 3. * rows))
    axe_idx = 0

    for spectrum_idx, spectrum in enumerate(spectra):
        axe = fig.add_subplot(
            rows, columns, subplot_position[axe_idx])
        for q in range(len(spectrum)):
            axe.step(
                velocity_axis[spectrum_idx],
                spectrum,
                'k',
                where='mid',
            )
        if C_V_initial[0] != False : 
            ymin = np.nanmin(spectrum) 
            ymax = np.nanmax(spectrum)
            axe.vlines(C_V_initial, 
                       ymin = ymin, 
                       ymax = ymax, 
                       colors = 'r', 
                       linestyles = 'dashed', 
                       label = r'Initial $C_V$')
            
            if delta_V[0] != False : 
                axe.vlines(
                        max(C_V_initial) + delta_V[axe_idx], 
                       ymin = ymin, 
                       ymax = ymax, 
                       colors = 'g', 
                       linestyles = 'dashed', 
                       label = r'Studied bandwidth')
                axe.vlines(
                        min(C_V_initial) - delta_V[axe_idx], 
                       ymin = ymin, 
                       ymax = ymax, 
                       colors = 'g', 
                       linestyles = 'dashed')
            
        axe.set_title(f'{suplot_titles[axe_idx]}')
        
        axe_idx += 1

    axs = fig.axes
    for c in range(1, columns+1):
        axs[-c].set_xlabel(r'Velocity [km/s]')

    for r in range(0, total_subplots, columns):
        axs[r].set_ylabel(r'Intensity [K]')

    fig.tight_layout()
    # for legend 
    if C_V_initial[0] != False :
        axe.legend(bbox_to_anchor=(1.2, 0.95), framealpha = 1)

    if save : 
        fig.savefig(f'{name_fig}{FORMAT_FILE}')

    return fig

def show_LoS_model(
        measured_spectra: List[np.array],
        measured_velocity_axis: List[float],
        spectra: List[np.array],
        velocity_axis: List[float],
        suplot_titles: List[str],
        s_V: List[float],
        C_V: List[float],
        velocity_resolution: List[float],
        optimal_Tex: List[np.array],
        optimal_tau: List[np.array],
        freqs: List[float],
        name_fig: str, 
        peak_only: Optional[List[bool]] = [], 
        number_of_C_V_components: Optional[int] = 1, 
        number_of_layers_per_clump: Optional[int] = 1,
        save: Optional[bool] = False,  
        FORMAT_FILE: Optional[str] = '.pdf'
) -> plt.Figure:

    if len(peak_only) == 0 : 
        peak_only = [False for i in range(len(measured_spectra))]
    
    cm1 = mcol.LinearSegmentedColormap.from_list("", ["r", "b"])
    cnorm = mcol.Normalize(vmin=0, vmax=len(optimal_Tex[0]))
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])

    total_subplots = len(spectra)
    rows = max(1, total_subplots // 2)
    columns = total_subplots // rows
    columns += total_subplots % rows
    subplot_position = range(1, total_subplots + 1)

    fig = plt.figure(figsize=(3.8 * columns, 3. * rows))
    axe_idx = 0

    s_V = np.array(s_V).reshape((1, len(s_V)))
    C_V = np.array(C_V).reshape((1, len(C_V)))

    for spectrum_idx, spectrum in enumerate(measured_spectra):
        axe = fig.add_subplot(
            rows, columns, subplot_position[axe_idx])

        # measure
        axe.step(
            measured_velocity_axis[spectrum_idx],
            spectrum,
            c='k',
            where='mid',
            label='Measure',
            linewidth=2.
        )

        velocity_channels_line = velocity_axis[spectrum_idx]
        velocity_resolution_line = velocity_resolution[spectrum_idx]
        freq = freqs[spectrum_idx]
        Tex = optimal_Tex[spectrum_idx]
        tau = optimal_tau[spectrum_idx]
        Tex = np.array(Tex).reshape((1, len(Tex)))
        tau = np.array(tau).reshape((1, len(tau)))

        # model
        s_line = compute_radiative_tranfer_equation(
            Tex,
            tau,
            freq,
            velocity_channels_line.reshape((1, len(velocity_channels_line))),
            velocity_resolution_line,
            s_V,
            C_V,
            decomposition=True, 
            number_of_C_V_components = number_of_C_V_components, 
            number_of_layers_per_clump = number_of_layers_per_clump, 
            peak_only = peak_only[spectrum_idx])

        if not peak_only[spectrum_idx]:  # not PEAK_ONLY
            for spectrum_idx, spectrum in enumerate(s_line):
                if spectrum_idx == 0:
                    label = r'Model'
                    color = 'green'
                    linestyle = 'solid'
                    axe.step(velocity_channels_line,
                             spectrum[0, :],
                             c=color,
                             where='mid',
                             label=label,
                             linewidth=2.,
                             linestyle=linestyle,
                             )
                else:
                    if spectrum_idx == len(s_line) - 1:
                        label = r'CMB'
                        color = 'gray'
                        linestyle = 'dashed'
                    else:
                        label = r'Layer 'f'{spectrum_idx}'
                        color = cpick.to_rgba(spectrum_idx)
                        linestyle = 'solid'

                    axe.plot(velocity_channels_line,
                             spectrum[0, :],
                             c=color,
                             label=label,
                             linewidth=1,
                             linestyle=linestyle
                             )
        else:
            for spectrum_idx, spectrum in enumerate(s_line):

                for clump_idx in range(spectrum.shape[1]) : 

                    if spectrum_idx == 0:
                        label = r'Model'
                        color = 'green'
                        linestyle = 'solid'

                        axe.plot(velocity_channels_line[clump_idx],
                            spectrum[0, clump_idx],
                            'x',
                            c=color,
                            linewidth=2.,
                            linestyle=linestyle,
                            label = label
                            ) 
                    else:
                        if spectrum_idx == len(s_line) - 1:
                            label = r'CMB'
                            color = 'gray'
                            linestyle = 'dashed'
                        else:
                            label = r'Layer 'f'{spectrum_idx}'
                            color = cpick.to_rgba(spectrum_idx)
                            linestyle = 'solid'

                        axe.plot(velocity_channels_line[clump_idx],
                                spectrum[0, clump_idx],
                                'x',
                                c=color,
                                linewidth=1,
                                linestyle=linestyle, 
                                label = label
                                )

        axe.set_title(f'{suplot_titles[axe_idx]}')
        axe_idx += 1

    axs = fig.axes
    for c in range(1, columns+1):
        axs[-c].set_xlabel(r'Velocity [km/s]')

    for r in range(0, total_subplots, columns):
        axs[r].set_ylabel(r'Intensity [K]')

    fig.tight_layout()

    #axe.legend(loc='upper right')         
    axe.legend(bbox_to_anchor=(1.2, 0.95), framealpha = 1)

    if save : 
        fig.savefig(f'{name_fig}{FORMAT_FILE}')

    return fig

# %%
# maps

def plot_maps(
    maps: List[np.array],
    titles: List[str],
    vmin_vmax: Optional[bool | List[List]] = False,
    cmap: Optional[List[str]] = ['jet'],
    array_coordinates: Optional[bool] = False,
    norm: Optional[List[bool]] = [False], 
    contour : Optional[bool] = False, 
    contour_map: Optional[np.ndarray] = False,
    contour_values: Optional[List[float]] = [0.], 
    colorbar_fraction: Optional[float] = 0.04, 
    format_colorbar: Optional[List[None]] = [ticker.FuncFormatter(fmt)], 
    ticks: Optional[List] = [False], 
    minorticks_on: Optional[bool] = False, 
    face_color: Optional[str] = 'white', 
    extend: Optional[bool] = 'neither', 
    pixel: Optional[List] = None
):
    total_subplots = len(maps)
    rows = max(1, total_subplots // 2)
    columns = total_subplots // rows
    columns += total_subplots % rows
    subplot_position = range(1, total_subplots + 1)

    if norm[0] == False : 
        norm = [False for i in range(total_subplots)]    
    if ticks[0] == False : 
        ticks = [False for i in range(total_subplots)]
    if len(cmap) == 1 : 
        cmap = [cmap[0] for i in range(total_subplots)]
    if len(format_colorbar) == 1 : 
        format_colorbar = [format_colorbar[0] for i in range(total_subplots)]

    fig = plt.figure(figsize=(3.8 * columns, 3. * rows))
    pad = 0.05

    for map_idx, map in enumerate(maps):
        map = map.astype(float)
        axe = fig.add_subplot(
            rows, columns, subplot_position[map_idx])
        axe.format_coord = lambda x, y: 'x={:.0f}, y={:.0f}'.format(x, y)
        axe.set_title(titles[map_idx])
        axe.set_facecolor(face_color)

        if vmin_vmax != False:
            vmin, vmax = vmin_vmax[map_idx][0], vmin_vmax[map_idx][1]
        else:
            if norm[map_idx] != False :
                vmin_vmax = False
            
        if norm[map_idx] != False:
            if vmin_vmax != False:
                img = axe.imshow(map,
                                 cmap=cmap[map_idx],
                                 origin='lower',
                                 vmin=vmin,
                                 vmax=vmax,
                                 norm=norm[map_idx]
                                 )
            else:
                img = axe.imshow(map,
                                 cmap=cmap[map_idx],
                                 origin='lower',
                                 norm=norm[map_idx]
                                 )
        else:
            if vmin_vmax != False:
                img = axe.imshow(map,
                                 cmap=cmap[map_idx],
                                 origin='lower',
                                 vmin=vmin,
                                 vmax=vmax,
                                 )
            else:
                img = axe.imshow(map,
                                 cmap=cmap[map_idx],
                                 origin='lower',
                                 )

        if format_colorbar[map_idx] == '' : 
            if ticks[map_idx] == False : 
                colorBar = fig.colorbar(
                    img,
                    ax=axe,
                    fraction=colorbar_fraction,
                    pad=pad,
                    norm = norm[map_idx], 
                    extend = extend
                )
        else : 
            if ticks[map_idx] == False : 
                colorBar = fig.colorbar(
                    img,
                    ax=axe,
                    fraction=colorbar_fraction,
                    pad=pad,
                    norm = norm[map_idx],
                    format=format_colorbar[map_idx], 
                    extend = extend
                )
            else : 
                colorBar = fig.colorbar(
                    img,
                    ax=axe,
                    fraction=colorbar_fraction,
                    pad=pad,
                    norm = norm[map_idx],
                    format=format_colorbar[map_idx], 
                    ticks = ticks[map_idx], 
                    extend = extend
                )
        if minorticks_on != False : 
            if minorticks_on[map_idx] != False : 
                colorBar.ax.minorticks_on()

        if contour: 
            img = axe.contour(contour_map, 
                              contour_values,
                              linewidths=2,
                              origin = 'lower',
                              colors = 'w'
                              )
            img = axe.contour(contour_map, 
                              contour_values,
                              linewidths=1,
                              origin = 'lower',
                              colors = 'k'
                              )

        if pixel is not None : 
            axe.scatter(pixel[1], pixel[0], 
                        c = 'k', 
                        s = 200,
                        marker = 'x',
                     label=f"({from_float_to_string(pixel[0], 'float', 1, 0)}, {from_float_to_string(pixel[1], 'float', 1, 0)})")
            axe.legend()

        if array_coordinates:

            min_i, max_i = -0.5, map.shape[0]-0.5  # vertical axis
            min_j, max_j = -0.5, map.shape[1]-0.5  # horizontal axis

            axe.set_xlim(min_j, max_j)
            axe.set_ylim(min_i, max_i)

            axe.set_xticks(np.arange(min_j + 0.5, max_j, 5))
            axe.set_yticks(np.arange(min_i + 0.5, max_i, 5))
            axe.tick_params(axis = 'both', labelsize = 6)

        else:
            axe.axes.xaxis.set_ticklabels([])
            axe.axes.yaxis.set_ticklabels([])

    plt.tight_layout()

    return fig