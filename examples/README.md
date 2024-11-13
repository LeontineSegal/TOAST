# Example

This folder contains notebooks illustrating how using `TOAST` to analyze multi-molecular lines, through the case of study of the Horsehead nebula dataset. 

- ``analyze_one_pixel``: allows to familiarize yourself with `TOAST` by analyzing a single line-of-sight with various model of clouds.
- ``analyze_map``: illustrates how handling several pixels to derive maps of estimations (kinetic temperature, density, column density, etc.), with parallel processing, eventually.
- ``compute_accuracy``: illustrate how computing accuracy on the estimation results obtained in the ``analyze_map`` example.   

Because the Horsehead dataset is composed of the lines of $^{12}\text{CO}$, $^{13}\text{CO}$, $\text{C}^{18}\text{O}$ and $\text{HCO}^{+}$, $\text{H}^{13}\text{CO}^{+}$, please add the files `*.dat` provided in `examples/Lamda/` to your own folder `RADEX/Lamda/`. 

The dataset is in the folder `data-horsehead` and contains the folders: 

- ``raw-ppv``: the position-position-velocity cubes (`.fits` files),
- ``ppv``: the position-position-velocity cubes (`.fits` files) modified following our needs (e.g., the studied bandwidth is reduced around where signal is detected to decrease the number of velocity channels to process),
- ``inputs``: the inputs to provide to `TOAST`. They have been extract from the `.fits` ppv cubes and formalized into `.npy` data (see the routine `prepare_dataset.py`).
- `grids-Tex-tau`: contains $T_{ex}$ and $\tau$ grids in the physical space to explore (called the Finite Parameter Space `FPS.npy`), for all molecular lines studied.

# To be checked when using `TOAST` to your own dataset

Please, make sure that the following points are checked for each molecular species you are considering : 

1) Download the file `.dat` from the Leiden Atomic and Molecular Database (`LAMDA`) at <https://home.strw.leidenuniv.nl/~moldata/> and add it to the folder `RADEX/Lamda/` (e.g., `12co.dat` for $^{12}\text{CO}$).

2) Add the name of the file `.dat` to the function `get_dat_file` in 
    ```console 
    toolboxs/toolbox_radex/toolbox_radex.py 
    ```

    Following your needs, modify the function `convert_name_mol_line` in 
    ```console 
    toolboxs/toolbox_fits/toolbox_fits.py 
    ```

3) Compute the `RADEX` grids of $\tau$ and $T_{ex}$ of each of the studied transitions of the molecular species. `TOAST` indeed performs a search of the solution in a gridded space before refining it with a gradient descent.
We provide a python routine `launch_radex_grids.py` to compute those grids.

4) Derive the map of the dispersion of the thermal noise of the molecular line and add it as a `.fit` file in the folder `data-*/raw-ppv/noise/`.

    For instance, let `12co10.fits` the ppv cube of the molecular line $^{12}\text{CO}$ (J = 1 - 0) that you want to analyze. The noise map can be derived by using `CUBE` (https://www.iram.fr/IRAMFR/GILDAS/) such as
    ```console
    $ cube 
    CUBE > import 12co10.fits 
    CUBE > noise 12co10:cube /RANGE vmin vmax km/s
    CUBE > export
    ```
    Here /RANGE vmin vmax correspond to the spectral range containing signal.

Finally, your dataset (`.fit` files of ppv cubes, maps of thermal noise, etc.) has then to be formalized in such way that it can be used by `TOAST`. 
We provide a routine `prepare_dataset.py` allowing one to do so, but since it was developed to suit the Horsehead nebula dataset, you may need to adjust it to your own dataset. 
The routine creates and saves the formalized data in the folder `data-*/inputs`. 

Finally, checked that you computed and saved the following quantities :

- In `data-*/grids-Tex-tau/radex*/` :
    - `FPS.npy` : the Finite Parameter Space (FPS) made of {$T_{kin}$, $n_{H_2}$, column densities, FWHM} to be explored $\rightarrow$ see `launch_radex_grids.py`
- In `data-*/inputs/` : 
    - `colden_ratios.npy` : column density ratios assumed when the `RADEX` grids of $T_{ex}$ and $\tau$ have been computed $\rightarrow$ see `launch_radex_grids.py`
    - `FoV.npy` : The binary map of the field of view to study $\rightarrow$ see `analyze_one_pixel.ipynb`
    - `map_C_V.npy` : The map of the systemic velocity of the cloud, required to initialize the model fitting optimization $\rightarrow$ see `analyze_one_pixel.ipynb`