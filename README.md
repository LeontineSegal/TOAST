# TOAST 

`TOAST` (mulTilayer clOud for simultAneouS analysis of mulTi-molecular lines) is a radiative transfer based-on code that aims to infer the physical, chemical and kinematic properties of a gas cloud with a heterogeneous environment along the line-of-sight (e.g. temperature or density gradient) from the simultaneous analysis of multiple molecular lines.

`TOAST` enables to derive maps of estimated properties over a given field of view, with a pixel-by-pixel approach (multiprocessing option included). It provides the confidence intervals on the estimation results. 

The generalized approaches (model fitting, quantification of accuracies) are detailed in \[1\], study where `TOAST` is customized to analyze the Horsehead nebula through the observations of $\text{CO}$ and $\text{HCO}^+$ isotopologues (dataset from the ORION-B Large Program, https://www.iram.fr/~pety/ORION-B/, P.I.: J. Pety \& M. Gerin).

## How using `TOAST`

The general using is the following :  

1) Modify the python file `model.py` as a text file, to customize the model of the cloud $\\$
    - by choosing the molecular lines to fit, 
    - by setting its number of layers along the sight-line, 
    - by injecting constraints on abundances and/or kinematics, 
    - etc.
2) Then, run the shell command 
    ```console
    $ python launch_model_fitting.py
    ```
    to launch the model fitting procedure. Once the optimization procedure is done, the results are saved in the folder `outputs/`. Estimation results are saved in `outputs/estimations/`. The corresponding confidence intervals can be then computed by launching the routine `launch_crb.py`.

# Installation 

## Installing `RADEX`

`TOAST` uses the non-local thermodynamic equilibrium (non-LTE) radiative transfer code `RADEX` from [2] to derive opacity ($\tau$) and excitation temperature ($T_{ex}$) of molecular lines.
Follow the installation steps detailed here https://personal.sron.nl/~vdtak/radex/index.shtml#top.

### Setting up `RADEX` for `TOAST` 

1) Replace the file `RADEX/src/io.f` :

Because `TOAST` performs a gradient descent when optimizing the model fitting to the measures, we increased the number of digits for $\tau$ and $T_{ex}$. Please, replace the file `RADEX/src/io.f` by the one provided in 
`radex-for-toast/io-for-toast.f`.

2) Modify the file `RADEX/src/radex.inc` : 

Set the number of minimal/maximal iterations such as
``` console
c     Numerical parameters
c
      integer miniter,maxiter
      parameter(miniter=50)     ! minimum number of iterations
      parameter(maxiter=9999)   ! maximum number of iterations
```

3) Rename the executable file : 

    In `RADEX/src/radex.inc`, `RADEX` allows one to choose a geometry for the escape probability approximation among : *uniform sphere* (assumed by default), *expanding sphere (LVG)*, and *plane parallel slab (shock)*. 

    After having executed `RADEX`, please rename the executable file `RADEX/bin/radex` following the geometry you chose, such as : 
    
    - *uniform sphere* : radex $\rightarrow$ radexUnifSphere
    - *expanding sphere (LVG)* : radex $\rightarrow$ radexExpSphere
    - *plane parallel slab (shock)* : radex $\rightarrow$ radexParaSlab 

## Routinely encountered issues

-
    Check the PATH of the folder `RADEX/Lamda` folder in `RADEX/src/radex.inc` :
    ``` console
    parameter(radat   = '/usr/*/Radex/Lamda/')
    ```

## Installing `TOAST` environment

Clone the source code

```console
$ git clone https://git.iram.fr/segal/toast.git
```

## Installing `Python` environment

To avoid conflicts between your Python versions, we suggest you to create a virtual environment.

First, place yourself in the repository `TOAST`
```console
$ cd TOAST
```

1) Make sure the version `3.11.10` of Python is installed on your machine, by typing `python` and hit tab (you should see appeared python3.11, install it otherwise).

2) Install `pip`
    ```console
    $ python3 -m pip install --user virtualenv
    ```

3) Create the virtual environment (here named .env)
    ```console
    $ python3 -m venv .env
    ```

4) Activate the virtual environment
    ```console
    $ source .env/bin/activate
    ```

    You can check whether the python environment has changed by typing 
    ```console
    which python
    ```
    and that no package are installed yet when typing
    ```console
    pip list
    ```

5) Install the packages required for `TOAST`
    ```console
    $ pip install -r requirements.txt
    ```

To deactivate the virtual environment, type
```console
$ deactivate
```

# Get started 
To get started with `TOAST`, check out the notebooks in the `examples` folder, where an analysis of this Horsehead nebula dataset is proposed. 

# References

[1] Ségal, L. \& Roueff, A. \&  Pety, J. \& Gerin, M. \& Roueff, E \& Javier R. Goicoechea \& Bešlić, I. \& Coudé, S. \& Einig, L. \& Mazurek, H. \& Orkisz, J. H. \& Palud, P. \& G. Santa-Maria, M. \& Zakardjian, A. \& Bardeau, S. \& Bron, E. \& Chainais, P. \& Demyk, K. \& de Souza Magalhẽs, V.  \& Gratier, P. \& V. Guzmán, V. \& Hughes, A. \& Languignon, D. \& Levrier, F. \& Le Bourlot, J. \& Le Petit, F. \& Darek C. Lis \& Liszt, H. S. \& Peretto, N. \& Sievers, A. \& Thouvenin, P.-A., "Toward a robust physical and chemical characterization of heterogeneous lines of sight: The case of the Horsehead nebula", *Astronomy and Astrophysics*, doi:10.1051/0004-6361/202451567.

[2] Van der Tak, F. F. S. \& Black, J. H. \& Schöier, F. L. \& Jansen, D. J. \& and van Dishoeck, E. F., “A computer program for fast non-LTE analysis of interstellar line spectra. With diagnostic plots to interpret observed line intensity ratios”, *Astronomy and Astrophysics*, vol. 468, no. 2, pp. 627–635, 2007. doi:10.1051/0004-6361:20066820.



