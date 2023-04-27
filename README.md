# Code repository for the Generating Function Method

This repository contains a python code library and scripts implemented for the development of the Generating Function Method (GFM),
which intends to describe and analyze the main current iterative PinT algorithms within a common framework (Parareal, MGRIT, PFASST, STMG, ...).

It is associated to the following publication :

> _Gander, M. J., Lunet, T., Ruprecht, D., & Speck, R. (2023). [Gander, M. J., Lunet, T., Ruprecht, D., & Speck, R. (2022). A Unified Analysis Framework for Iterative Parallel-in-Time Algorithms](https://arxiv.org/pdf/2203.16069.pdf). To appear in SIAM Journal of Scientific Computing._

Scripts used to generate the figures in the paper below are available in the [outputs](./outputs/README.md) folder.

# How to run the code

> :mega: Instructions are given for Unix-based system (Linux, MacOSX, ...). Sadly, Windows systems are not supported (and will probably never ... but you can still use [WSL](https://learn.microsoft.com/de-de/windows/wsl/install) on Windows to emulate a Ubuntu system and then use the instructions below).

The library currently depends on classical python packages like [Numpy](https://numpy.org/) and [Matplotlib](https://matplotlib.org/),
plus an external package for the generation of quadrature nodes for collocation methods, [Pythag](https://gitlab.com/tlunet/pythag).
More informations on how Pythag can be installed and used [here](./doku/pythag.md).

Additionally, figures are cropped using `pdfcrop`, which should be installed prior on your system.

If you have any issue with running the code, 
don't hesitate to open an [issue](https://github.com/Parallel-in-Time/gfm/issues) in the Github interface. 

## Repository structure

- [doku](./doku) : Markdown documentation pages.
- [gfm](./gfm) : Python library for the gfm package.
- [notes](./notes) : Jupyter notebooks with small numerical investigations on PinT methods using the gfm library.
- [outputs](./outputs) : scripts to reproduce the figures of a manuscript on the GFM method (in preparation).
- [tutorials](./tutorials) : Jupyter notebooks with example and tutorials on how to use the gfm library.

## Acknowledgements

This repository results from a collaboration between Jülich Supercomputing Centre 
([Robert SPECK](https://www.fz-juelich.de/SharedDocs/Personen/IAS/JSC/EN/staff/speck_r.html)), 
Hamburg University of Technology 
([Daniel RUPRECHT](https://www.mat.tuhh.de/home/druprecht/?homepage_id=druprecht),
[Thibaut LUNET](https://www.mat.tuhh.de/home/tlunet/?homepage_id=tlunet)) 
and the University of Geneva ([Martin J. GANDER](https://www.unige.ch/~gander/)), 
as part of the [Time-X project](https://www.timex-eurohpc.eu/). 

<p align="center">
  <img src="./doku/logo_JSC.jpg" height="55"/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./doku/tuhh-logo.png" height="55"/> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./doku/logo_sec-math.png" height="70"/> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./doku/LogoTime-X.png" height="75"/>
</p>

This project has received funding from the [European High-Performance Computing Joint Undertaking](https://eurohpc-ju.europa.eu/) (JU)
under grant agreement No 955701 ([Time-X](https://www.timex-eurohpc.eu/)).
The JU receives support from the European Union’s Horizon 2020 research and innovation programme and Belgium, France, Germany, and Switzerland.
This project also received funding from the 
[German Federal Ministry of Education and Research](https://www.bmbf.de/bmbf/en/home/home_node.html) (BMBF) grant 16HPC048.

<p align="center">
  <img src="./doku/EuroHPC.jpg" height="105"/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./doku/logo_eu.png" height="95" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./doku/BMBF_gefoerdert_2017_en.jpg" height="105" />
</p>

