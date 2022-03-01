## Installation of the Pythag package

This package is currently hosted in a Gitlab repository [here](https://gitlab.com/tlunet/pythag), 
where it is continuously developed.
Originally, this package has been developed to implement Gauss quadrature rules, and is used here to
provide a fast and generic way to compute quadrature nodes for collocation methods.

There is no pip or conda package associated to pythag, so one should "manually" download it 
using git, and get the associated version with the tag `pub/gfm-01` :
```bash
git clone https://gitlab.com/tlunet/pythag.git pythag
cd pythag
git checkout pub/gfm-01
```
Then, a last manipulation is required to get it installed on your system.

### Using Anaconda distribution

Let say you have Anaconda installed on the `${CONDA_DIR}` directory, and the git repository for Pythag is located in `${PYTHAG_DIR}`.
Then you can simply create a symbolic link to the Pythag library where all the conda package are stored :
```bash
ln -sf ${PYTHAG_DIR}/pythag ${CONDA_DIR}/lib/python${PYTHON_VERSION}/site-packages
```
where `${PYTHON_VERSION}` are the two first version digits of your current python installation (e.g "3.7", or "3.8").
Those can be retrieved using the following command :
```bash
PYTHON_VERSION=$(python -c 'import sys; v=sys.version_info; print(f"{v[0]}.{v[1]}")')
```
or simply looking at the content of `${CONDA_DIR}/lib/`.

> :warning: if you cloned the repository as the example above, then `${PYTHAG_DIR}/pythag` should be something like `[..]/pythag/pythag`

### Using standard Python with pip

Modify your `PYTHONPATH` environment variable to include the path to the directory containing the pythag package, e.g append in your `.bashrc` file :
```bash
export PYTHONPATH="${PYTHAG_DIR}:${PYTHONPATH}"
```
where `${PYTHAG_DIR}` is the path of the git repository for Pythag.

### On Windows systems

Not tested yet ... but are you really sure you want to do computational mathematics on Windows ?