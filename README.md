![](docs/logo.png)
=============
[![codecov](https://codecov.io/gh/PolymerGuy/recon/branch/master/graph/badge.svg?token=7J4EH3C399)](https://codecov.io/gh/PolymerGuy/recon)
[![CircleCI](https://circleci.com/gh/PolymerGuy/recon.svg?style=svg&circle-token=3403eba7b905e1a626d1c797ed5ca4e3daba76df)](https://circleci.com/gh/PolymerGuy/recon)
[![MIT License][license-shield]][license-url]
[![Documentation Status](https://readthedocs.org/projects/recolo/badge/?version=latest)](https://recolo.readthedocs.io/en/latest/?badge=latest)


About this project
------------------
Tools for load reconstruction using the virtual fields method. This Python package provides the tools neccesarry
for reconstructing distributed pressure loads acting on thin plates based on kinematic fields.

A virtual lab is also provided, allowing synthetic data to be generated based on input from finite element simulations.

Example kinematic fields pressure is shown below:
![alt text](docs/kinematics.gif)

which gives the following pressure field:
![alt text](docs/pressure.gif)

The documentation is hosted at https://recolo.readthedocs.io/en/latest/


Getting Started
---------------
Clone the repo by using `git`:

```bash
git clone https://github.com/PolymerGuy/recon.git
```

when in the folder with the repo, make a virtual environment and install all dependencies:

```bash
# Make virtual environment using venv
python -m venv env
# Activate the virtual environment
source ./env/bin/activate
# Install dependencies
pip install -r requirements.txt
```

To check that everything is working, run all tests:
```bash
pytest recon
```


Building the documentation from source
--------------------------------------
```bash
# Enter the documentation folder
cd docs
# Rebuild docs
make html
```

The documentation is now found in ./_build_/html


Example
-------



How to contribute
-----------------
* Fork the repo
* Add nice things to the code
* Make a pull-request to the -dev branch
* Celebrate!

How to cite us
--------------
If you use this toolkit in your scientific work, consider citing one or more of the following:

- Kaufmann,*et al.*, ["Virtual fields for load reconstuction in shock-tube experiments"](https://www.dead.link.com), *"International Journal of Impact Engineering*, May 2022. ([open access](https://www.dead.link.com))



[license-shield]: https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square
[license-url]: https://choosealicense.com/licenses/mit