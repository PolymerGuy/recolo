Recon
=============
[![codecov](https://codecov.io/gh/PolymerGuy/recon/branch/master/graph/badge.svg)](https://codecov.io/gh/PolymerGuy/recon)
[![CircleCI](https://circleci.com/gh/PolymerGuy/recon.svg?style=svg&circle-token=badgeToken)](https://circleci.com/gh/PolymerGuy/recon)
[![MIT License][license-shield]][license-url]

About this project
------------------
Tools for load reconstruction using the virtual fields method.


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

[license-shield]: https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square
[license-url]: https://choosealicense.com/licenses/mit