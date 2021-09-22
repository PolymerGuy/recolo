Installation
=============
In order to get started with Recolo, you need to install it on your computer.

By cloning the repo:
---------------------

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes.

Prerequisites::

    This toolkit is tested on Python 3.7
    We recommend the use of virtualenv

Start to clone this repo to your preferred location::

   git clone https://github.com/PolymerGuy/recolo.git


We recommend that you always use virtual environments, either by virtualenv or by Conda env

Virtual env::

    python -m venv env
    source ./env/bin/activate #On Linux and Mac OS
    env\Scripts\activate.bat #On Windows
    pip install -r requirements.txt


You can now run an example::

    $ python ./examples/AbaqusExperiments/Reconstruction_minimal.py

Running the tests
------------------
The tests should always be launched to check your installation.
These tests are integration and unit tests

If you cloned the repo, you have to call pytest from within the folder::

    pytest recolo
