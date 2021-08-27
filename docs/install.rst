Installation
=============
In order to get started with Recolo, you need to install it on your computer.
There are two main ways to to this:

*   You can install it via a package manager like PIP
*   You can  clone the repo


Install via package manager:
----------------------------

The toolkit is available via PIP, and the instructions below shows how a virtual environment can be created
and the toolkit installed.

Prerequisites::

    This toolkit is tested on Python 3.7
    We recommend the use of virtualenv

Installing::

    python -m venv env
    source ./env/bin/activate #On Linux and Mac OS
    env\Scripts\activate.bat #On Windows
    pip install recolo

Now the toolkit is installed and ready for use.

Run the tests::

    pytest recon

If you want to check out the examples, then download the files in the examples folder and run the examples.

By cloning the repo:
---------------------

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes.

Prerequisites::

    This toolkit is tested on Python 3.7
    We recommend the use of virtualenv

Start to clone this repo to your preferred location::

   git clone https://github.com/PolymerGuy/recon.git



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

    pytest recon
