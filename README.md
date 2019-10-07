# Neurocodes
Public repository of developed scripts to read, process and create models.

We use data from the public repository provided by [BadenLab](https://badenlab.org/) more specifically the following databases:

* [Inhibition decorrelates visual feature representations in the inner retina](http://dx.doi.org/10.5061/dryad.rs2qp)
* [The functional diversity of retinal ganglion cells in the mouse](http://dx.doi.org/10.5061/dryad.d9v38)

The name of the files you should have in your data directory are the following:

For Bipolar cells:

* FrankeEtAl\_BCs\_2017\_v1.mat
* FrankeEtAl\_BCs\_2017\_noise\_raw.mat

For Ganglion cells:

* BadenEtAl\_RGCs\_2016\_v1.mat

This repository is set up in a manner where every function the user interacts with is contained in the `main.py` file, the resulting trained models are stored in the `models` folder.

# Data Structure

<!--
We can talk a little about the class we developed to parse the data and
its problems here
-->
