Minimum Wasserstein Estimates (MWE)
===================================

Description
-----------

This repo contains code that can be used to reproduce the results of our paper
*Group level MEG/EEG source imaging via optimal transport:
minimum Wasserstein estimates*, IPMI 2019.

MWE is a model for M-EEG source localization for a group of subjects. It
leverages functional similarities across subjects using an Optimal transport
regularization.

User guide
----------

1. install **sparse-mtr:** sparse-mtr is a small package for MWE and all benchmark models.

.. code:: bash

  cd smtr
  python setup.py develop

2. run all preprocessing scripts `compute_*.py`. Make sure that paths to the data are
adapted in all the script `compute_*.py`, `config.py` and `brain_utils`.
The script `pick_labels.py` allows to select labels for sources simulation.

3. **ipmi-figures** contains scripts to reproduce all the experiments of the paper.
To plot, run first `results/merge_csv.py` with the correct file paths and then `plot_simulation.py`.
