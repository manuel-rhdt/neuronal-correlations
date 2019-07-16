# Coarse–graining, fixed points, and scaling in a large population of neurons

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/manuel-rhdt/neuronal-correlations/master) **[Show Notebook in Github](Notebook.ipynb)**

Work in progress implementing the data analysis from the paper.

Usage:

- Look at `Notebook.ipynb` to see plots, etc. The notebook assumes a file named `data.tif` to exist at the working directory.
- `neurenorm.py` contains the main code.
- `test.py` contains tests that the code is working as intended.

The pixel size of the images is 814 × 788.

The data and code are partially inspired by

    @article{giovannucci2019caiman,
      title={CaImAn: An open source tool for scalable Calcium Imaging data Analysis},
      author={Giovannucci, Andrea and Friedrich, Johannes and Gunn, Pat and Kalfon, Jeremie and Brown, Brandon L and Koay, Sue Ann and Taxidis, Jiannis and Najafi, Farzaneh and Gauthier, Jeffrey L and Zhou, Pengcheng and Khakh, Baljit S and Tank, David W and Chklovskii, Dmitri B and Pnevmatikakis, Eftychios A},
      journal={eLife},
      volume={8},
      pages={e38173},
      year={2019},
      publisher={eLife Sciences Publications Limited}
    }


# What to do
- [x] Make a fit for the rank plot (see how the critical exponents change for random clustering)
- [ ] Try to cluster neighbouring neurons (disregarding correlations)
- [ ] reproduce the author's final plot