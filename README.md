## The following project has been adapted by M. Pyrina, 2026 from the following:

## Generative diffusion-based downscaling for climate
#### Robbie A. Watt & Laura A. Mansfield      <https://arxiv.org/abs/2404.17752> using the implementation by T. Karras et al. (<https://arxiv.org/abs/2206.00364>) and code addapted from <https://github.com/NVlabs/edm>.

This repo contains code to go alongside "Joint Bias Correction and Downscaling of Subseasonal Forecasts via Diffusion Models" (2026) preprint. In this work, we apply a diffusion based model (DM) to a downscaling and bias correction task using ECMWF subseasonal temperature hindcast data and gridded temperature observations from MeteoSwiss over Switzerland.


## File structure
* src_mean: contains code used to train, validate, and test the DM model using the ensemble mean (used in the manuscript)
* src_mems: contains code used to train, validate, and test the DM model using the ensemble members (not used in the manuscript, only partly tested)
* Model_chpt: contains model checkpoints for the src_mean training

### Data
We are using ECMWF subseasonal hindcast data and gridded observational data from MeteoSwiss.

### Dependencies
python>=3.9, torch, tensorboard, xarray, netcdf4, cartopy, matplotlib, scipy, numpy

## Citation of original code
```
@misc{watt2024generative,
      title={Generative Diffusion-based Downscaling for Climate}, 
      author={Robbie A. Watt and Laura A. Mansfield},
      year={2024},
      eprint={2404.17752},
      archivePrefix={arXiv},
      primaryClass={physics.ao-ph}
}
```

## Current work
```
@misc{pyrina2026_ch_downscaling,
      title={Joint Bias Correction and Downscaling of Subseasonal Forecasts via Diffusion Models}, 
      author={M. Pyrina, A. Imamovic, D. Büeler, C. Spirig, D. I. V. Domeisen},
      year={2026},
      eprint=xxx,
      archivePrefix={arXiv},
      primaryClass=xxx
}
```

