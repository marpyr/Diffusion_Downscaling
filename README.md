### The following project has been adopted by M. Pyrina, 2026 from the following:

## Generative diffusion-based downscaling for climate
## Robbie A. Watt & Laura A. Mansfield      <https://arxiv.org/abs/2404.17752> using the implementation by T. Karras et al. (<https://arxiv.org/abs/2206.00364>) and code addapted from <https://github.com/NVlabs/edm>.

![plot](./example.png)

This repo contains code to go alongside "Joint Bias Correction and Downscaling of Subseasonal Forecasts via Diffusion Models" (2026) preprint. In this work, we apply a diffusion based model (DM) to a downscaling and bias correction task using ECMWF subseasonal temperature hindcast data and gridded temperature observations from MeteoSwiss over Switzerland.


## File structure
* src_mean: contains code used to train the DM model using the ensemble mean (used in the manuscript)
* src_mems: contains code used to train the DM model using the ensemble members (not used in the manuscript, only partly tested)
* inference_mean: contains inference scripts for the ensemble_mean training and generates 50 members from the DM distribution
* inference_mems: contains inference scripts for the ensemble_members training and applies the DM mean learned function to downscale the actuall ECMWF hindcasts (not used in the manuscript, only partly tested) 
* Model_chpt: contains model checkpoints for the src_mean training

## Usage
### Data
We are using ECMWF subseasonal hindcast data and gridded observational data from MeteoSwiss.

### Dependencies
python>=3.9, torch, tensorboard, xarray, netcdf4, cartopy, matplotlib, scipy, numpy

### Training
To train the diffusion model from scratch, simply run the `src_mean/TrainDiffusion.py` script from the project root directory.

### Inference (NOT ADAPTED YET)
After training, the inference scripts can be run in the following order:
1. `save_test_truth.py`: this script simply processes the true test data to save it into one file for easier comparison to other variables
2. `save_test_preds.py`: this script runs through all test data and saves the output into one file. You need to run this for each model. `modelname=UNet` for the standard UNet, `modelname=LinearInterpolation` for linear interpolation of coarse resolution variables onto the high resolution grid (i.e., the inputs to the model) and `modelname=Diffusion` for the diffusion model. When running the Diffusion model, we generate many possible samples in a loop, each seeded with a different random number, currently we loop over `rngs=range(0, 30)`.

After running the above scripts, you should have files saved as `output/{modelname}/samples_2018-2023.nc` (or for diffusion, these are saved as `output/Diffusion/samples_{i}_2018-2023.nc` where `i` indexes the different generated samples).

Plotting scripts:
* `plot_timestep_examples.py` plots maps of methods for each timestep (used for Fig. 1).
* `plot_error_metrics.py` plots maps of error metrics across full test dataset (Fig. 2) and prints the mean across the domain.
* `plot_spectrum.py` plots the power spectrum for all methods (Fig. 3)


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

