import torch
import torchvision.transforms as T
import numpy as np
import xarray as xr

class UpscaleDataset(torch.utils.data.Dataset):
    """
    Dataset class for ensemble downscaling SWITZERLAND. 
    """

    def __init__(self, coarse_data_dir, highres_data_dir, month, year_start, year_end,
                 in_shape=(16, 32), # my coarse CH shape 
                 out_shape=(128, 256), # my fine CH shape expanded from 103, 241
                 constant_variables=None,
                 constant_variables_filename=None, mask_path=None
                 ):
        """
        :param coarse_data_dir: path to coarse-resolution dataset
        :param highres_data_dir: path to high-resolution dataset
        :param in_shape: shape of the low-resolution input
        :param out_shape: shape of the high-resolution output
        :param year_start: starting year
        :param year_end: ending year
        :param constant_variables: list of constant vars (e.g., orography)
        :param constant_variables_filename: filename for constants
        """

        # Subsetting region from the coarse data
        west, east = 2, 14.4
        south, north = 44.4, 50.4

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.constant_variables = constant_variables
        
        # IFS data are in Kelvin and have no nan values.
        self.coarse_filenames = [f"IFS_EXT_HC_{year}0{month}_T_2M.nc" for year in range(year_start, year_end)]
        self.coarse_filepaths = [coarse_data_dir + coarse_filename for coarse_filename in self.coarse_filenames]
        # TabsD data have been tranfrormed to Kelvin and instead of nan there are zeros. A static mask is additionaly used for training.
        self.highres_filenames = [f"TabsD_expanded_EXT_HC_{year}0{month}_T_2M.nc" for year in range(year_start, year_end)]
        self.highres_filepaths = [highres_data_dir + highres_filename for highres_filename in self.highres_filenames]

        # Open first coarse file
        print("Test - new upscale")
        ds_coarse_c = xr.open_dataset(coarse_data_dir + self.coarse_filenames[0], engine="netcdf4")
        ds_coarse = ds_coarse_c.sel(x_1=slice(west, east)).sel(y_1=slice(south, north))
        
        # Only select T_2M
        self.varnames = ["T_2M"]

        # Dimensions
        self.ensemble_dim = ds_coarse.dims.get("epsd_1", 1)
        self.time_dim = ds_coarse.dims.get("time")
        self.lat_coarse = ds_coarse.coords["y_1"]
        self.lon_coarse = ds_coarse.coords["x_1"]

        self.n_var = len(self.varnames)

        # Load all coarse years
        ds_all_coarse_c = xr.open_mfdataset(self.coarse_filepaths, engine='netcdf4')
        ds_all_coarse = ds_all_coarse_c.sel(x_1=slice(west, east)).sel(y_1=slice(south, north))
        print(f"Loaded coarse data shape: {ds_all_coarse['T_2M'].shape}")

        # Now load high-resolution dataset
        ds_all_highres = xr.open_mfdataset(self.highres_filepaths, engine='netcdf4')
        print(f"Loaded high-resolution data shape: {ds_all_highres['TabsD'].shape}")

        # Save number of time steps
        self.ntime_total = ds_all_coarse.dims["time"]

        # Load coarse data
        self.data_coarse = torch.from_numpy(ds_all_coarse["T_2M"].values).float()
        self.data_coarse = self.data_coarse.unsqueeze(2)  # (time, member, channel, y, x)
        self.data_coarse = self.data_coarse.permute(0, 1, 2, 3, 4)  # (time, member, channel, y, x)
        self.data_coarse = self.data_coarse.reshape(-1, self.n_var, len(self.lat_coarse), len(self.lon_coarse))

        # Load high-res fine data
        self.data_fine = torch.from_numpy(ds_all_highres["TabsD"].values).float()
        self.data_fine = self.data_fine.unsqueeze(1)  # (time, channel, y, x)
        self.data_fine = self.data_fine.repeat_interleave(self.ensemble_dim, dim=0)  # Repeat for each ensemble member
        self.data_fine = self.data_fine.reshape(-1, self.n_var, self.out_shape[0], self.out_shape[1])

        print(f"Final coarse shape: {self.data_coarse.shape}")
        print(f"Final fine shape: {self.data_fine.shape}")

        # Interpolation operators
        self.coarsen = T.Resize(self.in_shape,
                                interpolation=T.InterpolationMode.BILINEAR,
                                antialias=True)
        self.upscale = T.Resize(self.out_shape,
                                interpolation=T.InterpolationMode.BILINEAR,
                                antialias=True)

        # Precompute coarse input
        coarse_temp = self.coarsen(self.data_coarse)
        coarse_upscaled = self.upscale(coarse_temp)

        self.inputs = coarse_upscaled
        print("Input shape (should be [N, 1, H, W]):", self.inputs.shape)
        self.targets = self.data_fine
        
        if mask_path is not None:
            print("Loading static mask...")
            mask_ds = xr.open_dataset(mask_path)
            mask_array = mask_ds["mask"].values.astype(np.float32)  # [lat, lon]
            self.mask = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            self.mask = self.mask.expand(self.targets.shape[0], -1, -1, -1)    # [N, 1, H, W]
            assert self.mask.shape[-2:] == self.out_shape, f"Mask shape {self.mask.shape[-2:]} != out_shape {self.out_shape}"
        else:
            self.mask = None

        self.fine = self.data_fine
        self.coarse = coarse_upscaled

        # Constant variables
        if self.constant_variables is not None and constant_variables_filename is not None:
            print("Loading constant variables...")
            ds_const = xr.open_dataset(coarse_data_dir + constant_variables_filename, engine="netcdf4")
            const_vars = []

            for var in self.constant_variables:
                const = torch.from_numpy(ds_const[var].values).float()
                # Broadcast to time * member dimension
                const = const.unsqueeze(0).expand(self.inputs.shape[0], -1, -1)
                const_vars.append(const)

            const_vars = torch.stack(const_vars, dim=1)  # (time*member, n_const_vars, lat, lon)
            self.inputs = torch.cat([self.inputs, const_vars], dim=1)  # concat along channels

        print("Dataset ready.")


    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return {
            "inputs": self.inputs[idx],
            "targets": self.targets[idx],
            "fine": self.fine[idx],
            "coarse": self.coarse[idx],
            "mask": self.mask[idx] if self.mask is not None else None,
        }

    def residual_to_fine_image(self, residual, coarse):
        """
        Combines the predicted residual with the coarse input to produce the final high-resolution prediction.
        :param residual: Predicted residual (tensor of shape B x C x H x W)
        :param coarse: Upscaled coarse input (tensor of shape B x C x H x W)
        :return: Predicted fine-resolution tensor
        """
        return residual + coarse

    def plot_batch(self, coarse, fine, predicted):
        """
        Plots coarse input, ground truth fine output, and model prediction for visual inspection.
        :param coarse: Upscaled coarse inputs
        :param fine: Ground truth fine-resolution images
        :param predicted: Model predictions
        :return: Figure and axes
        """
        import matplotlib.pyplot as plt

        n_samples = min(4, coarse.shape[0])  # Show at most 4 samples
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))

        for i in range(n_samples):
            for j, data in enumerate([coarse, fine, predicted]):
                ax = axes[i, j] if n_samples > 1 else axes[j]
                im = ax.imshow(data[i, 0].cpu().numpy(), cmap="coolwarm", origin='lower')
                ax.axis("off")
                if i == 0:
                    title = ["Upscaled Coarse", "Ground Truth", "Prediction"][j]
                    ax.set_title(title, fontsize=12)

        return fig, axes

