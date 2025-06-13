import torch
import torchvision.transforms as T
import numpy as np
import xarray as xr

class UpscaleDataset_old_ch(torch.utils.data.Dataset):
    """
    Dataset class for ensemble downscaling. 
    """

    def __init__(self, data_dir, month=0815,
                 in_shape=(11, 18),  # my coarse CH shape
                 out_shape=(103, 241),  # my fine shape file
                 year_start=2002, year_end=2021,
                 constant_variables=None,
                 constant_variables_filename=None
                 ):
        """
        :param data_dir: path to dataset
        :param in_shape: shape of the low-resolution input
        :param out_shape: shape of the high-resolution output
        :param year_start: starting year
        :param year_end: ending year
        :param constant_variables: list of constant vars (e.g., orography)
        :param constant_variables_filename: filename for constants
        """
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.constant_variables = constant_variables

        self.filenames = [f"IFS_EXT_HC_{year}{month}_T_2M.nc" for year in range(year_start, year_end)]

        # Open first file
        ds = xr.open_dataset(data_dir + self.filenames[0], engine="netcdf4")

        # Dimensions
        self.ensemble_dim = ds.dims.get("epsd_1", 1)
        self.time_dim = ds.dims.get("time")
        self.lat_fine = ds.coords["x_1"]
        self.lon_fine = ds.coords["y_1"]

        self.varnames = list(ds.data_vars.keys())  # assuming variables are all what you want
        self.n_var = len(self.varnames)

        # Load all years
        #ds_all = ds
        #for file in self.filenames[1:]:
            #ds = xr.open_dataset(data_dir + file, engine="netcdf4")
            #ds_all = xr.concat([ds_all, ds], dim="time")
        
        ds_all = 
        print(f"Loaded data shape: {ds_all[self.varnames[0]].shape}")
        self.ntime_total = ds_all.dims["time"]

        # Assume shape: (time, member, lat, lon)
        self.data = torch.stack(
            [torch.from_numpy(ds_all[var].values).float() for var in self.varnames],
            dim=2  # (time, member, channel, lat, lon)
        )

        # Bring data into (time * member, channel, lat, lon)
        self.data = self.data.permute(0,1,2,3,4)  # (time, member, channel, lat, lon)
        self.data = self.data.reshape(-1, self.n_var, len(self.lat_fine), len(self.lon_fine))

        print(f"Final data shape after stacking: {self.data.shape}")

        # Interpolation operators
        self.coarsen = T.Resize(self.in_shape,
                                interpolation=T.InterpolationMode.BILINEAR,
                                antialias=True)
        self.upscale = T.Resize(self.out_shape,
                                interpolation=T.InterpolationMode.BILINEAR,
                                antialias=True)

        # Precompute coarse and residual
        coarse_temp = self.coarsen(self.data)
        coarse_upscaled = self.upscale(coarse_temp)
        residual = self.data - coarse_upscaled

        self.inputs = coarse_upscaled
        self.targets = residual
        self.fine = self.data
        self.coarse = coarse_upscaled

        # Constant variables
        if self.constant_variables is not None and constant_variables_filename is not None:
            print("Loading constant variables...")
            ds_const = xr.open_dataset(data_dir + constant_variables_filename, engine="netcdf4")
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
        }
