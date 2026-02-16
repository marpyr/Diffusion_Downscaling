from torch.utils.data import DataLoader
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import Network
from DatasetCH import *
from TrainDiffusion import *

# -------------------------------
# Paths
# -------------------------------

# "Traditional" downscaling test, training from data of week_1

# I take the best model from the ensemble members training (src_mems), but apply the learned funtion of "deterministic" diffussion to the ifs file that contains all members. In this way I perform a "traditional" downscaling, so I do not generate members through the diffussion process, but downscale the existing 10 IFS members.

# NOTE: Here the upscaledataset function (DatasetCH.py) takes as input all IFS members and not their mean


best_model_path = "/.../Model_dif/norm_Test_5/best_model.pt"
ifs_dir = '/.../mpydownscaling/DATA/'
obs_dir = '/.../mpydownscaling/DATA/'
mask_path = '/.../mpydownscaling/DATA/TabsD_mask_static.nc'
results_dir = "/.../Results_dif/norm_Test_5/test_predictions"
os.makedirs(results_dir, exist_ok=True)

# -------------------------------
# Device
# -------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)


# -------------------------------
# Define sampling function
# -------------------------------
@torch.no_grad()
def sample_model_EDS(input_batch, model, device, dataset, num_steps=40,
                     sigma_min=0.002, sigma_max=80, rho=7, S_churn=40,
                     S_min=0, S_max=float('inf'), S_noise=1):
    """Performs diffusion sampling to generate fine-resolution output."""
    images_input = input_batch["inputs"].to(device)
    coarse, fine = input_batch["coarse"], input_batch["fine"]
    labels = input_batch["label"].to(device)

    sigma_min = max(sigma_min, model.sigma_min)
    sigma_max = min(sigma_max, model.sigma_max)

    # Initialize Gaussian noise for the residual
    init_noise = torch.randn(
        (images_input.shape[0], 1, images_input.shape[2], images_input.shape[3]),
        dtype=torch.float64, device=device
    )

    # Discretize timesteps
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
               * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([model.round_sigma(t_steps),
                         torch.zeros_like(t_steps[:1])])  # t_N = 0

    x_next = init_noise.to(torch.float64) * t_steps[0]

    # Main diffusion loop
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        # Temporary noise increase
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = model.round_sigma(t_cur + gamma * t_cur)
        x_hat = (x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur))

        # Euler step
        denoised = model(x_hat, t_hat, images_input, labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # 2nd-order correction
        if i < num_steps - 1:
            denoised = model(x_next, t_next, images_input, labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    # Convert model residual output to full fine-resolution prediction
    predicted = dataset.residual_to_fine_image(x_next.detach().cpu(), coarse)
    return coarse, fine, predicted


# -------------------------------
# Load test dataset
# -------------------------------
dataset_test = UpscaleDataset(
    coarse_data_dir=ifs_dir,
    highres_data_dir=obs_dir,
    year_start=2020,
    year_end=2021,
    month=815,
    constant_variables=None,
    constant_variables_filename=None,
    mask_path=None
)

dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=4)


# -------------------------------
# Load trained model
# -------------------------------
network = Network.EDMPrecond(
    img_resolution=(256, 128),
    in_channels=2,
    out_channels=1,
    label_dim=1
).to(device)

network.load_state_dict(torch.load(best_model_path, map_location=device))
network.eval()
print("Model loaded successfully.")


# -------------------------------
# Inference loop
# -------------------------------
all_coarse, all_fine, all_predicted, all_labels = [], [], [], []

print("Starting inference...")
for batch in tqdm(dataloader_test, desc="Inference"):
    coarse, fine, pred_fine = sample_model_EDS(batch, network, device, dataset_test)

    # Collect outputs
    all_coarse.append(coarse.numpy())
    all_fine.append(fine.numpy())
    all_predicted.append(pred_fine.numpy())
    all_labels.append(batch["label"].numpy())

# Concatenate all batches
all_coarse = np.concatenate(all_coarse, axis=0)
all_fine = np.concatenate(all_fine, axis=0)
all_predicted = np.concatenate(all_predicted, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# -------------------------------
# Save results
# -------------------------------
save_path = os.path.join(results_dir, "predictions_test5_set.npz")
np.savez_compressed(
    save_path,
    coarse=all_coarse,
    fine=all_fine,
    predicted=all_predicted,
    labels=all_labels
)

print(f"Predictions saved at: {save_path}")

