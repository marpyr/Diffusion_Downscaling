import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import os
import Network
from DatasetCH import *
from TrainDiffusion import *

# Make dirs
mdir="/.../Model_dif/norm_Test_4"
rdir="/.../Results_dif/norm_Test_4"
os.makedirs(mdir, exist_ok=True)
os.makedirs(rdir, exist_ok=True)
# Define the tensorboard writer
writer = SummaryWriter(mdir) # was runs_unet

# TRAIN THE MODEL
batch_size = 16
learning_rate = 1e-5
num_epochs = 2000
resume_from_epoch = 120
accum = 4

print('cuda is available : ',torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# a tensor of shape [B, C, H, W] mean that c=8, image resol=(H, W) 

network = Network.EDMPrecond(
        img_resolution=(256, 128),
        in_channels=2,
        out_channels=1,
        label_dim=1
    ).to(device)

# RESUME TRAINING
# -------------------------------------------------------------
resume_path = f"{mdir}/last_checkpoint.pt"
scaler = torch.cuda.amp.GradScaler()
optimiser = torch.optim.AdamW(network.parameters(), lr=learning_rate)

if os.path.exists(resume_path):
    print(f"Resuming training from {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)
    network.load_state_dict(checkpoint["model_state"])
    optimiser.load_state_dict(checkpoint["optimizer_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    resume_from_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
else:
    print("Starting training from scratch.")
    resume_from_epoch = 0
    best_val_loss = float("inf")
# -------------------------------------------------------------


# define the datasets
ifs_dir = '/.../mpydownscaling/DATA/'
obs_dir = '/.../mpydownscaling/DATA/'
mask_dir = '/.../mpydownscaling/DATA/TabsD_mask_static.nc'

# training data 2002-2018
dataset_train = UpscaleDataset(coarse_data_dir = ifs_dir, highres_data_dir = obs_dir,
year_start=2002, year_end=2019, month=815,  
constant_variables=None, constant_variables_filename=None, mask_path=None)

# i say test but its validation data 2019 (later run actual test on 2020)
dataset_test = UpscaleDataset(coarse_data_dir = ifs_dir, highres_data_dir = obs_dir,
year_start=2019, year_end=2020, month=815,  
constant_variables=None, constant_variables_filename=None, mask_path=None)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)

# Train
#scaler = torch.cuda.amp.GradScaler()
#optimiser = torch.optim.AdamW(network.parameters(), lr=learning_rate)

loss_fn = EDMLoss()
losses = []

for step in range(resume_from_epoch, num_epochs):
    # model_save
    model_save_path = f"{mdir}/dif_model_epoch_{step}.pt"
    # fig_save
    fig_save_path = f"{rdir}/dif_model_{step}.png"
    # best modes
    mbest = f"{mdir}/best_dif_model_epoch_{step}.pt"

    epoch_loss = training_step(network, loss_fn, optimiser,
                                   dataloader_train, scaler, step,
                                   accum, writer, device=device)
    losses.append(epoch_loss)
    # log the epoch loss
    writer.add_scalar("Loss/epoch_loss", epoch_loss, step)
    
    # Save the model weights
    torch.save(network.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    if losses[-1] == min(losses):
        torch.save(network.state_dict(), mbest)

    ###########
    # Track also validation loss to plot it

    network.eval()
    val_losses = []
    with torch.no_grad():
        for batch in dataloader_test:
            image_input = batch["inputs"].to(device)
            image_output = batch["targets"].to(device)
            labels = batch["label"].to(device)
            mask = batch.get("mask", None)
            mask = mask.to(device) if mask is not None else None

            val_loss = loss_fn(
                net=network,
                images=image_output,
                conditional_img=image_input,
                labels=labels,
                mask=mask
            )
            val_losses.append(val_loss.item())

    val_loss_mean = np.mean(val_losses)
    writer.add_scalar("Loss/val_epoch", val_loss_mean, step)

    # Save full state (last checkpoint) so that we resume training from here
    checkpoint = {
        "epoch": step,
        "model_state": network.state_dict(),
        "optimizer_state": optimiser.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val_loss": best_val_loss,
    }

    torch.save(checkpoint, f"{mdir}/last_checkpoint.pt")

    # save the best model weights during validation, to be used for evaluation/ deployment
    if val_loss_mean < best_val_loss:
        best_val_loss = val_loss_mean
        best_path = f"{mdir}/best_model.pt"
        torch.save(network.state_dict(), best_path)
        print(f"New best model saved at epoch {step} | val_loss={val_loss_mean:.4f}")

    ###########
        
    # Plot and save
    if step % 5 == 0:
        (fig, ax), (base_error, pred_error), predicted_numpy_array = sample_model_dif(network, dataloader_test, device=device)
        plt.show()
        fig.savefig(fig_save_path, dpi=300)
        plt.close(fig)
        writer.add_scalar("Error/base", base_error, step)
        writer.add_scalar("Error/pred", pred_error, step)

        # Save prediction arrays for reproducibility
        # Grab one batch again to align with the predictions
        batch = next(iter(dataloader_test))
        coarse = batch["coarse"].cpu().numpy()
        fine   = batch["fine"].cpu().numpy()
        labels = batch["label"].cpu().numpy()
    
        np.savez_compressed(
            f"{rdir}/predictions_{step}.npz",
            coarse=coarse,
            fine=fine,
            predicted=predicted_numpy_array,
            labels=labels,   # store lead times
        )

