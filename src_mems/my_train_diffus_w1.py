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

# =====================================
# Make directories
# =====================================
mdir = "/.../Model_dif/norm_Test_5"
rdir = "/.../Results_dif/norm_Test_5"
os.makedirs(mdir, exist_ok=True)
os.makedirs(rdir, exist_ok=True)

writer = SummaryWriter(mdir)

# =====================================
# Training setup
# =====================================
batch_size = 16
learning_rate = 1e-5
num_epochs = 2000
resume_from_epoch = 120
accum = 4

print("cuda is available :", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================
# Model
# =====================================
network = Network.EDMPrecond(
    img_resolution=(256, 128),
    in_channels=2,
    out_channels=1,
    label_dim=1,
).to(device)

# =====================================
# Resume training (if checkpoint exists)
# =====================================
resume_path = f"{mdir}/last_checkpoint.pt"
optimiser = torch.optim.AdamW(network.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()

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

# =====================================
# Define datasets
# =====================================
ifs_dir = "/.../mpydownscaling/DATA/"
obs_dir = "/.../mpydownscaling/DATA/"
mask_dir = "/.../mpydownscaling/DATA/TabsD_mask_static.nc"

# --- Training data (2002–2018) ---
dataset_train = UpscaleDataset(
    coarse_data_dir=ifs_dir,
    highres_data_dir=obs_dir,
    year_start=2002,
    year_end=2019,
    month=815,
    constant_variables=None,
    constant_variables_filename=None,
    mask_path=None,
)

# --- Validation data (2019) ---
dataset_test = UpscaleDataset(
    coarse_data_dir=ifs_dir,
    highres_data_dir=obs_dir,
    year_start=2019,
    year_end=2020,
    month=815,
    constant_variables=None,
    constant_variables_filename=None,
    mask_path=None,
)

# =====================================
# Restrict to first 7 lead times
# =====================================
print("Restricting datasets to first 7 lead times...")
if hasattr(dataset_train, "data_coarse"):
    dataset_train.data_coarse = dataset_train.data_coarse[:7]
if hasattr(dataset_train, "data_fine"):
    dataset_train.data_fine = dataset_train.data_fine[:7]

if hasattr(dataset_test, "data_coarse"):
    dataset_test.data_coarse = dataset_test.data_coarse[:7]
if hasattr(dataset_test, "data_fine"):
    dataset_test.data_fine = dataset_test.data_fine[:7]

# =====================================
# Dataloaders
# =====================================
dataloader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, num_workers=4
)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=True, num_workers=4
)

# =====================================
# Training loop
# =====================================
loss_fn = EDMLoss()
losses = []

for step in range(resume_from_epoch, num_epochs):
    model_save_path = f"{mdir}/dif_model_epoch_{step}.pt"
    fig_save_path = f"{rdir}/dif_model_{step}.png"
    mbest = f"{mdir}/best_dif_model_epoch_{step}.pt"

    # Train one epoch
    epoch_loss = training_step(
        network, loss_fn, optimiser, dataloader_train, scaler, step, accum, writer, device=device
    )
    losses.append(epoch_loss)
    writer.add_scalar("Loss/epoch_loss", epoch_loss, step)

    # Save model
    torch.save(network.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    if losses[-1] == min(losses):
        torch.save(network.state_dict(), mbest)

    # =====================================
    # Validation
    # =====================================
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
                mask=mask,
            )
            val_losses.append(val_loss.item())

    val_loss_mean = np.mean(val_losses)
    writer.add_scalar("Loss/val_epoch", val_loss_mean, step)

    # Save checkpoint
    checkpoint = {
        "epoch": step,
        "model_state": network.state_dict(),
        "optimizer_state": optimiser.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, f"{mdir}/last_checkpoint.pt")

    # Save best model
    if val_loss_mean < best_val_loss:
        best_val_loss = val_loss_mean
        best_path = f"{mdir}/best_model.pt"
        torch.save(network.state_dict(), best_path)
        print(f"New best model saved at epoch {step} | val_loss={val_loss_mean:.4f}")

    # =====================================
    # Plot and save predictions
    # =====================================
    if step % 5 == 0:
        (fig, ax), (base_error, pred_error), predicted_numpy_array = sample_model_dif(
            network, dataloader_test, device=device
        )
        plt.show()
        fig.savefig(fig_save_path, dpi=300)
        plt.close(fig)

        writer.add_scalar("Error/base", base_error, step)
        writer.add_scalar("Error/pred", pred_error, step)

        # Save sample predictions
        batch = next(iter(dataloader_test))
        coarse = batch["coarse"].cpu().numpy()
        fine = batch["fine"].cpu().numpy()
        labels = batch["label"].cpu().numpy()

        np.savez_compressed(
            f"{rdir}/predictions_{step}.npz",
            coarse=coarse,
            fine=fine,
            predicted=predicted_numpy_array,
            labels=labels,
        )

