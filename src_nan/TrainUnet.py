import torch
import torchvision
from DatasetCH import UpscaleDataset
import matplotlib.pyplot as plt
from Network import UNet
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np 

def train_step(model, loss_fn, data_loader, optimiser, scaler, step, accum=4,
               writer=None, device="cuda"):
    """
    Function for a single training step.
    :param model: instance of the Unet class
    :param loss_fn: loss function
    :param data_loader: data loader
    :param optimiser: optimiser to use
    :param scaler: scaler for mixed precision training
    :param step: current step
    :param accum: number of steps to accumulate gradients over
    :param writer: tensorboard writer
    :param device: device to use
    :return: loss value
    """
    print("THIS IS WITH NAN")

    model.train()
    step_loss = 0 # Initialize step_loss here

    with tqdm(total=len(data_loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {step}")

        epoch_losses = []
        for i, batch in enumerate(data_loader):
            tq.update(1)

            image_input = batch["inputs"].to(device)
            print(f"Batch input shape: {image_input.shape}")
            image_output = batch["targets"].to(device) # This is your ground truth fine image
            print(f"Batch output shape: {image_output.shape}")
            #day = batch["doy"].to(device)
            #hour = batch["hour"].to(device)
            #condition_params = torch.stack((day, hour), dim=1)

            # forward unet
            with torch.cuda.amp.autocast():
                model_out = model(image_input,
                                  class_labels=None)

                # --- START MODIFICATION for NaN handling in Loss Calculation ---
                # Identify non-NaN values in the ground truth images (image_output)
                non_nan_mask = ~torch.isnan(image_output)

                # Calculate the element-wise squared difference between prediction and ground truth
                squared_diff = (model_out - image_output)**2
                
                # Apply the mask to select only the squared differences from non-NaN pixels
                valid_squared_diff = squared_diff[non_nan_mask]

                # Calculate the mean only over valid pixels
                if valid_squared_diff.numel() > 0: # Ensure there are valid pixels to avoid division by zero
                    loss = torch.mean(valid_squared_diff)
                else:
                    # If no valid pixels in the current batch, return a zero loss
                    loss = torch.tensor(0.0, device=image_output.device, dtype=image_output.dtype)
                # --- END MODIFICATION ---

            # backpropagation
            scaler.scale(loss).backward()
            step_loss += loss.item() # Accumulate loss for print/tensorboard

            if (i + 1) % accum == 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)

                if writer is not None:
                    writer.add_scalar("Loss/train", step_loss / accum,
                                     step * len(data_loader) + i)
                step_loss = 0 # Reset accumulated loss after optimization step

            epoch_losses.append(loss.item()) # Append the actual loss for the batch
            tq.set_postfix_str(s=f"Loss: {loss.item():.4f}")


        mean_loss = sum(epoch_losses) / len(epoch_losses)
        tq.set_postfix_str(s=f"Loss: {mean_loss:.4f}")

    return mean_loss


@torch.no_grad()
def sample_model(model, dataloader, device="cuda"):
    """
    Function for sampling the model.
    :param model: instance of the Unet class
    :param dataloader: data loader
    """

    model.eval()

    # Get n_images from the dataloader
    batch = next(iter(dataloader))
    images_input = batch["inputs"].to(device)
    coarse, fine = batch["coarse"], batch["fine"] # coarse here is the original low-res
    
    #condition_params = torch.stack((batch["doy"].to(device),batch["hour"].to(device)), dim=1)
    residual = model(images_input, class_labels=None)
    predicted = dataloader.dataset.residual_to_fine_image(
        residual.detach().cpu(), coarse)
    
    print(f"Predicted output shape: {predicted.shape}")
    print(f"Predicted output min: {predicted.min().item()}, max: {predicted.max().item()}")
    print(f"Predicted output mean: {predicted.mean().item()}, std: {predicted.std().item()}")

    # Add these print statements:
    print(f"Shape of `coarse` (from batch, eg 8, 1, 11, 18: {coarse.shape}")
    print(f"Shape of `fine ground truth` (from batch), eg 8, 1, 128, 256: {fine.shape}")
    print(f"Shape of `images_input` (to model), is same as `coarse`: {images_input.shape}") # Should be same as `coarse`

    fig, ax = dataloader.dataset.plot_batch(coarse, fine, predicted)

    plt.subplots_adjust(wspace=0, hspace=0)

    # --- START MODIFICATION for NaN handling and MSE in Error Calculation ---
    # Ensure all tensors for error calculation are on CPU for consistent NaN handling
    # predicted is already on CPU due to .detach().cpu() above
    fine_cpu = fine.cpu()
    coarse_cpu = coarse.cpu()

    # Create a mask for non-NaN values in the ground truth (fine_cpu)
    non_nan_mask = ~torch.isnan(fine_cpu)

    # Apply the mask to all tensors to get only valid (non-NaN) pixels
    masked_fine = fine_cpu[non_nan_mask]
    masked_coarse = coarse_cpu[non_nan_mask]
    masked_predicted = predicted[non_nan_mask] # `predicted` is already on CPU

    # Calculate MSE for non-NaN values
    if masked_fine.numel() > 0: # Ensure there are valid pixels
        base_error = torch.mean((masked_fine - masked_coarse) ** 2).item() # MSE for coarse vs fine
        pred_error = torch.mean((masked_fine - masked_predicted) ** 2).item() # MSE for predicted vs fine
    else:
        # If no valid pixels, set errors to 0.0 or float('nan') as appropriate
        base_error = 0.0
        pred_error = 0.0
    # --- END MODIFICATION ---

    #return (fig, ax), (base_error.item(), pred_error.item())
    return (fig, ax), (base_error, pred_error), predicted.numpy() # Convert to numpy for easy access


def main():
    batch_size = 8
    learning_rate = 3e-5
    num_epochs = 10000
    accum = 8

    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # define the ml model
    #unet_model = UNet((256, 128), 5, 3, label_dim=0, use_diffuse=False)
    unet_model = UNet((256, 128), 1, 1, label_dim=0, use_diffuse=False)
    unet_model.to(device)

    # define the datasets
    ifs_dir = '/s2s/mpyrina/ECMWF_MCH/Europe_eval/s2s_hind_2022/all/'
    obs_dir = '/net/cfc/s2s_nobackup/mpyrina/TABSD_ifs_like/'

    dataset_train = UpscaleDataset(coarse_data_dir = ifs_dir, highres_data_dir = obs_dir,
    year_start=2005, year_end=2008, month=815,  
    constant_variables=None, constant_variables_filename=None)

    dataset_test = UpscaleDataset(coarse_data_dir = ifs_dir, highres_data_dir = obs_dir,
    year_start=2009, year_end=2010, month=815,  
    constant_variables=None, constant_variables_filename=None)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)


    scaler = torch.cuda.amp.GradScaler()

    # define the optimiser
    optimiser = torch.optim.AdamW(unet_model.parameters(), lr=learning_rate)

    # Define the tensorboard writer
    writer = SummaryWriter("./runs_unet")

    loss_fn = torch.nn.MSELoss() # This is instantiated as a standard MSELoss, but our train_step will handle the masking

    # train the model
    losses = []
    for step in range(6, num_epochs):
        epoch_loss = train_step(
            unet_model, loss_fn, dataloader_train, optimiser,
            scaler, step, accum, writer)
        losses.append(epoch_loss)

        if (step + 0) % 5 == 0:
            (fig, ax), (base_error, pred_error) = sample_model(
                unet_model, dataloader_test)
            fig.savefig(f"./results_unet/{step}.png", dpi=300)
            plt.close(fig)

            writer.add_scalar("Error/base", base_error, step)
            writer.add_scalar("Error/pred", pred_error, step)

        # save the model
        if losses[-1] == min(losses):
            torch.save(unet_model.state_dict(), f"./Models_unet/{step}.pt")

if __name__ == "__main__":
    main()