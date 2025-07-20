import torch
import Network
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from DatasetCH import UpscaleDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import xarray as xr


# Loss class taken from EDS_Diffusion/loss.py
class EDMLoss:

    # The model learns to denoise D_yn = y + n (which is the fine image plus noise) based on the coarse input.
    
    # Initializes constants used in the Epsilon Denoising Matching (EDM) loss.
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, conditional_img=None, labels=None,
             augment_pipe=None, mask=None):
        # Learn to denoise noisy inputs
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        # This line generates a random noise scale sigma from a log-normal distribution.
        #rnd_normal is standard normal noise (~ N(0, 1)), shape [batch_size, 1, 1, 1].
        #P_mean and P_std define the mean and std in log-space of sigma.
        #Applying .exp() transforms it from log-space to real space.
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            # This weighting ensures the model learns well across the whole range of noise levels,
        # by computing a per-sample loss weight, based on the noise level sigma.
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data)**2

        # Apply optional data augmentation
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma

        # PASS label to the model!
        D_yn = net(y + n, sigma, conditional_img, labels,
               augment_labels=augment_labels)

        mse = (D_yn - y) ** 2
        if mask is not None:
            loss = weight * mse * mask
            loss = loss.sum() / mask.sum().clamp(min=1.0)
            print("THE MASK IS USED")
        else:
            loss = weight * mse
            loss = loss.mean()
            print("THE MASK IS NOT USED")

        return loss

def training_step(model, loss_fn, optimiser, data_loader, scaler, step,
                  accum=4, writer=None, device="cuda"):
    #  Uses mixed-precision (scaler) and gradient accumulation.
    """
    Function for a single training step.
    :param model: Instance of the Unet class
    :param loss_fn: Loss function
    :param optimiser: Optimiser to use
    :param data_loader: Data loader
    :param scaler: Scaler for mixed precision training
    :param step: Current step
    :param accum: Number of steps to accumulate gradients over
    :param writer: Tensorboard writer
    :param device: Device to use
    :return: Loss value
    """

    model.train()
    with tqdm(total=len(data_loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {step}")

        epoch_losses = []
        step_loss = 0
        for i, batch in enumerate(data_loader):
            tq.update(1)
            image_input = batch["inputs"].to(device)
            image_output = batch["targets"].to(device)
            labels = batch["label"].to(device)
            #mask = batch["mask"].to(device) if batch["mask"] is not None else None
            # to make sure that mask is a tensor before calling .to(device)
            mask = batch.get("mask", None)
            mask = mask.to(device) if mask is not None else None


            #day = batch["doy"].to(device)
            #hour = batch["hour"].to(device)
            #condition_params = torch.stack((day, hour), dim=1)

            # forward unet
            with torch.cuda.amp.autocast():
                ## Compute loss with mixed precision
                loss = loss_fn(net=model, images=image_output,
                               conditional_img=image_input, labels=labels, mask=mask) # i removed labels=condition_params
                loss = torch.mean(loss)

            # backpropagation
            scaler.scale(loss).backward()
            step_loss += loss.item()

            if (i + 1) % accum == 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)

                if writer is not None:
                    writer.add_scalar("Loss/train", step_loss / accum,
                                      step * len(data_loader) + i)
                step_loss = 0

            epoch_losses.append(loss.item())
            tq.set_postfix_str(s=f"Loss: {loss.item():.4f}")
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        tq.set_postfix_str(s=f"Loss: {mean_loss:.4f}")
    return mean_loss

#This is the sampling process for generating high-res outputs from low-res inputs using the trained diffusion model.
#Samples noise, Applies Euler + 2nd order correction steps, Each step denoises with the model to gradually refine the image
@torch.no_grad()
def sample_model_dif(model, dataloader, num_steps=40, sigma_min=0.002,
                 sigma_max=80, rho=7, S_churn=40, S_min=0,
                 S_max=float('inf'), S_noise=1, device="cuda"):

    batch = next(iter(dataloader))
    labels = batch["label"].to(device)
    images_input = batch["inputs"].to(device) # These are my upscaled coarse data
    print(f"Batch input shape: {images_input.shape}")
    # The output PREDICTION is a residual that the network learns to denoise — i.e., the difference between high-res and coarse.
    coarse, fine = batch["coarse"], batch["fine"]

    #condition_params = torch.stack(
        #(batch["doy"].to(device),
         #batch["hour"].to(device)), dim=1)

    sigma_min = max(sigma_min, model.sigma_min)
    sigma_max = min(sigma_max, model.sigma_max)

    #init_noise = torch.randn((images_input.shape[0], 3, images_input.shape[2],
                              #images_input.shape[3]),
                             #dtype=torch.float64, device=device)
    init_noise = torch.randn((images_input.shape[0], 1, images_input.shape[2], # Change 3 to 1 here
                              images_input.shape[3]),
                             dtype=torch.float64, device=device)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64,
                                device=init_noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
               * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([model.round_sigma(t_steps),
                         torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = init_noise.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = model.round_sigma(t_cur + gamma * t_cur)
        x_hat = (x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise *
                 torch.randn_like(x_cur))

        # Euler step.
        denoised = model(x_hat, t_hat, images_input, labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = model(x_next, t_next, images_input, labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    # The prediction is not the full high-res image but a correction to the coarse image (i.e., residual).
    #Converts the model's output (residual) into a full high-resolution image.
    predicted = dataloader.dataset.residual_to_fine_image(
        x_next.detach().cpu(), coarse)

    print(f"Predicted output shape: {predicted.shape}")
    print(f"Predicted output min: {predicted.min().item()}, max: {predicted.max().item()}")
    print(f"Predicted output mean: {predicted.mean().item()}, std: {predicted.std().item()}")

    fig, ax = dataloader.dataset.plot_batch(coarse, fine, predicted)

    plt.subplots_adjust(wspace=0, hspace=0)
    base_error = torch.mean(torch.abs(fine - coarse))
    pred_error = torch.mean(torch.abs(fine - predicted))

    return (fig, ax), (base_error.item(), pred_error.item()), predicted.cpu().numpy() 


def main():
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 10000
    accum = 8

    import os
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./Model", exist_ok=True)

    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ## Initialize diffusion model
    #network = Network.EDMPrecond((256, 128), 8, 3, label_dim=0) # label referes to the doy,hour .. condition params, since this doesnt exist i put to zero
    network = Network.EDMPrecond(
    img_resolution=(256, 128),   # (W_out, H_out) — note this is (width, height)
    in_channels=2,                # base channel size (can be tuned) NUM OF PRED VARS
    out_channels=1,     # target variable channels (e.g. temperature)
                        # in this case 2, as 1 for coarse image and one for fine image (Since you're always concatenating the target + coarse image, in_channels must be 2)
    label_dim=1         # set to 0 if you're not using extra inputs like day/hour
)
    network.to(device)

    # define the datasets
    ifs_dir = '/s2s/mpyrina/ECMWF_MCH/Europe_eval/s2s_hind_2022/all/'
    obs_dir = '/net/cfc/s2s_nobackup/mpyrina/TABSD_ifs_like/'
    mask_dir = '/net/cfc/s2s_nobackup/mpyrina/TABSD_ifs_like/TabsD_mask_static.nc'

    dataset_train = UpscaleDataset(coarse_data_dir = ifs_dir, highres_data_dir = obs_dir,
    year_start=2005, year_end=2008, month=815,  
    constant_variables=None, constant_variables_filename=None, mask_path=mask_dir)

    dataset_test = UpscaleDataset(coarse_data_dir = ifs_dir, highres_data_dir = obs_dir,
    year_start=2009, year_end=2010, month=815,  
    constant_variables=None, constant_variables_filename=None, mask_path=mask_dir)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)

    scaler = torch.cuda.amp.GradScaler()

    # define the optimiser
    optimiser = torch.optim.AdamW(network.parameters(), lr=learning_rate)

    # Define the tensorboard writer
    writer = SummaryWriter("./runs")

    # define loss function
    loss_fn = EDMLoss()

    # train the model
    losses = []
    for step in range(0, num_epochs):
        epoch_loss = training_step(network, loss_fn, optimiser,
                                   dataloader_train, scaler, step,
                                   accum, writer)
        losses.append(epoch_loss)
        
        # save the model
        if losses[-1] == min(losses):
            torch.save(network.state_dict(), f"./Model/{step}.pt")
        
        if (step + 0) % 5 == 0:
            # Plot and save
            (fig, ax), (base_error, pred_error), predicted_numpy_array = sample_model_dif(network, dataloader_test, device=device)
            plt.show()
            fig.savefig(f"./results/{step}.png", dpi=300)
            plt.close(fig)
            writer.add_scalar("Error/base", base_error, step)
            writer.add_scalar("Error/pred", pred_error, step)




if __name__ == "__main__":
    main()

