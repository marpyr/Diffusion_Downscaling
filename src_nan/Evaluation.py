import torch
from tqdm import tqdm

def evaluate_model(model, loss_fn, dataloader_test, device="cuda"):
    """
    Evaluates the model's performance over the entire test dataset.
    :param model: The trained UNet model.
    :param loss_fn: The loss function (e.g., torch.nn.MSELoss()).
    :param dataloader_test: DataLoader for the test dataset.
    :param device: Device to perform computations on ('cuda' or 'cpu').
    :return: Tuple of (average_pred_error, average_base_error)
    """
    model.eval()  # Set the model to evaluation mode
    total_pred_error = 0.0
    total_base_error = 0.0
    num_batches = 0

    with torch.no_grad():  # Disable gradient calculation for inference
        for batch in tqdm(dataloader_test, desc="Evaluating Model"):
            image_input = batch["inputs"].to(device)
            # Ensure `coarse` is the upscaled coarse image as per our previous discussion
            # If you implemented the fix in DatasetCH.py, `batch["coarse"]` will be upscaled.
            # Otherwise, you might need to upscale `image_input` here for base_error calculation
            # if you want to compare against an upscaled coarse.
            coarse_for_error_calc = batch["coarse"].to(device) # This should be the upscaled coarse from DatasetCH.py

            fine = batch["targets"].to(device)

            predicted = model(image_input) # Get the model's prediction

            # If your model predicts a residual, combine it with the coarse image
            # Make sure your DatasetCH.py's residual_to_fine_image is imported if needed here
            # For a direct UNet (like in TrainUnet.py), predicted is often the fine image directly.
            # If your UNet predicts residual, you'll need:
            # from DatasetCH import UpscaleDataset # assuming it's available
            # predicted_fine = UpscaleDataset(None,None,0,0,0).residual_to_fine_image(predicted, coarse_for_error_calc)
            # Else, if UNet directly predicts fine image:
            predicted_fine = predicted


            # Calculate errors using absolute difference as in your sample_model
            batch_base_error = torch.mean(torch.abs(fine - coarse_for_error_calc)).item()
            batch_pred_error = torch.mean(torch.abs(fine - predicted_fine)).item()

            total_base_error += batch_base_error
            total_pred_error += batch_pred_error
            num_batches += 1

    avg_base_error = total_base_error / num_batches
    avg_pred_error = total_pred_error / num_batches

    return avg_pred_error, avg_base_error