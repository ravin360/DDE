import torch
def compute_metrics(predicted, target, thresholds=[1.0, 2.0, 3.0]):
    # Compute MAE
    mae = torch.mean(torch.abs(predicted - target)).item()

    # Compute RMSE
    rmse = torch.sqrt(torch.mean((predicted - target) ** 2)).item()

    # Compute Bad Pixel Error for each threshold
    bad_pixel_errors = {}
    for threshold in thresholds:
        bad_pixel_percentage = torch.mean((torch.abs(predicted - target) > threshold).float()).item() * 100
        bad_pixel_errors[f"Bad{threshold}"] = bad_pixel_percentage

    return mae, rmse, bad_pixel_errors