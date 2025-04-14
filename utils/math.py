import torch


def z_score(x, mean, std):
    """
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: torch array, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: torch array, z-score normalized array.
    """
    return (x - mean) / std


def un_z_score(x_normed, mean, std):
    """
    Undo the Z-score calculation
    :param x_normed: torch array, input array to be un-normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    """
    return x_normed * std + mean


def MAPE(v, v_, threshold=0.1):
    """
    Filtered Mean Absolute Percentage Error.
    Ignores values where ground truth is below a threshold.
    :param v: torch tensor, ground truth.
    :param v_: torch tensor, prediction.
    :param threshold: scalar, minimum value of ground truth to consider.
    :return: torch scalar, MAPE (%), averaged over filtered values.
    """
    mask = torch.abs(v) >= threshold
    if torch.sum(mask) == 0:
        return torch.tensor(float("nan"))  # or 0, or raise an exception
    return torch.sum(torch.abs(v_ - v)[mask]) / torch.sum(torch.abs(v)[mask]) * 100


def RMSE(v, v_):
    """
    Mean squared error.
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch scalar, RMSE averages on all elements of input.
    """
    return torch.sqrt(torch.mean((v_ - v) ** 2))


def MAE(v, v_):
    """
    Mean absolute error.
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch scalar, MAE averages on all elements of input.
    """
    return torch.mean(torch.abs(v_ - v))


# Function to compute MAPE
def MAPE_split(v, v_):
    # Get lengths
    n = len(v)

    # Split the vectors into 3 parts
    split_1 = n // 3
    split_2 = 2 * (n // 3)

    v_split_1 = v[:split_1]
    v_split_2 = v[:split_2]
    v_split_3 = v

    v_pred_split_1 = v_[:split_1]
    v_pred_split_2 = v_[:split_2]
    v_pred_split_3 = v_

    # Compute MAPE for each split
    mape_1 = torch.mean(
        torch.abs((v_pred_split_1 - v_split_1)) / (v_split_1 + 1e-15) * 100
    )
    mape_2 = torch.mean(
        torch.abs((v_pred_split_2 - v_split_2)) / (v_split_2 + 1e-15) * 100
    )
    mape_3 = torch.mean(
        torch.abs((v_pred_split_3 - v_split_3)) / (v_split_3 + 1e-15) * 100
    )

    return mape_1, mape_2, mape_3


# Function to compute WMAPE
def WMAPE_split(v, v_):
    # Get lengths
    n = len(v)

    # Split the vectors into 3 parts
    split_1 = n // 3
    split_2 = 2 * (n // 3)

    v_split_1 = v[:split_1]
    v_split_2 = v[:split_2]
    v_split_3 = v

    v_pred_split_1 = v_[:split_1]
    v_pred_split_2 = v_[:split_2]
    v_pred_split_3 = v_

    # Compute WMAPE for each split
    wmape_1 = (
        torch.sum(torch.abs(v_pred_split_1 - v_split_1))
        / (torch.sum(v_split_1) + 1e-15)
        * 100
    )
    wmape_2 = (
        torch.sum(torch.abs(v_pred_split_2 - v_split_2))
        / (torch.sum(v_split_2) + 1e-15)
        * 100
    )
    wmape_3 = (
        torch.sum(torch.abs(v_pred_split_3 - v_split_3))
        / (torch.sum(v_split_3) + 1e-15)
        * 100
    )

    return wmape_1, wmape_2, wmape_3


# Function to compute MAE
def MAE_split(v, v_):
    # Get lengths
    n = len(v)

    # Split the vectors into 3 parts
    split_1 = n // 3
    split_2 = 2 * (n // 3)

    v_split_1 = v[:split_1]
    v_split_2 = v[:split_2]
    v_split_3 = v

    v_pred_split_1 = v_[:split_1]
    v_pred_split_2 = v_[:split_2]
    v_pred_split_3 = v_

    # Compute MAE for each split
    mae_1 = torch.mean(torch.abs(v_pred_split_1 - v_split_1))
    mae_2 = torch.mean(torch.abs(v_pred_split_2 - v_split_2))
    mae_3 = torch.mean(torch.abs(v_pred_split_3 - v_split_3))

    return mae_1, mae_2, mae_3


# Function to compute RMSE
def RMSE_split(v, v_):
    # Get lengths
    n = len(v)

    # Split the vectors into 3 parts
    split_1 = n // 3
    split_2 = 2 * (n // 3)

    v_split_1 = v[:split_1]
    v_split_2 = v[:split_2]
    v_split_3 = v

    v_pred_split_1 = v_[:split_1]
    v_pred_split_2 = v_[:split_2]
    v_pred_split_3 = v_

    # Compute RMSE for each split
    rmse_1 = torch.sqrt(torch.mean((v_pred_split_1 - v_split_1) ** 2))
    rmse_2 = torch.sqrt(torch.mean((v_pred_split_2 - v_split_2) ** 2))
    rmse_3 = torch.sqrt(torch.mean((v_pred_split_3 - v_split_3) ** 2))

    return rmse_1, rmse_2, rmse_3


# Function to compute MSE
def MSE_split(v, v_):
    # Get lengths
    n = len(v)

    # Split the vectors into 3 parts
    split_1 = n // 3
    split_2 = 2 * (n // 3)

    v_split_1 = v[:split_1]
    v_split_2 = v[:split_2]
    v_split_3 = v

    v_pred_split_1 = v_[:split_1]
    v_pred_split_2 = v_[:split_2]
    v_pred_split_3 = v_

    # Compute MSE for each split
    mse_1 = torch.mean((v_pred_split_1 - v_split_1) ** 2)
    mse_2 = torch.mean((v_pred_split_2 - v_split_2) ** 2)
    mse_3 = torch.mean((v_pred_split_3 - v_split_3) ** 2)

    return mse_1, mse_2, mse_3
