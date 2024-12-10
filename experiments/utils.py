from typing import Union, Tuple, Optional, Any

import numpy as np
from numpy.typing import NDArray
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.callbacks import TFMProgressBar



def calculate_score(actual: NDArray, predicted: NDArray, axis: Optional[Union[int, Tuple[int, ...]]] = None,
                    mean_collapse: bool = False, metric_type: str = 'smape') -> Union[float, NDArray]:
    """
    Calculate various error metrics between actual and predicted values.

    :param actual: Array of actual values
    :param predicted: Array of predicted values
    :param axis: Axis or axes along which to compute the score
    :param mean_collapse: Whether to return a single mean value
    :param metric_type: Type of metric to calculate ('smape', 'wmape', 'r2', 'rmse', 'mae')
    :return: Calculated score
    """
    if metric_type == 'smape':
        return calculate_smape(actual, predicted, axis, mean_collapse)
    if metric_type == 'wmape':
        return calculate_wmape(actual, predicted, axis, mean_collapse)
    if metric_type == 'r2':
        return calculate_r2(actual, predicted, axis, mean_collapse)
    if metric_type == 'rmse':
        return rmse(actual, predicted, axis=axis, mean_collapse=mean_collapse)
    if metric_type == 'mae':
        return calculate_mae(actual, predicted, axis=axis, mean_collapse=mean_collapse)


def calculate_mae(actual: NDArray, predicted: NDArray, axis: Optional[Union[int, Tuple[int, ...]]] = None,
                  mean_collapse: Optional[bool] = None) -> Union[float, NDArray]:
    """
    Calculate Mean Absolute Error (MAE).

    :param actual: Array of actual values
    :param predicted: Array of predicted values
    :param axis: Axis or axes along which to compute the MAE
    :param mean_collapse: Whether to return a single mean value
    :return: MAE score
    """
    if axis is None:
        axis = tuple(range(actual.ndim))
    absolute = np.abs(actual - predicted)
    mae = np.mean(absolute, axis=axis)
    if mean_collapse:
        return np.mean(mae)
    return mae


def calculate_r2(actual: NDArray, predicted: NDArray, axis: Optional[Union[int, Tuple[int, ...]]] = None,
                 mean_collapse: bool = False) -> Union[float, NDArray]:
    """
    Calculate R-squared (R2) score.

    :param actual: Array of actual values
    :param predicted: Array of predicted values
    :param axis: Axis or axes along which to compute the R2 score
    :param mean_collapse: Whether to return a single mean value
    :return: R2 score
    """
    if axis is None:
        axis = tuple(range(actual.ndim))
    numerator = np.sum((actual - predicted) ** 2, axis=axis)
    denominator = np.sum((actual) ** 2, axis=axis)
    r2_score = np.where(denominator != 0, 1 - (numerator / denominator), 1)
    if mean_collapse:
        return np.mean(r2_score)
    return r2_score


def calculate_coef_score(predicted_coefs: NDArray, true_coefs: NDArray) -> float:
    """
    Calculate coefficient score based on squared differences.

    :param predicted_coefs: Array of predicted coefficients
    :param true_coefs: Array of true coefficients
    :return: Coefficient score
    """
    absolute_difference = true_coefs - predicted_coefs
    score = np.sum(absolute_difference ** 2)
    return score


def rmse(actual: NDArray, predicted: NDArray, axis: Optional[Union[int, Tuple[int, ...]]] = None,
         mean_collapse: bool = False) -> Union[float, NDArray]:
    """
    Calculate Root Mean Squared Error (RMSE).

    :param actual: Array of actual values
    :param predicted: Array of predicted values
    :param axis: Axis or axes along which to compute the RMSE
    :param mean_collapse: Whether to return a single mean value
    :return: RMSE score
    """
    if axis is None:
        axis = tuple(range(actual.ndim))
    if mean_collapse:
        return np.mean(np.sqrt(np.sum((actual - predicted) ** 2, axis=axis)))
    return np.sqrt(np.sum((actual - predicted) ** 2, axis=axis))


def calculate_smape(actual: NDArray, predicted: NDArray, axis: Optional[Union[int, Tuple[int, ...]]] = None,
                    mean_collapse: bool = False) -> Union[float, NDArray]:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).

    :param actual: Array of actual values
    :param predicted: Array of predicted values
    :param axis: Axis or axes along which to compute the SMAPE
    :param mean_collapse: Whether to return a single mean value
    :return: SMAPE score
    """
    if axis is None:
        axis = tuple(range(actual.ndim))
    numerator = np.abs(actual - predicted) * 200
    denominator = np.abs(actual) + np.abs(predicted) + 0.1
    smape = np.mean(numerator / denominator, axis=axis)
    if mean_collapse:
        return np.mean(numerator / denominator)
    return smape


def calculate_wmape(actual: NDArray, predicted: NDArray, axis: Optional[Union[int, Tuple[int, ...]]] = None,
                    mean_collapse: bool = False) -> Union[float, NDArray]:
    """
    Calculate Weighted Mean Absolute Percentage Error (WMAPE).

    :param actual: Array of actual values
    :param predicted: Array of predicted values
    :param axis: Axis or axes along which to compute the WMAPE
    :param mean_collapse: Whether to return a single mean value
    :return: WMAPE score
    """
    if axis is None:
        axis = tuple(range(actual.ndim))
    denominator = np.abs(np.sum(actual, axis=axis))
    numerator = np.sum(np.abs(actual - predicted), axis=axis) * 100
    if mean_collapse:
        return np.mean(numerator / denominator)
    return numerator / denominator


def to_lagged_tensor(data: np.ndarray, lags: int = 1) -> np.ndarray:
    """
    Convert a 3D numpy array into a lagged tensor format.

    This function takes a 3D array representing multiple time series (trajectories)
    and creates a 4D lagged tensor. Each slice along the second axis of the output
    represents the data at a specific lag.

    :param data: Input data array of shape (num_trajectories, time_steps, features)
    :param lags: Number of lags to include (default: 1)
    :return: Lagged tensor of shape (num_trajectories, lags+1, new_time_steps, features)
    """

    # Extract dimensions from the input data
    num_trajectories, time_steps, num_features = data.shape

    # Calculate the new number of time steps after lagging
    new_time_steps = time_steps - lags

    # Initialize the output lagged tensor with NaN values
    lagged_tensor = np.full((num_trajectories, lags + 1, new_time_steps, num_features), np.nan)

    # Fill the lagged tensor
    for lag in range(lags, -1, -1):
        lagged_tensor[:, lag, :, :] = data[:, lag:new_time_steps + lag, :]

    return lagged_tensor


def extract_features_and_labels(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features (X) and labels (y) from lagged tensor data.

    :param data: Lagged tensor data of shape (num_trajectories, lags+1, time_steps, features)
    :return: Tuple of (X, y) where:
             X is features of shape (num_trajectories, lags, time_steps, features)
             y is labels of shape (num_trajectories, 1, time_steps, features)
    """
    X = data[:, :-1, :, :]  # All lags except the last one
    y = data[:, -1:, :, :]  # Only the last lag (current values)
    return X, y


def reshape_lagged_tensor(lagged_tensor: np.ndarray) -> np.ndarray:
    """
    Reshape a lagged tensor for model input.

    This function takes a 4D lagged tensor and reshapes it into a 3D tensor
    where the first dimension represents lags, and the second dimension
    combines trajectories and time steps.

    :param lagged_tensor: 4D numpy array of shape (num_trajectories, lags_plus_one, time_steps, features)
    :return: Reshaped 3D numpy array of shape (lags_plus_one, num_trajectories * time_steps, features)
    """
    # Extract dimensions from the input tensor
    num_trajectories, lags_plus_one, time_steps, num_features = lagged_tensor.shape

    # Reshape the tensor:
    # 1. Transpose to bring lags_plus_one to the first dimension
    # 2. Reshape to combine num_trajectories and time_steps
    reshaped_tensor = lagged_tensor.transpose(1, 0, 2, 3).reshape(lags_plus_one, num_trajectories * time_steps,
                                                                  num_features)

    return reshaped_tensor


def reshape_and_transpose(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape and transpose X and y for model input.

    :param X: Features array
    :param y: Labels array
    :return: Tuple of reshaped and transposed (X, y)
    """
    X_reshaped = np.transpose(reshape_lagged_tensor(X), (1, 0, 2))
    y_reshaped = np.squeeze(np.transpose(reshape_lagged_tensor(y), (1, 0, 2)), axis=1)
    return X_reshaped, y_reshaped


def process_lagged_data(data: np.ndarray, n_lags: int, train_steps: int, val_steps: int) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process lagged data into train, validation, and test sets.

    :param data: Input data of shape (num_trajectories, time_steps, features)
    :param n_lags: Number of lags to use
    :param train_steps: Number of steps for training data
    :param val_steps: Number of steps for validation data
    :return: Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Create lagged tensor
    lagged_data = to_lagged_tensor(data, n_lags)

    # Split data into train, validation, and test sets
    train_data = lagged_data[:, :, :train_steps, :]
    val_data = lagged_data[:, :, train_steps:train_steps + val_steps, :]
    test_data = lagged_data[:, :, train_steps + val_steps:, :]

    # Extract features and labels for each set
    X_train, y_train = extract_features_and_labels(train_data)
    X_val, y_val = extract_features_and_labels(val_data)
    X_test, y_test = extract_features_and_labels(test_data)

    # Reshape and transpose train and validation data
    X_train, y_train = reshape_and_transpose(X_train, y_train)
    X_val, y_val = reshape_and_transpose(X_val, y_val)

    # Reshape test data
    X_test = np.transpose(X_test[0, :, :, :], (1, 0, 2))
    y_test = np.squeeze(np.transpose(y_test[0, :, :, :], (1, 0, 2)), axis=1)

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_params(input_chunk_length, output_chunk_length, full_training=True):
    """
    Create parameters for model training.

    :param input_chunk_length: Length of input sequences
    :param output_chunk_length: Length of output sequences
    :param full_training: Whether to use full training or a reduced version
    :return: Dictionary of training parameters
    """
    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=1e-5,
        mode="min",
    )

    if full_training:
        limit_train_batches = None
        limit_val_batches = None
        max_epochs = 200
        batch_size = 256
    else:
        limit_train_batches = 20
        limit_val_batches = 10
        max_epochs = 40
        batch_size = 64

    progress_bar = TFMProgressBar(
        enable_sanity_check_bar=False, enable_validation_bar=False
    )
    pl_trainer_kwargs = {
        "gradient_clip_val": 1,
        "max_epochs": max_epochs,
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": limit_val_batches,
        "accelerator": "auto",
        "callbacks": [early_stopper, progress_bar],
    }

    optimizer_cls = torch.optim.Adam
    optimizer_kwargs = {
        "lr": 1e-4,
    }

    lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
    lr_scheduler_kwargs = {"gamma": 0.999}

    likelihood = QuantileRegression()
    loss_fn = None

    return {
        "input_chunk_length": input_chunk_length,
        "output_chunk_length": output_chunk_length,
        "use_reversible_instance_norm": True,
        "optimizer_kwargs": optimizer_kwargs,
        "pl_trainer_kwargs": pl_trainer_kwargs,
        "lr_scheduler_cls": lr_scheduler_cls,
        "lr_scheduler_kwargs": lr_scheduler_kwargs,
        "likelihood": likelihood,
        "loss_fn": loss_fn,
        "save_checkpoints": True,
        "force_reset": True,
        "batch_size": batch_size,
        "random_state": 42,
        "add_encoders": {
            "cyclic": {
                "future": ["hour", "dayofweek", "month"]
            }
        },
    }


def get_forecast(simulation: Any, past_data: np.ndarray, future_steps: int) -> np.ndarray:
    """
    Generate forecasts for multiple trajectories using a SVAR (Structural Vector Autoregression) process.

    This function takes a simulation object (SVAR process),
    past data for multiple trajectories, and generates forecasts for a specified
    number of future steps.

    :param simulation: An object containing a SVAR process with a forecast method
    :param past_data: 3D numpy array of shape (num_trajectories, time_steps, features)
                      representing past data for multiple trajectories
    :param future_steps: Number of steps to forecast into the future
    :return: 3D numpy array of shape (num_trajectories, future_steps, features)
             containing forecasts for all trajectories
    """
    forecasts = []

    # Iterate over each trajectory in the past data
    for i in range(past_data.shape[0]):
        # Generate forecast for this trajectory
        forecast_trajectory = simulation.svar_process.forecast(past_data[i], steps=future_steps)
        forecasts.append(forecast_trajectory)

    # Convert list of forecasts to a numpy array
    return np.array(forecasts)


def clone_effects(n_simul_test: int, effects: np.ndarray) -> np.ndarray:
    """
    Clone the effects array to create a trajectory of simulations.
    :param: n_simul_test (int): Number of simulations to run.
    :param: effects (np.ndarray): Array of effects to be cloned.
    :return: np.ndarray: A 3D array representing the trajectory of simulations.
    """
    trajectory = np.zeros((n_simul_test, *effects.shape))
    for i in range(n_simul_test):
        trajectory[i] = effects
    return trajectory

def additive_forecasting(test_forecast, predicted_forecast, test_effects, predicted_effects):
    test_if = test_forecast + test_effects
    predicted_if = predicted_forecast + predicted_effects
    return test_if, predicted_if