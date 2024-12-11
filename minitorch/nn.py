from typing import Tuple

from .tensor import Tensor
import numpy as np


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError("Need to implement for Task 4.3")
    # Calculate new dimensions
    new_height = height // kh
    new_width = width // kw

    # Create a view of the input tensor with shape:
    # (batch, channel, new_height, kh, new_width, kw)
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Permute the dimensions to get:
    # (batch, channel, new_height, new_width, kh * kw)
    output = reshaped.permute(0, 1, 2, 4, 3, 5)
    final_shape = (batch, channel, new_height, new_width, kh * kw)
    output = output.contiguous().view(*final_shape)

    return output, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D.

    Args:
    ----
        input: Tensor of shape (batch, channel, height, width)
        kernel: Tuple of (kernel_height, kernel_width)

    Returns:
    -------
        Tensor of shape (batch, channel, height/kernel_height, width/kernel_width)

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel

    # Check if dimensions are compatible
    assert height % kh == 0
    assert width % kw == 0

    # Use tile to reshape the input tensor
    tiled, new_height, new_width = tile(input, kernel)

    # Calculate the average by summing and dividing by kernel size
    kernel_size = kh * kw
    output = tiled.sum(4) / kernel_size  # Using explicit dimension 4 instead of -1

    return output.view(batch, channel, new_height, new_width)


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D.

    Args:
    ----
        input: Tensor of shape (batch, channel, height, width)
        kernel: Tuple of (kernel_height, kernel_width)

    Returns:
    -------
        Tensor of shape (batch, channel, height/kernel_height, width/kernel_width)

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel

    # Check if dimensions are compatible
    assert height % kh == 0
    assert width % kw == 0

    # Use tile to reshape the input tensor
    tiled, new_height, new_width = tile(input, kernel)

    # Calculate the max over the kernel dimension (dim 4)
    output = tiled.max(4)

    return output.view(batch, channel, new_height, new_width)


def max_func(input: Tensor, kernel: int) -> Tensor:
    """Run the max function which is implemented in tensor.py"""
    return input.max(kernel)


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    """Compute the softmax of a tensor along the specified dimension.

    Args:
    ----
        input: Tensor of arbitrary shape.
        dim: The dimension along which to compute the softmax. Default is the last dimension.

    Returns:
    -------
        Tensor of the same shape as `input`, where values are normalized probabilities along the specified dimension.

    """
    # Handle negative dimension index
    if dim < 0:
        dim = input.dims + dim

    # Step 1: Get the maximum value along the dimension for numerical stability
    max_vals = input.max(dim)

    # Step 2: Subtract the max for numerical stability
    shifted_input = input - max_vals

    # Step 3: Compute the exponential
    exp_vals = shifted_input.exp()

    # Step 4: Sum the exponential values along the dimension
    sum_exp_vals = exp_vals.sum(dim)

    # Step 5: Normalize the exponentials to produce softmax probabilities
    softmax_output = exp_vals / sum_exp_vals

    return softmax_output


def logsoftmax(input: Tensor, dim: int = -1) -> Tensor:
    """Compute the log of the softmax as a tensor.

    Uses the log-sum-exp trick for numerical stability:
    log(softmax(x)) = log(exp(x)/sum(exp(x)))
                    = x - log(sum(exp(x)))
                    = x - max(x) - log(sum(exp(x - max(x))))

    Args:
    ----
        input: Tensor of arbitrary shape
        dim: The dimension along which to compute the logsoftmax (default: -1, last dimension)

    Returns:
    -------
        Tensor of same shape as input containing log probabilities

    """
    # Handle negative dimension index
    if dim < 0:
        dim = input.dims + dim

    # Step 1: Get the maximum value along the dimension for numerical stability
    max_vals = input.max(dim)

    # Step 2: Subtract the max for numerical stability
    shifted_input = input - max_vals

    # Step 3: Compute the exponential of the shifted values
    exp_vals = shifted_input.exp()

    # Step 4: Sum the exponential values along the dimension
    sum_exp = exp_vals.sum(dim)

    log_sum = sum_exp.log()  # take log of the sum
    final_result = shifted_input - log_sum

    return final_result


def dropout(input: Tensor, rate: float = 0.5, ignore: bool = False) -> Tensor:
    """Randomly drop units from the input tensor with probability `rate` during training.

    Args:
    ----
        input: Tensor of any shape
        rate: Probability of dropping a unit (setting it to 0), between 0 and 1
        ignore: If True, dropout is disabled and the input is returned as-is

    Returns:
    -------
        Tensor of same shape as input with some entries randomly set to 0

    """
    # Early returns
    if ignore:
        return input
    if rate < 0 or rate > 1:
        raise ValueError("Dropout rate must be between 0 and 1")
    if rate == 1.0:
        return input.zeros(input.shape)
    if rate == 0.0:
        return input

    # Create random mask using numpy for better efficiency
    prob = 1.0 - rate  # probability of keeping a unit
    mask_array = (np.random.random(len(input._tensor._storage)) < prob).astype(
        np.float64
    )

    # Convert numpy array to tensor
    mask = input.zeros(input.shape)
    mask._tensor._storage[:] = mask_array

    # Scale up values by 1/(1-rate) to maintain expected sum during training
    scale = 1.0 / prob

    return input * mask * scale
