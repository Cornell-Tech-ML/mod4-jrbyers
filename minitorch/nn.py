from typing import Tuple

from .tensor import Tensor


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

    # Calculate the new height and width after pooling
    new_height = height // kh
    new_width = width // kw

    # Reshape the input tensor to facilitate pooling
    # The shape we want: batch x channel x new_height x new_width x (kernel_height * kernel_width)

    # Step 1: View the tensor to extract sliding blocks
    # First, we will reshape the height and width dimensions so that each block of size (kh, kw) becomes a new axis.
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Step 2: Permute the new axes so that the kernel elements are the last dimension
    reshaped = reshaped.permute(
        0, 1, 2, 4, 3, 5
    )  # Move the kernel dimensions to the end

    # Step 3: Reshape the tensor to have the last dimension as the flattened kernel (kh * kw)
    reshaped = reshaped.contiguous().view(
        batch, channel, new_height, new_width, kh * kw
    )

    # Return the reshaped tensor along with the new height and width
    return reshaped, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width after performing average pooling.

    """
    # Step 1: Use tile to reshape the input tensor for pooling
    reshaped, new_height, new_width = tile(input, kernel)
    pooled = reshaped.mean(dim=4)  # Average along the kernel dimension (last dimension)

    # remove last singleton dimension
    pooled = pooled.view(input.shape[0], input.shape[1], new_height, new_width)
    return pooled


# TODO: Implement for Task 4.3.
