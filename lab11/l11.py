#cnn stratch
import numpy as np

def conv2d(image, kernel, stride=1, padding=0): #Performing 2D Convolution over a single-channel image.

    if padding > 0:
        image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)

    H, W = image.shape
    Kh, Kw = kernel.shape

    # Compute output spatial dimensions
    out_H = (H - Kh) // stride + 1
    out_W = (W - Kw) // stride + 1

    output = np.zeros((out_H, out_W))

    # Slide kernel over image patches
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride

            patch = image[h_start:h_start + Kh, w_start:w_start + Kw]
            output[i, j] = np.sum(patch * kernel)

    return output


def maxpool2d(image, pool_size=(2, 2), stride=2):  #Performing 2D Max Pooling downsampling.

    H, W = image.shape
    Ph, Pw = pool_size

    out_H = (H - Ph) // stride + 1
    out_W = (W - Pw) // stride + 1

    output = np.zeros((out_H, out_W))

    # Extract maximum value per pool window
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride

            patch = image[h_start:h_start + Ph, w_start:w_start + Pw]
            output[i, j] = np.max(patch)

    return output

#  Execution Example 
np.random.seed(69)

# Input image (32x32) and Kernel (3x3)
input_image = np.random.randn(32, 32)
kernel_3x3 = np.array([
    [2, 0, -1],
    [1, 0, -2],
    [0, 0, -1]
])

# 1. Convolution (Valid Padding: output shrinks from 32x32 to 30x30)
conv_output = conv2d(input_image, kernel_3x3, stride=1, padding=0)

# 2. Max Pooling (2x2 pool, stride 2: output shrinks from 30x30 to 15x15)
pool_output = maxpool2d(conv_output, pool_size=(2, 2), stride=2)

print(f"Input Image Shape:       {input_image.shape}")
print(f"Post-Conv Output Shape:  {conv_output.shape}")
print(f"Post-Pool Output Shape:  {pool_output.shape}")
