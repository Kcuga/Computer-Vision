import cv2
import numpy as np
import matplotlib.pyplot as plt


def convolution(image, kernel):
    # Get image dimensions
    height, width = image.shape
    # Get kernel dimensions
    k_height, k_width = kernel.shape
    # Calculate padding
    pad_height = k_height // 2
    pad_width = k_width // 2
    # Pad the image
    padded_img = pad_image(image, pad_height, pad_height, pad_width, pad_width)
    # Initialize result
    result = np.zeros(image.shape).astype(dtype=np.float32)
    # Perform convolution
    for i in range(height):
        for j in range(width):
            result[i, j] = np.sum(padded_img[i:i + k_height, j:j + k_width] * kernel)
    return result


def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)), (size, size))
    return kernel / np.sum(kernel)


def pad_image(image, top, bottom, left, right, value=0):
    padded_image = []
    for i in range(len(image) + top + bottom):
        if top <= i < top + len(image):
            padded_row = [value] * left + list(image[i - top]) + [value] * right
        else:
            padded_row = [value] * (len(image[0]) + left + right)
        padded_image.append(padded_row)
    return np.array(padded_image)


def normalize_kernel(kernel):
    # Normalize the kernel
    normalized_kernel = kernel / np.sum(np.abs(kernel))
    return normalized_kernel


def edge_detection(image):
    # Define Sobel kernels for horizontal and vertical gradients
    sobel_kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)  # Specify the data type as float64

    sobel_kernel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float64)  # Specify the data type as float64

    # Convolve the image with the Sobel kernels
    gradient_x = convolution(image, sobel_kernel_x)
    gradient_y = convolution(image, sobel_kernel_y)

    # Compute edge strength image (gradient magnitude)
    edge_strength_image = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Scale the edge strength image to the range [0, 255]
    edge_strength_image_scaled = (edge_strength_image - np.min(edge_strength_image)) / (
            np.max(edge_strength_image) - np.min(edge_strength_image)) * 255

    # Convert the scaled edge strength image to uint8 datatype
    edge_strength_image_scaled_uint8 = edge_strength_image_scaled.astype(np.uint8)

    return edge_strength_image_scaled_uint8


def threshold_edges(edge_strength_image, threshold_value=100):
    # Perform thresholding
    edges_threshold = (edge_strength_image > threshold_value) * 255
    return edges_threshold


def plot_histogram(image, width=15, height=8):
    # Set the figure size
    plt.figure(figsize=(width, height))

    plt.hist(image.flatten(), bins=np.arange(0, 256, 1), color='c')  # Adjust bin size for more detail
    plt.title('Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(0, 256, 10))  # Add more ticks on the x-axis
    plt.show()


# Load the image
image = cv2.imread('kitty.bmp', cv2.IMREAD_GRAYSCALE)

# Define a simple 3x3 averaging kernel
smoothing_kernel = np.ones((11, 11)) / 121
# Convolve the image with the average kernel
smoothed_image = convolution(image, smoothing_kernel)

# Define weighted average smoothing kernel
weighted_average_smoothing_kernel = np.array([[1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
                                                    [1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 1],
                                                    [1, 2, 2, 3, 4, 4, 3, 2, 2, 1, 1],
                                                    [2, 2, 3, 4, 5, 5, 4, 3, 2, 2, 1],
                                                    [2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 2],
                                                    [2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 2],
                                                    [2, 2, 3, 4, 5, 5, 4, 3, 2, 2, 1],
                                                    [1, 2, 2, 3, 4, 4, 3, 2, 2, 1, 1],
                                                    [1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 1],
                                                    [1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1]]) / 264


# Convolve the image with the weighted average kernel
smoothed_image_weighted_average = convolution(image, weighted_average_smoothing_kernel)

# Define a Gaussian smoothing kernel with sigma = 1.5 and size = 3
gaussian_kernel_size = 11
gaussian_sigma = 4
gaussian_smoothing_kernel = gaussian_kernel(gaussian_kernel_size, gaussian_sigma)
# Convolve the image with the Gaussian kernel
smoothed_image_gaussian = convolution(image, gaussian_smoothing_kernel)

# Compute edge strength image
# Select between: smoothed_image, smoothed_image_weighted_average, smoothed_image_gaussian
edge_strength_image = edge_detection(smoothed_image_weighted_average)

# Plot histogram
plot_histogram(edge_strength_image)

# Threshold edge
threshold_value = 24
edges_threshold = threshold_edges(edge_strength_image, threshold_value)

# Display all images inline
plt.figure(figsize=(15, 10))  # Adjust the figure size as needed

# Original Image
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.xticks([])
plt.yticks([])

# Smoothed Image with Normal Kernel
plt.subplot(2, 3, 2)
plt.imshow(smoothed_image, cmap='gray')
plt.title('Smoothed Image (3x3 Kernel)')
plt.xticks([])
plt.yticks([])

# Smoothed Image with Weighted Average Kernel
plt.subplot(2, 3, 3)
plt.imshow(smoothed_image_weighted_average, cmap='gray')
plt.title('Smoothed Image (Weighted Average Kernel)')
plt.xticks([])
plt.yticks([])

# Smoothed Image with Gaussian Kernel
plt.subplot(2, 3, 4)
plt.imshow(smoothed_image_gaussian, cmap='gray')
plt.title('Smoothed Image (Gaussian Kernel)')
plt.xticks([])
plt.yticks([])

# Edge Detection Result
plt.subplot(2, 3, 5)
plt.imshow(edge_strength_image, cmap='gray')
plt.title('Edge Detection Result')
plt.xticks([])
plt.yticks([])

# Thresholded Image
plt.subplot(2, 3, 6)
plt.imshow(edges_threshold, cmap='gray')
plt.title(f'Thresholded Image (Threshold: {threshold_value})')  # Include the threshold value in the title
plt.xticks([])
plt.yticks([])

plt.tight_layout(pad=3.0)  # Increase pad for more spacing between subplots
plt.show()