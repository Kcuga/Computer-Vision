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

    # Normalize the kernels
    sobel_kernel_x_normalized = normalize_kernel(sobel_kernel_x)
    sobel_kernel_y_normalized = normalize_kernel(sobel_kernel_y)

    # Convolve the image with the Sobel kernels
    gradient_x = convolution(image, sobel_kernel_x)
    gradient_y = convolution(image, sobel_kernel_y)

    print(gradient_x)
    print(gradient_y)
    cv2.imshow('Gradient X', gradient_x)
    cv2.imshow('Gradient y', gradient_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Compute edge strength image (gradient magnitude)
    edge_strength_image = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Scale the edge strength image to the range [0, 255]
    edge_strength_image_scaled = (edge_strength_image - np.min(edge_strength_image)) / (
                np.max(edge_strength_image) - np.min(edge_strength_image)) * 255


    # Convert the scaled edge strength image to uint8 datatype
    edge_strength_image_scaled_uint8 = edge_strength_image_scaled.astype(np.uint8)

    print(edge_strength_image_scaled)
    cv2.imshow('Scaled Edge Strength Image', edge_strength_image_scaled_uint8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return edge_strength_image_scaled_uint8


def threshold_edges(edge_strength_image, threshold_value = 56):
    # Perform thresholding
    edges_threshold = (edge_strength_image > threshold_value) * 255
    return edges_threshold

def plot_histogram(image):
    plt.hist(image.flatten(), bins=range(256), color='c')
    plt.title('Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

# Load the image
image = cv2.imread('kitty.bmp', cv2.IMREAD_GRAYSCALE)

# Define a simple 3x3 averaging kernel
smoothing_kernel = np.ones((3, 3)) / 9
# Convolve the image with the average kernel
smoothed_image = convolution(image, smoothing_kernel)
print(smoothed_image)

# Define weighted average smoothing kernel
weighted_average_smoothing_kernel = np.array([[1, 2, 1],
                                              [2, 4, 2],
                                              [1, 2, 1]]) / 16
# Convolve the image with the weighted average kernel
smoothed_image_weighted_average = convolution(image, weighted_average_smoothing_kernel)
print(smoothed_image_weighted_average)

# Compute edge strength image
edge_strength_image = edge_detection(image)

# Plot histogram
plot_histogram(edge_strength_image)

# Threshold edges
edges_threshold = threshold_edges(edge_strength_image)

# Display the original and processed images
cv2.imshow('Original Image', image.astype(np.uint8))
cv2.imshow('Smoothed Image', smoothed_image.astype(np.uint8))
cv2.imshow('Weighted Average Smoothed Image', smoothed_image_weighted_average.astype(np.uint8))
cv2.imshow('Edge Strength Image', edge_strength_image.astype(np.uint8))
cv2.imshow('Threshold Edges', edges_threshold.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
