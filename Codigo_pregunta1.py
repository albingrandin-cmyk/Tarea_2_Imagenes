import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.ndimage import uniform_filter

# Constructin synthetic image 256x256 pixels
# Defining matrix size, coordinates and center
n = 256
x, y = np.ogrid[:n, :n]
center_y, center_x = (n-1)/2.0, (n-1)/2.0

# Centered square 128x128 pixels
mask_square_full = (x >= 64) & (x < 192) & (y >= 64) & (y < 192)

# Centered circle, (x-center)^2 + (y-center)^2 <= r^2, r =32
mask_circle = (x-center_x)**2 + (y - center_y)**2 <= 32**2

# Centered square - centered circle
mask_square = mask_square_full & ~mask_circle

# Background, everyting outside the centered square
mask_bg = ~mask_square_full

# Creating image with given intensity values
img = np.zeros((n,n))
img[mask_bg] = 0.15
img[mask_square] = 0.45
img[mask_circle] = 0.8

# Image visualization
fig, axes = plt.subplots(2, 2, figsize=(16,4))

axes[0][0].imshow(img, cmap="gray", vmin=0, vmax=1) # vmin and vmax locking the colourscale to absolute black = 0 and absolute white = 1. Very important for scaling
axes[0][0].set_title("Synthetic image")
axes[0][1].imshow(mask_bg, cmap="gray")
axes[0][1].set_title("Background")
axes[1][0].imshow(mask_square, cmap="gray")
axes[1][0].set_title("Centered square with circle")
axes[1][1].imshow(mask_circle, cmap="gray")
axes[1][1].set_title("Circle")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.savefig("Synthehtic_image_different_parts")
plt.show()

# Random seed, the meaning of life 
seed = 42
rng_num = np.random.default_rng(seed)
n = 40

# Poisson-noise
img_noisy = rng_num.poisson(n*img)/n

plt.imshow(img_noisy, cmap="gray", vmin=0, vmax=1)
plt.axis("off")
plt.savefig("Noisy_image")
plt.show()

def gaussian_kernel(sigma):
    """
    Creating an isometric Gaussian Kernel
    
    Parameter:
        sigma(float): The sigma value for the Kernel

    Return:
        kernel(np.darray): A 2D-matrix with the Gauss-weights

    """
    # if sigma <= 0 return identity filter
    if sigma <= 0:
        return np.array([[1.0]])

    # Radius = 3*sigma to include ca 99.73% of the Gauss-distribution
    r = int(np.ceil(3.0*sigma))
    # If sigma << 1
    r = max(1,r)
    # You want an odd kernel size to have a clear center
    # Usually put to 2*3sigma + 1
    kern_size = 2*r + 1

    
    ax = np.arange(-r, r+1)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    # Normalizing kernel
    kernel = kernel / np.sum(kernel)
    return kernel

def apply_gaussian_kernel(img, sigma, mode="reflect"):
    """
    Applying a gaussian kernel with sigma=sigma on image=img
    mode = "reflected" to a avoid a dark frame

    Parameters:
        img(np.ndarray): the image that we want to filter
        sigma(float): standard deviation for the gaussian filter
        mode(str): How convolve is suppose to handle the frame of the image

    Return:
        filtered_image(np.ndarray): the filtered image after applying the Gaussian kernel


    """

    kernel = gaussian_kernel(sigma)

    filtered_image = convolve(img, kernel, mode=mode)

    return filtered_image

def evaluate_sigma_values(img_ideal, img_noisy, sigma_values, mask=None):
    """
    Function to evaluate real sigma values and return the rmse for these values

    Parameters:
        img_ideal(np.ndarray): the ideal image for comparison
        img_noisy(np.ndarray): the image we want to filter
        sigma_values(array-like): the list of all sigma values to evaluate

    Return:
        rmse_values(np.ndarray): array with all the rmse values
        filtered_dict(dictionary): dictionary with the sigma values as keys and the filtered image with said sigma value as value
    
    """

    rmse_values = []
    filtered_dict = {}

    for sigma in sigma_values:
        sigma = float(sigma)
        img_filtered = apply_gaussian_kernel(img_noisy, sigma)

        if mask is not None:
            rmse = np.sqrt(np.mean((img_ideal[mask]-img_filtered[mask])**2))
        else:
            rmse = np.sqrt(np.mean((img_ideal-img_filtered)**2))

        rmse_values.append(rmse)
        filtered_dict[sigma] = img_filtered

    return np.array(rmse_values), filtered_dict

# Testing on whole image without taking into account the different intensities
"""print("Choose range of sigma values to test")
min_s = float(input("Min: "))
max_s = float(input("Max: "))
n = int(input("Number of values: "))

sigma_values = np.linspace(min_s, max_s, n) """
sigma_values = np.linspace(0.0, 3.0)

rmse_array, _ = evaluate_sigma_values(img, img_noisy, sigma_values)

best_sigma_idx = np.argmin(rmse_array)
best_sigma = sigma_values[best_sigma_idx]
min_rmse = rmse_array[best_sigma_idx]

print(f"Best sigma value from {sigma_values[0]} to {sigma_values[-1]} = {best_sigma:.2f}\nThe minimum rmse value: {min_rmse:.5f}")

# Testing with taking into account of the different intensities
rmse_array_bg, _ = evaluate_sigma_values(img, img_noisy, sigma_values, mask_bg)
rmse_array_circ, _ = evaluate_sigma_values(img, img_noisy, sigma_values, mask_circle)
rmse_array_sq, _ = evaluate_sigma_values(img, img_noisy, sigma_values, mask_square)

best_sigma_bg = sigma_values[np.argmin(rmse_array_bg)]
best_sigma_circ = sigma_values[np.argmin(rmse_array_circ)]
best_sigma_sq = sigma_values[np.argmin(rmse_array_sq)]

print(f"Best sigma for Background: {best_sigma_bg:.2f}")
print(f"Best sigma for Circle: {best_sigma_circ:.2f}")
print(f"Best sigma for Square: {best_sigma_sq:.2f}")
print(f"Best sigma overall: {best_sigma:.2f}")

def apply_adaptive_gaussian_filter(img_noisy, sigma_map, n_levels=25):
    """
    Function to apply an adaptive gaussian filter on a noisy image using interpolated values of optimal sigma for each pixel
    The the sigma map is disretized into n_levels to make the function faster

    Parameters:
        img_noisy(np.ndarray): the noisy image we want to filter
        sigma_map(np.ndarray): all interpolated values calculated depending on a few precalculated optimal sigma values
        n_levels(int): amount of discretization levels

    Return:
        img_adaptive(np.ndarray): the adaptively filtered image
    """

    # Creating a 1D vector with discretized values of the sigma_map to increase computer performance
    s_min, s_max = np.min(sigma_map), np.max(sigma_map)
    levels = np.linspace(s_min, s_max, n_levels)


    # Calculating a 3D matrix (nx256x256) where the noise image has filterd with the n different sigma values
    filtered_stack = np.array([apply_gaussian_kernel(img_noisy, sigma) for sigma in levels])

    # Creating empty matrix to save data
    img_adaptive = np.zeros_like(img_noisy)

    # Interpolation: Checking were s_value is in levels and interpolates the intensity value for the corresponding index in filtered_stack
    for i in range(img_noisy.shape[0]):
        for j in range(img_noisy.shape[1]):
            s_value = sigma_map[i,j]

            img_adaptive[i][j] = np.interp(s_value, levels, filtered_stack[:, i, j])

    return img_adaptive

# Defining best sigma for different mu points
mu_points = [0.15, 0.45, 0.80]
sigma_points = [best_sigma_bg, best_sigma_sq, best_sigma_circ,]

# Average value of intensities for the noisy image
# The sigma value chosen to estimate the local average intensity is the before calculated best
# overall sigma value.
mu_hat = apply_gaussian_kernel(img_noisy, best_sigma)

# The interpolated sigma values that have been evaluated knowing the optimal sigma values for three sifferent intensities
sigma_map = np.interp(mu_hat, mu_points, sigma_points)

# applying filter and calculating RMSE
img_adaptive = apply_adaptive_gaussian_filter(img_noisy, sigma_map)
rmse_adaptive = np.sqrt(np.mean((img - img_adaptive)**2))



# -- Experiment and Analysis --
print(f"Globally best RMSE: {min_rmse:.5f}\nAdaptive filter RMSE: {rmse_adaptive:.5f}")
print(f"Reference image RMSE: {rmse_array[0]:.5f}")

plt.figure(figsize=(10,6))

plt.plot(sigma_values, rmse_array_bg, label=f"Background: ($\mu=0.15$)", color="navy", linewidth=2)
plt.plot(sigma_values, rmse_array_sq, label=f"Square: ($\mu=0.45$)", color="green", linewidth=2)
plt.plot(sigma_values, rmse_array_circ, label=f"Circle: ($\mu=0.80$)", color="red", linewidth=2)

plt.plot(sigma_values, rmse_array, label=f"global", color="black", linewidth=2)

plt.scatter(best_sigma_bg, rmse_array_bg[np.argmin(rmse_array_bg)], color="navy", s=100, zorder=5, 
            label=f"Min BG ($\sigma={best_sigma_bg:.2f}$)")
plt.scatter(best_sigma_sq, rmse_array_sq[np.argmin(rmse_array_sq)], color="green", s=100, zorder=5, 
            label=f"Min SQ ($\sigma={best_sigma_sq:.2f}$)")
plt.scatter(best_sigma_circ, rmse_array_circ[np.argmin(rmse_array_circ)], color="red", s=100, zorder=5, 
            label=f"Min CIRC ($\sigma={best_sigma_circ:.2f}$)")

plt.scatter(best_sigma, rmse_array[best_sigma_idx], color="black", marker="x", s=130, linewidths=2.5, zorder=5, 
            label=f"Min overall ($\sigma={best_sigma:.2f}$)")

plt.xlabel(r"$\sigma$", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.title(r"$\text{RMSE}(\sigma)$-specific and overall curves", fontsize=14)
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("Comparison_rmse")
plt.show()

"""
As we can clearly see in the graphs the different regions have different optimal sigma values
But the RMSE for the global filter and the adaptive filter have very similar results.

sigma_map = np.interp(mu_hat, mu_points, sigma_points) In this line we define sigma = F(mu_hat)
np.interp(x,xp,fp) is choosen because it gives a smooth transfer between values to avoid image
artefacts
"""

# RMSE comparison
print(f"min RMSE for Background: {min(rmse_array_bg):.5f}")
print(f"min RMSE for Circle: {min(rmse_array_circ):.5f}")
print(f"min RMSE for Square: {min(rmse_array_sq):.5f}")
print(f"min RMSE overall: {min_rmse:.5f}")

plt.imshow(img_adaptive, cmap="gray", vmin=0, vmax=1)
plt.axis("off")
plt.savefig("Adaptive_image")
plt.show()
"""
The image looks blurry but no clear discontinuities, halos or abrupt transitions can be observed
"""

# Choose a pixel from each one of the three regions. The same kernel is used for all pixels
# Circle (128,128)
print(f"Mu_hat for pixel (128,128) = {mu_hat[128,128]:.2f}")
print(f"Sigma for pixel (128,128) = {sigma_map[128,128]:.2f}")

# Square (100,100)
print(f"Mu_hat for pixel (100,100) = {mu_hat[100,100]:.2f}")
print(f"Sigma for pixel (100,100) = {sigma_map[100,100]:.2f}")

# Background (200,200)
print(f"Mu_hat for pixel (200,200) = {mu_hat[200,200]:.2f}")
print(f"Sigma for pixel (200,200) = {sigma_map[200,200]:.2f}")