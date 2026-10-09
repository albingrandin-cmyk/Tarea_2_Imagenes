import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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

# Creating image
img = np.zeros((n,n))
img[mask_bg] = 0.15
img[mask_square] = 0.45
img[mask_circle] = 0.8

seed = 42
rng_num = np.random.default_rng(seed)
noise = rng_num.normal(loc=0.0, scale=0.05, size=img.shape)
img_noisy = np.clip(img + noise, 0.0, 1.0)


# We use anisotrophic diffusion to make the image less blurry without smudging edges
def anisotrophic_diffusion(image, num_iter, dt, c_func, return_details=False, **c_params):
    """
    The function executes explicit anisotrophic diffusion on a 2D image

    Parameters:
        image(np.ndarray): The image we want to filter
        num_iter(int): The number of iterations we want to run the anisotrophic diffusion
        dt(float): The time step for the explicit update
        c_func(callable): The function for the diffusion coefficient
        return_details(Bool, optional): If True, returns the final c-map along 
            with the filtered image. Defaults to False.
        **c_params(dict): A dictionary of the explicit parameters to c_func
    
    Returns:
np.ndarray or tuple: 
        - If return_details is False: Returns the filtered image (np.ndarray).
        - If return_details is True: Returns a tuple (c_map, filtered_image).
    """


    I = image.copy()

    for k in range(num_iter):
        # Creating a frame of 1 pixel around the image to guarantee that every pixel has 4 neighbours
        I_pad=np.pad(I, pad_width=1, mode="edge")

        # ___ The Gradient is calculated here ___
        nabla_N = I_pad[:-2,1:-1] - I
        nabla_S = I_pad[2:, 1:-1] - I
        nabla_E = I_pad[1:-1, 2:] - I
        nabla_W = I_pad[1:-1, :-2] - I

                    
        if c_func == c_laplacian:
            I_smoothed = gaussian_filter(I, sigma=0.8)
            I_pad_sm = np.pad(I_smoothed, pad_width=1, mode="edge")
            
            # ___ If the c function is the Laplacian the Laplacian is caluclated ___
            lap_sm = (I_pad_sm[:-2, 1:-1] + I_pad_sm[2:, 1:-1] + 
                      I_pad_sm[1:-1, 2:] + I_pad_sm[1:-1, :-2] - 4 * I_smoothed)
            
            # ___ The diffusioncoefficients are calculated here ___
            cN = c_func(np.abs(nabla_N), lap_sm, **c_params)
            cS = c_func(np.abs(nabla_S), lap_sm, **c_params)
            cE = c_func(np.abs(nabla_E), lap_sm, **c_params)
            cW = c_func(np.abs(nabla_W), lap_sm, **c_params)
        else:
            # ___ The diffusioncoefficients are also calculated here ___
            cN = c_func(np.abs(nabla_N), **c_params)
            cS = c_func(np.abs(nabla_S), **c_params)
            cE = c_func(np.abs(nabla_E), **c_params)
            cW = c_func(np.abs(nabla_W), **c_params)
        
        #c_max = max(np.max(cN), np.max(cS), np.max(cE), np.max(cW))
        #if dt > (1.0 / (4.0 * c_max)):
            #raise ValueError(f"Too big dt\nNeed to be less than {(1.0 / (4.0 * c_max))}")
        c_map = (cN + cS + cE + cW) / 4.0
        # ___ The final image is then updated here ___
        I += dt*(cN*nabla_N + cS*nabla_S + cE*nabla_E + cW*nabla_W)
    if return_details:
        return (
            c_map,
            I,
        )

    return I

def c_TV(grad_mag, epsilon=10**(-2)):
    return 1 / (np.sqrt(np.abs(grad_mag)**2 + epsilon**2))

# Two alternatives


# Combining the first-order gradient and second-order derivative
def c_laplacian(grad_mag, laplacian, k_g=0.1, k_l=0.15):
    E = (grad_mag / k_g)**2 + (np.abs(laplacian) / k_l)**2
    return 1.0/((1.0 + E)**2)


# A rational damping function of fourth order with the gradient quotient g/k
def c_highorder(grad_mag, k=0.1):
    E = grad_mag/k
    return 1.0/(1.0 + E**4)

# -- Experiment and Analysis --
epsilon_values = [10 ** (-3), 10 ** (-2), 10 ** (-1), 0.5]
T_total = 0.25

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for idx, e in enumerate(epsilon_values):
    dt = 0.9 * (e / 4.0)
    num_iter = max(1, int(np.round(T_total / dt)))


    c_map, _, _, img_filtered = anisotrophic_diffusion(
        img_noisy,
        num_iter=num_iter,
        dt=dt,
        c_func=c_TV,
        return_details = True,
        epsilon=e,
    )

    # Relative flow
    c_map_norm = c_map * e


    axes[0, idx].imshow(img_filtered, cmap="gray", vmin=0, vmax=1)
    axes[0, idx].set_title(
        f"$\\epsilon = {e}$\n({num_iter} iter, $\\Delta t={dt:.5f}$)"
    )
    axes[0, idx].axis("off")

    axes[1, idx].imshow(c_map_norm, cmap="magma", vmin=0, vmax=1)
    axes[1, idx].set_title(r"$c_{TV} \cdot \epsilon$ (Relative flow)")
    axes[1, idx].axis("off")

plt.tight_layout()
plt.savefig("C_TV_map")
plt.show()
# Showcasing numeric instability where dt is to big
# Parametrar: epsilon = 0.01 dt should be dt <= 0.0025. Have to comment out the error control in anisotrophic_diffusion
epsilon = 0.01


img_stable = anisotrophic_diffusion(img_noisy, num_iter=100, dt=0.002, c_func=c_TV, epsilon=epsilon)


img_unstable = anisotrophic_diffusion(img_noisy, num_iter=10, dt=0.01, c_func=c_TV, epsilon=epsilon)


fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img_stable, cmap='gray')
axes[0].set_title(f"Stable ($\Delta t = 0.002 \leq \\frac{{\epsilon}}{{4}}$)")
axes[0].axis('off')

axes[1].imshow(img_unstable, cmap='gray')
axes[1].set_title(f"Unstable ($\Delta t = 0.01 > \\frac{{\epsilon}}{{4}}$)")
axes[1].axis('off')

plt.tight_layout()
plt.savefig("Comparison_unstable_stable")
plt.show()



c_map_high, _, _, _ = anisotrophic_diffusion(
    img_noisy, num_iter=1, dt=0.001, c_func=c_highorder, return_details=True, k=0.1
)

c_map_lap, _, _, _ = anisotrophic_diffusion(
    img_noisy, num_iter=1, dt=0.001, c_func=c_laplacian, return_details=True, k_g=0.1, k_l=0.15
)


fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")

axes[0].imshow(c_map_high, cmap='magma', vmin=0, vmax=1)
axes[0].set_title(r"$c$-map: Highorder ($k=0.1$)")
axes[0].axis('off')

im2 = axes[1].imshow(c_map_lap, cmap='magma', vmin=0, vmax=1)
axes[1].set_title(r"$c$-map: Laplacian ($k_g=0.1, k_l=0.15$)")
axes[1].axis('off')

plt.colorbar(im2, ax=axes, shrink=0.8, label="Diffusionscoefficient $c$")
plt.savefig("comparison_highorder_laplacian.png")
plt.show()


def calc_rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))

# Defining methods and parameters
methods = [
    {"label": "Noisy Image", "key": "Noisy", "img": img_noisy, "col": "gray"},
    {"label": r"TV ($\epsilon=0.01$)", "key": "TV", "func": c_TV, "kwargs": {"num_iter": 100, "dt": 0.002, "epsilon": 0.01}, "col": "C0"},
    {"label": r"Highorder ($k=0.08$)", "key": "Highorder", "func": c_highorder, "kwargs": {"num_iter": 50, "dt": 0.15, "k": 0.08}, "col": "C1"},
    {"label": r"Stable Laplacian", "key": "Laplacian", "func": c_laplacian, "kwargs": {"num_iter": 50, "dt": 0.15, "k_g": 0.08, "k_l": 0.15}, "col": "C2"},
]

# Running simulations and calculating RMSE
for m in methods:
    if "func" in m:
        m["img"] = anisotrophic_diffusion(img_noisy, c_func=m["func"], **m["kwargs"])
    m["rmse"] = calc_rmse(m["img"], img)

fig = plt.figure(figsize=(15, 9))
row_idx, x_range = 128, np.arange(45, 85)

for idx, m in enumerate(methods):
    ax = fig.add_subplot(2, 4, idx + 1)
    ax.imshow(m["img"], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"{m['label']}\nRMSE: {m['rmse']:.4f}")
    ax.axis('off')


ax_prof = fig.add_subplot(2, 1, 2)
ax_prof.plot(x_range, img[row_idx, x_range], 'k--', label='Original (True border)', linewidth=2)

for m in methods:
    ax_prof.plot(x_range, m["img"][row_idx, x_range], color=m["col"], 
                 alpha=0.7 if m["key"] == "Noisy" else 1.0, 
                 label=m["key"], linewidth=1.8)

ax_prof.set_title("1D Intensity-profile across the border (Row 128, Pixel 45–85)")
ax_prof.set_xlabel("Pixel-position (x)")
ax_prof.set_ylabel("Intensity")
ax_prof.grid(True, linestyle=':', alpha=0.6)
ax_prof.legend()

plt.tight_layout()
plt.savefig("comparison_adaptive_image.png")
plt.show()

# Seleccione un píxel dentro de una región relativamente homogénea y otro próximo a un borde. Para una
# iteración, muestre las diferencias direccionales utilizadas, el valor de c y la actualización resultante en ambos
# casos.

# Creating padded image
pad = np.pad(img_noisy, pad_width=1, mode="edge")

# Parameters choosen 
k, dt = 0.08, 0.15
pixlar = {"Pixel A (Homogen)": (100, 80), "Pixel B (Border)": (128, 64)}

# Calculation
for name, (r, c) in pixlar.items():
    I = img_noisy[r, c]

    neighbours = np.array([pad[r, c + 1], pad[r + 2, c + 1], pad[r + 1, c + 2], pad[r + 1, c]])


    dN = neighbours - I
    c_val = 1.0 / (1.0 + (np.abs(dN) / k) ** 4)
    delta_I = dt * np.sum(c_val * dN)

    print(f"=== {name} (row {r}, column {c}) ===")
    print(f"Starting values I:      {I:.4f}")
    print(f"Differences (dN):  {np.round(dN, 4)}   # [N, S, E, W]")
    print(f"Coefficients (c): {np.round(c_val, 4)}   # [N, S, E, W]")
    print(f"(delta_I): {delta_I:.5f}")
    print(f"New Values I(1):   {I + delta_I:.4f}\n")