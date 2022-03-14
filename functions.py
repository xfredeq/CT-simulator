import math

import numpy as np
import skimage.color
import skimage.draw


def _convert_coords(coords, r):
    return coords[0] + r, -coords[1] + r


def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def adjust_image(image):
    if len(image.shape) > 2:
        image = skimage.color.rgb2gray(image)

    if np.max(image) > 1:
        image = normalize(image)

    y, x = image.shape
    if y < x:
        fill = np.zeros(((x-y)//2, x))
        image = np.vstack((fill, image, fill))
    elif x < y:
        fill = np.zeros((y, (y-x)//2))
        image = np.hstack((fill, image, fill))

    dimension = min(image.shape)
    image = image[0:dimension, 0:dimension]

    return image


def make_sinogram(image, alpha_step=4, phi=180, n=200):
    alpha_step = np.deg2rad(alpha_step)
    phi = np.deg2rad(phi)

    img_size = image.shape[0]
    r = 0.5 * img_size

    iterations = math.ceil(2 * np.pi / alpha_step)
    sinogram = np.zeros((iterations, n))

    alpha = 0
    for iteration in range(iterations):
        x_e = r * np.cos(alpha)
        y_e = r * np.sin(alpha)
        x_e, y_e = _convert_coords((x_e, y_e), r)

        for i in range(n):
            x_d = r * np.cos(alpha + np.pi - phi / 2 + i * (phi / (n - 1)))
            y_d = r * np.sin(alpha + np.pi - phi / 2 + i * (phi / (n - 1)))
            x_d, y_d = _convert_coords((x_d, y_d), r)

            coords = skimage.draw.line_nd([x_e, y_e], [x_d, y_d])

            # Niestety mogą zdarzyć się koordy poza obrazem
            coords[0][coords[0] >= img_size] -= 1
            coords[1][coords[1] >= img_size] -= 1

            points = image[coords[1], coords[0]]

            # Średnia, nie suma -> bez artefaktów
            sinogram[iteration, i] = np.mean(points)

        alpha += alpha_step

    sinogram = normalize(sinogram)
    return sinogram


def _make_kernel(size):
    half_size = math.floor(size / 2)
    kernel = np.zeros(half_size)
    center = math.floor(half_size / 2)

    for i in range(0, half_size):
        k = i - center
        if k % 2 != 0:
            kernel[i] = (-4 / (np.pi * np.pi)) / (k * k)

    kernel[center] = 1
    return kernel


def filter_sinogram(sinogram):
    # Jaki rozmiar?? Na razie liczba detektorów
    kernel = _make_kernel(sinogram.shape[1])

    for i in range(len(sinogram)):
        sinogram[i, :] = np.convolve(sinogram[i, :], kernel, mode="same")

    return sinogram


def reconstruct_image(sinogram, alpha_step, phi, n, img_size):
    alpha_step = np.deg2rad(alpha_step)
    phi = np.deg2rad(phi)

    image = np.zeros((img_size, img_size))
    r = 0.5 * img_size

    iterations = math.ceil(2 * np.pi / alpha_step)

    alpha = 0
    for iteration in range(iterations):
        x_e = r * np.cos(alpha)
        y_e = r * np.sin(alpha)
        x_e, y_e = _convert_coords((x_e, y_e), r)

        for i in range(n):
            x_d = r * np.cos(alpha + np.pi - phi / 2 + i * (phi / (n - 1)))
            y_d = r * np.sin(alpha + np.pi - phi / 2 + i * (phi / (n - 1)))
            x_d, y_d = _convert_coords((x_d, y_d), r)

            coords = skimage.draw.line_nd([x_e, y_e], [x_d, y_d])

            # Niestety mogą zdarzyć się koordy poza obrazem
            coords[0][coords[0] >= img_size] -= 1
            coords[1][coords[1] >= img_size] -= 1

            image[coords[1], coords[0]] += sinogram[iteration, i]

        alpha += alpha_step

    image = normalize(image)

    # Ostrożnie
    #image[:][image[:] < 0.2] = 0
    #image[:][image[:] > 0.8] = 1

    return image
