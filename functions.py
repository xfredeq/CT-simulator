import numpy as np
import math
import skimage.draw, skimage.color

def _convert_coords(coords, r):
    x,y = coords
    return (x + r, -y + r)

def crop_image(image):
    image = skimage.color.rgb2gray(image).astype(np.float)
    dimension = min(image.shape)
    image = image[0:dimension, 0:dimension]
    return image

# Nie dziala jeszcze ...
def make_sinogram(image, alpha_step=0.0175, phi=math.pi/2, n=180):
    r = 0.5 * image.shape[0] # TODO: Okrąg wpisany czy opisany?

    sinogram = np.zeros((math.ceil(phi/alpha_step), n))

    alpha = 0
    count = 0
    while alpha <= phi:

        x_e = r * math.cos(alpha)
        y_e = r * math.sin(alpha)
        x_e, y_e = _convert_coords((x_e, y_e), r)

        for i in range(n):
            x_d = r * math.cos(alpha + math.pi - phi/2 + i * (phi/(n-1)))
            y_d = r * math.sin(alpha + math.pi - phi/2 + i * (phi/(n-1)))
            x_d, y_d = _convert_coords((x_d, y_d), r)

            coords = skimage.draw.line_nd([x_e, y_e], [x_d, y_d])
            
            coords[0][coords[0] >= image.shape[0]] -= 1
            coords[1][coords[1] >= image.shape[0]] -= 1
            
            points = image[coords[1], coords[0]]

            points_sum = np.sum(points)
            sinogram[count, i] = points_sum
        
        alpha += alpha_step
        count += 1

    sinogram *= 1.0/sinogram.max()
    return sinogram