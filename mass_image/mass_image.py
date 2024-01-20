import numpy as np
from PIL import Image
import random

def mandelbrot(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    C = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    C = C[:, 0] + 1j * C[:, 1]
    N = [mandelbrot(c, max_iter) for c in C]
    N = np.array(N).reshape(width, height)
    return N

def generate_random_image(max_iter, num_images):
    i = 0
    while(i < num_images):
        # Randomly select a region within the bounds
        center_x = random.uniform(-2, 2)
        center_y = random.uniform(-2, 2)
        width = 0.2
        xmin, xmax = center_x - width/2, center_x + width/2
        ymin, ymax = center_y - width/2, center_y + width/2

        mandelbrot_image = mandelbrot_set(xmin, xmax, ymin, ymax, 256, 256, max_iter)
        image = Image.fromarray(np.uint8(mandelbrot_image / max_iter * 255), 'L')
        
        # Check if the image is not entirely black or white
        if not np.all(mandelbrot_image == 0) and not np.all(mandelbrot_image == max_iter):
            i += 1
            image.save(f"mandelbrot_random_{i}.png")
            print("Image " + str(i) + " Saved Successfully!")

# Example usage
generate_random_image(1000, 10000)
