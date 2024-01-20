import numpy as np
from PIL import Image
import random

def mandelbrot(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return 0  # Point is not in the Mandelbrot set
        z = z*z + c
    return 1  # Point is in the Mandelbrot set

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
    while i < num_images:
        center_x = random.uniform(-2, 2)
        center_y = random.uniform(-2, 2)
        width = 0.01
        xmin, xmax = center_x - width/2, center_x + width/2
        ymin, ymax = center_y - width/2, center_y + width/2

        mandelbrot_image = mandelbrot_set(xmin, xmax, ymin, ymax, 256, 256, max_iter)
        image = Image.fromarray(np.uint8(mandelbrot_image * 255), 'L')
        
        black_percentage = np.sum(mandelbrot_image == 0) / mandelbrot_image.size
        white_percentage = np.sum(mandelbrot_image == 1) / mandelbrot_image.size

        if black_percentage >= 0.01 and white_percentage >= 0.01:
            i += 1
            image.save(f"photos/mandelbrot_random_{i}.png")
            print(f"Image {i} Saved Successfully: Black {black_percentage*100:.2f}%, White {white_percentage*100:.2f}%")
        else:
            print(f"Image Skipped: Black {black_percentage*100:.2f}%, White {white_percentage*100:.2f}%")

# Example usage
generate_random_image(100, 10)
