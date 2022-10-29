import os
import string
import random
import torch
from torchvision.io import write_png
from planet_generation_perlin import gen_random_cube_grid, grids_to_planet
from conf import data_path, img_size, layers

if not os.path.isdir(data_path):
    os.mkdir(data_path)


def generate_images(num_imgs: int):
    batch_size = 128

    if num_imgs % batch_size != 0:
        raise RuntimeError("Batch size must divide number of images")

    num_batches = int(num_imgs / batch_size)
    for i in range(num_batches):
        print(f"Generating batch {i}/{num_batches}...")
        grids = [
            torch.stack([gen_random_cube_grid(num_cells) for _ in range(batch_size)])
            for num_cells in layers
        ]
        planets = grids_to_planet(grids, batch_size, img_size).to('cpu')
        for j in range(batch_size):
            name = ''.join([random.choice(string.ascii_lowercase) for _ in range(16)])
            print(f"Saving image {i*batch_size + j + 1}/{num_imgs} ({name})")
            img_path = os.path.join(data_path, name + ".png")
            metadata_path = os.path.join(data_path, name + ".pt")
            assert not os.path.isfile(img_path)
            write_png((planets[j] * 255.9).type(torch.uint8)[None, :, :], img_path)
            torch.save([grid[j] for grid in grids], metadata_path)
    print("Done")


if __name__ == '__main__':
    generate_images(64000)
