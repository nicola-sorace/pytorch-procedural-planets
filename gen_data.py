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
    for i in range(num_imgs):
        name = ''.join([random.choice(string.ascii_lowercase) for _ in range(16)])
        img_path = os.path.join(data_path, name + ".png")
        metadata_path = os.path.join(data_path, name + ".pt")
        assert not os.path.isfile(img_path)
        print(f"Generating '{name}' ({i+1}/{num_imgs})")

        grids = [
            gen_random_cube_grid(num_cells)
            for num_cells in layers
        ]
        planet = grids_to_planet(grids, img_size)
        write_png((planet * 255.9).type(torch.uint8)[None, :, :], img_path)
        torch.save(grids, metadata_path)
    print("Done")


if __name__ == '__main__':
    generate_images(10000)