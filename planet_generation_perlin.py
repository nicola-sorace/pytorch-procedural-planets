"""
Procedural planet generation using Perlin noise

ISO spherical coordinates used throughout:
 - polar `theta` in [0, pi)
 - azimuthal `phi` in [0, 2pi)
"""

#%% Imports
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple
from enum import IntEnum

colormap = cm.turbo


class Face(IntEnum):
    LEFT = 0
    FRONT = 1
    RIGHT = 2
    BACK = 3
    TOP = 4
    BOTTOM = 5


def lerp(a, b, f):
    return (1 - f) * a + f * b


def smootherstep(f):
    return f**3 * (f * (f * 6 - 15) + 10)


def dot(a, b):
    return a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1] + a[:, :, 2] * b[:, :, 2]


def gen_random_cube_grid(num_cells: int) -> torch.Tensor:
    """
    Generate a grid of random values for each face of a cube, where values
    at corners and edges match across neighbors
    :param num_cells: Number of cells along each axis (note that values are computed at nodes, not cells)
    :return: Tensor of dimensions (6, num_cells + 1, num_cells + 1, 2); one grid of 2D vectors for each of the 6 faces.
    """
    # Corner and edge values are shared
    corner_vals = torch.rand(8, 3) * 2 - 1
    edge_vals = torch.rand(
        12, num_cells - 1, 3
    ) * 2 -1
    # The rest is unique to each face
    face_vals = torch.rand(6, num_cells - 1, num_cells - 1, 3) * 2 - 1

    def concat_face(f, c1, c2, c3, c4, e1, e2, e3, e4, flip_top=False, flip_bottom=False):
        top_left = corner_vals[c1:c1+1]
        top_edge = edge_vals[e1]
        top_right = corner_vals[c2:c2+1]
        left_edge = edge_vals[e4:e4+1]
        right_edge = edge_vals[e2:e2+1]
        bottom_left = corner_vals[c4:c4+1]
        bottom_edge = edge_vals[e3]
        bottom_right = corner_vals[c3:c3+1]

        if flip_top:
            top_edge = top_edge.flip(0)
        if flip_bottom:
            bottom_edge = bottom_edge.flip(0)

        r3 = torch.concat((top_left, top_edge, top_right))[:, None, :]
        r2 = torch.concat((left_edge, face_vals[f], right_edge))
        r1 = torch.concat((bottom_left, bottom_edge, bottom_right))[:, None, :]
        return torch.concat((r1, r2, r3), dim=1)

    # Finally, assemble into a grid of values for each face
    left = concat_face(Face.LEFT, 0, 1, 2, 3, 0, 1, 2, 3, flip_top=True)
    front = concat_face(Face.FRONT, 1, 5, 6, 2, 9, 5, 10, 1)
    right = concat_face(Face.RIGHT, 5, 4, 7, 6, 4, 7, 6, 5, flip_bottom=True)
    back = concat_face(Face.BACK, 4, 0, 3, 7, 8, 3, 11, 7)
    top = concat_face(Face.TOP, 0, 4, 5, 1, 8, 4, 9, 0, flip_top=True)
    bottom = concat_face(Face.BOTTOM, 2, 6, 7, 3, 10, 6, 11, 2, flip_bottom=True)

    return torch.stack((left, front, right, back, top, bottom))


def theta_phi_to_face_x_y(theta, phi):
    # First convert spherical coordinates to cartesian
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    x_abs = torch.abs(x)
    y_abs = torch.abs(y)
    z_abs = torch.abs(z)

    faces = torch.empty(theta.shape, dtype=torch.long)
    face_xs = torch.empty(theta.shape)
    face_ys = torch.empty(theta.shape)

    back_faces = (y > 0) & (y > x_abs) & (y > z_abs)
    left_faces = (x < 0) & (-x > z_abs) & (-x > y_abs)
    right_faces = (x > 0) & (x > z_abs) & (x > y_abs)
    top_faces = (z > 0) & (z > x_abs) & (z > y_abs)
    bottom_faces = (z < 0) & (-z > x_abs) & (-z > y_abs)
    front_faces = ~back_faces & ~left_faces & ~right_faces & ~top_faces & ~bottom_faces

    faces[left_faces] = Face.LEFT
    faces[front_faces] = Face.FRONT
    faces[right_faces] = Face.RIGHT
    faces[back_faces] = Face.BACK
    faces[top_faces] = Face.TOP
    faces[bottom_faces] = Face.BOTTOM

    face_xs[left_faces] = (-y[left_faces] / x_abs[left_faces] + 1) / 2
    face_xs[front_faces] = (x[front_faces] / y_abs[front_faces] + 1) / 2
    face_xs[right_faces] = (y[right_faces] / x_abs[right_faces] + 1) / 2
    face_xs[back_faces] = (-x[back_faces] / y_abs[back_faces] + 1) / 2
    face_xs[top_faces] = (x[top_faces] / z_abs[top_faces] + 1) / 2
    face_xs[bottom_faces] = (x[bottom_faces] / z_abs[bottom_faces] + 1) / 2

    face_ys[left_faces] = (z[left_faces] / x_abs[left_faces] + 1) / 2
    face_ys[front_faces] = (z[front_faces] / y_abs[front_faces] + 1) / 2
    face_ys[right_faces] = (z[right_faces] / x_abs[right_faces] + 1) / 2
    face_ys[back_faces] = (z[back_faces] / y_abs[back_faces] + 1) / 2
    face_ys[top_faces] = (y[top_faces] / z_abs[top_faces] + 1) / 2
    face_ys[bottom_faces] = (-y[bottom_faces] / z_abs[bottom_faces] + 1) / 2

    return faces, face_xs, face_ys


def face_x_y_to_x_y_z(faces, fxs, fys):
    fzs = torch.ones(fxs.shape)
    xs = torch.empty(fxs.shape)
    ys = torch.empty(fxs.shape)
    zs = torch.empty(fxs.shape)

    # Normalize
    mag = torch.sqrt(fxs*fxs + fys*fys + fzs*fzs)
    fxs = fxs / mag
    fys = fys / mag
    fzs = fzs / mag

    xs[faces == Face.LEFT] = -fzs[faces == Face.LEFT]
    ys[faces == Face.LEFT] = -fxs[faces == Face.LEFT]
    zs[faces == Face.LEFT] = fys[faces == Face.LEFT]

    xs[faces == Face.FRONT] = fxs[faces == Face.FRONT]
    ys[faces == Face.FRONT] = -fzs[faces == Face.FRONT]
    zs[faces == Face.FRONT] = fys[faces == Face.FRONT]

    xs[faces == Face.RIGHT] = fzs[faces == Face.RIGHT]
    ys[faces == Face.RIGHT] = fxs[faces == Face.RIGHT]
    zs[faces == Face.RIGHT] = fys[faces == Face.RIGHT]

    xs[faces == Face.BACK] = -fxs[faces == Face.BACK]
    ys[faces == Face.BACK] = fzs[faces == Face.BACK]
    zs[faces == Face.BACK] = fys[faces == Face.BACK]

    xs[faces == Face.TOP] = fxs[faces == Face.TOP]
    ys[faces == Face.TOP] = fys[faces == Face.TOP]
    zs[faces == Face.TOP] = fzs[faces == Face.TOP]

    xs[faces == Face.BOTTOM] = fxs[faces == Face.BOTTOM]
    ys[faces == Face.BOTTOM] = -fys[faces == Face.BOTTOM]
    zs[faces == Face.BOTTOM] = -fzs[faces == Face.BOTTOM]

    return torch.stack((xs, ys, zs)).permute(1, 2, 0)


def theta_phi_to_val(grid, num_cells, theta, phi):
    faces, xs, ys = theta_phi_to_face_x_y(theta, phi)
    xs = xs * num_cells
    ys = ys * num_cells
    min_x, max_x, frac_x = (
        torch.max(torch.FloatTensor([0]), torch.floor(xs)).long(),
        torch.min(torch.FloatTensor([num_cells]), torch.ceil(xs)).long(),
        smootherstep(torch.fmod(xs, torch.FloatTensor([1])))
    )
    min_y, max_y, frac_y = (
        torch.max(torch.FloatTensor([0]), torch.floor(ys)).long(),
        torch.min(torch.FloatTensor([num_cells]), torch.ceil(ys)).long(),
        smootherstep(torch.fmod(ys, torch.FloatTensor([1])))
    )
    val00, val01, val10, val11 = (
        grid[faces, min_x, min_y],
        grid[faces, min_x, max_y],
        grid[faces, max_x, min_y],
        grid[faces, max_x, max_y]
    )
    corn00, corn01, corn10, corn11 = (
        face_x_y_to_x_y_z(faces, 2 * min_x.float() / num_cells - 1, 2 * min_y.float() / num_cells - 1),
        face_x_y_to_x_y_z(faces, 2 * min_x.float() / num_cells - 1, 2 * max_y.float() / num_cells - 1),
        face_x_y_to_x_y_z(faces, 2 * max_x.float() / num_cells - 1, 2 * min_y.float() / num_cells - 1),
        face_x_y_to_x_y_z(faces, 2 * max_x.float() / num_cells - 1, 2 * max_y.float() / num_cells - 1),
    )

    px = torch.sin(theta) * torch.cos(phi)
    py = torch.sin(theta) * torch.sin(phi)
    pz = torch.cos(theta)

    p = torch.stack((px, py, pz)).permute(1, 2, 0)

    off00, off01, off10, off11 = (
        p - corn00, p - corn01, p - corn10, p - corn11
    )
    val00, val01, val10, val11 = (
        dot(val00, off00), dot(val01, off01), dot(val10, off10), dot(val11, off11)
    )
    return lerp(lerp(val00, val10, frac_x), lerp(val01, val11, frac_x), frac_y)


def grid_to_planet(grid: torch.Tensor, num_cells: int, img_size: Tuple[int, int]) -> torch.Tensor:
    theta = torch.linspace(0, torch.pi, img_size[1])
    phi = torch.linspace(0, 2*torch.pi, img_size[0])
    theta, phi = torch.meshgrid(theta, phi)

    return theta_phi_to_val(grid, num_cells, theta, phi)


def planet_plot_3d(planet):
    print("Generating 3d plot...")
    theta = torch.linspace(0, torch.pi, planet.shape[0])
    phi = torch.linspace(0, 2*torch.pi, planet.shape[1])
    phi, theta = torch.meshgrid(phi, theta)

    # The Cartesian coordinates of the unit sphere
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    # Normalize values
    fmax, fmin = planet.max(), planet.min()
    planet = (planet - fmin) / (fmax - fmin)

    # Set the aspect ratio to 1 so our sphere looks spherical
    # fig, ax = plt.subplots(subplot_kw={'projection': '3d', 'aspect': 'equal'})
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colormap(planet.mT))

    # Turn off the axis planes
    ax.set_axis_off()
    plt.show()
    print("Plot generated")


if __name__ == '__main__':
    num_cells = 10
    # img_size = (128, 64)
    img_size = (512, 256)
    # img_size = (1024, 512)
    grid = gen_random_cube_grid(num_cells)
    planet = grid_to_planet(grid, num_cells, img_size)
    plt.imshow(planet, cmap=colormap)
    plt.show()
    planet_plot_3d(planet)
