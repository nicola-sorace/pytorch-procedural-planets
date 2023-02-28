> This project was mostly intended as a learning exercise.
> The goal was to implement a U-Net in PyTorch and use it in a new application.

# Problem statement

Consider a procedural terrain generation algorithm, such as [Perlin noise](https://en.wikipedia.org/wiki/Perlin_noise).
This algorithm takes grids of random values as input, and outputs visually smooth terrain height-maps by interpolating between these values.

The goal of this investigation is to train a machine learning framework which reverses this process:
Given a terrain height-map image, obtain a set of vector grids which produces the same terrain when passed through the procedural generation algorithm.

# Procedural Generation

The procedural terrain algorithm which will be used is a variation on Perlin noise.
Unlike standard Perlin noise, the generated terrain is spherical, representing an entire planetary surface.
This is achieved by combining 6 planar grids to form a cube, and then mapping that cube to a sphere.
In order to interpolate smoothly in this space, 3D vectors are used (even though the surface is 2D).

# Neural Network structure

The network primarily uses a U-Net structure to convert height-map values into Perlin node values.
However, this is surrounded by some pre- and post-processing steps in order to manipulate the data into suitable forms.

## Pre-processing

Terrain images are passed to the network in the form of 6 faces forming a cube, rather than a single equirectangular projection.
There are two reasons why this is preferred:

1. The "cube-faces" form matches the structure of the desired output (a cube of vector grids).
2. Each face can be processed independently and equivalently, as there should be no structural difference between them.
This is in contrast to equirectangular images, where each region of the image would be subject to drastically different warping.

Since these faces should be independent, this new "face" dimension is flattened into the network's batch dimension, meaning that each face is fed into the network independently of other faces.

For each layer of Perlin noise, it is noted that the value of a node vector is more closely linked to far-away pixels than to nearby pixels.
In fact, pixels at a nodal point will always be zero, providing no information at all.
When calculating the value of a vector, the most useful information should be near the center of nearby cells (which is to say at a distance of half the grid spacing along each axis).
With this in mind, additional channels are added to the input image: four channels for each Perlin layer, corresponding to the planet being rotated in such a way that one of the four nearby cell centers is aligned with the nodal point.
In this way, information about nearby cells is brought spatially closer to where it is needed (just in a different channel).
In addition, these extra channels allow the network to "peek" into nearby faces, as information would otherwise be limited to the current face.

## Post-processing

The network outputs a grid of values of equal size to the input face, but with number of channels equal to the number of desired Perlin layers times three.
Finally, each group of three channels is down-sampled (using an average-pooling layer) to match the number of grid vectors for that layer.
Hence, each of the three channels becomes one element of the output vectors for that layer.

Depending on the choice of loss function, these vectors can either be outputted as-is, or passed back into the procedural generation function to produce a final terrain image.

# Loss function

Two loss functions were attempted: Mean Squared Error between input and output image, and Mean Squared Error between target and output vectors.

It was found that MSE between target and output vectors was vastly more successful.
In addition to producing better results in fewer epochs, skipping the image generation step resulted in much lower training time per epoch.

# Results

The network achieves great results on unseen Perlin images, with loss indistinguishable from training images:

![Network performance on unseen procedural terrain (targets on the left, outputs on the right)](readme_img/test-original.png)

However, it remains unable to generalize to terrain that was not produced via this method:

![Network performance on a real height-map of the earth (target above, output below)](readme_img/validation-original.png)

It is hypothesized that this is due to Perlin noise being insufficiently flexible to reproduce arbitrary terrain.
Although Perlin terrain is *seemingly* random, in actuality it is limited in the images it can produce.
The most obvious evidence of this is the fact that all values at grid nodes must be zero.

A modification was attempted in which each grid node was given a constant offset value, which is added to that node's value before final interpolation.
This improved results, but only marginally:

![Performance on unseen images with constant offset value (targets on the left, outputs on the right)](readme_img/test-constant-offset.png)
![Performance on real height-map with constant offset value (target above, output below)](readme_img/validation-constant-offset.png)

Here it seems likely that the network has learnt how to find a suitable constant offset value (which is somewhat trivial), but not the correct vectors.

In conclusion, it seems that the network is successful at solving the task for layered Perlin noise, but that layered Perlin noise is not representative of real terrain height-maps.
Some modification to Perlin noise may solve this issue.
Another problem is the fact that real planetary height-maps are in very short supply, meaning there is insufficient real-world training data if procedural noise cannot be used.

