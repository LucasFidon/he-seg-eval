"""Lookup tables used by surface distance metrics."""

# neighbour_code_to_normals is a lookup table.
# For every binary neighbour code
# (2x2x2 neighbourhood = 8 neighbours = 8 bits = 256 codes)
# it contains the surface normals of the triangles (called "surfel" for
# "surface element" in the following). The length of the normal
# vector encodes the surfel area.
#
# created using the marching_cube algorithm
# see e.g. https://en.wikipedia.org/wiki/Marching_cubes
# pylint: disable=line-too-long
neighbour_code_to_normals = [
    [[0, 0, 0]],
    [[0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125]],
    [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
    [[0.125, -0.125, 0.125]],
    [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
    [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[-0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
    [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125]],
    [[-0.5, 0.0, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
    [[0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25]],
    [[-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, -0.5, 0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, 0.0, -0.5], [0.25, 0.25, 0.25], [-0.125, -0.125, -0.125]],
    [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [0.125, -0.125, -0.125]],
    [[0.125, 0.125, 0.125], [0.375, 0.375, 0.375], [0.0, -0.25, 0.25], [-0.25, 0.0, 0.25]],
    [[0.125, -0.125, -0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[0.375, 0.375, 0.375], [0.0, 0.25, -0.25], [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
    [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25], [0.125, 0.125, 0.125]],
    [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25]],
    [[0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
    [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25]],
    [[0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.125, -0.125, 0.125], [-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
    [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25], [0.125, -0.125, 0.125]],
    [[-0.375, -0.375, 0.375], [-0.0, 0.25, 0.25], [0.125, 0.125, -0.125], [-0.25, -0.0, -0.25]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.125, 0.125, -0.125], [-0.125, -0.125, 0.125]],
    [[0.125, -0.125, 0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125], [0.125, -0.125, 0.125]],
    [[0.0, 0.25, -0.25], [0.375, -0.375, -0.375], [-0.125, 0.125, 0.125], [0.25, 0.25, 0.0]],
    [[-0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
    [[0.0, 0.5, 0.0], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
    [[0.0, 0.5, 0.0], [0.125, -0.125, 0.125], [-0.25, 0.25, -0.25]],
    [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]],
    [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], [0.125, -0.125, 0.125]],
    [[-0.375, -0.375, -0.375], [-0.25, 0.0, 0.25], [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
    [[0.125, 0.125, 0.125], [0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
    [[0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
    [[-0.125, 0.125, 0.125], [0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
    [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.375, 0.375, -0.375], [-0.25, -0.25, 0.0], [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
    [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
    [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
    [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
    [[-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25]],
    [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.375, -0.375, 0.375], [0.0, -0.25, -0.25], [-0.125, 0.125, -0.125], [0.25, 0.25, 0.0]],
    [[-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
    [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
    [[-0.25, 0.25, -0.25], [-0.25, 0.25, -0.25], [-0.125, 0.125, -0.125], [-0.125, 0.125, -0.125]],
    [[-0.25, 0.0, -0.25], [0.375, -0.375, -0.375], [0.0, 0.25, -0.25], [-0.125, 0.125, 0.125]],
    [[0.5, 0.0, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
    [[-0.0, 0.0, 0.5], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
    [[-0.25, -0.0, -0.25], [-0.375, 0.375, 0.375], [-0.25, -0.25, 0.0], [-0.125, 0.125, 0.125]],
    [[0.0, 0.0, -0.5], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
    [[-0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
    [[0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
    [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25], [-0.125, 0.125, 0.125]],
    [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
    [[0.125, -0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[0.25, 0.0, 0.25], [-0.375, -0.375, 0.375], [-0.25, 0.25, 0.0], [-0.125, -0.125, 0.125]],
    [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
    [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.125, -0.125, 0.125]],
    [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
    [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25], [0.125, -0.125, -0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
    [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.5, 0.0, -0.0], [0.25, -0.25, -0.25], [0.125, -0.125, -0.125]],
    [[-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
    [[0.375, -0.375, 0.375], [0.0, 0.25, 0.25], [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
    [[0.0, -0.5, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[-0.375, -0.375, 0.375], [0.25, -0.25, 0.0], [0.0, 0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
    [[0.125, 0.125, 0.125], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
    [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
    [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
    [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
    [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], [0.125, 0.125, 0.125]],
    [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
    [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
    [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
    [[0.125, 0.125, 0.125], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
    [[-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
    [[-0.375, -0.375, 0.375], [0.25, -0.25, 0.0], [0.0, 0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.0, -0.5, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[0.375, -0.375, 0.375], [0.0, 0.25, 0.25], [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
    [[-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
    [[0.5, 0.0, -0.0], [0.25, -0.25, -0.25], [0.125, -0.125, -0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25], [0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
    [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.125, -0.125, 0.125]],
    [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[0.125, 0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
    [[0.25, 0.0, 0.25], [-0.375, -0.375, 0.375], [-0.25, 0.25, 0.0], [-0.125, -0.125, 0.125]],
    [[0.125, -0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25], [-0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
    [[-0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
    [[0.0, 0.0, -0.5], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
    [[-0.25, -0.0, -0.25], [-0.375, 0.375, 0.375], [-0.25, -0.25, 0.0], [-0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
    [[-0.0, 0.0, 0.5], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
    [[0.5, 0.0, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
    [[-0.25, 0.0, -0.25], [0.375, -0.375, -0.375], [0.0, 0.25, -0.25], [-0.125, 0.125, 0.125]],
    [[-0.25, 0.25, -0.25], [-0.25, 0.25, -0.25], [-0.125, 0.125, -0.125], [-0.125, 0.125, -0.125]],
    [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
    [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[0.375, -0.375, 0.375], [0.0, -0.25, -0.25], [-0.125, 0.125, -0.125], [0.25, 0.25, 0.0]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25]],
    [[-0.125, -0.125, 0.125], [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
    [[-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125]],
    [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
    [[0.125, 0.125, 0.125], [-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
    [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [-0.125, -0.125, 0.125]],
    [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
    [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
    [[-0.375, 0.375, -0.375], [-0.25, -0.25, 0.0], [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
    [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
    [[0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
    [[0.125, 0.125, 0.125], [0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
    [[-0.375, -0.375, -0.375], [-0.25, 0.0, 0.25], [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
    [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], [0.125, -0.125, 0.125]],
    [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]],
    [[0.0, 0.5, 0.0], [0.125, -0.125, 0.125], [-0.25, 0.25, -0.25]],
    [[0.0, 0.5, 0.0], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
    [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
    [[-0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.0, 0.25, -0.25], [0.375, -0.375, -0.375], [-0.125, 0.125, 0.125], [0.25, 0.25, 0.0]],
    [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125], [0.125, -0.125, 0.125]],
    [[0.125, -0.125, 0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.125, 0.125, -0.125], [-0.125, -0.125, 0.125]],
    [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.375, -0.375, 0.375], [-0.0, 0.25, 0.25], [0.125, 0.125, -0.125], [-0.25, -0.0, -0.25]],
    [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25], [0.125, -0.125, 0.125]],
    [[0.125, -0.125, 0.125], [-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
    [[0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25]],
    [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.125, -0.125, 0.125]],
    [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25]],
    [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25], [0.125, 0.125, 0.125]],
    [[0.375, 0.375, 0.375], [0.0, 0.25, -0.25], [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
    [[0.125, -0.125, -0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[0.125, 0.125, 0.125], [0.375, 0.375, 0.375], [0.0, -0.25, 0.25], [-0.25, 0.0, 0.25]],
    [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
    [[-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, 0.0, -0.5], [0.25, 0.25, 0.25], [-0.125, -0.125, -0.125]],
    [[0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, -0.5, 0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25]],
    [[0.125, -0.125, -0.125]],
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
    [[-0.5, 0.0, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125]],
    [[0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
    [[0.125, 0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[-0.125, 0.125, 0.125]],
    [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
    [[0.125, -0.125, 0.125]],
    [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
    [[-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125]],
    [[0, 0, 0]]]
# pylint: enable=line-too-long