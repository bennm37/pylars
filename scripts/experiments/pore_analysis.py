# from pylars.domain import generate_rv_circles
# import numpy as np
# from scipy.stats import lognorm
# import matplotlib.pyplot as plt
# import porespy as ps
# import openpnm as op


# def voxelify(cetnroids, radii, bound, n_voxels):
#     im = np.zeros((n_voxels, n_voxels))
#     x = np.linspace(-bound, bound, n_voxels)
#     y = np.linspace(-bound, bound, n_voxels)
#     X, Y = np.meshgrid(x, y)
#     for centroid, radii in zip(centroids, radii):
#         im[
#             np.where(
#                 (X - centroid.real) ** 2 + (Y - centroid.imag) ** 2
#                 < radii**2
#             )
#         ] = 1
#     return im


# rv = lognorm.rvs
# rv_args = {"s": 0.5, "scale": 0.27, "loc": 0.0}
# length = 16.0
# porosity = 0.95
# centroids, radii = generate_rv_circles(
#     porosity=porosity,
#     rv=rv,
#     rv_args=rv_args,
#     length=length,
#     min_dist=0.05,
# )
# im = voxelify(centroids, radii, length / 2, 100)
# snow_output = ps.networks.snow2(im)
# pn = op.io.network_from_porespy(snow_output.network)
# print(pn)

import numpy as np
import porespy as ps
import openpnm as op
import matplotlib.pyplot as plt

ps.visualization.set_mpl_style()
np.random.seed(10)
im = ps.generators.blobs(shape=[400, 400], porosity=0.6, blobiness=2)
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(im)
snow_output = ps.networks.snow2(im, voxel_size=1)
print(snow_output)
# pn = op.io.network_from_porespy(snow_output.network)
# print(pn)
