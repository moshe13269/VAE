import numpy as np

v1 = np.asarray([0.0, 0.25, 0.5, 0.75])
v2 = np.asarray([0.0, 0.43])


def denormalized_vector(vector):
    vector[0] = v1[np.argmin(np.abs(v1 - vector[0]))]
    vector[1] = v1[np.argmin(np.abs(v1 - vector[1]))]
    vector[2] = v2[np.argmin(np.abs(v2 - vector[2]))]
    return vector

