# James Brinkley,    A11763002
# Tyler Pitruzzello, A11717294

import math
import random
import numpy as np


def get_mean_std(index_array):
    mean = np.mean(index_array)
    std = np.std(index_array, dtype=np.float64)
    return mean, std


def apply_z_scale(training_set, mean_array, std_array):
    for element in training_set:
        for i in range(0, len(mean_array) - 1):
            element[i] = (element[i] - mean_array[i]) / std_array[i]

def undo_z_scale(training_set, mean_array, std_array):
    for element in training_set:
        for i in range(0, len(mean_array) - 1):
            element[i] = (element[i] * std_array[1]) + mean_array[i]

def get_k_neighbors(training_set, test_data, k):
    k_nearest_neighbors = [(0, 0)] * k
    for i in range(0, k):
        distance = 0
        for j in range(0, len(test_data) - 1):
            distance += math.pow(training_set[i][j] - test_data[j], 2)
        distance = math.sqrt(distance)
        k_nearest_neighbors[i] = i, distance
    k_nearest_neighbors = sorted(k_nearest_neighbors, key=lambda element: element[1])
    for i in range(k, len(training_set)):
        distance = 0
        for j in range(0, len(test_data) - 1):
            distance += math.pow((training_set[i][j] - test_data[j]), 2)
        distance = math.sqrt(distance)
        if distance < k_nearest_neighbors[k - 1]:
            k_nearest_neighbors[k - 1] = i, distance
            k_nearest_neighbors = sorted(k_nearest_neighbors, key=lambda element: element[1])
        nearest_neighbors_data = []
        for element in k_nearest_neighbors:
            nearest_neighbors_data.append(training_set[element[0]])
        return nearest_neighbors_data


def get_average(data_set):
    mean_array = []

    for i in range(0, len(data_set[0]) - 1):
        attribute_array = []
        for j in range(0, len(data_set)):
            observation = data_set[j]
            attribute_array.append(observation[i])
        mean = np.mean(attribute_array)
        mean_array.append(mean)

    return mean_array


def get_index_of_new_centroid(cluster):
    mean_array = get_average(cluster)
    centroid_index = get_closest(cluster, mean_array)
    return centroid_index


def get_closest(cluster, mean_array):
    # return index of closest element in data_set to the average
    closest_index = 0

    # return index of set that the data_entry will be switched too
    for i in range(0, len(cluster)):
        if get_euclidean_distance(cluster[i], mean_array) < \
                get_euclidean_distance(cluster[closest_index], mean_array):
            closest_index = i
    return closest_index


def get_nearest_centroid_index(cluster_list, data_entry):
    closest_centroid_index = 0

    # return index of set that the data_entry will be switched too
    for i in range(0, len(cluster_list)):
        if get_euclidean_distance(cluster_list[i][0], data_entry) < \
                get_euclidean_distance(data_entry, cluster_list[closest_centroid_index][0]):
            closest_centroid_index = i
    return closest_centroid_index


def get_euclidean_distance(point_a, point_b):
    distance = 0
    for j in range(0, len(point_a) - 1):
        distance += math.pow((point_a[j] - point_b[j]), 2)
    return math.sqrt(distance)

def get_wcss(cluster):
    wcss_sum = 0
    for i in range(1, len(cluster)):
        wcss_sum += math.pow(get_euclidean_distance(cluster[i], cluster[0]), 2)

    return wcss_sum

def qr_decompose (X_train):
    n = len(X_train)
    q_accumulated = np.identity(n)
    r_last = np.mat(X_train)
    X_train = np.mat(X_train)
    rows, cols = X_train.shape
    # 1. get the i-th column of matrix X_train put in Zi
    for i in range(0, cols):

        e_i = make_e_vector(n)
        z_i = np.mat(r_last[i:, i])

        # 2. Calculate Vi by:
        if z_i[0, 0] > 0:
            e_i *= np.linalg.norm(z_i)
            e_i[0, 0] *= -1
            v_i = np.subtract(e_i, z_i)
        else:
            v_i = (np.linalg.norm(z_i) * e_i) - z_i

        p_i = np.subtract(np.identity(n), (np.multiply((v_i * v_i.T), (2.0 / ((v_i.T * v_i)[0,0])))))
        r, c = p_i.shape

        sub_r = r_last[(rows - r):, (rows - c):]

        r_last[rows - r:, rows - c:] = np.dot(p_i, sub_r)
        n -= 1
    q = q_accumulated.T

    r = r_last
    return q, r

def back_solve (R, label_values):
    rows, cols = R.shape
    index = cols - 1
    betas = [1] * cols
    for i in range(index, -1, -1):
        sum = 0
        for j in range(index, -1, -1):
            if j == i:
                betas[j] = (label_values[j, 0] - sum) / R[i, j]
                break
            else:
                sum += betas[j] * R[i, j]
    return np.mat(betas).T

def make_e_vector(n):
    e = [0.0] * n
    e[0] = 1.0
    return np.transpose(np.mat(e))

def make_q_matrix(n, p_i):
    q = np.identity(n)
    rows, cols = p_i.shape
    q[n - rows:, n - cols:] = p_i
    return np.mat(q)