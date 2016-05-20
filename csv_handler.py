import math
import random
import numpy as np

def getTrainingSet(indexArray, rng):
    obsvLeft = float(418)
    dataLeft = float(4177)

    lineCount = 0

    with open("abalone.data", "r+") as file:
        for line in file:
            randThresh = rng.random()
            if randThresh < obsvLeft/dataLeft:
                indexArray[lineCount] += 1
                obsvLeft -= 1

            if obsvLeft == 0:
                break

            dataLeft -= 1
            lineCount += 1

def getMeanStd(indexArray):
    mean = np.mean(indexArray)
    std = np.std(indexArray, dtype=np.float64)
    return mean, std

def apply_z_scale(training_set, mean_array, std_array):
    for element in training_set:
        for i in range(0, len(mean_array) - 1):
            element[i] = (element[i] - mean_array[i]) / std_array[i]

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