import csv
import numpy as np
import random as rng
import csv_handler

whole_data = []
with open("abalone.data", "r+") as file:
    for line in file:
        data = []
        sex_index = True
        for element in line.split(','):
            if sex_index:
                sex_index = False
                if element == "M" or element == "m":
                    data.append(1.0)
                    data.append(0.0)
                    data.append(0.0)
                if element == "F" or element == "f":
                    data.append(0.0)
                    data.append(1.0)
                    data.append(0.0)
                if element == "I" or element == "i":
                    data.append(0.0)
                    data.append(0.0)
                    data.append(1.0)
            else:
                data.append(float(element))
        whole_data.append(data)


np.random.seed(99)
np.random.shuffle(whole_data)
train_num = int(len(whole_data) * 0.9)

training_set = whole_data[:train_num]
test_set = whole_data[train_num:]

mean_array = []
std_array = []

for i in range(0, len(training_set[0]) - 1):
    attribute_array = []
    for j in range(0, len(training_set)):
        observation = training_set[j]
        attribute_array.append(observation[i])
    mean = np.mean(attribute_array)
    std = np.std(attribute_array)
    mean_array.append(mean)
    std_array.append(std)

csv_handler.apply_z_scale(training_set, mean_array, std_array)
csv_handler.apply_z_scale(test_set, mean_array, std_array)

k = 6
# get k centroids method
#      - shuffle the set
#      - pick top K entries...these are our first centroids
np.random.shuffle(training_set)

# make K number of lists to hold clusters
#   - centroids always at front of set list
clusters_list = [[] for x in range(0, k)]

for i in range(0, len(clusters_list)):
    clusters_list[i].append(training_set.pop(0))

# Step through entire training dataset
#   - assign each entry to the cluster who's centroid it is closest too
for i in range(0, len(training_set)):
    data_point = training_set.pop(0)
    nearest_centroid_index = csv_handler.get_nearest_centroid_index(clusters_list, data_point)
    clusters_list[nearest_centroid_index].append(data_point)

# Recalculate the centroid for each cluster by taking the average of the cluster
#   - Step through each cluster and reassign each entry to the the cluster who's new centroid it is closest too
element_switched = True
while element_switched:
    element_switched = False
    for i in range(0, len(clusters_list)):
        index_of_new_centroid = csv_handler.get_index_of_new_centroid(clusters_list[i])
        clusters_list[i].insert(0, clusters_list[i].pop(index_of_new_centroid))

    for i in range(0, len(clusters_list)):
        original_size = len(clusters_list[i])

        for j in range(1, original_size):
            data_point = clusters_list[i].pop(1)
            nearest_centroid_index = csv_handler.get_nearest_centroid_index(clusters_list, data_point)

            if nearest_centroid_index != i:
                element_switched = True
            clusters_list[nearest_centroid_index].append(data_point)

# Repeat the step above until no entry changes cluster


#change test set into a matrix
test_set = np.array([[float(r) for r in row] for row in test_set])

#iterate over clusters to output WCSS & perform linear regression
# TODO DOn't need to print them out anymore
for cluster in clusters_list:
    # Calculate and print WCSS and Centroid
    print(csv_handler.get_wcss(cluster))
    print(cluster[0])

#

# This is where we need to iterate over our test data and predict its last column
# maybe pre calculate Beta's of each cluster before we iterate and hold them in a correspoding
# array to the cluster list.
