import csv
import numpy as np
import random as rng
import csv_handler

'''
RMSEs
regression-0.05.csv:
0.0514199089124
regression-A.csv:
0.102839817825
regression-B.csv:
0.308519453474
regression-C.csv:
0.154259726737
abalone.data:
2.28
'''
# load our data
#with open('regression-C.csv', 'rb') as csvfile:
#reader = csv.reader(csvfile, delimiter=',')
#data = np.array([[float(r) for r in row] for row in reader])


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

# TODO EXCLUDING LAST COLUMN -- -UNSURE
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

k = 4
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
        clusters_list[i].insert(0, clusters_list.pop(index_of_new_centroid))

    for i in range(0, len(clusters_list)):
        original_size = len(clusters_list[i])

        for j in range(1, original_size):
            data_point = clusters_list[i].pop(1)
            nearest_centroid_index = csv_handler.get_nearest_centroid_index(clusters_list, data_point)

            if nearest_centroid_index != i:
                element_switched = True
            clusters_list[nearest_centroid_index].append(data_point)

# Repeat the step above until no entry changes cluster


# Calculate and print WCSS and Centroid

#
'''
data = np.array([[float(r) for r in row] for row in whole_data])

np.random.seed(99)
np.random.shuffle(data)
train_num = int(data.shape[0] * 0.9)

X_train = data[:train_num, :-1]
Y_train = data[:train_num, -1]
data_train = data[:train_num, :]
X_test = data[train_num:, :-1]
Y_test = data[train_num:, -1]
# linear least square Y = X beta
Q, R = p3_helper.qr_decompose(data_train)
Y_primes = R[:, -1]
R = R[:, :-1]
beta = p3_helper.back_solve(R, Y_primes)#np.dot(Q.T, np.mat(Y_train).T))
print beta
y_final = np.dot(X_test, beta)
diff = np.subtract(y_final, np.mat(Y_test).T)
y_array = np.squeeze(np.asarray((diff)))
y_array = np.square(y_array)
mean = np.mean(y_array)
print np.sqrt(mean)
'''