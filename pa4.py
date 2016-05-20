
import csv
import numpy as np
import random as rng
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
            # SEX INDEX ONLY FOR ABALONE, must comment out if doing miscategorization
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


# whole_data[i] = np.asarray(whole_data[i])

np.random.seed(99)
np.random.shuffle(whole_data)
train_num = len(whole_data)

print train_num


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