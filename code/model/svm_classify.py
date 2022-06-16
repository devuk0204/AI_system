from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

subway = pd.read_csv('/home/devuk/code/AI_system/data/feature_target.csv')

accuracy_set = []

t_feature = subway.iloc[:, :29].values.tolist()
t_target = subway.iloc[:, 29].values.tolist()

train_feature, test_feature, train_target, test_target = train_test_split(t_feature, t_target, 
                                                                          random_state = 40, shuffle = True, train_size = 0.7)
mean = np.mean(train_feature)
std = np.std(train_feature)
train_scaled = (train_feature - mean) / std
test_scaled = (test_feature - mean) / std

learning_rate = [0.01, 0.1, 1]
C_params = [1, 10, 100 ,1000]
params = [['gamma', 'C', 'train_set_accuracy', 'test_set_accuracy']]

for i in learning_rate :
    for j in C_params :
        s = svm.SVC(kernel = 'rbf', gamma = i, C = j)
        s.fit(train_scaled, train_target)
        train_accuray = s.score(train_scaled, train_target)
        test_accuracy = s.score(test_scaled, test_target)
        params.append([i, j, train_accuray, test_accuracy])
        print(i, j)

for i in range(13) :
    print('gamma : ' + str(params[i][0]) + '\t' + 'C : ' + str(params[i][1]) + '\n'
          + 'train_accuracy : ' + str(params[i][2]) + '\t' + 'test_accuracy : ' + str(params[i][3]))
        
        

"""
for j in learning_rate :
    #s = svm.SVC(kernel = 'rbf', gamma = j)
    s = svm.SVC(kernel = 'linear', gamma = j)
        
    s.fit(train_scaled, train_target)
    res = s.predict(test_scaled)

    conf = np.zeros((10, 10))
    for j in range(len(res)) :
        conf[res[j]][test_target[j]] += 1
    print(conf)

    correct = 0
    for k in range(10) :
        correct += conf[k][k]
    accuracy = correct/len(res)
    print("Accuracy is ", accuracy * 100, "%. \n")
    accuracy_set.append(accuracy * 100)

print(accuracy_set, '\n')

accuracy_temp = 0
for i in range(len(accuracy_set)):
    accuracy_temp = accuracy_temp + accuracy_set[i]
accuracy_avg = accuracy_temp/len(accuracy_set)
print("Accuracy average is : ", accuracy_avg, "%")
"""

