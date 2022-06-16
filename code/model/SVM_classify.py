from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

subway = pd.read_csv('/home/devuk/code/AI_system/data/feature_target.csv')

accuracy_set = []

#feature, target extract
feature = subway.iloc[:, :30].values.tolist()
target = subway.iloc[:, 30].values.tolist()

train_feature, test_feature, train_target, test_target = train_test_split(feature, target, 
                                                                          random_state = 40, shuffle = True, train_size = 0.7)

#standard score scailing
mean = np.mean(train_feature)
std = np.std(train_feature)
train_scaled = (train_feature - mean) / std
test_scaled = (test_feature - mean) / std

learning_rate = [0.01, 0.1, 1]
C_params = [1, 10, 100 ,1000]
params = [['gamma', 'C', 'train_set_accuracy', 'test_set_accuracy', 'f1_score']]

for i in learning_rate :
    for j in C_params :
        s = svm.SVC(kernel = 'rbf', gamma = i, C = j)
        s.fit(train_scaled, train_target)
        train_accuray = round(s.score(train_scaled, train_target), 6)
        test_accuracy = round(s.score(test_scaled, test_target),6)
        pred = s.predict(test_scaled)
        f1 = round(f1_score(test_target, pred), 6)
        params.append([i, j, train_accuray, test_accuracy, f1])
        print(i, j)
        

for i in range(13) :
    print('gamma : ' + str(params[i][0]) + '\t' + 'C : ' + str(params[i][1]) + '\n'
          + 'train_accuracy : ' + str(params[i][2]) + '\t' + 'test_accuracy : ' + str(params[i][3]) + '\n'
          + 'f1_score : ' + str(params[i][4]) + '\n')