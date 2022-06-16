from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

subway = pd.read_csv('/home/devuk/code/AI_system/data/feature_target.csv')

feature = subway.iloc[:, :30].to_numpy()
target = subway.iloc[:, 30].to_numpy()


train_feature, test_feature, train_target, test_target = train_test_split(feature, target, 
                                                                          random_state = 53, shuffle = True, train_size = 0.7)

#standard score scailing
mean = np.mean(train_feature)
std = np.std(train_feature)
train_scaled = (train_feature - mean) / std
test_scaled = (test_feature - mean) / std

k_param = range(1, 11)
train_accuracy_set = []
test_accuracy_set = []
f1_set = []

for i in k_param :
    kn = KNeighborsClassifier(n_neighbors = i)
    
    kn.fit(train_scaled, train_target)
    
    train_accuracy_set.append(round(kn.score(train_scaled, train_target), 6))
    test_accuracy_set.append(round(kn.score(test_scaled, test_target), 6))

    pred = kn.predict(test_scaled)
    f1_set.append(round(f1_score(test_target, pred), 6))

for i in range(10) :
    print('n : ' + str(i+1) + '\n' 
          + 'train_accuracy : ' + str(train_accuracy_set[i]) + '\t' + 'test_accuracy : ' + str(test_accuracy_set[i]) + '\n' 
          + 'f1_score : ' + str(f1_set[i]) + '\n')