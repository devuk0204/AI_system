from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

k_param = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
train_accuracy_set = []
test_accuracy_set = []
f1_set = []

def f1_score() :
    #모델 평가
    pred = kn.predict(test_scaled)
    preds_1d = pred.flatten()
    pred_class = np.where(preds_1d > 0.5, 1, 0)
    target_names = ['not_transfer_station', 'transfer_station']
    f1_temp = classification_report(test_target, pred_class, target_names = target_names)
    return f1_temp

for i in k_param :
    kn = KNeighborsClassifier(n_neighbors = i)
    
    kn.fit(train_scaled, train_target)
    
    train_accuracy_set.append(round(float(kn.score(train_scaled, train_target)), 6))
    test_accuracy_set.append(round(float(kn.score(test_scaled, test_target)), 6))
    print(kn.score(train_scaled, train_target))

    pred = kn.predict(test_scaled)
    f1_set.append(f1_score())

for i in range(10) :
    print('n : ' + str((i+1)*5) + '\n' 
          + 'train_accuracy : ' + str(train_accuracy_set[i]) + '\t' + 'test_accuracy : ' + str(test_accuracy_set[i]) + '\n' 
          + str(f1_set[i]) + '\n')