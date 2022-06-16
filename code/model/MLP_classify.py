import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt



subway = pd.read_csv('/home/devuk/code/AI_system/data/feature_target.csv')

feature = subway.iloc[:, :29].to_numpy()
target = subway.iloc[:, 29].to_numpy()


train_feature, test_feature, train_target, test_target = train_test_split(feature, target, shuffle = True, random_state = 66, stratify = target, train_size = 0.8)

#standard score scailing
mean = np.mean(train_feature)
std = np.std(train_feature)
train_scaled = (train_feature - mean) / std
test_scaled = (test_feature - mean) / std

mlp = Sequential()

activation_set = ['sigmoid', 'relu']
batch_size_set = [64, 128, 256]
learning_rate_set = [0.001, 0.01, 0.1]
result_set = []

early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0.0005, patience = 10, mode = 'min')

mlp = Sequential()

def plot(optimizer) :
    #정확률 곡선
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid()
    plt.show()
    #plt.savefig('/home/devuk/code/AI_system/result_image/' + str(i) + '_' + str(j) + '_' + str(optimizer) + '_' + str(k) + '_accuracy' + '.png')

    #손실함수 곡선
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid()
    plt.show()
    #plt.savefig('/home/devuk/code/AI_system/result_image/' + str(i) + '_' + str(j) + '_' + str(optimizer) + '_' + str(k) + '_loss' + '.png')
    
def f1_score() :
    #모델 평가
    pred = mlp.predict(test_scaled)
    preds_1d = pred.flatten()
    pred_class = np.where(preds_1d > 0.5, 1, 0)
    target_names = ['not_transfer_station', 'transfer_station']
    f1_temp = classification_report(test_target, pred_class, target_names = target_names)
    return f1_temp

mlp.add(Dense(29, activation = 'relu', input_shape = (29,), kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros'))
#mlp.add(Dense(50, activation = i, kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros'))
mlp.add(Dense(100, activation = 'relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros'))
#mlp.add(Dense(50, activation = i, kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros'))
mlp.add(Dense(1, activation = 'relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros'))
mlp.compile(optimizer = optimizers.Adam(learning_rate = 0.001), loss = 'mean_squared_error', metrics=['accuracy'])
hist = mlp.fit(train_scaled, train_target, batch_size = 128, epochs = 200, validation_data= (test_scaled, test_target), callbacks = [early_stopping])
plot('Adam')
res = mlp.evaluate(test_scaled, test_target, verbose = 0)
accuracy = res[1]
print(res)
f1_temp = f1_score()
print(f1_temp)
result_set.append([1, 1, 'Adam', 1,  accuracy, f1_temp])
"""
for i in activation_set :
    mlp.add(Dense(29, activation = i, input_shape = (29,), kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros'))
    #mlp.add(Dense(50, activation = i, kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros'))
    mlp.add(Dense(100, activation = i, kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros'))
    #mlp.add(Dense(50, activation = i, kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros'))
    mlp.add(Dense(1, activation = i, kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros'))
    
    for j in learning_rate_set :
        mlp.compile(optimizer = optimizers.Adam(learning_rate = j), loss = 'mean_squared_error', metrics=['accuracy'])
        
        for k in batch_size_set :
            hist = mlp.fit(train_scaled, train_target, batch_size = k, epochs = 200, validation_data= (test_scaled, test_target), callbacks = [early_stopping])
            plot('Adam')
            res = mlp.evaluate(test_scaled, test_target, verbose = 0)
            accuracy = res[1]
            print(res)
            f1_temp = f1_score()
            result_set.append([i, j, 'Adam', k,  accuracy, f1_temp])
            print(i, j, k)

df = pd.DataFrame(result_set, columns = ['activation', 'learning_rate', 'optimizer', 'batch_size', 'accuracy', 'f1_score'])
df.to_csv('/home/devuk/code/AI_system/data/result.csv', index = False)
"""