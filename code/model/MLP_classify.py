import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.callbacks import EarlyStopping



subway = pd.read_csv('/home/devuk/code/AI_system/data/feature_target.csv')

feature = subway.iloc[:, :29].to_numpy()
target = subway.iloc[:, 29].to_numpy()


train_feature, test_feature, train_target, test_target = train_test_split(feature, target, 
                                                                          random_state = 40, shuffle = True, train_size = 0.7)

#standard score scailing
mean = np.mean(train_feature)
std = np.std(train_feature)
train_scaled = (train_feature - mean) / std
test_scaled = (test_feature - mean) / std

mlp = Sequential()

mlp.add(Dense(29, activation = 'relu', input_shape = (29,),
              kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros'))
mlp.add(Dense(50, activation = 'relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros'))
mlp.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros'))

mlp.compile(optimizer = optimizers.Adam(lr=0.01), loss = 'mse', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, patience = 10, mode = 'min')
hist = mlp.fit(train_scaled, train_target, batch_size = 64, epochs = 200, validation_data= (test_scaled, test_target), callbacks = [early_stopping])


res = mlp.evaluate(test_scaled, test_target, verbose=0)
print(res)

#pred = mlp.predict(test_scaled, test_target)
#f1 = f1_score(test_target, pred)
#print(f1)


import matplotlib.pyplot as plt

#정확률 곡선
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid()
plt.show()

#손실함수 곡선
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.grid()
plt.show()
