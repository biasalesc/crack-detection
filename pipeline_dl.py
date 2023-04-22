#%% Imports
import numpy as np
import pandas as pd
import os
# from keras.preprocessing import image
import keras.utils as image
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

from func import mean_performance, DL_model

#%% DL models pipeline

n_images = ['25', '50', '100', '200', '1000', '4000'] # number of images
n_splits = 10 # number of train_test_splits
models = ['CNN1', 'CNN2', 'CNN3', 'CNN4']


results = []
for n in n_images:

    train_crack = [] 
    train_not_crack = [] 

    # loading no crack images
    path = "images/Negative_"+n
    for i in os.listdir(path):
        img_path = path + "/" + i
        img = image.load_img(img_path, target_size=(227, 227)) 
        x = image.img_to_array(img)
        x = np.dot(x[...,:3], [0.2989, 0.5870, 0.1140]) # change from RGB to grayscale
        x = x/255 # normalizing
        train_not_crack.append(x)

    train_not_crack = np.array(train_not_crack)
    print('No crack shape: ', train_not_crack.shape)

    # loading crack images
    path = "images/Positive_"+n
    for i in os.listdir(path):
        img_path = path + "/" + i
        img = image.load_img(img_path, target_size=(227, 227)) 
        x = image.img_to_array(img)
        x = np.dot(x[...,:3], [0.2989, 0.5870, 0.1140]) # change from RGB to grayscale
        x = x/255 # normalizing
        train_crack.append(x)

    train_crack = np.array(train_crack)
    print('Crack shape: ', train_crack.shape)

    # Concatenate crack and no crack arrays
    x = np.concatenate((train_crack, train_not_crack), axis = 0)
    number = len(train_not_crack) + len(train_crack)
    y = np.zeros((number)) 

    # Adding label and shuffle
    for i in range(len(train_crack)):
        y[i] = 1
    x, y = shuffle(x, y, random_state=42)      
    x = np.reshape(x, [-1,227,227,1])
    print(np.shape(x))
    print(np.shape(y))
    y = np_utils.to_categorical(y, 2)

    for key in models:

        print('Running {}'.format(key))
        m = DL_model(key=key)
        callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        m.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

        for j in range(n_splits):
            print("Split {}".format(j+1))
            mean_ba = []
            x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=j)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

            m.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, callbacks=[callback], verbose=1)
            
            y_pred = m.predict(x_test)
            test_acc = balanced_accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
            report = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
            print("Test balanced accuracy: %.4f" % test_acc)
            results.append([n, key, test_acc, report[0][0], report[0][1], report[1][0], report[1][1]])

data = results
# Save all results to csv
df = pd.DataFrame(data, columns=['Number of Images', 'Model', 'Balanced Accuracy', 'cm_00', 'cm_01', 'cm_10', 'cm_11'])
df.to_csv('output/Results_DL'+'.csv')

# Save mean results across models and data size to csv
data_mean = mean_performance(data)
df_mean = pd.DataFrame(data_mean, columns=['Number of Images', 'Model', 'Balanced Accuracy', 'cm_00', 'cm_01', 'cm_10', 'cm_11'])
df_mean = df_mean.sort_values(by=['Number of Images'])
df_mean.to_csv('output/Results_DL_mean'+'.csv')
# %%
