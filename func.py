# Imports
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import time
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from matplotlib import pyplot as plt
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense
from keras.layers import Conv2D

# Important functions

## Read and organize data
def AssignData(file, print_columns=True):
  '''
  Function to read data from csv file and assign it to variables

  Inputs:
  - file (filepath): .xlsx features file
  - print_columns (bool): True to print file columns

  Return:
  - x: features
  - y: label
  '''
  data = pd.read_excel(file)
  y = np.array(data['Label'])
  x = data.drop(columns = ['Label', 'Area',
       'MajorAxisLength', 'CrackLength', 'Name'])
  if print_columns:
    print('File columns:', x.columns)
  x = np.array(x)

  return x, y

def tstamp(extension=''):
    '''returns a timestamp based on current date and time'''
    d = datetime.datetime.now().strftime('%Y%m%d%H%M%S') + extension
    return d

## Metrics
def mean_performance(data, target_names=(['no crack', 'crack'])):
  '''
  Function to calculate the mean performance of different models for each dataset size. 
  Metrics: balanced accuracy, confusion matrix, training time, prediction time.
  '''

  if np.shape(data)[1] == 7:
    df = pd.DataFrame(data, columns=['Number of Images', 'Model', 'Balanced Accuracy', 'cm_00', 'cm_01', 'cm_10', 'cm_11'])
  elif np.shape(data)[1] == 9:
    df = pd.DataFrame(data, columns=['Number of Images', 'Model', 'Balanced Accuracy', 'cm_00', 'cm_01', 'cm_10', 'cm_11', 'Training time', 'Prediction time'])
  
  df = df.groupby(['Number of Images'])
  results = []
  #for name, group1 in mean_ba:
  for name, group1 in df:
    a = group1.groupby(['Model']).mean()
    for i in range(len(a.index)):
      metrics = [name, a.iloc[i].name, a.iloc[i]['Balanced Accuracy'], a.iloc[i]['cm_00'], a.iloc[i]['cm_01'], a.iloc[i]['cm_10'], a.iloc[i]['cm_11']]
      if np.shape(data)[1] == 9:
         metrics.append(a.iloc[i]['Training time'])
         metrics.append(a.iloc[i]['Prediction time'])
      results.append(metrics)

  return results

## Plots
def getp(x, mult=1):
    '''returns percentage confusion matrix'''
    y = []
    for i in range(len(x)):
        temp = []
        for j in range(len(x[i])):
            temp.append(mult*x[i][j]/np.sum(x[i]))
        y.append(temp)
    return y

def plot_confusion_matrix(conf_mat, figsize=(10,7), filename=None, title=False, 
                          label=True, perc=True, classes=(['0', '1', '2'])):
    '''plots confusion matrix'''
    conf_mat_p = getp(conf_mat)
    if perc:
        cmat = getp(conf_mat, mult=100)
    else:
        cmat = conf_mat

    plt.matshow(conf_mat_p, cmap="Blues", vmin=0, vmax=1)
    tick_marks = np.arange(len(classes))
    plt.yticks(tick_marks, '', fontsize=14)
    if title:
        plt.title('Confusion matrix', y=1.08, fontsize=16)
    if label:
        plt.ylabel('True label', fontsize=18)
        plt.yticks(tick_marks, classes, fontsize=14)
    plt.xlabel('Predicted', fontsize=18)
    plt.xticks(tick_marks, classes, fontsize=14)
    

    for (i,j), z in np.ndenumerate(cmat):
        if conf_mat_p[i][j] > .7:
            if perc:
                plt.text(j, i, '{:0.2f}%'.format(z), ha='center', va='center', fontsize=18, color='white')
            else:
                plt.text(j, i, '{:0.0f}'.format(z), ha='center', va='center', fontsize=22, color='white')
        else:
            if perc:
                plt.text(j, i, '{:0.2f}%'.format(z), ha='center', va='center', fontsize=18, color='black')
            else:
                plt.text(j, i, '{:0.0f}'.format(z), ha='center', va='center', fontsize=22, color='black')
    if filename:
        full_filename = os.path.join('img/',filename)
        plt.savefig(full_filename)
    else:
        plt.show()

## Pipelines
def run_models(X, y, models, params, x_test=None, y_test=None, n_splits=1, test_size=0.20, 
               cv = 4, scoring='balanced_accuracy', target_names=(['no crack', 'crack']), 
               standardization=False, GS=True, Test=True, n_images=None):
  '''
  Run train_test_split + grid search + prediction + test on a set of ML models.

  Inputs:
  - X: np.array of features
  - y: np.array of labels
  - models: dictionary of models
  - params: dictionary of models parameters and values for grid search
  - x_test: additional features data for test, only if Test=True
  - y_test:additional label data for test, only if Test=True
  - n_splits: number of train_test splits
  - test_size: percentage of data that should be used for test
  - CV: cross validation splits
  - scoring: metric to evaluate model quality
  - target_names: labels
  - standardization: if standardScaler should be applied to data
  - GS: if grid search
  - Test: if aditional data for test is provided
  - n_images: number of images provided

  '''
  
  results = []
  results_test = []

  for _ in range(n_splits):
    print("Split {}".format(_+1))
    # split data
    # stratify=y
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=test_size, random_state=_)
    classes, samples_classes = np.unique(trainy, return_counts = True)

    if standardization == True:
      # data sandardization
      scaler = StandardScaler().fit(trainX)
      trainX = scaler.transform(trainX)
      testX = scaler.transform(testX)

    for key in models.keys():

      #Grid Search
      if GS == True:
        print("Running GridSearchCV for %s." % key)
        model = models[key]
        param = params[key]
        gs = GridSearchCV(model, param, cv=cv,
                              verbose=1, scoring=scoring,
                              return_train_score=True)
        gs.fit(trainX,trainy)
        # Mean cross-validated score of the best_estimator
        print("Best parameter (CV score=%0.3f):" % gs.best_score_)
        print(gs.best_params_)

        best_par = {}
        for par, val in gs.best_params_.items():
            best_par[par] = [val]
        
        # Calculate training time
        t = time.process_time()

        refit = GridSearchCV(model, best_par, cv=cv,
                              verbose=1, scoring=scoring,
                              return_train_score=True)
        refit.fit(trainX,trainy)

        elapsed_time = time.process_time() - t
         
      else:
        gs = models
        gs.fit(trainX, trainy)

      # evaluate model
      t = time.process_time()
      y_pred = gs.predict(testX)
      prediction_time = time.process_time() - t
      fittedModel = gs
      test_acc = balanced_accuracy_score(testy,y_pred)
      report = confusion_matrix(testy, y_pred, labels=[0,1])
      print("Test balanced accuracy: %.4f" % test_acc)
      results.append([n_images, key, test_acc, report[0][0], report[0][1], report[1][0], report[1][1], elapsed_time, prediction_time])

      # Test
      if Test == True:
        if standardization == True:
          #data sandardization
          scaler = StandardScaler().fit(x_test)
          x_test = scaler.transform(x_test)
        y_pred = gs.predict(x_test)
        test_acc = balanced_accuracy_score(y_test,y_pred)
        report = confusion_matrix(y_test, y_pred)
        results_test.append([n_images, key, test_acc, report[0][0], report[0][1], report[1][0], report[1][1]])
      print('\n')

  return results, results_test

def DL_model(key='CNN1'):

    m = None

    if key == 'CNN1':
        m = Sequential()
        m.add(Conv2D(64, (3, 3), padding='same',
                        input_shape =(227, 227, 1)))
        m.add(Activation('relu'))
        m.add(Conv2D(64, (3, 3)))
        m.add(Activation('relu'))
        m.add(MaxPooling2D(pool_size=(2, 2)))

        m.add(Activation('relu'))
        m.add(Conv2D(64, (3, 3)))
        m.add(Activation('relu'))
        m.add(MaxPooling2D(pool_size=(2, 2)))

        m.add(Conv2D(64, (3, 3), padding='same'))
        m.add(Activation('relu'))
        m.add(Conv2D(64, (3, 3)))
        m.add(Activation('relu'))

        m.add(Flatten())
        m.add(Dense(512))
        m.add(Activation('relu'))
        m.add(Dense(512))
        m.add(Activation('relu'))
        m.add(Dense(2))
        m.add(Activation('softmax'))

    if key == 'CNN2':
        m = Sequential()
        m.add(Conv2D(32, (3, 3), padding='same',
                        input_shape =(227, 227, 1)))
        m.add(Activation('relu'))
        m.add(Conv2D(32, (3, 3)))
        m.add(Activation('relu'))
        m.add(MaxPooling2D(pool_size=(2, 2)))

        m.add(Activation('relu'))
        m.add(Conv2D(16, (3, 3)))
        m.add(Activation('relu'))
        m.add(MaxPooling2D(pool_size=(2, 2)))

        m.add(Conv2D(8, (3, 3), padding='same'))
        m.add(Activation('relu'))
        m.add(Conv2D(4, (3, 3)))
        m.add(Activation('relu'))

        m.add(Flatten())
        m.add(Dense(128))
        m.add(Activation('relu'))
        m.add(Dense(32))
        m.add(Activation('relu'))
        m.add(Dense(2))
        m.add(Activation('softmax'))

    if key == 'CNN3':
        m = Sequential()
        m.add(Conv2D(16, (3, 3), padding='same',
                        input_shape =(227, 227, 1)))

        m.add(MaxPooling2D(pool_size=(2, 2)))

        m.add(Conv2D(8, (3, 3), padding='same'))
        m.add(Activation('relu'))
        m.add(Conv2D(4, (3, 3)))
        m.add(Activation('relu'))

        m.add(Flatten())
        m.add(Dense(64))
        m.add(Activation('relu'))
        m.add(Dense(16))
        m.add(Activation('relu'))
        m.add(Dense(2))
        m.add(Activation('softmax'))

    if key == 'CNN4':
        m = Sequential()
        m.add(Conv2D(16, (3, 3), padding='same',
                        input_shape =(227, 227, 1)))

        m.add(MaxPooling2D(pool_size=(2, 2)))

        m.add(Conv2D(8, (3, 3), padding='same'))
        m.add(Activation('relu'))


        m.add(Flatten())
        m.add(Dense(128))
        m.add(Activation('relu'))
        m.add(Dense(2))
        m.add(Activation('softmax'))
    
    return m


