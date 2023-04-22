#%% Imports
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from func import AssignData, run_models, tstamp, mean_performance

#%% Classification models and parameters for grid search
svc = SVC()
mlp = MLPClassifier(max_iter=100000)
adaboost = AdaBoostClassifier()
rf = RandomForestClassifier()
knn = KNeighborsClassifier()

models = {
    'SVC': Pipeline(steps=[('svc', svc)]),
    
    'MLP': Pipeline(steps=[('mlp', mlp)]),
    
    'AdaBoost': Pipeline(steps=[('adaboost', adaboost)]),
    
    'Random Forest': Pipeline(steps=[('rf', rf)]),

    'KNN' : Pipeline(steps=[('knn', knn)])
    
}


params = {
    'SVC': [
         {'svc__kernel':['linear'], 'svc__C':[1,10,100,1000]},
         {'svc__kernel':['rbf'], 'svc__C':[1,10,100,1000], 
          'svc__gamma':[1,0.1,0.001,0.0001]}, ],

    'MLP': {'mlp__solver': ['sgd','adam'],
                   'mlp__hidden_layer_sizes': [(100,),(100,100), (100,50), (100,100,100)],
                   'mlp__alpha': 10.0 ** -np.arange(1, 5)},   
                   
    'AdaBoost': {'adaboost__n_estimators': [50, 100, 200]},
    
    'Random Forest': {'rf__n_estimators': [50, 100, 200]},

    'KNN': {'knn__n_neighbors': [3,5,11],
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan']}

}

file = ["25_AllFeaturesRand.xlsx", 
        "50_AllFeaturesRand.xlsx", 
        "100_AllFeaturesRand.xlsx", 
        "200_AllFeaturesRand.xlsx",
        "1000_AllFeaturesRand.xlsx",
        "4000_AllFeaturesRand.xlsx"] # feature files for different dataset sizes

n = ['25', '50', '100', '200', '1000', '4000'] # number of images
n_splits = 10 # number of train_test_splits
target_names = ['no crack', 'crack']

# Run
data = []
data_mean = []

for i in range(len(file)): # for each data file
  
  x, y = AssignData('features/'+file[i])
  print('-------------------------------------------------------------------')
  print('Executing file {} with {} images'.format(file[i], n[i]))
  classes, samples_classes = np.unique(y, return_counts = True)
  n_classes = len(classes)
  print('Classes: ', classes)
  print('Samples: ', samples_classes)
  print('\n')

  results, results_test = run_models(x, y, n_splits=n_splits, models=models, params=params, standardization=True, Test=False, n_images=n[i])
  data = data + results

# Save all results to csv
df = pd.DataFrame(data, columns=['Number of Images', 'Model', 'Balanced Accuracy', 'cm_00', 'cm_01', 'cm_10', 'cm_11', 'Training Time', 'Prediction Time'])
df.to_csv('output/Results_ML'+'.csv')

# Save mean results across models and data size to csv
data_mean = mean_performance(data)
df_mean = pd.DataFrame(data_mean, columns=['Number of Images', 'Model', 'Balanced Accuracy', 'cm_00', 'cm_01', 'cm_10', 'cm_11', 'Training Time', 'Prediction Time'])
df_mean = df_mean.sort_values(by=['Number of Images'])
df_mean.to_csv('output/Results_ML_mean'+'.csv')
# %%
