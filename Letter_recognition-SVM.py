import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
#from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#import matplotlib.pyplot as plt

letter_data = pd.read_csv("Letter-recognition.csv")
letter_data.head(10)

#Understanding Dimensions
letter_data.shape

#Structure of the dataset
letter_data.info

#Exploring the data
letter_data.describe()

# check for NA values in dataset
letter_data.isnull().sum()  
letter_data.isnull().values.any()
letter_data.isnull().values.sum()

# Split the data into train and test set
X = letter_data.drop("letter", axis = 1).values
Y = letter_data.letter.values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 4, stratify = Y)

#Using Linear Kernel
model = SVC(C = 1, kernel  = 'linear')
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
model.score(X_test, Y_test)

#creating confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)


#Using RBF Kernel
model = SVC(C = 1, kernel  = 'rbf')
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
model.score(X_test, Y_test)


# hyperparameter tuning and n cross validation

# Performing 5-fold cross validation
folds = KFold(n_splits = 5, shuffle = True, random_state = 4)
model = SVC(kernel  = 'rbf')
cv_results = cross_val_score(model, X_train, Y_train, cv = folds, scoring = 'accuracy') 

print(cv_results)


# hyperparamater tuning: choosing optimum value of C and gamma
params = {"C": (0.025, 0.05), "gamma": (0.1,0.5,1,2)}
model = SVC(kernel  = 'rbf')
model_cv = GridSearchCV(estimator = model, param_grid = params, scoring= 'accuracy', cv = folds, verbose = 1)            
model_cv.fit(X_train, Y_train)                  
model_cv.best_score_
model_cv.best_params_

mean_accuracy = model_cv.cv_results_['mean_test_score']
#
#mean_accuracy.reshape(5,5)


list(model_cv.cv_results_['params'])
c_labels = sorted(list(set([item['C'] for item in model_cv.cv_results_['params']])))
gamma_labels = sorted(list(set([item['gamma'] for item in model_cv.cv_results_['params']])))


mycmap = sns.light_palette("red", reverse=False, as_cmap=True)
ax = sns.heatmap(data=pd.DataFrame(mean_accuracy.reshape(5,5), index=c_labels, columns=gamma_labels), annot = np.round(mean_accuracy.reshape(5,5), 5), cmap=mycmap)



