# Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
 
df = pd.read_csv(url, header=None) 
 
# replacing column names based on: https://archive.ics.uci.edu/ml/datasets/heart+Disease
df.columns = ['age','sex','cp','trestbps','chol','fbs',
              'restecg','thalach','exang','oldpeak','slope','ca','thal','hd']
df
df.shape


df['hd'].value_counts()


# search for missing Data
 
df.dtypes

df[df['ca'] == '?']

df[df['thal'] == '?']

df['thal'].unique()

df['ca'].unique()

len(df.loc[(df['ca'] == '?') | (df['thal'] == '?')])

df.loc[(df['ca'] == '?') | (df['thal'] == '?')]


df_no_missing = df.loc[(df['ca'] != '?') & (df['thal'] != '?')]
df_no_missing

print("Based on the value counts,", round(((55+36+35+13) / (303-6)) * 100),"percent indicated to have some kind of heart disease.")


# Split dataframe into dependent(y) & independent(X) variables
 
X = df_no_missing.drop('hd', axis=1).copy() # dropping dependent variable(hd) and assigned to 'X'
X    #independent vars
y = df_no_missing['hd'].copy()  #independent vars
y.value_counts()


# check columns for proper data types, to change categorically later
X.dtypes

X['ca'].unique()


# test/check categorizing an attribute using pd.get_dummies()
 
pd.get_dummies(X, columns=['cp'])
X

X['restecg'].value_counts()
X

X['slope'].value_counts()
X

X['thal'].value_counts()

pd.set_option('max_columns', None)
 
    
# encoding (cp,restecg,slope,thal) categorically
 
X_encoded = pd.get_dummies( X, columns=['cp','restecg','slope','thal'] )
X_encoded

X_encoded.dtypes

#drop the ca having 'object' datatype
X_encoded['ca_'] = X_encoded[['ca']].astype(float)
X_encoded

X_encoded.drop('ca', axis=1, inplace=True)
 
X_encoded.dtypes


# convert heart disease(hd) category to 1 & 0 only
 
y.unique()

'''
diagnosis of heart disease (hd)
-- Value 0: < 50% diameter narrowing
-- Value 1: > 50% diameter narrowing
'''
# convert y to two values only
 
y_not_zero_index = y > 0 # get the index of non zero value in y
 
y[y_not_zero_index] = 1  # set y to 1
 
y.unique()               # verify if its only (1, 0)

##################################### Preliminary Classification Tree  #############################################
# split data into training & testing
 
X_train,X_test, y_train,y_test = train_test_split(X_encoded, y, random_state=42) # default split .7-train & .3-test
 
# create decision tree and fit to the training data
 
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(X_train, y_train)            # fit the model
clf_dt

# plot the dt - decision tree (training set)
 
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt,filled = True,
            rounded = True,
            class_names = ['No HD','Yes HD'],
            feature_names = X_encoded.columns)
 
# prelim Confusion Matrix
 
plot_confusion_matrix(clf_dt, X_test, y_test, display_labels=['Does Not have HD', 'Has HD'])
# result using default partition .70 & .30 test_size



############################################  Cost Complexity Pruning - CCP ########################################

clf_dt.cost_complexity_pruning_path(X_train, y_train)

# CCP - Cost Complexity Pruning: Visualize alpha
# pruning to solve overfitting
# omit the max value or alpha with ccp_alphas=ccp_alphas[:-1], so it would not prune everything, leaving us with only root instead of a tree.
 
path = clf_dt.cost_complexity_pruning_path(X_train, y_train)  # values of alpha
ccp_alphas = path.ccp_alphas  # extract different values for alpha
ccp_alphas = ccp_alphas[:-1]  # exclude max value of alpha
 
clf_dts = []  # create an array to put the decision tree into
 
## create one decision tree for each value of alpha and store it in the array clf_dts
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf_dt.fit(X_train, y_train)
    clf_dts.append(clf_dt)
    
path.ccp_alphas

#value of ccp_alpha
path.ccp_alphas[-1]  # max value of ccp_alpha

clf_dt


train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]  # training_rp
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]     # testing_rp
 
    
fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title('Accuracy vs alpha for Training & Testing sets')
ax.plot(ccp_alphas, train_scores, marker='o', label='train', drawstyle='steps-post')
ax.plot(ccp_alphas, test_scores, marker='o', label='test', drawstyle='steps-post')
ax.legend()
plt.show()


for clf_dt in clf_dts:
    training_rp = [clf_dt.score(X_train, y_train)]
    print(training_rp)

len(train_scores)

for clf_dt in clf_dts:
    testing_rp = clf_dt.score(X_test, y_test)
    print(testing_rp)

len(test_scores)

''' 
accuracy for the Testing Dataset hits its max value of alpha is about 0.016. 
after this value for alpha, accuracy of the Training Dataset drops off and that suggests 
to set the ccp_alpha = 0.016 or lesser
'''

## To know the best Training & Testing Dataset, we can use 10-Fold Cross Validation - cross_val_score()
# Cross Validation using ccp_alpha = 0.016 (eye balled value)
 
clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=0.016) 
 
    
# use 5-fold cv cross validation
 
scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
df = pd.DataFrame(data={'tree': range(5), 'accuracy': scores})
 
df.plot(x='tree', y='accuracy', marker='o', linestyle='--')
''' 
graph above shows that using different training & Testing Dataset with the same alpha, 
resulted in different accuracies, suggesting that alpha is sensitive to the datasets. 
Use Cross validation to find the optimal value for ccp_alpha 
'''

# for each alpha value, run 5-fold cv. Then store the mean and standard deviation of 
# the scores(accuracy) for each call to cross_val_score in alpha_loop_values.
 
alpha_loop_values = []
 
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])
 

# plot of the means & std of the scores
 
alpha_results = pd.DataFrame(alpha_loop_values, columns=['alpha','mean_accuracy','std'])
alpha_results.plot(x='alpha', y='mean_accuracy', yerr='std', marker='o', linestyle='--')


# using cross validation, we can see that overall, we need to set the ccp_alpha 
# closer to 0.014
# store the best alpha value to build the best decision tree 
 
ideal_ccp_alpha = alpha_results[ (alpha_results['alpha'] > 0.014) & (alpha_results['alpha'] < 0.015) ] 
ideal_ccp_alpha


# Build, Evaluate, Draw & Interpret the Final Classification Tree
 
clf_dt_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=0.014225)
clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)
plot_confusion_matrix(clf_dt_pruned, X_test, y_test, display_labels=['Does not have HD', 'Has HD'])

# try find the alpha value using Grid Search #
 
    
# draw the pruned decision tree
 
plt.figure(figsize=(15,7.5))
plot_tree(clf_dt_pruned, filled=True, rounded=True, class_names=['No HD', 'Yes HD'], feature_names=X_encoded.columns)

'''
The variable (column name) and the threshold for splitting the observations. 
For example, in the tree's root, we use CA to split the observations. 
All observations with CA <= 0.5 go to the left and all observations with CA > 0.5 go to the right. 
gini is the gini index or score for that node
'''

'''
1 age: age in years
2 sex: sex
(1 = male; 0 = female)
3 cp: chest pain type
  -- Value 1: typical angina
  -- Value 2: atypical angina
  -- Value 3: non-anginal pain
  -- Value 4: asymptomatic
4 trestbps: resting blood pressure (in mm Hg on admission to the hospital)
5 chol: serum cholestoral in mg/dl
6 fbs: (fasting blood sugar > 120 mg/dl)
  (1 = true; 0 = false)
7 restecg: resting electrocardiographic results
  -- Value 0: normal
  -- Value 1: having ST-T wave abnormality 
  -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
8 thalach: maximum heart rate achieved
9 exang: exercise induced angina
      (1 = yes; 0 = no)
10 oldpeak = ST depression induced by exercise relative to rest
11 slope: the slope of the peak exercise ST segment
  -- Value 1: upsloping
  -- Value 2: flat
  -- Value 3: downsloping
12 ca: number of major vessels
  (0-3) colored by flourosopy
13 thal: 3 = normal
    6 = fixed defect
    7 = reversable defect
14 hd: diagnosis of heart disease (angiographic disease status)
  -- Value 0: < 50% diameter narrowing
  -- Value 1: > 50% diameter narrowing
'''

clear
