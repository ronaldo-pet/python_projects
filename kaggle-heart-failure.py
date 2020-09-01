import pandas as pd

pd.set_option('max_columns', None)
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv'
df=pd.read_csv(url, header=0)
df
df.columns = df.columns.str.lower()
################################  2 X's features to try separately  ##########
# X = df[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
#        'ejection_fraction', 'high_blood_pressure', 'platelets',
#        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']]

y = df['death_event']

X = df[['creatinine_phosphokinase','ejection_fraction','platelets', 
        'serum_creatinine','serum_sodium','age','sex']]

##############################################################################
##################  split data to train/test. ################################

from sklearn.model_selection import train_test_split

ts = .15
rs = 42
X_train,X_test, y_train,y_test = train_test_split(
                                X, y, test_size=ts, random_state=rs)
print('Training samples:', X_train.shape[0]) 
print('Testing samples:', X_test.shape[0])

##############################################################################
##############################################################################

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,\
    ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.linear_model import Perceptron,LogisticRegression,\
    SGDClassifier,RidgeClassifierCV

from sklearn.metrics import accuracy_score

##########################################

dtc = DecisionTreeClassifier(random_state=rs).fit(X_train, y_train)
rfc = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
bag = BaggingClassifier().fit(X_train, y_train)
ext = ExtraTreesClassifier(n_estimators=300).fit(X_train, y_train)
ada = AdaBoostClassifier(n_estimators=100, random_state=rs).fit(X_train, y_train)
knn_clf = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2)
knn_clf.fit(X_train, y_train)
gaussian_nb = GaussianNB().fit(X_train, y_train)
bernoulli_nb = BernoulliNB().fit(X_train, y_train)
mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(2, 1),max_iter=1000)
mlp.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, stratify=y,random_state=rs)
mlp_clf = MLPClassifier(random_state=rs, max_iter=200).fit(X_train, y_train)
lr_clf = LogisticRegression(random_state=rs).fit(X_train, y_train)
svm_clf = SVC(gamma=.1, kernel='rbf', probability=True).fit(X_train, y_train)
nsvc = NuSVC().fit(X_train, y_train)
rdg_clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train, y_train)
gb = GradientBoostingClassifier(random_state=rs).fit(X_train, y_train)
sgd_clf = make_pipeline(StandardScaler(),
                    SGDClassifier(max_iter=1000, tol=1e-3)).fit(X_train,y_train)
lsvc = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=rs, tol=1e-5)).fit(X_train, y_train)

##############################################################################

dtc = DecisionTreeClassifier(random_state=rs).fit(X_train, y_train)
rfc = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
bag = BaggingClassifier().fit(X_train, y_train)
ext = ExtraTreesClassifier(n_estimators=300).fit(X_train, y_train)
ada = AdaBoostClassifier(n_estimators=100, random_state=rs).fit(X_test, y_test)
knn_clf = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
knn_clf.fit(X_train, y_train)
gaussian_nb = GaussianNB().fit(X_train, y_train)
bernoulli_nb = BernoulliNB().fit(X_train, y_train)
mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(2, 1))
mlp.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, stratify=y,random_state=rs)
mlp_clf = MLPClassifier(random_state=rs, max_iter=500).fit(X_train, y_train)
lr_clf = LogisticRegression(random_state=rs).fit(X_train, y_train)
svm_clf = SVC(gamma=.1,kernel='rbf',probability=True,max_iter=2000).fit(X_train, y_train)
nsvc = NuSVC().fit(X_train, y_train)
rdg_clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1])
rdg_clf.fit(X_train, y_train)
gb = GradientBoostingClassifier(random_state=rs).fit(X_train, y_train)
sgd_clf = make_pipeline(StandardScaler(),
                    SGDClassifier(max_iter=1000, tol=1e-3))
sgd_clf.fit(X_train,y_train)
lsvc = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=rs, tol=1e-5)).fit(X_train, y_train)

##############################################################################
DTC = accuracy_score(y_test, dtc.predict(X_test))
RFC = accuracy_score(y_test, rfc.predict(X_test))
BAG = accuracy_score(y_test, bag.predict(X_test))
EXT = accuracy_score(y_test, ext.predict(X_test))
ADA = accuracy_score(y_test, ada.predict(X_test))
KNN = accuracy_score(y_test, knn_clf.predict(X_test))
NBG = accuracy_score(y_test, gaussian_nb.predict(X_test))
NBB = accuracy_score(y_test, bernoulli_nb.predict(X_test))
MLP = accuracy_score(y_train, mlp.predict(X_train))
MLPs = mlp_clf.score(X_train, y_train)
LOG = lr_clf.score(X_train,y_train)
SVM = accuracy_score(y_test, svm_clf.predict(X_test))
NSVC = nsvc.score(X_train, y_train)
RDG = accuracy_score(y_test, rdg_clf.predict(X_test))
GB = accuracy_score(y_test, gb.predict(X_test))
SGD = sgd_clf.score(X_train,y_train)
LSVC = accuracy_score(y_test, lsvc.predict(X_test))
###################################################
dt = dtc.score(X_test, y_test)
rf = rfc.score(X_test, y_test)
bg = bag.score(X_test, y_test)
et = ext.score(X_test, y_test)
ad = ada.score(X_test, y_test)
kn = knn_clf.score(X_test, y_test)
ng = gaussian_nb.score(X_test, y_test)
nb = bernoulli_nb.score(X_test, y_test)
m1 = mlp.score(X_test, y_test)
m2 = mlp_clf.score(X_test, y_test)
lg = lr_clf.score(X_test,y_test)
svm = svm_clf.score(X_test, y_test)
nsvc = nsvc.score(X_test, y_test)
rd = rdg_clf.score(X_test, y_test)
gb = gb.score(X_test, y_test)
sg = sgd_clf.score(X_test, y_test)
ls= lsvc.score(X_test, y_test)

##########################  view in DataFrame  ###################################
acc_score = pd.DataFrame({'model':['DTC','RFC','BAG','EXT','ADA','KNN','NBG','NBB','MLP',
                                   'MLPs','LOG','SVM','NSVC','RDG','GB','SGD','LSVC'], 
                        'train-set-score':[DTC,RFC,BAG,EXT,ADA,KNN,NBG,NBB,MLP,MLPs,LOG,
                                           SVM,NSVC,RDG,GB,SGD,LSVC],
                        'test-set-score':[dt,rf,bg,et,ad,kn,ng,nb,m1,m2,
                                          lg,svm,nsvc,rd,gb,sg,ls]})
round(acc_score, 3)

#######################################################
#############  confusion matrix plot  #################
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
title = "Model's Confusion Matrix"
model = bag
disp = plot_confusion_matrix(model, X_test, y_test,
                            display_labels=('no_HF','with_HF'),
                            cmap=plt.cm.Purples,
                            normalize=None)
disp.ax_.set_title(title)
print(title)
print(disp.confusion_matrix)
plt.show()
############  to DataFrame  ###########################
pd.set_option('max_rows',None)
values = pd.DataFrame()
values['actual'] = y_test
values['predicted'] = model.predict(X_test)
values

##############################################################################
########################  Model Evaluation F1_Score  #########################
'''
F1 score is a measure of a test’s accuracy. It considers both the precision and 
the recall of the test to compute the score. The F1 score can be interpreted as a 
weighted average of the precision and recall, where an F1 score reaches its best 
value at 1 and worst at 0
'''

from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, \
        RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

models = [
    DecisionTreeClassifier(random_state=rs, max_depth=15),
    SVC(gamma='auto'), NuSVC(gamma='auto'), LinearSVC(),
    SGDClassifier(max_iter=100, tol=1e-3), KNeighborsClassifier(),
    LogisticRegression(solver='lbfgs'), LogisticRegressionCV(cv=3),
    BernoulliNB(),
    BaggingClassifier(), ExtraTreesClassifier(n_estimators=200),
    RandomForestClassifier(n_estimators=200), AdaBoostClassifier(),
    GradientBoostingClassifier(random_state=rs),
    MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(2, 1), max_iter=2000)]

def score_model(X, y, estimator, **kwargs):
    """
    Test various estimators.
    """
    y = LabelEncoder().fit_transform(y)
    model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('estimator', estimator),
                    ])

    # Instantiate the model and visualizer
    model.fit(X, y, **kwargs)

    expected  = y
    predicted = model.predict(X)

    # Compute and return F1 (harmonic mean of precision and recall)
    print("{}: {}".format(estimator.__class__.__name__, f1_score(expected, predicted)))

print('''The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst at 0''')
# Training Set
print("=====================")
print("Training Set F1-Score")
print("=====================")
for model in models:
    score_model(X_train, y_train, model)
    
# Testing Set
print("====================")
print("Testing Set F1-Score")
print("====================")
for model in models:   
    score_model(X_test, y_test, model)


##############################################################################
###########################  confusion_matrix plot  ##########################
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
############ models
dtc
rfc
bag
ext
ada
knn_clf
gaussian_nb
bernoulli_nb
mlp
mlp_clf
lr_clf
sgd_clf
lsvc
svm_clf
nsvc
rdg_clf
############
title = "Decision Tree: Preliminary Confusion Matrix"
model = dtc
disp = plot_confusion_matrix(model, X_test, y_test,
                            display_labels=('no_HF','with_HF'),
                            cmap=plt.cm.Purples,
                            normalize=None)
disp.ax_.set_title(title)
print(title)
print(disp.confusion_matrix)
plt.show()

##############################################################################
############### further steps if using Decision Tree Classifier  #############

from sklearn.tree import plot_tree

# (training set)
plt.figure(figsize=(15, 7.5))
plot_tree(dtc,filled = True,
            rounded = True,
            class_names = ['No heart failure','With heart failure'],
            feature_names = X_train.columns)

##############################################################################
# Cost Complexity Pruning - CCP - to avoid overfitting

dtc.cost_complexity_pruning_path(X_train, y_train)

# remove the impurities
pruned = dtc.cost_complexity_pruning_path(X_train, y_train)  # values of alpha
ccp_alphas = pruned.ccp_alphas  # extract different values for alpha
ccp_alphas = ccp_alphas[:-1]    # excluding max value of alpha (last value)
ccp_alphas
'''
create one decision tree for each value of alpha and store it in the array dtcs
'''
dtcs = [] 
for ccp_alpha in ccp_alphas:
    dtc = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    dtc.fit(X_train, y_train)
    dtcs.append(dtc)
pruned.ccp_alphas

pruned.ccp_alphas[-1]  # excluding max value of ccp_alpha (last value)

dtc

############################## check the score accuracy  #####################
train_scores = [dtc.score(X_train, y_train) for dtc in dtcs]  # training_
test_scores = [dtc.score(X_test, y_test) for dtc in dtcs]     # testing_

fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title('Accuracy vs alpha for Training & Testing sets')
ax.plot(ccp_alphas, train_scores, marker='o', label='train', drawstyle='steps-post')
ax.plot(ccp_alphas, test_scores, marker='o', label='test', drawstyle='steps-post')
ax.legend()
plt.show()

from sklearn.model_selection import cross_val_score
# Cross Validation using ccp_alpha = 0.1 (eye balled value)
dtc = DecisionTreeClassifier(random_state=rs, ccp_alpha=0.1) 
# use 5-fold cv cross validation
scores = cross_val_score(dtc, X_train, y_train, cv=5)
dframe = pd.DataFrame(data={'tree': range(5), 'accuracy': scores})
dframe.plot(x='tree', y='accuracy', marker='o', linestyle='--')

'''
for each alpha value, run 5-fold cv. Then store the mean and standard deviation 
of the scores(accuracy) for each call to cross_val_score in alpha_loop_values.
'''
import numpy
alpha_loop_values = []
for ccp_alpha in ccp_alphas:
    dtc = DecisionTreeClassifier(random_state=rs, ccp_alpha=ccp_alpha)
    scores = cross_val_score(dtc, X_train, y_train, cv=5)
    alpha_loop_values.append([ccp_alpha, numpy.mean(scores), numpy.std(scores)])

# graph of the means & std of the scores
alpha_results = pd.DataFrame(alpha_loop_values, 
                            columns=['alpha','mean_accuracy','std'])
alpha_results

alpha_results.plot(x='alpha', 
                    y='mean_accuracy', yerr='std', marker='o', linestyle='--')

'''
using cross validation, set the ccp_alpha closer to 0.012 value
then store the best alpha value to build the best decision tree 
'''
best_ccp_alpha = alpha_results[(alpha_results['alpha']>0.01) & (alpha_results['alpha']< 0.09)] 
best_ccp_alpha

# Build, Evaluate, Draw & Interpret the Final Classification Tree using the ideal alpha
alpha = 0.016649
dtc_pruned = DecisionTreeClassifier(random_state=rs, ccp_alpha=alpha)
dtc_pruned = dtc_pruned.fit(X_train, y_train)

plot_confusion_matrix(dtc_pruned, X_test, y_test, display_labels=['No_HF', 'with_HF'])

# draw the pruned decision tree
plt.figure(figsize=(15,7.5))
plot_tree(dtc_pruned, 
            filled=True, 
            rounded=True, 
            class_names=['No_HF', 'with_HF'], 
            feature_names=X_train.columns)

##############################################################################
