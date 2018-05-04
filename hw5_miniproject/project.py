import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.decomposition import PCA

from sklearn.metrics import classification_report,confusion_matrix
#Install XGBoost using Anaconda python version 3.x. Command: conda install -c conda-forge xgboost
import xgboost as xgb
from  sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


#loading data

print("Loading Data......")
print(" ")

df = pd.read_csv('./responses.csv')


#Dropping rows with NaN value

print("Dropping empty rows...")
print(" ")
df = df.dropna(axis=0, how='any')

#Feature Explosion of categorical attributes

print("Converting categorical features to numeric features using feature explosion...")
print(" ")
df=pd.get_dummies(df)

df1=df.drop(['Empathy'], axis=1)
#print (df1.shape)
X=df1
Y = df["Empathy"]



#-----Feature Selection Methods-----#

#Method 1: Tree Based Feature Selection

clf1 = ExtraTreesClassifier()
clf1 = clf1.fit(X, Y)
model1 = SelectFromModel(clf1, prefit=True)

X_new1 = model1.transform(X)


#Method 2: L1-based Feature Selection

print("Using Regularisation using L1 based Feature selection method for Feature Extraction...")
print(" ")

clf2 = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y)
model2= SelectFromModel(clf2, prefit=True)
X_new2 = model2.transform(X)


#Method 3: Dimensionality Reduction using Principle Component Analysis (PCA)

pca = PCA(n_components=2)
pca.fit(X)
X_new=pca.transform(X)
clf2 = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_new, Y)
model2= SelectFromModel(clf2, prefit=True)
X_new3 = model2.transform(X_new)



#Selecting the transformed data we got using L1 Based Feature selection technique

print("Splitting Data into TRAIN/ DEV/ TEST dataset in ratio 90:10:10 ")
print(" ")

train=X_new2
classes=Y
length=train.shape[0]


f=int(0.8 * length)
l=int(0.9 * length)

#------Train Dataset-----#
xtrain=train[0:f]
ytrain=classes[0:f]


#-------Development Dataset-----#
xdev=train[f:l]
ydev=classes[f:l]


#---------Test Dataset-------#
xtest=train[l:]
ytest=classes[l:]

#-----Different Baseline Models------#


print("Fitting Baseline Models... ")
print(" ")

#Strategy1: MOST FREQUENT - Always predicts the most frequent label in the training set.

bl1 = DummyClassifier(strategy='most_frequent',random_state=0)
bl1=bl1.fit(xtrain, ytrain)
p1=bl1.score(xtest, ytest)
print("Accuracy of Baseline model using MOST FREQUENT strategy: ", p1)

#Strategy2: UNIFORM - Generates predictions uniformly at random.

bl2 = DummyClassifier(strategy='uniform',random_state=0)
bl2=bl2.fit(xtrain, ytrain)
p2=bl2.score(xtest, ytest)
print("Accuracy of Baseline model using UNIFORM strategy: ", p2)

#Strategy3: STRATIFIED - Generates random predictions by respecting the training set class distribution.
bl3 = DummyClassifier(strategy='stratified',random_state=0)
bl3=bl3.fit(xtrain, ytrain)
p3=bl3.score(xtest, ytest)
print("Accuracy of Baseline model using STRATIFIED strategy: ", p3)


def find_accuracies(clf,xtrain, ytrain, xtest, ytest):
    predicted1 = clf.predict ( xtrain )
    predicted2 = clf.predict ( xtest )
    trainacc = 0.0
    testacc = 0.0
    for i in range (100):
        trainaccuracy = np.mean(ytrain==predicted1)
        testaccuracy = np.mean(ytest==predicted2)
        trainacc += trainaccuracy
        testacc += testaccuracy
    return predicted1, predicted2, trainacc, testacc


#------Function to find correctly and incorrectly classified development data set-----#
def find_corincor_devdata(clf,xdev,ydev):
    predicted=clf.predict(xdev)
    correct=[]
    incorrect=[]
    for i in range ( int(len(ydev)/4) ): #Finding only few examples of correctly and incorrectly classified development data set
        if predicted[i] == ydev[i]:
            correct.append(xdev[i])
        else:
            incorrect.append(xdev[i])
    #print(correct)
    #print(incorrect)
    return correct,incorrect


#----------------------Classification Models--------------------#
print(" ")
print("----------------------------------------Stochastic Gradient Descent (SGD)-----------------------------------")
#Model 1: Using Stochastic Gradient Descent

clf1 = linear_model.SGDClassifier(max_iter=1000)
clf1= clf1.fit(xtrain, ytrain)
predtrain,predtest,trainacc,testacc=find_accuracies(clf1,xtrain,ytrain,xtest,ytest)
print(" ")
print ( "Training accuracy for SGD model: ", trainacc / 100.0 )
print ( "Testing accuracy for SGD model: ", testacc / 100.0 )

#Now, since the model is not giving much good accuracy against baseline models on test data, we will tune hyperparamters
#using validation data and Gridsearch CV approach for all models.


params_SGD={
            'loss': ['hinge','log'],
            'penalty': ['l1','l2'],
            'max_iter':[1000,2000,3000]
            }

gs1=GridSearchCV(clf1, params_SGD)

#Fitting the model using development dataset since its a conventional technique to use dev data set for hyperparameter tuning.
gs1.fit(np.array(xdev), np.array(ydev))

print("Best Parameters for SGD:", gs1.best_params_)

print(" ")
print("---Accuracies for SGD model after hyperparameter tuning---")

gs1=linear_model.SGDClassifier(loss='log', max_iter=1000, penalty='l1')
gs1.fit(xtrain,ytrain)
#We notice that even after hyperparameter tuning, model fails to give good results.
predtrain,predtest,trainacc,testacc=find_accuracies(gs1,xtrain,ytrain,xtest,ytest)
print ( "Training accuracy: ", trainacc / 100.0 )
print ( "Testing accuracy: ", testacc / 100.0 )


print("---------------------------------Support Vector Classification-------------------------------------------")
#Model 2: Support Vector Classification
clf2 = SVC()
clf2.fit(xtrain, ytrain)
predtrain,predtest,trainacc,testacc=find_accuracies(clf2,xtrain,ytrain,xtest,ytest)
print(" ")
print ( "Training accuracy for SVC model: ", trainacc / 100.0 )
print ( "Testing accuracy for SVC model: ", testacc / 100.0 )

#We see that the model is overfitting, so there is a need to tune hyperparameters to avoid overfitting

params_SVC= {'C': [1, 10, 100, 1000],
         'gamma': [0.001, 0.001],
         'kernel': ['linear','rbf']}


gs2=GridSearchCV(clf2, params_SVC)

gs2.fit(np.array(xdev), np.array(ydev))

print("Best Parameters for SVC:", gs2.best_params_)
gs2=SVC(C= 10, gamma= 0.001, kernel= 'rbf')
gs2.fit(xtrain,ytrain)

#Finding the examples in the development data set that were wrongly classified even after parameter tuning
gs_dev=gs2
gs_dev.fit(xtrain, ytrain)
corrlist,incorrlist=find_corincor_devdata(gs_dev,np.array(xdev),np.array(ydev))
print(" ")
print ( "Few Examples from Development Set correctly classified are as follow:")
print(corrlist)
print(" ")
print ( "Few Examples from Development Set incorrectly classified are as follow:")
print(incorrlist)

print(" ")
print("---Accuracies for SVC model after hyperparameter tuning---")

#Now, model overfits less but there is decrement in the test accuracy.
predtrain,predtest,trainacc,testacc=find_accuracies(gs2,xtrain,ytrain,xtest,ytest)
print ( "Training accuracy: ", trainacc / 100.0 )
print ( "Testing accuracy: ", testacc / 100.0 )

print(" ")

print ( '  A. Confusion matrix:\n', confusion_matrix ( ytest, predtest ), '\n' )
print ( '  B. Classification Report:\n', classification_report ( ytest, predtest ), '\n' )

print("---------------------------------------------Logistic Regression-------------------------------------------")

#Model 3: Logistic Regression

clf3 = linear_model.LogisticRegression()
clf3.fit(xtrain, ytrain)
predtrain,predtest,trainacc,testacc=find_accuracies(clf3,xtrain,ytrain,xtest,ytest)
print(" ")
print ( "Training accuracy for Logistic Regression model: ", trainacc / 100.0 )
print ( "Testing accuracy for Logistic Regression model: ", testacc / 100.0 )


params_logistic= {
            'penalty': ['l1','l2'],
            'C': [1, 10, 100, 1000],
            'max_iter':[100,1000,2000,3000],
        }


gs3=GridSearchCV(clf3, params_logistic)

gs3.fit(np.array(xdev), np.array(ydev))

print("Best Parameters for Logistic Regression:", gs3.best_params_)
gs3=linear_model.LogisticRegression(C= 1, max_iter= 100, penalty= 'l1')
gs3.fit(xtrain,ytrain)

print(" ")
print("---Accuracies for Logistic Regression model after hyperparameter tuning---")

predtrain,predtest,trainacc,testacc=find_accuracies(gs3,xtrain,ytrain,xtest,ytest)
print ( "Training accuracy: ", trainacc / 100.0 )
print ( "Testing accuracy: ", testacc / 100.0 )

print("---------------------------------------------RandomForest----------------------------------------------------------")

#Model 4: Random Forest

clf4 = RandomForestClassifier()
clf4.fit(xtrain, ytrain)
predtrain,predtest,trainacc,testacc=find_accuracies(clf4,xtrain,ytrain,xtest,ytest)
print(" ")
print ( "Training accuracy for Random Forest model: ", trainacc / 100.0 )
print ( "Testing accuracy for Random Forest model: ", testacc / 100.0 )

random= {
            'n_estimators': [403,500,600],
            'random_state': [200,420,500,666],
            'n_jobs':[-1]
        }


gs4=GridSearchCV(clf4,random)

gs4.fit(np.array(xdev), np.array(ydev))

print("Best Parameters for Random Forest Model:", gs4.best_params_)
gs4=RandomForestClassifier(n_estimators= 403, n_jobs= -1, random_state= 666)
gs4.fit(xtrain,ytrain)

print(" ")
print("---Accuracies for Random Forest model after hyperparameter tuning---")

predtrain,predtest,trainacc,testacc=find_accuracies(gs4,xtrain,ytrain,xtest,ytest)
print ( "Training accuracy: ", trainacc / 100.0 )
print ( "Testing accuracy: ", testacc / 100.0 )

print("----------------------------------Ensemble of Classifiers--------------------------------------------")
#Model 5: Ensembling Different Classifiers
#Almost give the same results as that of Random Forest model
ensemble_voting=VotingClassifier([("SGD",gs1),("SVC",gs2),("logistic",gs3),("RandomForest",clf4)],weights=[1,1,1,2])
ensemble_voting.fit(xtrain,ytrain)
predtrain,predtest,trainacc,testacc=find_accuracies(ensemble_voting,xtrain,ytrain,xtest,ytest)
print(" ")
print ( "Training accuracy using ensembling method: ", trainacc / 100.0 )
print ( "Testing accuracy using ensembling method: ", testacc / 100.0 )

print ( '  A. Confusion matrix:\n', confusion_matrix ( ytest, predtest ), '\n' )
print ( '  B. Classification Report:\n', classification_report ( ytest, predtest ), '\n' )



print("---------------------------------------------XGBoost----------------------------------------------------------")
#Model 5: XGBoost
clf5 = xgb.XGBClassifier()
clf5.fit(xtrain, ytrain)

predtrain,predtest,trainacc,testacc=find_accuracies(clf5,xtrain,ytrain,xtest,ytest)
print(" ")
print ( "Training accuracy using XGBoost method: ", trainacc / 100.0 )
print ( "Testing accuracy using XGBoost method: ", testacc / 100.0 )


#Parameter Tuning
parameters = {'nthread':[4,5],
              'objective':['binary:logistic'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [500,1000],
              'missing':[-999],
              'seed': [1337]}


gs5 = GridSearchCV(clf5, parameters)

gs5.fit(np.array(xdev), np.array(ydev))

print("Best Parameters for XGBoost Model:", gs5.best_params_)
gs5=xgb.XGBClassifier(colsample_bytree= 0.7, learning_rate= 0.05, max_depth= 6, min_child_weight= 11, missing= -999, n_estimators= 500, nthread= 4, objective= 'binary:logistic', seed= 1337, silent= 1, subsample= 0.8)
gs5.fit(xtrain,ytrain)



predtrain,predtest,trainacc,testacc=find_accuracies(gs5,xtrain,ytrain,xtest,ytest)

print(" ")
print("---Accuracies for XGBoost model after hyperparameter tuning---")
print ( "Training accuracy: ", trainacc / 100.0 )
print ( "Testing accuracy: ", testacc / 100.0 )


print("---------------------------------------------AdaBoost---------------------------------------------------------")

#Model 6: AdaBoost
clf6 = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=5), n_estimators=300,
    algorithm="SAMME.R", learning_rate=1, random_state=65)
clf6.fit(xtrain, ytrain)

predtrain,predtest,trainacc,testacc=find_accuracies(clf6,xtrain,ytrain,xtest,ytest)
print(" ")
print ( "Training accuracy using AdaBoost method: ", trainacc / 100.0 )
print ( "Testing accuracy using Adaboost method: ", testacc / 100.0 )



print(" ")
print("Program Ended with displaying all model's accuracies")

