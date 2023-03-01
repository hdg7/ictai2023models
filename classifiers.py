from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import timeit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import numpy as np
import warnings
import os


def training(X_train, Y_train):
        #We are going to create our machine learning models with the classifiers
        #Classifiers
        #This is a list of models and each of them is going to be a classifier
        models=[]
        models.append(("Tree",DecisionTreeClassifier()))
        models.append(("KNN",KNeighborsClassifier()))
        models.append(("LDiscrimination",LinearDiscriminantAnalysis()))
        models.append(("NB",GaussianNB()))
        models.append(("SVM",SVC(gamma="auto")))
        models.append(("LRegression",LogisticRegression(solver="liblinear",multi_class="ovr")))
        models.append(("RandomForest",RandomForestClassifier()))
        models.append(("GradientBoosting",GradientBoostingClassifier()))
        models.append(("AdaBoost",AdaBoostClassifier()))
        models.append(("XGBoost",XGBClassifier()))
        models.append(("NNet",MLPClassifier(random_state=1, max_iter=300)))
        #models.append(("OneRule",StackingClassifier()))
        #This list will accumulate the results
        results=[]
        names = []
        times=[]

        for name, model in models:
                #Normally you divide the training data in 10 blocks (or n blocks) and you use 9 for training and one
                #for testing, then you change the blocks 10 times and you choose form the 10 models that you have 
                #created the best one. This reduces overfitting
                start=timeit.default_timer()
                cv_fold= StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
                cv_results= cross_val_score(model, X_train,Y_train,cv=cv_fold, scoring="accuracy")
                model.fit(X_train,Y_train)
                stop=timeit.default_timer()
                results.append(cv_results)
                names.append(name)
                times.append(stop-start)
                print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
                print("Time: ",stop-start)
        return results, names, times,models

#Set up for the dierctories
os.mkdir("models")
os.mkdir("results")
os.mkdir("times")


iris = datasets.load_iris()
X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, shuffle=True)
rIris,nIris,tIris,mIris=training(X_train,Y_train)
with open(r'classifiersNames.txt', 'w') as fp:
        for name in nIris:
                fp.write("%s\n" % name)

for i,model in enumerate(mIris):
       pickle.dump(model, open('models/Iris_model_'+nIris[i], 'wb'))
       pickle.dump(rIris[i], open('results/Iris_results_'+nIris[i], 'wb'))
       pickle.dump(tIris[i], open('times/Iris_times_'+nIris[i], 'wb'))

digits = datasets.load_digits()
X = digits.data
Y = digits.target
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, shuffle=True)
rDigits,nDigits,tDigits,mDigits=training(X_train,Y_train)

for i,model in enumerate(mDigits):
        pickle.dump(model, open('models/Digits_model_'+nDigits[i], 'wb'))
        pickle.dump(rDigits[i], open('results/Digits_results_'+nDigits[i], 'wb'))
        pickle.dump(tDigits[i], open('times/Digits_times_'+nDigits[i], 'wb'))

wine = datasets.load_wine()
X = wine.data
Y = wine.target
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, shuffle=True)
rWine,nWine,tWine,mWine=training(X_train,Y_train)

for i,model in enumerate(mWine):
        pickle.dump(model, open('models/Wine_model_'+nWine[i], 'wb'))
        pickle.dump(rWine[i], open('results/Wine_results_'+nWine[i], 'wb'))
        pickle.dump(tWine[i], open('times/Wine_times_'+nWine[i], 'wb'))

bcancer = datasets.load_breast_cancer()
X = bcancer.data
Y = bcancer.target
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, shuffle=True)
rBCancer,nBCancer,tBCancer,mBCancer=training(X_train,Y_train)

for i,model in enumerate(mBCancer):
        pickle.dump(model, open('models/BCancer_model_'+nBCancer[i], 'wb'))
        pickle.dump(rBCancer[i], open('results/BCancer_results_'+nBCancer[i], 'wb'))
        pickle.dump(tBCancer[i], open('times/BCancer_times_'+nBCancer[i], 'wb'))

faces=datasets.fetch_olivetti_faces()
X = faces.data
Y = faces.target
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, shuffle=True)
rFaces,nFaces,tFaces,mFaces=training(X_train,Y_train)

for i,model in enumerate(mFaces):
        pickle.dump(model, open('models/Faces_model_'+nFaces[i], 'wb'))
        pickle.dump(rFaces[i], open('results/Faces_results_'+nFaces[i], 'wb'))
        pickle.dump(tFaces[i], open('times/Faces_times_'+nFaces[i], 'wb'))

#################################
#This bit is missing #
#################################

import types


for i,elem in enumerate(rIris):
    print(nIris[i], "&",
        "{:.1f} ({:.1f})".format(elem.mean()*100,elem.std()*100), "&",
        "{:.1f} ({:.1f})".format(rWine[i].mean()*100,rWine[i].std()*100), "&",
        "{:.1f} ({:.1f})".format(rDigits[i].mean()*100,rDigits[i].std()*100), "&",
        "{:.1f} ({:.1f})".format(rBCancer[i].mean()*100,rBCancer[i].std()*100), "&",
        "{:.1f} ({:.1f})".format(rFaces[i].mean()*100,rFaces[i].std()*100), "\\\\",
         )


###########################################
# The part with MLighter ##########
###########################################

import sys



sys.path.append(os.environ["MLIGHTER_HOME"]+"/backend/")

from MLighter import MLighter


#Retrieving data and models
iris = datasets.load_iris()
nIris=[]
mIris=[]
with open(r'classifiersNames.txt', 'r') as fp:
        for line in fp:
                nIris.append(line[:-1])
                mIris.append(pickle.load(open("models/Iris_model_"+line[:-1],'rb')))

def getRobustness(name,dataset,models):
        parameters = {"name":name}
        session = MLighter(parameters)
        session.uploadDataset("structured",datasetName=name,actualData=dataset.data,targetData=dataset.target)
        session.uploadModel("sklearn",models[0][0],actualModel=models[0][1])
        session.chooseStrategy("noise")
        session.chooseTransformation("discreet")
        try:
                config={"numberVariants":1,"shift":0,"noise":1,"features":np.repeat(1,len(dataset.feature_names))}
        except AttributeError:
                config={"numberVariants":1,"shift":0,"noise":1,"features":np.repeat(1,len(faces.data[0:1][0]))}
        session.setupTransformation(config)
        session.data.transform(session.transformation)
        variants = session.data.getVariants()
        Y_var=session.prediction(variants)
        print("Misclassification==",(Y_var != dataset.target).sum())

        eData=[]
        erData=[]
        for model in models:
                solutionsAbs=[]
                solutionsRel=[]
                for i in range(10):
                        session.uploadModel("sklearn",model[0],actualModel=model[1])
                        session.data.transform(session.transformation)
                        variants = session.data.getVariants()
                        print(session.prediction(variants))
                        Y_var=session.prediction(variants)
                        Y_predict=session.prediction(dataset.data)
                        print(model[0])
                        solutionsAbs.append((Y_var != dataset.target).sum()/len(dataset.target))
                        solutionsRel.append((Y_var != Y_predict).sum()/len(Y_predict))
                eData.append(np.repeat(1, len(solutionsAbs))-np.array(solutionsAbs))
                erData.append(np.repeat(1, len(solutionsRel))-np.array(solutionsRel))
                print("Misclassification Abs==",np.array(solutionsAbs).mean())
                print("Misclassification Rel==",np.array(solutionsRel).mean())
        return eData,erData


iris = datasets.load_iris()
nIris=[]
mIris=[]
with open(r'classifiersNames.txt', 'r') as fp:
        for line in fp:
                nIris.append(line[:-1])
                mIris.append(pickle.load(open("models/Iris_model_"+line[:-1],'rb')))
absIris,relIris=getRobustness("iris",iris,mIris)

wine = datasets.load_wine()
nWine=[]
mWine=[]
with open(r'classifiersNames.txt', 'r') as fp:
        for line in fp:
                nWine.append(line[:-1])
                mWine.append(pickle.load(open("models/Wine_model_"+line[:-1],'rb')))
absWine,relWine=getRobustness("wine",wine,mWine)


digits = datasets.load_digits()
nDigits=[]
mDigits=[]
with open(r'classifiersNames.txt', 'r') as fp:
        for line in fp:
                nDigits.append(line[:-1])
                mDigits.append(pickle.load(open("models/Digits_model_"+line[:-1],'rb')))
absDigits,relDigits=getRobustness("digits",digits,mDigits)

bcancer = datasets.load_breast_cancer()
nBCancer=[]
mBCancer=[]
with open(r'classifiersNames.txt', 'r') as fp:
        for line in fp:
                nBCancer.append(line[:-1])
                mBCancer.append(pickle.load(open("models/BCancer_model_"+line[:-1],'rb')))
absBCancer,relBCancer=getRobustness("bcancer",bcancer,mBCancer)

faces=datasets.fetch_olivetti_faces()
nFaces=[]
mFaces=[]
with open(r'classifiersNames.txt', 'r') as fp:
        for line in fp:
                nFaces.append(line[:-1])
                mFaces.append(pickle.load(open("models/Faces_model_"+line[:-1],'rb')))
absFaces,relFaces=getRobustness("faces",faces,mFaces)



for i,elem in enumerate(absIris):
    print(nIris[i], "&",
        "{:.1f} ({:.1f})".format(elem.mean()*100,elem.std()*100), "&",
        "{:.1f} ({:.1f})".format(absWine[i].mean()*100,absWine[i].std()*100), "&",
        "{:.1f} ({:.1f})".format(absDigits[i].mean()*100,absDigits[i].std()*100), "&"
        "{:.1f} ({:.1f})".format(absBCancer[i].mean()*100,absBCancer[i].std()*100), "&",
        "{:.1f} ({:.1f})".format(absFaces[i].mean()*100,absFaces[i].std()*100), "\\\\",
         )

for i,elem in enumerate(relIris):
    print(nIris[i], "&",
        "{:.1f} ({:.1f})".format(elem.mean()*100,elem.std()*100), "&",
        "{:.1f} ({:.1f})".format(relWine[i].mean()*100,relWine[i].std()*100), "&",
        "{:.1f} ({:.1f})".format(relDigits[i].mean()*100,relDigits[i].std()*100), "&"
        "{:.1f} ({:.1f})".format(relBCancer[i].mean()*100,relBCancer[i].std()*100), "&",
        "{:.1f} ({:.1f})".format(relFaces[i].mean()*100,relFaces[i].std()*100), "\\\\",
         )


for i,elem in enumerate(absIris):
    print(nIris[i], "&",
        "{:.1f} ({:.1f})".format(elem.mean()*100,elem.std()*100), "&",
        "{:.1f} ({:.1f})".format(relIris[i].mean()*100,relIris[i].std()*100), "&",
        "{:.1f} ({:.1f})".format(absWine[i].mean()*100,absWine[i].std()*100), "&",
        "{:.1f} ({:.1f})".format(relWine[i].mean()*100,relWine[i].std()*100), "&",
        "{:.1f} ({:.1f})".format(absDigits[i].mean()*100,absDigits[i].std()*100), "&"
        "{:.1f} ({:.1f})".format(relDigits[i].mean()*100,relDigits[i].std()*100), "&"
        "{:.1f} ({:.1f})".format(absBCancer[i].mean()*100,absBCancer[i].std()*100), "&",
        "{:.1f} ({:.1f})".format(relBCancer[i].mean()*100,relBCancer[i].std()*100), "&",
        "{:.1f} ({:.1f})".format(absFaces[i].mean()*100,absFaces[i].std()*100), "\\\\",
        "{:.1f} ({:.1f})".format(relFaces[i].mean()*100,relFaces[i].std()*100), "\\\\",
         )
