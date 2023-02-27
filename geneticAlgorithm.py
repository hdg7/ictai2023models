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
import random
import os.path



###########################################
# The part with MLighter ##########
###########################################

import sys
import os

try:
        os.mkdir("resultsGen")
except:
        pass
try:
        os.mkdir("resultsGen/AdvGens")
except:
        pass
try:
        os.mkdir("resultsGen/selection")
except:
        pass
try:
        os.mkdir("resultsGen/AdvGlobal")
except:
        pass


sys.path.append(os.environ["MLIGHTER_HOME"]+"/backend/")
from MLighter import MLighter


def getRobustness(name,dataset,models):
        parameters = {"name":name}
        session = MLighter(parameters)
        session.uploadDataset("structured",datasetName=name,actualData=dataset.data,targetData=dataset.target)
        session.uploadModel("sklearn",models[0][0],actualModel=models[0][1])
        session.chooseStrategy("search")
        session.chooseTransformation("genAlg")
        configOr=session.transformation.ga_config()
        try:
                configEx={"numberVariants":1,"numtuples":len(dataset.feature_names),"shift":0,"noise":1,"predictor":session.prediction_proba,"features":np.repeat(1,len(dataset.feature_names))}
        except AttributeError:
                configEx={"numberVariants":1,"numtuples":len(faces.data[0:1][0]),"shift":0,"noise":1,"predictor":session.prediction_proba,"features":np.repeat(1,len(faces.data[0:1][0]))}
        config={**configOr,**configEx}
        config["numgen"]=50
#        session.setupTransformation(config)
#        session.data.transform(session.transformation)
#        variants = session.data.getVariants()
        #print(session.prediction(variants))
#        Y_var=session.prediction(variants)
        #print(mIris[0][0])
        config["population_size"]=100
        config["lambda_sel"]=50
        config["mu_sel"]=50
        values=random.sample(range(0,len(dataset.target)),20)
        eData=[]
        erData=[]
        for model in models:
                if(os.path.exists("resultsGen/AdvGens/"+name+"_"+model[0]+"_results_19")):
                   continue
                solutionsAbs=[]
                solutionsRel=[]
                for j,i in enumerate(values):
                   if(os.path.exists("resultsGen/AdvGens/"+name+"_"+model[0]+"_results_"+str(j))):
                      continue                
                   config["oriVariant"]=i
                   weights=np.repeat(1.0,len(np.unique(dataset.target)))
                   weights[dataset.target[i]]=-1.0
                   weights2=tuple(i for i in weights)
                   config["weights"]=weights2
                   params={}
                   params["oriVariant"]=i
                   params["weights"]=weights2
                   params["numberVariants"]=config["numberVariants"]
                   params["numtuples"]=config["numtuples"]
                   params["shift"]=config["shift"]
                   params["noise"]=config["noise"]
                   params["features"]=config["features"]
                   params["class"]=dataset.target[config["oriVariant"]]
                   params["oriPrediction"]=model[1].predict([dataset.data[config["oriVariant"]]])[0]
                                                  
                   #mIris[0][1].predict_proba(iris.data[0:1])[0]
                   session.setupTransformation(config)
                   session.uploadModel("sklearn",model[0],actualModel=model[1])
                   session.data.transform(session.transformation)
                   print(session.transformation.logbook)
                   pickle.dump(session.transformation.logbook, open('resultsGen/AdvGens/'+name+"_"+model[0]+'_results_'+str(j), 'wb'))
                   pickle.dump(params, open('resultsGen/selection/'+name+"_"+model[0]+'_config_'+str(j), 'wb'))
                   variants = session.data.getVariants()
                   print(session.prediction(variants))
                   Y_var=session.prediction(variants)
                   oriVec=np.repeat(dataset.target[config["oriVariant"]],len(Y_var))
                   print(weights2)
                   print(oriVec)
                   print("Misclassification==",(Y_var != oriVec).sum())
                   print(Y_var)
                   #Y_predict=session.prediction(dataset.data)
                   print(model[0])
                   solutionsAbs.append((Y_var != oriVec).sum()/len(oriVec))
                   #solutionsRel.append((Y_var != Y_predict).sum()/len(Y_predict))
                pickle.dump(np.repeat(1, len(solutionsAbs))-np.array(solutionsAbs), open('resultsGen/AdvGlobal/'+name+"_"+model[0]+'_results', 'wb'))
                
                eData.append(np.repeat(1, len(solutionsAbs))-np.array(solutionsAbs))
                #erData.append(np.repeat(1, len(solutionsRel))-np.array(solutionsRel))
                print("Misclassification Abs==",np.array(solutionsAbs).mean())
#                print("Misclassification Rel==",np.array(solutionsRel).mean())
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


digits = datasets.load_digits()
nDigits=[]
mDigits=[]
with open(r'classifiersNames.txt', 'r') as fp:
        for line in fp:
                nDigits.append(line[:-1])
                mDigits.append(pickle.load(open("models/Digits_model_"+line[:-1],'rb')))
print(getRobustness("digits",digits,mDigits))



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
