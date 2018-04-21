import numpy as np
from sklearn import preprocessing 
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.naive_bayes import GaussianNB  
from sklearn.neighbors import KNeighborsClassifier
import sica
import sfa



def show_accuracy(predictLabel,Label):
    Label = np.ravel(Label).tolist()
    predictLabel = predictLabel.tolist()
    count = 0
    for i in range(len(Label)):
        if Label[i] == predictLabel[i]:
            count += 1
    print(round(count/len(Label),5))

def Faceidentifier( trainDataSimplified,trainLabel,testDataSimplified,testLabel):
    print("=====================================")  
    
    print("KNeighborsClassifier")
    clf2 = KNeighborsClassifier(n_neighbors = 11)
    clf2.fit(trainDataSimplified,np.ravel(trainLabel))
    predictTestLabel2 = clf2.predict(testDataSimplified)
    show_accuracy(predictTestLabel2,testLabel)
    print()
    
    print("GaussianNB")
    clf5 = GaussianNB()
    clf5.fit(trainDataSimplified,np.ravel(trainLabel))
    predictTestLabel5 = clf5.predict(testDataSimplified)
    show_accuracy(predictTestLabel5,testLabel)
    print()
    

if __name__ == '__main__': 
    
    data1 = pd.read_csv('/Users/zhuxiaoxiansheng/Desktop/c2k_data_comma.csv')
    
    label = data1['label']
    data1 = data1.drop('label',axis=1)
    
    traindata = data1.iloc[0:int(0.9*data1.shape[0]),:].values
    testdata = data1.iloc[int(0.9*data1.shape[0]):,:].values
    
    trainlabel = label.iloc[0:int(0.9*data1.shape[0])].values
    testlabel = label.iloc[int(0.9*data1.shape[0]):].values
     
    print(traindata.shape,len(trainlabel),testdata.shape,len(testlabel))    
    
  
    scaler = preprocessing.StandardScaler().fit(traindata)
    traindata = scaler.transform(traindata)
    testdata = scaler.transform(testdata)
    
    n = 16
    
    print('//===========================pca==========================')
    pca = PCA(n)
    traindata_pca = pca.fit_transform(traindata)
    testdata_pca = pca.transform(testdata)
    Faceidentifier(traindata_pca,trainlabel,testdata_pca,testlabel)

    print('//===========================sfa==========================')
    sfa = sfa.SFA()
    traindata_sfa = sfa.fit_transform(traindata.T,conponents =n).T
    testdata_sfa = sfa.transform(testdata.T).T
    Faceidentifier(traindata_sfa,trainlabel,testdata_sfa,testlabel)
    
    print('//===========================fastica==========================')
    fastica = FastICA(n)
    traindata_fastica = fastica.fit_transform(traindata)
    testdata_fastica = fastica.transform(testdata)
    Faceidentifier(traindata_fastica,trainlabel,testdata_fastica,testlabel)

    for i in range(0,9):
        if i == 0:
            b = 0.1
        elif i == 1:
            b = 0.2
        elif i == 2:
            b = 0.5
        elif i == 3:
            b = 0.8
        elif i == 4:
            b = 1
        elif i == 5:
            b = 2
        elif i == 6:
            b = 5
        elif i == 7:
            b = 8
        elif i == 8:
            b = 10

            
        print('//===========================sica==========================')
        print(b)
        sica1 = sica.SICA(a = b,conponent = n)
        traindata_sica = sica1.fit_transform(traindata.T).T
        testdata_sica = sica1.transform(testdata.T).T
        Faceidentifier(traindata_sica,trainlabel,testdata_sica,testlabel)





























