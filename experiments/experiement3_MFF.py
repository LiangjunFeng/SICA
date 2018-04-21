import numpy as np
from sklearn import preprocessing 
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.naive_bayes import GaussianNB  
from sklearn.neighbors import KNeighborsClassifier
import scipy.io as sio 
import ssica
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
    fault1_1 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/CVACaseStudy/CVACaseStudy/FaultyCase1.mat')['Set1_1']
    fault2_1 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/CVACaseStudy/CVACaseStudy/FaultyCase2.mat')['Set2_1']
    fault3_1 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/CVACaseStudy/CVACaseStudy/FaultyCase3.mat')['Set3_1']
    fault4_1 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/CVACaseStudy/CVACaseStudy/FaultyCase4.mat')['Set4_1']
    fault5_1 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/CVACaseStudy/CVACaseStudy/FaultyCase5.mat')['Set5_1']   
    fault6_1 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/CVACaseStudy/CVACaseStudy/FaultyCase6.mat')['Set6_1']

    traindata1 = fault1_1[0:int(0.8*fault1_1.shape[0]),:]
    traindata2 = fault2_1[0:int(0.8*fault2_1.shape[0]),:]
    traindata3 = fault3_1[0:int(0.8*fault3_1.shape[0]),:]
    traindata4 = fault4_1[0:int(0.8*fault4_1.shape[0]),:]
    traindata5 = fault5_1[0:int(0.8*fault5_1.shape[0]),:]
    traindata6 = fault6_1[0:int(0.8*fault6_1.shape[0]),:]
    
    traindata = np.vstack([traindata1,traindata2,traindata3,traindata4,traindata5,traindata6])
    trainlabel = [0]*traindata1.shape[0]+[1]*traindata2.shape[0]+[2]*traindata3.shape[0]+[3]*traindata4.shape[0]+[4]*traindata5.shape[0]+[5]*traindata6.shape[0]
    
    testdata1 = fault1_1[int(0.8*fault1_1.shape[0]):,:]
    testdata2 = fault2_1[int(0.8*fault2_1.shape[0]):,:]
    testdata3 = fault3_1[int(0.8*fault3_1.shape[0]):,:]
    testdata4 = fault4_1[int(0.8*fault4_1.shape[0]):,:]
    testdata5 = fault5_1[int(0.8*fault5_1.shape[0]):,:]
    testdata6 = fault6_1[int(0.8*fault6_1.shape[0]):,:]
    
    testdata = np.vstack([testdata1,testdata2,testdata3,testdata4,testdata5,testdata6])
    testlabel = [0]*testdata1.shape[0]+[1]*testdata2.shape[0]+[2]*testdata3.shape[0]+[3]*testdata4.shape[0]+[4]*testdata5.shape[0]+[5]*testdata6.shape[0]
    

    scaler = preprocessing.StandardScaler()
    scaler.fit(traindata)
    traindata = scaler.transform(traindata)
    testdata = scaler.transform(testdata)
    
    print(traindata.shape,len(trainlabel),testdata.shape,len(testlabel))

    n = 16
    print('//===========================pca==========================')
    pca = PCA(n)
    traindata_pca = pca.fit_transform(traindata)
    testdata_pca = pca.transform(testdata)
    Faceidentifier(traindata_pca,trainlabel,testdata_pca,testlabel)

    print('//===========================fastica==========================')
    fastica = FastICA(n)
    traindata_fastica = fastica.fit_transform(traindata)
    testdata_fastica = fastica.transform(testdata)
    Faceidentifier(traindata_fastica,trainlabel,testdata_fastica,testlabel)
    

    print('//===========================sfa==========================')
    sfa = sfa.SFA()
    traindata_sfa = sfa.fit_transform(traindata.T,conponents =n).T
    testdata_sfa = sfa.transform(testdata.T).T
    Faceidentifier(traindata_sfa,trainlabel,testdata_sfa,testlabel)

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
        sica1 = ssica.SSICA(a = b,conponent = n)
        traindata_sica = sica1.fit_transform(traindata.T,trainlabel).T
        testdata_sica = sica1.transform(testdata.T).T
        Faceidentifier(traindata_sica,trainlabel,testdata_sica,testlabel)





