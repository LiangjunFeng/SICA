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
    traindata0 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d00.mat')['data'].T
    traindata1 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d01.mat')['data'].T
    traindata2 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d02.mat')['data'].T
    traindata3 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d03.mat')['data'].T
    traindata4 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d04.mat')['data'].T
    traindata5 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d05.mat')['data'].T
    traindata6 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d06.mat')['data'].T
    traindata7 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d07.mat')['data'].T
    traindata8 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d08.mat')['data'].T
    traindata9 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d09.mat')['data'].T
    traindata10 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d10.mat')['data'].T
    traindata11 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d11.mat')['data'].T
    traindata12 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d12.mat')['data'].T
    traindata13 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d13.mat')['data'].T
    traindata14 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d14.mat')['data'].T
    traindata15 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d15.mat')['data'].T
    traindata16 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d16.mat')['data'].T
    traindata17 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d17.mat')['data'].T
    traindata18 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d18.mat')['data'].T
    traindata19 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d19.mat')['data'].T
    traindata20 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d20.mat')['data'].T
    traindata21 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d21.mat')['data'].T
  
    testdata0 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d00_te.mat')['data'].T    
    testdata1 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d01_te.mat')['data'].T
    testdata2 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d02_te.mat')['data'].T
    testdata3 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d03_te.mat')['data'].T
    testdata4 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d04_te.mat')['data'].T
    testdata5 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d05_te.mat')['data'].T
    testdata6 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d06_te.mat')['data'].T    
    testdata7 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d07_te.mat')['data'].T
    testdata8 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d08_te.mat')['data'].T
    testdata9 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d09_te.mat')['data'].T
    testdata10 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d10_te.mat')['data'].T
    testdata11 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d11_te.mat')['data'].T
    testdata12 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d12_te.mat')['data'].T
    testdata13 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d13_te.mat')['data'].T
    testdata14 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d14_te.mat')['data'].T
    testdata15 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d15_te.mat')['data'].T
    testdata16 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d16_te.mat')['data'].T
    testdata17 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d17_te.mat')['data'].T    
    testdata18 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d18_te.mat')['data'].T
    testdata19 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d19_te.mat')['data'].T
    testdata20 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d20_te.mat')['data'].T
    testdata21 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/TE_mat_data/d21_te.mat')['data'].T
    


    traindata = np.vstack([traindata0,traindata1,traindata2,traindata3,traindata4,traindata5,traindata6,traindata7,traindata8,traindata9,traindata10,traindata11,traindata12,traindata13,traindata14,traindata15,traindata16,traindata17,traindata18,traindata19,traindata20,traindata21])
    testdata = np.vstack([testdata0,testdata1,testdata2,testdata3,testdata4,testdata5,testdata6,testdata7,testdata8,testdata9,testdata10,testdata11,testdata12,testdata13,testdata14,testdata15,testdata16,testdata17,testdata18,testdata19,testdata20,testdata21])
    
    trainlabel = []
    for i in range(500+480*21):
        if i < 500:
            trainlabel.append(0)
        else:
            trainlabel.append(1+int((i-500)/480))
    trainlabel = np.mat(trainlabel).T
    
    testlabel = []
    for i in range(960*22):
        testlabel.append(int(i/960))
    testlabel = np.mat(testlabel).T
     
    scaler = preprocessing.StandardScaler()
    scaler.fit(traindata)
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
        sica1 = ssica.SSICA(a = b,conponent = n)
        traindata_sica = sica1.fit_transform(traindata.T,trainlabel).T
        testdata_sica = sica1.transform(testdata.T).T
        Faceidentifier(traindata_sica,trainlabel,testdata_sica,testlabel)






