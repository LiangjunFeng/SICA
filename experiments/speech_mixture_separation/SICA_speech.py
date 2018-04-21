import wave
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

import scipy.io as sio 
import pandas as pd

def LoadSoundSet(path):
    filename= os.listdir(path) #得到文件夹下的所有文件名称 
    data = []
    for i in range(len(filename)):
        f = wave.open(path+filename[i],'rb')
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = f.readframes(nframes)#读取音频，字符串格式
        waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
        waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化

        data += waveData.tolist()
    time = np.arange(0,nframes*len(filename))*(1.0 / framerate)
    return time.tolist(),data

def LoadSound(path):
    f = wave.open(path,'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)#读取音频，字符串格式
    waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
    waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
    time = np.arange(0,nframes*nchannels)*(1.0 / framerate)
    return time.tolist(),waveData.tolist() 

def ShowRes(data):
    print("//=========================================================//")
    x = np.linspace(0,data.shape[1],data.shape[1])
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(8.5, 0.6*data.shape[0])
    for i in range(data.shape[0]):
        axes = plt.subplot(data.shape[0],1,i+1)
        axes.set_frame_on(False) 
        axes.set_axis_off()
        plt.plot(x,data[i,:].T,color = 'black')
    plt.show()
    print("//========================================================//")

    
def getData():
    file1 = "/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/LDC2017S07.clean.wav" 
    file2 = "/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/LDC2017S10.embed.wav"
    file3 = "/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/LDC93S1.wav"

    noise1 = np.random.standard_normal(25681) 
    time2,noise2 = LoadSound(file2)
    
    time1,data1 = LoadSound(file1)
    time2,data2 = LoadSound(file3)
    data3 = sio.loadmat(u"/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/voice.mat")['voice']
    
    time1 = time1[1000:-1000]
    data1 = np.mat(data1[1000:-1000])  
    data2 = np.mat(data2[3000:3000+len(time1)])
    data3 = np.mat(data3[0,5000:5000+len(time1)])
    noise2 = np.mat(noise2[0:len(time1)])
    s1 = np.mat(np.random.standard_normal(25681))
    
    data = np.zeros((6,len(time1)))

    s1.sort()
      
    data[0,:] = data1
    data[1,:] = data2
    data[2,:] = data3
    data[3,:] = noise2
    data[4,:] = noise1
    data[5,:] = s1
    
    return data

def getRandomW(length,height):
    W = random.random(size=(length,height))
    return W



if __name__ == '__main__':    
#    data = getData() 
#    np.save("speech.npy",data)
    data = np.load("speech.npy")
    A = getRandomW(6,6)
    dataMerage = A.dot(data)     
    print("//==========================original data=========================//")
    ShowRes(data)
    print("//============================mixed data=========================//")
    ShowRes(dataMerage)
    
    ica = FastICA()
    a = ica.fit_transform(dataMerage.T).T
    print("//=============================FastICA===========================//")
    ShowRes(a)

    sica1 = sica.SICA(a = 20)
    res = sica1.fit_transform(dataMerage)
    print("//=============================SICA=============================//")
    ShowRes(res)


