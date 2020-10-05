# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 23:28:30 2020

@author: 沛芯&沛纹
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
from scipy.io import wavfile # get the api
import numpy as np
from scipy import signal
import wave

'''不管輸入多少都切30秒'''

fs, data = wavfile.read(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Yuche_hungry.wav') 
data = data[0:5292000]# 44100 * seconds
# mix dual sound
data = (data[:,0]+data[:,1])/2
''' 
原始音檔處理- 2 min
fs, data = wavfile.read(r'G:\Proiect_fullness\收集之資料\0126\0126_Peiwen_hungry.wav') 
data = data[0:5292000]
# mix dual sound
data = (data[:,0]+data[:,1])/2
'''
# How long is the data (second)
long_sec = 120
duration=[]
for i in range(1,len(data)+1):
    duration.append(i/fs) 

#plt.plot(duration[1:44100],data_mix[1:44100])


'''------------------------------濾波--------------------------------'''

H=[]
for i in range(0,len(data)):
    H.append(0)

aa = int(300*len(data)/fs)
bb = int(3000*len(data)/fs)

for i in range(aa,bb):
    H[i]=1
    
H = np.array(H)   
    
F =  fft(data.T)
Y = H*F
y = ifft(Y).real


'''-----------------------------STFT-----------------------------------'''
# f:采样频率数组,t:段时间数组,stft_result:STFT结果
f, t, stft_result = signal.spectrogram(y,fs,noverlap=240)

# 每隔幾秒取一次
# 每0.5秒一格, 共30秒, 60格(30/0.5) 
time = 0.25
time_x = []
cell = int(long_sec/time)

for i in range(1,cell+1):
    time_x.append(i*time)

new_tablet = np.zeros((129,cell))

num = np.size(stft_result,1)


# 82672/60格 = 1378 每1378筆資料放1格
for n in range(0,(round(num/round(num/cell))-1)*(round(num/cell)+1),round(num/cell)):
    for m in range(0,129): #1~129
        cal_1 =int(n/round(num/cell))
        cal_2 = int(n+1)
        cal_3 = int(n+round(num/cell))
        
        new_tablet[m,cal_1] = sum(stft_result[m,cal_2:cal_3])

merge_tablet = np.zeros((9,cell)) #9維頻段
for i in range(0,18,2):
    i_cal = int(i/2)
    merge_tablet[i_cal,] = new_tablet[i,]+new_tablet[i+1,]

result = pd.DataFrame(merge_tablet.T)
result.columns=['0~344','344~689','689~1033', '1033~1378','1378~1722','1722~2067','2067~2411','2411~2756','2756~3100']
#result.to_csv('hsinting_80_4.csv',index=0)

'''--------------------------Normalize-------------------------------'''
from sklearn import preprocessing
result = preprocessing.scale(result)


result = round(result,2)
'''--------------------------load model---------------------------------'''

from keras.models import load_model
from sklearn.externals import joblib
#import joblib


# model = load_model('best_model.h5')
# model_binary = load_model('best_model_binary.h5')
# model_five = load_model('RF_label5_diff_thre_round_cut25_.plk')
# pred = model.predict_classes(result)
# pred_binary = model_binary.predict_classes(result)
# pred_five = model_five.predict_classes(result)

model_five = joblib.load('RF_label5_diff_thre_round_cut25_.plk')
# result = pd.DataFrame(result)
pred_five = model_five.predict(result)
# 0 is hungry, 1 is full, 2 is unknow
# pred_0=[]
# pred_1=[]
# pred_2=[]
# for i in range(0,len(pred)):
#     if pred[i] == 0:
#         pred_0.append(i/2)
#     if pred[i] == 1:
#         pred_1.append(i/2)
#     if pred[i] == 2:
#         pred_2.append(i/2)
# print('----------THREE labels------------')
# print('0:',len(pred_0))
# print('1:',len(pred_1))
# print('2:',len(pred_2))

# pred_binary_0=[]
# pred_binary_1=[]
# pred_binary_2=[]
# for i in range(0,len(pred_binary)):
#     if pred_binary[i] == 0:
#         pred_binary_0.append(i/2)
#     if pred_binary[i] == 1:
#         pred_binary_1.append(i/2)
# print('----------TWO labels------------')
# print('0:',len(pred_binary_0))
# print('1:',len(pred_binary_1))

pred_five_0=[]
pred_five_1=[]
pred_five_2=[]
pred_five_3=[]
pred_five_4=[]
for i in range(0,len(pred_five)):
    if pred_five[i] == 0:
        pred_five_0.append(i/2)
    if pred_five[i] == 1:
        pred_five_1.append(i/2)
    if pred_five[i] == 2:
        pred_five_2.append(i/2)
    if pred_five[i] == 3:
        pred_five_3.append(i/2)
    if pred_five[i] == 4:
        pred_five_4.append(i/2)
print('----------FIVE labels------------')
print('0:',len(pred_five_0))
print('1:',len(pred_five_1))
print('2:',len(pred_five_2))
print('3:',len(pred_five_3))
print('4:',len(pred_five_4))

'''
沛紋的比比看

# cos sim = -0.17
#飽的是腸音(index=3)
A=[-0.0505202,-0.0946091,-0.20633,-0.220005,-0.220868,-0.25568,-0.238121,-0.283774,-0.234926]
#餓的是腸音 (index=14) 
B=[-0.446552,-0.486021,-0.426559,-0.314548,-0.233767,-0.22422,0.116246,0.966448,0.928487]
np.dot(A,B) / (np.linalg.norm(A) * np.linalg.norm(B))


# cos sim =  -0.5
#飽的不是腸音 (index = 21)
A=[-0.763446,-0.774226,-0.583941,-0.339411,-0.233622,-0.278413,-0.245484,-0.260817,-0.2512]
#飽的是腸音 (index=3) 
B=[0.115093,0.575489,0.974287,0.0562762,-0.168235,-0.188291,-0.172056,-0.246108,-0.229093]
np.dot(A,B) / (np.linalg.norm(A) * np.linalg.norm(B))


#cos sim = 0.79
#飽的不是腸音 (index = 21)
A=[-0.763446,-0.774226,-0.583941,-0.339411,-0.233622,-0.278413,-0.245484,-0.260817,-0.2512]
#餓的不是腸音 18秒 (index=35)
B=[-0.529442,-0.555293,-0.462869,-0.31729,-0.190427,-0.195014,0.0244259,0.340677,0.163157]
np.dot(A,B) / (np.linalg.norm(A) * np.linalg.norm(B))


# cos sim = 0.8
#餓的是腸音 (index=14) 
A=[-0.446552,-0.486021,-0.426559,-0.314548,-0.233767,-0.22422,0.116246,0.966448,0.928487]
#餓的不是腸音 18秒 (index=35)
B=[-0.529442,-0.555293,-0.462869,-0.31729,-0.190427,-0.195014,0.0244259,0.340677,0.163157]
np.dot(A,B) / (np.linalg.norm(A) * np.linalg.norm(B))
'''

'''
yuche hungry
#0.15
#哲餓的是腸音(index5)
A = [0.222556,0.297408,-0.084254,-0.174108,-0.20199,-0.360361,0.14876,-0.117274,0.342176]
#紋飽的是腸音(index=3)
B = [-0.0505202,-0.0946091,-0.20633,-0.220005,-0.220868,-0.25568,-0.238121,-0.283774,-0.234926]
np.dot(A,B) / (np.linalg.norm(A) * np.linalg.norm(B))
'''























