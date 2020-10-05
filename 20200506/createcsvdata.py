# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:08:31 2020

@author: 沛紋
"""

def create_csv_data(sound):
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft,ifft
    from scipy.io import wavfile # get the api
    import numpy as np
    from scipy import signal
    global cell
    
    fs, data = wavfile.read(sound) 
    data = data[0:5292000] #120seconds
    # mix dual sound
    data = (data[:,0]+data[:,1])/2
    
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
    
    new_tablet = np.zeros((129,cell)) #129為頻率段
    
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

    return result


from sklearn import preprocessing

Peiwen_hungry = create_csv_data(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Peiwen_hungry.wav')
Peiwen_30 = create_csv_data(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Peiwen_30.wav')
Peiwen_50 = create_csv_data(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Peiwen_50.wav')
Peiwen_80 = create_csv_data(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Peiwen_80.wav')
Peiwen_hungry_nor = preprocessing.scale(Peiwen_hungry)
Peiwen_30_nor = preprocessing.scale(Peiwen_30)
Peiwen_50_nor = preprocessing.scale(Peiwen_50)
Peiwen_80_nor = preprocessing.scale(Peiwen_80)

Jarwy_hungry = create_csv_data(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Jarwy_hungry.wav')
Jarwy_30 = create_csv_data(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Jarwy_30.wav')
Jarwy_50 = create_csv_data(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Jarwy_50.wav')
Jarwy_80 = create_csv_data(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Jarwy_80.wav')
Jarwy_hungry_nor = preprocessing.scale(Jarwy_hungry)
Jarwy_30_nor = preprocessing.scale(Jarwy_30)
Jarwy_50_nor = preprocessing.scale(Jarwy_50)
Jarwy_80_nor = preprocessing.scale(Jarwy_80)

Yuche_hungry = create_csv_data(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Yuche_hungry.wav')
Yuche_30 = create_csv_data(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Yuche_30.wav')
Yuche_50 = create_csv_data(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Yuche_50.wav')
Yuche_80 = create_csv_data(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Yuche_80.wav')
Yuche_hungry_nor = preprocessing.scale(Yuche_hungry)
Yuche_30_nor = preprocessing.scale(Yuche_30)
Yuche_50_nor = preprocessing.scale(Yuche_50)
Yuche_80_nor = preprocessing.scale(Yuche_80)

chinghui_hungry = create_csv_data(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0129_chinghui_hungry.wav')
chinghui_30 = create_csv_data(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0129_chinghui_30.wav')
chinghui_50 = create_csv_data(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0129_chinghui_50.wav')
chinghui_80 = create_csv_data(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0129_chinghui_80.wav')
chinghui_hungry_nor = preprocessing.scale(chinghui_hungry)
chinghui_30_nor = preprocessing.scale(chinghui_30)
chinghui_50_nor = preprocessing.scale(chinghui_50)
chinghui_80_nor = preprocessing.scale(chinghui_80)


def cosine_similarity_calculate(standard_nor,yuche_hungry_nor,index_standard,index_yuche):
    import numpy as np
    cos_sim = np.dot(standard_nor[index_standard],yuche_hungry_nor[index_yuche]) / (np.linalg.norm(standard_nor[index_standard]) * np.linalg.norm(yuche_hungry_nor[index_yuche]))
    return cos_sim

def compare_similarity(x,y,index_standard,quantity_data):
    output=[]
    for index in range(0,quantity_data):
        index_y = index
        cos = cosine_similarity_calculate(x,y,index_standard,index_y)
        output.append(cos)
        
        y_list = y.tolist()
        STFT_value = []
        index=[]
        STFT_unknown=[]
        for i in range(0,len(output)):
            if abs(output[i])>0.8:
                STFT_value.append(y_list[i])
                index.append(i)
            else:
                STFT_unknown.append(y_list[i])
    
    return output,STFT_value,index,STFT_unknown

# -------------------------------------------------------------Hungry------------------------------------------------------------------------
index_x = 29
marksound = Peiwen_hungry_nor


all_cosine_peiwen_and_peiwen,STFT_peiwen_hungry,threshold_peiwen_index,STFT_peiwen_unknown = compare_similarity(marksound,Peiwen_hungry_nor,index_x,cell)
all_cosine_peiwen_and_Jarwy,STFT_Jarwy_hungry,threshold_Jarwy_index,STFT_Jarwy_unknown = compare_similarity(marksound,Jarwy_hungry_nor,index_x,cell)
all_cosine_peiwen_and_Yuche,STFT_Yuche_hungry,threshold_Yuche_index,STFT_Yuche_unknown = compare_similarity(marksound,Yuche_hungry_nor,index_x,cell)
all_cosine_peiwen_and_chinghui,STFT_chinghui_hungry,threshold_chinghui_index,STFT_chinghui_unknown = compare_similarity(marksound,chinghui_hungry_nor,index_x,cell)

'''abs相似度超過0.7'''  
import pandas as pd
# Hungry - Label 0
all_hungry_list=[]
all_hungry_list.extend(STFT_peiwen_hungry)
all_hungry_list.extend(STFT_Jarwy_hungry)
all_hungry_list.extend(STFT_Yuche_hungry)
all_hungry_list.extend(STFT_chinghui_hungry)
all_new_hungry_data = pd.DataFrame(all_hungry_list)
all_new_hungry_data['class'] = 0

# # -------------------------------------------------------------30%------------------------------------------------------------------------
# index_x_30 = 14
# marksound_30 = Peiwen_30_nor


# all_cosine_peiwen_and_peiwen_30,STFT_peiwen_30,threshold_peiwen_index_30,STFT_peiwen_unknown_30 = compare_similarity(marksound_30,Peiwen_30_nor,index_x_30,cell)
# all_cosine_peiwen_and_Jarwy_30,STFT_Jarwy_30,threshold_Jarwy_index_30,STFT_Jarwy_unknown_30 = compare_similarity(marksound_30,Jarwy_30_nor,index_x_30,cell)
# all_cosine_peiwen_and_Yuche_30,STFT_Yuche_30,threshold_Yuche_index_30,STFT_Yuche_unknown_30= compare_similarity(marksound_30,Yuche_30_nor,index_x_30,cell)
# all_cosine_peiwen_and_chinghui_30,STFT_chinghui_30,threshold_chinghui_index_30,STFT_chinghui_unknown_30 = compare_similarity(marksound_30,chinghui_30_nor,index_x_30,cell)

# # 30% - Label 1
# all_30_list=[]
# all_30_list.extend(STFT_peiwen_30)
# all_30_list.extend(STFT_Jarwy_30)
# all_30_list.extend(STFT_Yuche_30)
# all_30_list.extend(STFT_chinghui_30)
# all_new_30_data = pd.DataFrame(all_30_list)
# all_new_30_data['class'] = 1

# # -------------------------------------------------------------50%------------------------------------------------------------------------
# index_x_50 = 12
# marksound_50 = Peiwen_50_nor


# all_cosine_peiwen_and_peiwen_50,STFT_peiwen_50,threshold_peiwen_index_50,STFT_peiwen_unknown_50 = compare_similarity(marksound_50,Peiwen_50_nor,index_x_50,cell)
# all_cosine_peiwen_and_Jarwy_50,STFT_Jarwy_50,threshold_Jarwy_index_50,STFT_Jarwy_unknown_50 = compare_similarity(marksound_50,Jarwy_50_nor,index_x_50,cell)
# all_cosine_peiwen_and_Yuche_50,STFT_Yuche_50,threshold_Yuche_index_50,STFT_Yuche_unknown_50= compare_similarity(marksound_50,Yuche_50_nor,index_x_50,cell)
# all_cosine_peiwen_and_chinghui_50,STFT_chinghui_50,threshold_chinghui_index_50,STFT_chinghui_unknown_50 = compare_similarity(marksound_50,chinghui_50_nor,index_x_50,cell)

# # 50% - Label 2
# all_50_list=[]
# all_50_list.extend(STFT_peiwen_50)
# all_50_list.extend(STFT_Jarwy_50)
# all_50_list.extend(STFT_Yuche_50)
# all_50_list.extend(STFT_chinghui_50)
# all_new_50_data = pd.DataFrame(all_50_list)
# all_new_50_data['class'] = 2

# # -------------------------------------------------------------80%------------------------------------------------------------------------
index_x_80 = 196
marksound_80 = Peiwen_80_nor


all_cosine_peiwen_and_peiwen_80,STFT_peiwen_80,threshold_peiwen_index_80,STFT_peiwen_unknown_80 = compare_similarity(marksound_80,Peiwen_80_nor,index_x_80,cell)
all_cosine_peiwen_and_Jarwy_80,STFT_Jarwy_80,threshold_Jarwy_index_80,STFT_Jarwy_unknown_80 = compare_similarity(marksound_80,Jarwy_80_nor,index_x_80,cell)
all_cosine_peiwen_and_Yuche_80,STFT_Yuche_80,threshold_Yuche_index_80,STFT_Yuche_unknown_80= compare_similarity(marksound_80,Yuche_80_nor,index_x_80,cell)
all_cosine_peiwen_and_chinghui_80,STFT_chinghui_80,threshold_chinghui_index_80,STFT_chinghui_unknown_80 = compare_similarity(marksound_80,chinghui_80_nor,index_x_80,cell)


# 80% - Label 3
all_80_list=[]
all_80_list.extend(STFT_peiwen_80)
all_80_list.extend(STFT_Jarwy_80)
all_80_list.extend(STFT_Yuche_80)
all_80_list.extend(STFT_chinghui_80)
all_new_80_data = pd.DataFrame(all_80_list)
all_new_80_data['class'] = 3

# # -------------------------------------------------------------unknow------------------------------------------------------------------------
# unknow - Label 4
#  
all_unknow_list=[]
all_unknow_list.extend(STFT_peiwen_unknown)
all_unknow_list.extend(STFT_Jarwy_unknown)
all_unknow_list.extend(STFT_Yuche_unknown)
all_unknow_list.extend(STFT_chinghui_unknown)

all_unknow_list.extend(STFT_peiwen_unknown_30)
all_unknow_list.extend(STFT_Jarwy_unknown_30)
all_unknow_list.extend(STFT_Yuche_unknown_30)
all_unknow_list.extend(STFT_chinghui_unknown_30)

all_unknow_list.extend(STFT_peiwen_unknown_50)
all_unknow_list.extend(STFT_Jarwy_unknown_50)
all_unknow_list.extend(STFT_Yuche_unknown_50)
all_unknow_list.extend(STFT_chinghui_unknown_50)

all_unknow_list.extend(STFT_peiwen_unknown_80)
all_unknow_list.extend(STFT_Jarwy_unknown_80)
all_unknow_list.extend(STFT_Yuche_unknown_80)
all_unknow_list.extend(STFT_chinghui_unknown_80)

all_new_unknow_data = pd.DataFrame(all_unknow_list)
all_new_unknow_data['class'] = 4

mydataset = pd.concat([all_new_hungry_data,all_new_30_data,all_new_50_data,all_new_80_data,all_new_unknow_data])
mydataset.to_csv('mydataset_each_diff_threshold.csv',index=0)


def correlation(data):
    import matplotlib.pyplot as plt
    import seaborn as sn
    plt.figure()
    corrMatrix = data.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()
#correlation(mydataset)

def PCA(data,target):
    from sklearn.decomposition import PCA
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 讀取資料集的部分跳過
    x_data = np.array(data) # x 為四維的資料
    y_data = np.array(target) # y 為此資料的類別
    
    # 執行 PCA
    pca = PCA(n_components=3) # n_components 為要降到的維度，這邊降為二維
    pca.fit(x_data) # 使用 x_data 為來源資料
    result = pca.transform(x_data) # 使用 transform() 即可取得降維後的陣列
    print(result)
    # 使用 matplotlib 將結果繪出
    # 前兩個參數是各個點的 x,y 座標值
    # c 是按照每筆資料的類別數值(1~9)，自動著色
    plt.figure()
    plt.subplot(221)
    plt.scatter(result[:,0], result[:,1], c=y_data, s=25)
    plt.subplot(222)
    plt.scatter(result[:,1], result[:,2], c=y_data, s=25)
    plt.subplot(223)
    plt.scatter(result[:,0], result[:,2], c=y_data, s=25)
    
#PCA(mydataset.drop(['class'],axis='columns'),mydataset['class'])
