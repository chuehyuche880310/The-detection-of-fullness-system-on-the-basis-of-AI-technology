# -*- coding: utf-8 -*-
"""
Created on Sun May 31 19:45:00 2020

@author: ChuehYuChe
"""


import audio_set_function
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
from scipy.io import wavfile # get the api
import numpy as np
from scipy import signal
import wave

#minkai_hungry_1= audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\minkai _hungry_1.wav',0.25,120) #路徑,間隔,總秒數
#minkai_hungry_2= audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\minkai _hungry_2.wav',0.25,120) #路徑,間隔,總秒數
#minkai_hungry_3= audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\minkai _hungry_3.wav',0.25,120) #路徑,間隔,總秒數
#minkai_hungry_4= audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\minkai _hungry_4.wav',0.25,120) #路徑,間隔,總秒數
#minkai_hungry_5= audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\minkai _hungry_5.wav',0.25,120) #路徑,間隔,總秒數
#
#Minkai_hungry_0516= audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0516_Minkai_hungry.wav',0.25,120) #路徑,間隔,總秒數
#
#Minkai_30_0516= audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0516_Minkai_30.wav',0.25,120) #路徑,間隔,總秒數
#Minkai_50_0516= audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0516_Minkai_50.wav',0.25,120) #路徑,間隔,總秒數
#Minkai_80_0516= audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0516_Minkai_80.wav',0.25,120) #路徑,間隔,總秒數
#
#
#Peiwen_hungry_0518= audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0518_peiwen_hungry_1.wav',0.25,120) #路徑,間隔,總秒數
#Peiwen_30_0518= audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0518\0518_peiwen_30_1.wav',0.25,120) #路徑,間隔,總秒數
#Peiwen_50_0518= audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0518\0518_peiwen_50_2.wav',0.25,120) #路徑,間隔,總秒數
#Peiwen_80_0518= audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0518\0518_peiwen_80_1.wav',0.25,120) #路徑,間隔,總秒數
#--------------------------------------------------

#------驗證音檔-----------
Peiwen_hungry= audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Peiwen_hungry.wav',0.25,120) #路徑,間隔,總秒數
Jarwy_hungry = audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Jarwy_hungry.wav',0.25,120) #路徑,間隔,總秒數
Yuche_hungry = audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Yuche_hungry.wav',0.25,120) #路徑,間隔,總秒數
chinghui_hungry = audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0129_chinghui_hungry.wav',0.25,120) #路徑,間隔,總秒數
#hsinting_hungry = audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0203_hsinting_hungry.wav',0.25,120) #路徑,間隔,總秒數
#
Peiwen_30= audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Peiwen_30.wav',0.25,120) #路徑,間隔,總秒數
Jarwy_30 = audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Jarwy_30.wav',0.25,120) #路徑,間隔,總秒數
Yuche_30 = audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Yuche_30.wav',0.25,120) #路徑,間隔,總秒數
chinghui_30 = audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0129_chinghui_30.wav',0.25,120) #路徑,間隔,總秒數
#hsinting_30 = audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0203_hsinting_30.wav',0.25,120) #路徑,間隔,總秒數
#
Peiwen_50= audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Peiwen_50.wav',0.25,120) #路徑,間隔,總秒數
Jarwy_50 = audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Jarwy_50.wav',0.25,120) #路徑,間隔,總秒數
Yuche_50 = audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Yuche_50.wav',0.25,120) #路徑,間隔,總秒數
chinghui_50 = audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0129_chinghui_50.wav',0.25,120) #路徑,間隔,總秒數
#hsinting_50 = audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0203_hsinting_50.wav',0.25,120) #路徑,間隔,總秒數
#
Peiwen_80= audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Peiwen_80.wav',0.25,120) #路徑,間隔,總秒數
Jarwy_80 = audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Jarwy_80.wav',0.25,120) #路徑,間隔,總秒數
Yuche_80 = audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0126_Yuche_80.wav',0.25,120) #路徑,間隔,總秒數
chinghui_80 = audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0129_chinghui_80.wav',0.25,120) #路徑,間隔,總秒數
#hsinting_80 = audio_set_function.audio_set(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\美國蒐集五人的音檔\0203_hsinting_80.wav',0.25,120) #路徑,間隔,總秒數


#Dataframe 取 列
#def fetch_new_row(data_array):
#    
#    for i in range(len(data_array)):
#        sum_row=0
#        sum_row = sum(data_array[i])
#        for j in range(len(data_array[0])):
#            
#            data_array[i][j] = data_array[i][j]/sum_row 
#            a = data_array
#    
#    a = pd.DataFrame(a)
#    return a
#            
def color_denote(data_frame):
    color=[]
    for i in range(len(data_frame)):
        if data_frame[:,8][i] ==0:
           color.append('red')
        elif data_frame[:,8][i]==1:
            color.append('green')
            
        elif data_frame[:,8][i]==2:
            
            color.append('blue')
            
    return color
    
     
'''HUNGRY'''       
Peiwen_hungry_n = fetch_new_row(np.array(Peiwen_hungry))
Jarwy_hungry_n = fetch_new_row(np.array(Jarwy_hungry))
chinghui_hungry_n = fetch_new_row(np.array(chinghui_hungry))
Yuche_hungry_n = fetch_new_row(np.array(Yuche_hungry))


all_hungry_data = pd.concat([Peiwen_hungry,Jarwy_hungry,chinghui_hungry,Yuche_hungry],ignore_index=True)
all_hungry_data['class'] = 0


'''50%'''

Peiwen_50_n = fetch_new_row(np.array(Peiwen_50))
Jarwy_50_n = fetch_new_row(np.array(Jarwy_50))
chinghui_50_n = fetch_new_row(np.array(chinghui_50))
Yuche_50_n = fetch_new_row(np.array(Yuche_50))


all_50_data = pd.concat([Peiwen_50,Jarwy_50,chinghui_50,Yuche_50],ignore_index=True)
all_50_data['class'] = 1

'''80%'''

Peiwen_80_n = fetch_new_row(np.array(Peiwen_80))
Jarwy_80_n = fetch_new_row(np.array(Jarwy_80))
chinghui_80_n = fetch_new_row(np.array(chinghui_80))
Yuche_80_n = fetch_new_row(np.array(Yuche_80))

all_80_data = pd.concat([Peiwen_80,Jarwy_80,chinghui_80,Yuche_80],ignore_index=True)
all_80_data['class'] = 2


#------------------------------------
'''All data concat'''

all_data = pd.concat([all_hungry_data,all_50_data,all_80_data],ignore_index=True)

score = audio_set_function.fisher_score(all_data.values[:,0:8],all_data['class'].values)

color=[]
for i in range(len(np.array(all_data))):
    if all_data['class'][i]==0:
        color.append('red')
    elif all_data['class'][i]==1:
        color.append('orange')
    else:
        color.append('green')

#plt.figure()
#plt.scatter(all_data[6],all_data[3],c=color)
'''-----------------------'''

'''Peiwen_picture'''
#Peiwen_hungry_n['class'] = 0
#Peiwen_50_n['class'] = 1
#Peiwen_80_n['class'] = 2
#
#
#Peiwen_all = pd.concat([Peiwen_hungry_n,Peiwen_50_n,Peiwen_80_n],ignore_index=True)
#Peiwen_all_color =color_denote(Peiwen_all.values)
#
#
#score_peiwen = audio_set_function.fisher_score(Peiwen_all.values[:,0:8],Peiwen_all['class'].values)
#
##plt.figure()
##plt.scatter(Peiwen_all[6],Peiwen_all[7],c=Peiwen_all_color)
'''---------------'''
def deleteDuplicatedElementFromList2(chose,temp):
#    temp = original
    
    for i in chose:
        temp.remove(i)
        
    return temp


#def random_fetch_list()

from random import sample
#global choose,standard_point,rest_of_number


def change_every_parameter(standard_point,all_hungry_data):
    
#    standard_point = 400
    store_list = np.zeros(standard_point)
    dimension = 8
    
    
    all_hungry_process=all_hungry_data.values[:,0:dimension]
    all_hungry_process=all_hungry_process.tolist()
    
    temp=all_hungry_process[0:len(all_hungry_process)]
    choose = sample(all_hungry_process,standard_point)  #從所有List當中，隨機挑選100個list 
    rest_of_number = deleteDuplicatedElementFromList2(choose,temp)
    
    
    
    def iteration_each_points_first(rest_of_number,new_100):
        new_points = []
        count_accumulation = np.zeros(standard_point)
        
        average_accumulation = []
        for i in range(standard_point):
            average_accumulation.append([])
        
    
        for i in range(len(rest_of_number)):
            d1 = 0
            find_minmum = 0
         
            point_store=[] #每個點 對所有100點的儲存 共有100個
            for j in range(len(new_100)):
                
                d1= np.sqrt(np.sum(np.square(np.array(rest_of_number[i])-np.array(new_100[j]))))
                point_store.append(d1)
                
                
            find_minmum = point_store.index(min(point_store))
            average_accumulation[find_minmum] += rest_of_number[i]
            average_accumulation[find_minmum] = np.array(average_accumulation[find_minmum])
            count_accumulation[find_minmum] +=1
        
    #    print(average_accumulation)
        print(count_accumulation)
        print(np.sum(count_accumulation))
        
        for i in range(len(average_accumulation)):
            if count_accumulation[i] >0:
                new_points.append((average_accumulation[i]/count_accumulation[i]).tolist())
                
            else:
                new_points.extend(sample(rest_of_number,1))
                
        return new_points
    
    

    first_100 = iteration_each_points_first(rest_of_number,choose)
   
        
    def iteration_each_points_second(rest_of_number,new_100):
    
        def iteration_inside(new_100):
    
            new_points = []
            count_accumulation = np.zeros(standard_point)
            
            average_accumulation = []
            for i in range(standard_point):
                average_accumulation.append([])
            
        
            for i in range(len(rest_of_number)):
                d1 = 0
                find_minmum = 0
             
                point_store=[] #每個點 對所有100點的儲存 共有100個
                for j in range(len(new_100)):
                    
                    d1= np.sqrt(np.sum(np.square(np.array(rest_of_number[i])-np.array(new_100[j]))))
                    point_store.append(d1)
                    
                    
                find_minmum = point_store.index(min(point_store))
                average_accumulation[find_minmum] += rest_of_number[i]
                average_accumulation[find_minmum] = np.array(average_accumulation[find_minmum])
                count_accumulation[find_minmum] +=1
            
        #    print(average_accumulation)
          
            print(count_accumulation)
            print(np.sum(count_accumulation))
    
            
            for i in range(len(average_accumulation)):
                if count_accumulation[i] >0:
                    new_points.append((average_accumulation[i]/count_accumulation[i]).tolist())
                    
                else:
                    new_points.extend(sample(rest_of_number,1))
                 
                
    #    new_100=iteration_each_points(rest_of_number,new_points)
                
            return new_points
    
    
    
        def check_equal(new_100,new_points):
            
            if (new_100 == np.array(new_points)).all() == False:
            #        count_iteration =count_iteration+1
            #        print(count_iteration)
            
                new_points=iteration_inside(new_points)
    
            return new_points
         
        
        def again(old):
            after_interaion=iteration_inside(old)
            final_point = check_equal(old,after_interaion)
            if (final_point == np.array(after_interaion)).all() == False:
                again(final_point)
            
            return final_point
        
        
        
        new_points=iteration_inside(new_100)
        
        next_new_100 = check_equal(new_100,new_points)
        
        final_point = again(next_new_100)
    #    
    #    third_points=iteration_inside(final_point)
    #    check_equal(final_point,third_points)

            
        return pd.DataFrame(final_point)
    
    return iteration_each_points_second(rest_of_number,first_100)
    




import matplotlib.pyplot as plt




all_hungry_data_100=change_every_parameter(100,all_hungry_data)
all_50_data_100=change_every_parameter(100,all_50_data)
all_80_data_100=change_every_parameter(100,all_80_data)



plt.figure()
plt.scatter(all_hungry_data_100[0],all_hungry_data_100[1],color='red',label = 'hungry leader',marker = '*',s=100)
plt.legend()
plt.scatter(all_50_data_100[0],all_50_data_100[1],color='orange',label = 'fifty leader',marker = '*',s=100)
plt.legend()
plt.scatter(all_80_data_100[0],all_80_data_100[1],color='blue',label = 'eight leader',marker = '*',s=100)
plt.legend()
#plt.title('Four peoples leaders')
plt.title('Four people all data with 100 leaders')


all_hungry_data_100['class'] = 0
all_50_data_100['class'] = 1
all_80_data_100['class'] = 2

all_100_data = pd.concat([all_hungry_data_100,all_50_data_100,all_80_data_100],ignore_index=True)








all_hungry_data_150=change_every_parameter(150,all_hungry_data)
all_50_data_150=change_every_parameter(150,all_50_data)
all_80_data_150=change_every_parameter(150,all_80_data)

i=0
j=1

plt.figure()
plt.scatter(all_hungry_data_150[i],all_hungry_data_150[j],color='red',label = 'hungry leader',marker = '*',s=100)
plt.legend()
plt.scatter(all_50_data_150[i],all_50_data_150[j],color='orange',label = 'fifty leader',marker = '*',s=100)
plt.legend()
plt.scatter(all_80_data_150[i],all_80_data_150[j],color='blue',label = 'eight leader',marker = '*',s=100)
plt.legend()
#plt.title('Four peoples leaders')
plt.title('Four people all data with 150 leaders')

#all_hungry_data_150['class'] = 0
#all_50_data_150['class']=1
#all_80_data_150['class']=2
#all_150 = pd.concat([all_hungry_data_150,all_50_data_150,all_80_data_150],ignore_index=False)
#audio_set_function.fisher_score(np.array(all_150)[:,0:2],all_150['class'].values)


all_hungry_data_200=change_every_parameter(200,all_hungry_data)
all_50_data_200=change_every_parameter(200,all_50_data)
all_80_data_200=change_every_parameter(200,all_80_data)

plt.figure()
plt.scatter(all_hungry_data_200[0],all_hungry_data_200[1],color='red',label = 'hungry leader',marker = '*',s=100)
plt.legend()
plt.scatter(all_50_data_200[0],all_50_data_200[1],color='orange',label = 'fifty leader',marker = '*',s=100)
plt.legend()
plt.scatter(all_80_data_200[0],all_80_data_200[1],color='blue',label = 'eight leader',marker = '*',s=100)
plt.legend()
#plt.title('Four peoples leaders')
plt.title('Four people all data with 200 leaders')

'''-------------------------'''
#record_x_hungry=[]
#record_y_hungry=[]
#
#record_x_50=[]
#record_y_50=[]
#
#record_x_80=[]
#record_y_80=[]
#
#limit_x = 800000
#limit_y = 230000
#for i in range(len(all_hungry_data_150[0])):
#    if all_hungry_data_150[0][i] < limit_x and all_hungry_data_150[1][i]< limit_y:
#        record_x_hungry.append(all_hungry_data_150[0][i])
#        record_y_hungry.append(all_hungry_data_150[1][i])
#    if all_50_data_150[0][i]<limit_x and all_50_data_150[1][i]<limit_y:
#        record_x_50.append(all_50_data_150[0][i])
#        record_y_50.append(all_50_data_150[1][i])
#    if all_80_data_150[0][i]<limit_x and all_80_data_150[1][i]<limit_y:
#        record_x_80.append(all_80_data_150[0][i])
#        record_y_80.append(all_80_data_150[1][i])
        

#record_all_hungry_150 =pd.DataFrame([record_x_hungry,record_y_hungry]).T
#record_all_50_150 =pd.DataFrame([record_x_50,record_y_50]).T
#record_all_80_150 =pd.DataFrame([record_x_80,record_y_80]).T




#plt.figure()
#plt.scatter(record_all_hungry_150[0],record_all_hungry_150[1],color='red',label = 'hungry leader',marker = '*',s=100)
#plt.legend()
#plt.scatter(record_all_50_150[0],record_all_50_150[1],color='orange',label = 'fifty leader',marker = '*',s=100)
#plt.legend()
#plt.scatter(record_all_80_150[0],record_all_80_150[1],color='blue',label = 'eight leader',marker = '*',s=100)
#plt.legend()
##plt.title('Four peoples leaders')
#plt.title('Four people all data with red64 orange74 blue104 leaders')
#
#
#record_all_hungry_150['class'] = 0
#record_all_50_150['class'] = 1
#record_all_80_150['class'] = 2
#
#record_all_150 = pd.concat([record_all_hungry_150,record_all_50_150,record_all_80_150],ignore_index=True)

'''---------------------'''







import functions
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#functions.smote(record_all_150)
#csv_record_all_150_smote = pd.read_csv('all_data_150_leader_cut_smote.csv')
#

data_random=shuffle(all_100_data)

data_random_=data_random.drop(['class'],axis='columns')

traindata, testdata, target_train, target_test = train_test_split(data_random_, data_random['class'], test_size=0.3)

inputs_train = traindata
inputs_test = testdata

functions.SVM(inputs_train,target_train,inputs_test,target_test)  
functions.DT(inputs_train,target_train,inputs_test,target_test)
functions.KNN(inputs_train,target_train,inputs_test,target_test)




#color = []
#for i in range(len(data_random)):
#    if(data_random['class'].values[i]) ==0:
#        color.append('red')
#    elif(data_random['class'].values[i]) ==1:
#        color.append('orange')
#    else:
#         color.append('blue')
#plt.figure()
#plt.scatter(data_random['0'],data_random['1'],c= color,marker='*')










'''---------------------'''
#hungry_all_400 = second_100
#fifty_all_400 = second_100
#eighty_all_400 = second_100



#hungry_all_30 = second_100
#fifty_all_30 = second_100
#eighty_all_30 = second_100


#hungry_all_100 = pd.DataFrame(hungry_all_100)
#fifty_all_100 = pd.DataFrame(fifty_all_100)
#eighty_all_100 = pd.DataFrame(eighty_all_100)





plt.figure()

plt.scatter(all_hungry_data['344~689'],all_hungry_data['689~1033'],color='red',label = 'All people hungry')
plt.legend()
plt.scatter(all_50_data['344~689'],all_50_data['689~1033'],color='orange',label = 'All people 50%')
plt.legend()
plt.scatter(all_80_data['344~689'],all_80_data['689~1033'],color='blue',label = 'All people 80%')
plt.legend()

plt.figure()
plt.scatter(hungry_all_30[0],hungry_all_30[1],color='red',label = 'hungry leader',marker = '*',s=100)
plt.legend()
plt.scatter(fifty_all_30[0],fifty_all_30[1],color='orange',label = 'fifty leader',marker = '*',s=100)
plt.legend()
plt.scatter(eighty_all_30[0],eighty_all_30[1],color='blue',label = 'eight leader',marker = '*',s=100)
plt.legend()
#plt.title('Four peoples leaders')
plt.title('Four people all data with 30 leaders')


plt.figure()
plt.scatter(hungry_all_400[0],hungry_all_400[1],color='red',label = 'hungry leader',marker = '*',s=100)
plt.legend()
plt.scatter(fifty_all_400[0],fifty_all_400[1],color='orange',label = 'fifty leader',marker = '*',s=100)
plt.legend()
plt.scatter(eighty_all_400[0],eighty_all_400[1],color='blue',label = 'eight leader',marker = '*',s=100)
plt.legend()
#plt.title('Four peoples leaders')
plt.title('Four people all data with 400 leaders')








hungry_all_100['class']=0
fifty_all_100['class']=1
eighty_all_100['class'] = 2
leaders_concat = pd.concat([hungry_all_100,fifty_all_100,eighty_all_100],ignore_index=True)


leaders_concat.to_csv('four_people_leaders_no_cosine_similarity.csv',index=0)





score=audio_set_function.fisher_score(leaders_concat.values[:,0:8],leaders_concat['class'].values)


plt.title('Four people all data with leaders')




hungry_all_100














##Final_Result = np.array(second_100)
#Final_Result_2 = np.array(second_100)
#Final_Result = pd.DataFrame(Final_Result)
#Final_Result_2 = pd.DataFrame(Final_Result_2)



#plt.scatter(Final_Result[0],Final_Result[1],color='red')
#plt.scatter(Final_Result_2[0],Final_Result_2[1],color='blue')
#plt.title('Iteration two times validation (All hungry)')





    







































#fetch_new_row(Minkai_hungry_0516)
#fetch_new_row(Minkai_50_0516)
#fetch_new_row(Minkai_80_0516)
#
#fetch_new_row(Peiwen_hungry_0518)
#fetch_new_row(Peiwen_50_0518)
#fetch_new_row(Peiwen_80_0518)
#
#
#  
#        
#Minkai_hungry_0516 = pd.DataFrame(Minkai_hungry_0516)
#Minkai_50_0516 = pd.DataFrame(Minkai_50_0516)
#Minkai_80_0516 = pd.DataFrame(Minkai_80_0516)
#
#Peiwen_hungry_0518 = pd.DataFrame(Peiwen_hungry_0518)
#Peiwen_50_0518 =pd.DataFrame(Peiwen_50_0518)
#Peiwen_80_0518 =pd.DataFrame(Peiwen_80_0518)
#
#
#        
#Minkai_hungry_0516['class'] = 0
#Minkai_50_0516['class'] = 1
#Minkai_80_0516['class']=2
#
#Peiwen_hungry_0518['class'] = 0
#Peiwen_50_0518['class']=1
#Peiwen_80_0518['class']=2
#
#Minkai_concat_0_50_80 = pd.concat([Minkai_hungry_0516,Minkai_50_0516,Minkai_80_0516],ignore_index=True)
#Minkai_concat_0_50_80 = np.array(Minkai_concat_0_50_80)    
#
#
#Peiwen_concat_0_50_80=pd.concat([Peiwen_hungry_0518,Peiwen_50_0518,Peiwen_80_0518],ignore_index=True)
#Peiwen_concat_0_50_80 = np.array(Peiwen_concat_0_50_80)  
#
#audio_set_function.fisher_score(Minkai_concat_0_50_80[:,0:8],Minkai_concat_0_50_80[:,8])
#audio_set_function.fisher_score(Peiwen_concat_0_50_80[:,0:8],Peiwen_concat_0_50_80[:,8])
#
#
##Minkai_concat_0_50_80 = pd.DataFrame(Minkai_concat_0_50_80)
##Peiwen_concat_0_50_80 = pd.DataFrame(Peiwen_concat_0_50_80)
#
#
#
#color_Minkai = color_denote(np.array(Minkai_concat_0_50_80))
#color_Peiwen = color_denote(np.array(Peiwen_concat_0_50_80))
#
#
#plt.figure()
#
#plt.scatter(Minkai_concat_0_50_80[:,3],Minkai_concat_0_50_80[:,4],c=color_Minkai,marker='*',label = 'Minkai')
#plt.scatter(Peiwen_concat_0_50_80[:,3],Peiwen_concat_0_50_80[:,4],c=color_Peiwen,marker='v',label='Peiwen')
#
#plt.legend()
#
#
#plt.figure()
#plt.scatter(Peiwen_concat_0_80[:,3],Peiwen_concat_0_80[:,4],c=color)



