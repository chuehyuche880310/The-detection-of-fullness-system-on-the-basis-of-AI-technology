# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:48:33 2020

@author: 沛紋
"""
import numpy as np
import pandas as pd
def deleteDuplicatedElementFromList2(chose,temp):
#    temp = original
    
    for i in chose:
        temp.remove(i)
        
    return temp


#def random_fetch_list()

from random import sample
#global choose,standard_point,rest_of_number

standard_point = 100
store_list = np.zeros(standard_point)

dimension = 8

all_hungry_data = pd.read_csv('all_hungry_data.csv')
all_hungry_process=all_hungry_data.values[:,0:dimension]
all_hungry_process=all_hungry_process.tolist()

temp=all_hungry_process[0:len(all_hungry_process)]
choose = sample(all_hungry_process,100)  #從所有List當中，隨機挑選100個list 

rest_of_number = deleteDuplicatedElementFromList2(choose,temp)
    

#count_accumulation = np.zeros(standard_point) #累積跑一次的分數

#average_accumulation = []






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
        
            next_new_100=iteration_inside(new_points)

        return next_new_100
     
    
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
    
    return final_point



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
import matplotlib.pyplot as plt

first_100 = iteration_each_points_first(rest_of_number,choose)
second_100 = iteration_each_points_second(rest_of_number,first_100)

#Final_Result = np.array(second_100)
Final_Result_2 = np.array(second_100)
Final_Result = pd.DataFrame(Final_Result)
Final_Result_2 = pd.DataFrame(Final_Result_2)
plt.scatter(Final_Result[0],Final_Result[1],color='red')
plt.scatter(Final_Result_2[0],Final_Result_2[1],color='blue')
plt.title('Iteration two times validation (All hungry)')




