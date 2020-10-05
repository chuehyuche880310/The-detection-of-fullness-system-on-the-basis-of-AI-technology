# -*- coding: utf-8 -*-
"""
Created on Sun May  3 18:38:34 2020

@author: ChuehYuChe
"""

def find_cosin_value(file1,index1,file2,index2):
    import numpy as np
    value1=0
    value2=0
    value1 = file1[index1]
    value2 = file2[index2]
    
    cos_sim = np.dot(value1,value2) / (np.linalg.norm(value1) * np.linalg.norm(value2))
    
    
    print(abs(cos_sim))
    return cos_sim
#-----------
def find_value(file1,index):
    value=[]
    value = file1[index]
    
    return value

def find_cosine(value1,value2):
    import numpy as np
    cos_sim = np.dot(value1,value2)/(np.linalg.norm(value1) * np.linalg.norm(value2))
    return abs(cos_sim)


A=98
B=195
a=23
b=73
temp_index=[]
name=['80 bowel sound: '+str(A),'80 bowel sound: '+str(B),'30 NOT: '+str(a),'30 NOT: '+str(b)]
bowl_sound_1= find_value(Peiwen_80_nor,A)
bowl_sound_2= find_value(Peiwen_80_nor,B)
not_bowl_sound_1= find_value(Peiwen_80_nor,a)
not_bowl_sound_2= find_value(Peiwen_80_nor,b)

temp_index.append(bowl_sound_1)
temp_index.append(bowl_sound_2)
temp_index.append(not_bowl_sound_1)
temp_index.append(not_bowl_sound_2)


matrix = np.zeros([len(temp_index),len(temp_index)])
for i in range(len(temp_index)):
    for j in range(len(temp_index)):
        matrix[i,j] = find_cosine(temp_index[i],temp_index[j])
   
A1 = pd.DataFrame(matrix, index=[name[0],name[1],name[2],name[3]],columns=[name[0],name[1],name[2],name[3]])


#----------------------------------------------------------------



  
find_cosin_value(Peiwen_hungry_nor,14,Peiwen_hungry_nor,107) #餓腸腸
find_cosin_value(Peiwen_hungry_nor,14,Peiwen_hungry_nor,42) #餓腸不是腸
find_cosin_value(Peiwen_hungry_nor,107,Peiwen_hungry_nor,71) #餓腸不是腸
find_cosin_value(Peiwen_hungry_nor,71,Peiwen_hungry_nor,42) #餓不是腸不是腸

#-----------------------------

find_cosin_value(Peiwen_30_nor,7,Peiwen_30_nor,13) #30腸-腸
find_cosin_value(Peiwen_30_nor,77,Peiwen_30_nor,13) #30不是腸-腸
find_cosin_value(Peiwen_30_nor,113,Peiwen_30_nor,7) #30不是腸-腸
find_cosin_value(Peiwen_30_nor,77,Peiwen_30_nor,113) #30不是腸-不是腸 

find_cosin_value(Peiwen_50_nor,44,Peiwen_50_nor,6) #50腸-腸
find_cosin_value(Peiwen_50_nor,47,Peiwen_50_nor,6) #50不是腸-腸
find_cosin_value(Peiwen_50_nor,103,Peiwen_50_nor,44) #50不是腸-腸
find_cosin_value(Peiwen_50_nor,47,Peiwen_50_nor,103) #50不是腸-不是腸 

find_cosin_value(Peiwen_80_nor,17,Peiwen_80_nor,98) #50腸-腸
find_cosin_value(Peiwen_80_nor,23,Peiwen_80_nor,98) #50不是腸-腸
find_cosin_value(Peiwen_80_nor,73,Peiwen_80_nor,17) #50不是腸-腸
find_cosin_value(Peiwen_80_nor,23,Peiwen_80_nor,73) #50不是腸-不是腸 