# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 01:48:30 2020

@author: 沛芯&沛纹
"""

import pandas as pd

#all_new_hungry_data = pd.read_csv('hungry_nor_abs_80.csv')
#all_new_full_data = pd.read_csv('full_nor_abs_80.csv')
#data = pd.concat([all_new_hungry_data,all_new_full_data])
#data.to_csv('newdata_nor.csv',index = 0)
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.model_selection import train_test_split

#data_random = shuffle(data)
#data_random.to_csv('newdata_shuffle_abs80.csv',index = 0)
data_random = pd.read_csv('mydataset_each_diff_threshold.csv')

#data_random = pd.read_csv(r'G:\Proiect_fullness\收集之資料\0126\threshold8_round.csv')

#data_random = pd.read_csv('newdata_shuffle_abs80_label123.csv')

#data_random = pd.read_csv(r'G:\fullness_desktop\hungry_che_full_pei_data\allnewdata_shuffle_abs80_pei_f_che_H.csv')
data_random_=data_random.drop(['class'],axis='columns')

traindata, testdata, target_train, target_test = train_test_split(data_random_, data_random['class'], test_size=0.3, random_state=42)

inputs_train = traindata
inputs_test = testdata

def ROC_multiple(y_true,y_pred):

    import numpy as np
    from sklearn.metrics import roc_curve, roc_auc_score, auc
    from scipy import interp
    from itertools import cycle
    from sklearn.preprocessing import label_binarize
#    # Compute ROC curve and ROC area for each class

    nb_classes=5
    # Binarize the output
    y_true = label_binarize(y_true, classes=[i for i in range(nb_classes)])
    y_pred = label_binarize(y_pred, classes=[i for i in range(nb_classes)])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # one vs rest方式计算每个类别的TPR/FPR以及AUC
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= nb_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['red', 'green', 'blue'])
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    

def PCA(data,target):
    from sklearn.decomposition import PCA
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 讀取資料集的部分跳過
    x_data = np.array(data) # x 為四維的資料
#    y_data = np.array(target) # y 為此資料的類別
    
    # 執行 PCA
    pca = PCA(n_components=2) # n_components 為要降到的維度，這邊降為二維
    pca.fit(x_data) # 使用 x_data 為來源資料
    result = pca.transform(x_data) # 使用 transform() 即可取得降維後的陣列
    result = pd.DataFrame(result)
#    plt.scatter(result[:,0], result[:,1], c=y_data, s=25, alpha=0.4, marker='o')
    return result
    # 使用 matplotlib 將結果繪出
    # 前兩個參數是各個點的 x,y 座標值
    # c 是按照每筆資料的類別數值(1~9)，自動著色

    
#PCA_result = PCA(data_random_,data_random['class'])
#
#PCA_result['class'] = data_random['class']
#PCA_result.to_csv('PCA_label5.csv',index = 0)

def Draw_Confusion_Matrix(y_true,y_pred):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    from sklearn import metrics
    sns.set()
    f,ax = plt.subplots()
    C2 = confusion_matrix(y_true, y_pred, labels=[0, 1, 2,3,4])
#    C2 = confusion_matrix(y_true, y_pred, labels=[0, 1])
#    print (C2) # 打印出來看看 
    sns.heatmap(C2,annot=True, fmt='g',ax=ax) # 畫熱力圖
    ax.set_title( ' confusion matrix ' ) # 標題 
    ax.set_xlabel( ' predict ' ) # x軸 
    ax.set_ylabel( ' true ' )
    accuracy=metrics.accuracy_score(y_true, y_pred)
    print("accuracy: ",accuracy)
    print (metrics.classification_report(y_true, y_pred))
    
    
def cross_validation(model,inputs_train,target_train):
    from sklearn.model_selection import cross_val_score
    from sklearn import datasets
    import numpy as np
    scores = cross_val_score(model,inputs_train,target_train,cv=10,scoring='accuracy')
    print(scores)
    print(scores.mean())
    
    
'''SVM: 0.78'''    
def SVM(inputs_train,target_train,inputs_test,target_test):
    from sklearn.svm import SVC
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(inputs_train,target_train)
    y_pred = svm.predict(inputs_test)
    y_true = target_test
    Draw_Confusion_Matrix(y_true,y_pred)
    cross_validation(svm,data_random_,data_random['class'])
    

#SVM(inputs_train,target_train,inputs_test,target_test)
    
'''DT: 0.78''' 
def DT(inputs_train,target_train,inputs_test,target_test):
    from sklearn.externals import joblib
    from sklearn import ensemble, preprocessing, metrics
    model = ensemble.RandomForestClassifier(n_estimators = 100)
    model.fit(inputs_train, target_train)
#    model = tree.DecisionTreeClassifier()
#    model.fit(inputs_train, target_train)
    y_pred = model.predict(inputs_test)
    y_true = target_test
    Draw_Confusion_Matrix(y_true,y_pred)
    
    cross_validation(model,data_random_,data_random['class'])
    ROC_multiple(y_true,y_pred)
    joblib.dump(model , "RF_label5_diff_thre_round_cut25_.plk")

    
#     
#    clf2 = joblib.load("clf.plk")
    
#
DT(inputs_train,target_train,inputs_test,target_test)   


'''--------------------------------NN---------------------------------------'''
from matplotlib import pyplot as plt    
import numpy as np
from sklearn.datasets import make_moons
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot
from keras.models import load_model
def NN(traindata,testdata,target_train,target_test):
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.utils import np_utils
    
    
    inputs_train = traindata
    inputs_test = testdata
    #模型搭建阶段
    model= Sequential()
    model.add(Dense(50, activation='relu', input_dim=np.size(inputs_train,1)))
#    model.add(Dense(100, activation='relu'))
#    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    model.summary()
    # simple early stopping
    #patience：能夠容忍多少個epoch內都沒有improvement。
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = ModelCheckpoint('test.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    from keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.001, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.00001)
    train_history = model.fit(inputs_train, target_train,  validation_split=0.2, epochs=300,batch_size=150,verbose=1, callbacks=[reduce_lr,es,mc])
    # load the saved model
#    saved_model = load_model('best_model.h5')
    # evaluate the model
   
    #train_history = model.fit(x=inputs_train, y=target_train, validation_split=0.2, epochs=10, batch_size=800, verbose=2)  
    
    # 顯示訓練成果(分數)
    scores_ = model.evaluate(inputs_test, target_test)  
    print()  
    print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores_[1]*100.0))  
    
    # 預測(prediction)
    y_pred = model.predict_classes(inputs_test)
    y_true = target_test
    ROC_multiple(y_true,y_pred)
    Draw_Confusion_Matrix(y_true,y_pred)
    plt.figure()
    plt.plot(train_history.history['loss'])  
    plt.plot(train_history.history['val_loss'])  
    plt.title('Train History')  
    plt.ylabel('loss')  
    plt.xlabel('Epoch')  
    plt.legend(['loss', 'val_loss'], loc='upper left')  
    plt.show() 
   
#NN(traindata,testdata,target_train,target_test)

#import pandas as pd
#import seaborn as sn
#import matplotlib.pyplot as plt
#
#corrMatrix = data_random.corr()
#sn.heatmap(corrMatrix, annot=True)
#plt.show()