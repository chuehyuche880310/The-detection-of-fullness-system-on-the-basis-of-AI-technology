# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 03:03:58 2020

@author: ChuehYuChe
"""

from matplotlib import pyplot as plt    
import numpy as np
from sklearn.datasets import make_moons
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot
from keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle



def KNN(traindata,target_train,testdata,target_test):
    from sklearn.externals import joblib
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(traindata,target_train)
    y_true = target_test
    y_pred = knn.predict(testdata)
    Draw_Confusion_Matrix(y_pred,y_true)
    cross_validation(knn,traindata,target_train)
    ROC_multiple(y_true,y_pred)
    joblib.dump(knn , r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\main\20200607\KNN_no_cut_ROC68.plk')
#    draw_boundary(knn,'KNN')
#    return y_pred
    
    
    
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
    
def Draw_Confusion_Matrix(y_true,y_pred):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    from sklearn import metrics
    sns.set()
    f,ax = plt.subplots()
    C2 = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
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
    

def NN(traindata,testdata,target_train,target_test):

    inputs_train = traindata
    inputs_test = testdata
    #模型搭建阶段
    model= Sequential()
    model.add(Dense(9, activation='relu', input_dim=np.size(inputs_train,1)))
#    model.add(Dense(100, activation='relu'))
#    model.add(Dense(50, activation='relu'))
    model.add(Dense(7, activation='relu'))
#    model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    model.summary()
    # simple early stopping
    #patience：能夠容忍多少個epoch內都沒有improvement。
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = ModelCheckpoint(r"C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\main\20200517\5features_skew_whole_person_0518_smote.h5", monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    from keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.1, cooldown=0, min_lr=0.0001)
    train_history = model.fit(inputs_train, target_train,  validation_split=0.2, epochs=3000,batch_size=3000,verbose=1, callbacks=[reduce_lr,es,mc])
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

#def SVM(inputs_train,target_train,inputs_test,target_test):
#    from sklearn.svm import SVC
#    svm = SVC(kernel='rbf', probability=True)
#    svm.fit(inputs_train,target_train)
#    y_pred = svm.predict(inputs_test)
#    y_true = target_test
#    Draw_Confusion_Matrix(y_true,y_pred)
#    cross_validation(svm,data_random_,data_random['class'])
#    ROC_multiple(y_pred,y_true)
#    
#SVM(inputs_train,target_train,inputs_test,target_test)  

def SVM(inputs_train,target_train,inputs_test,target_test):
    from sklearn.svm import SVC
    from sklearn.externals import joblib #jbolib模块
    svm = SVC(kernel='rbf', probability=True,C=10,gamma=0.1) #minkai 兩類 C=1000,gamma=0.0000001
#    svm = SVC(kernel='poly', probability=True)
    svm.fit(inputs_train,target_train)
    y_pred = svm.predict(inputs_test)
    y_true = target_test
    Draw_Confusion_Matrix(y_true,y_pred)
    ROC_multiple(y_pred,y_true)
#    joblib.dump(svm, 'SVM_Minkai_hungry_80_92p.pkl')

#SVM(inputs_train,target_train,inputs_test,target_test)  

#from sklearn.externals import joblib
#import draw_boundary_new
#model_temp = joblib.load('SVM_Minkai_hungry_80_92p.pkl')
#draw_boundary_new.draw_boundary(model_temp)


from sklearn import tree
def DT(inputs_train,target_train,inputs_test,target_test):
    from sklearn.externals import joblib
    from sklearn import ensemble, preprocessing, metrics
    model = tree.DecisionTreeClassifier()
    model.fit(inputs_train, target_train)
    y_pred = model.predict(inputs_test)
    y_true = target_test
    Draw_Confusion_Matrix(y_pred,y_true)
#    
#    cross_validation(model,data_random_,data_random['class'])
    ROC_multiple(y_pred,y_true)
    
#    joblib.dump(model , "5features_skew_whole_person_0518_smote.plk")

#DT(inputs_train,target_train,inputs_test,target_test) 

def smote(csvdata):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=2)
    train_X = csvdata.drop(['class'],axis='columns')
    train_y = csvdata['class']
    traindata, testdata, target_train, target_test = train_test_split(train_X, train_y, test_size=0.3, random_state=42)    
    
    X_train_res, y_train_res = sm.fit_sample(traindata, target_train)
    
    print("Before OverSampling, counts of label '0': {}".format(sum(target_train==0)))
    print("Before OverSampling, counts of label '1': {}".format(sum(target_train==1)))
    print("Before OverSampling, counts of label '2': {}".format(sum(target_train==2)))
    print("Before OverSampling, counts of label '3': {}".format(sum(target_train==3)))
    print("Before OverSampling, counts of label '4': {} \n".format(sum(target_train==4)))
    
    print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
    
    print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
    print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
    print("After OverSampling, counts of label '2': {}".format(sum(y_train_res==2)))
    print("After OverSampling, counts of label '3': {}".format(sum(y_train_res==3)))
    print("After OverSampling, counts of label '4': {}".format(sum(y_train_res==4)))
    output = pd.DataFrame(X_train_res)
    output['class'] = y_train_res
    output.to_csv(r'C:\Users\ChuehYuChe\Desktop\北科專題StartFrom_20200502\main\20200531\all_data_150_leader_cut_smote.csv',index=0)
def draw_boundary(model,title):
    list_i = []
    list_j = []
    
    for i in range(0,800000,10000):
        
        for j in range(0,230000,10000):
            list_i.append(i)
            list_j.append(j)
            
    #        plt.scatter(i,j)
            
    all_X_100_df = pd.DataFrame(list_i,columns=['X'])
    all_Y_100_df = pd.DataFrame(list_j,columns=['Y'])

    
    all_100_data = pd.concat([all_X_100_df,all_Y_100_df,],axis=1)

    from sklearn.externals import joblib
#    model = joblib.load(model)
    

    y_pred = model.predict(all_100_data)
    
    color = []
    for i in y_pred:
        if i == 0:
            color.append("red")
        elif i == 1:
            color.append("orange")
        elif i==2:
            color.append("blue")

    
    plt.figure()
    plt.scatter(all_100_data['X'],all_100_data['Y'],c = color)
    plt.title(title)
#smote(data_random)