import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





dataframe = pd.read_csv('iris.csv')
#print(dataframe.head())
X = dataframe.iloc[:,0:4].values
y = dataframe.iloc[:,4].values
y=y.reshape(-1,1)
#print(X.shape)
#print(y.shape)

def cross_validation(X,y,k):
    i_test = np.arange(((k-1)*15),k*15)
    i_train = []
    #print(i_test)
    #print(i_train)
    for i in range(0,150):
        if i not in i_test:
            i_train.append(i)
    X_train = X[i_train,:]
    X_test = X[i_test,:]
    y_train = y[i_train,:]
    y_test = y[i_test,:]
    return X_train,X_test,y_train,y_test

for i in range(0,150):
    if y[i] == 'Iris-setosa':
        y[i] = 1
    if y[i] == 'Iris-virginica':
        y[i] = 3
    if y[i] == 'Iris-versicolor':
        y[i] = 2

#print(y[100:])
accu = []
beta_list = []
for intertion in range(1,11):
    X_train,X_test,y_train,y_test = cross_validation(X,y,intertion)
    #print("The shape "+ str(X_train.shape))

    #(ATA)-1ATY
    #print(X_train.T.shape)
    inverse = np.linalg.inv(np.dot(X_train.T,X_train))
    #print(inverse.shape)
    #print(inverse[:5,:])
    sub_result = np.dot(inverse,X_train.T)
    #print(sub_result[:5,:])
    #print(sub_result.shape)
    beta = np.dot(sub_result,y_train)
    #print(beta[:5,:])
    #print(beta.shape)
    beta_list.append(beta)
    #y_-x_b2
    #print(beta)

    #Analysis
    y_pred = np.dot(X_test,beta)

    #print(y_pred)
    #print("The shape of y is "+ str(y_pred.shape))

    for i in range(0,y_test.shape[0]):
        if y_pred[i]<=1.2:
            y_pred[i] = 1
        if y_pred[i]<=2.5 and y_pred[i]>1.2:
            y_pred[i]= 2
        if y_pred[i]<=3.6 and y_pred[i]>2.5:
            y_pred[i]=3
        #y[i] = int(y[i])

    #print(y_pred)


    y_true = np.empty((150,1),dtype=int)
    #print(y_true.shape)
    #y_pred2 = np.ones((150,1))
    for i in range(0,150):
        y_true[i] = int(y[i])
    #print(y_pred.shape)
    #print(y_true)
    #print(y_pred)

    '''
    from sklearn.metrics import confusion_matrix
    print(type(y))
    cm = confusion_matrix(y_true,y_pred)
    print(cm)
    '''
    count = 0
    for i in range(0,y_test.shape[0]):
        if y_test[i]==y_pred[i]:
            count =count +1
    #print("The accuracy is "+str(count/15))
    #print("The unique classes "+str(np.unique(y_pred)))
    accu.append((count/y_test.shape[0]))
print("The accuracy for all the folds\n")
print(accu)

print("The average accuracy is \n")
sum = 0
for i in range(0,len(accu)):
    sum = sum + accu[i]
print((sum/10)*100)
print("The List of Betas\n")
#print(beta_list)
for i in range(0,len(beta_list)):
    print("Fold "+str(i+1)+":\n")
    print(beta_list[i][0])
    print(beta_list[i][1])
    print(beta_list[i][2])
    print(beta_list[i][3])