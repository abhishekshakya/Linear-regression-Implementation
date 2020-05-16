import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def hypothesis(X,theta):
    y_ = np.dot(X,theta)
    return y_


def gradient(X,Y,theta):
    n = X.shape[1]
    m = X.shape[0]
    y_ = hypothesis(X,theta)
    # print(y_)
    grad = np.ones((X.shape[1],1))
 
    for j in range(n):
     	grad[j][0] = np.sum((y_ - Y)*X[:,j].reshape(-1,1))


    return grad/m


def gradient_descent(X,Y,learning_rate = 0.2, max_steps = 300):
    theta = np.zeros((X.shape[1],1))
    error_list = []
    for i in range(max_steps):
        grad = gradient(X,Y,theta)
        e = mse(X,Y,theta)
        error_list.append(e)
        # print(grad.shape)
        theta = theta - learning_rate*grad
        # break
        
    return theta,error_list


def mse(X,Y,theta):
    y_ = hypothesis(X,theta)
    m = X.shape[0]
    return np.sum((y_ - Y)**2)/m


def r2_score(y,y_):
	s = 1 - np.sum((y_-y)**2)/np.sum((y-np.mean(y))**2)
	return s*100



df = pd.read_csv('Train.csv')

# print(df.columns)
# print(df.shape)

X =  df.values[:,:-1]
Y = df.values[:,-1].reshape(X.shape[0],1)

print(X.shape)
print(Y.shape)

# print(df.describe())
#dataset is already normalised (mean = 0, std = 1)

#lets add col=[1,1,1,1...] in X

ones = np.ones((X.shape[0],1))
X = np.concatenate((ones,X),axis=1)
# print(X[:4,:4])
#okk ones col is added

theta,error = gradient_descent(X,Y)

plt.plot(error)
plt.show()
#showing how error is reducing

# y_ = hypothesis(X,theta)
# print(r2_score(Y,y_))###yey 96% accuracy



##----------------------Testing---------------------

test = pd.read_csv('Test.csv').values

ones = np.ones((test.shape[0],1))
test = np.concatenate((ones,test),axis=1)
# print(test)
y_ = hypothesis(test,theta)

pred_df=pd.DataFrame(y_)
pred_df.to_csv('pred.csv',index_label=['id','target'],index = True)



