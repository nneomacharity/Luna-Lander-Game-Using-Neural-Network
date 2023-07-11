#!/usr/bin/env python
# coding: utf-8

# In[529]:


import pandas as pd
import numpy as np


# In[530]:


### Pre-Processing of Data


# In[531]:


# importing and reading the file


# In[532]:


pp = pd.read_csv(r"C:\Users\DELL'\Desktop\Masters Assignment\Game Data.csv")


# In[533]:


# renaming the columns


# In[534]:


pp.columns= ['X_Dist', 'Y_Dist','X_Velo', 'Y_Velo']
pp


# In[535]:


pp.head()


# In[536]:


pp.describe()


# In[537]:


#Checking for null values/missing data


# In[538]:


pp.isnull().sum()


# In[539]:


# normalizing for a range of (0.0 to 1.0) using min-max scaler =(x-min(x))/(max(x)-min(x))


# In[540]:


pf = ((pp-pp.min()) / (pp.max() -pp.min()))
pf


# In[541]:


# Checking if data has been scaled


# In[542]:


(pf.min(), pf.max())


# In[543]:


pf.describe()


# In[544]:


pf.info()


# 

# In[545]:


# shuffling the data to avoid biasness using sample function


# In[546]:


pf = pf.sample(frac=1)


# In[547]:


# identifying the inputs and outputs data using the locate function 


# In[548]:


inPut = pf.iloc[:,0:2]
inPut


# In[549]:


output = pf.iloc[:,2:4]
output


# In[550]:


# Splitting the data into a Train and test sets using train_test_split function with a 0.80 ratio


# In[551]:


from sklearn.model_selection import train_test_split


# In[552]:


Big_Train_x, test_x, Big_Train_y,  test_y = train_test_split(inPut,output,train_size =0.8)


# In[553]:


Big_Train_x.shape


# In[554]:


Big_Train_y.shape


# In[555]:


test_x.shape


# In[556]:


test_y.shape 


# In[557]:


# Splitting the Big Train set, further into a  validation and a new train set with a 0.10 ratio


# In[558]:


train_x, vali_x, train_y, vali_y = train_test_split(Big_Train_x,Big_Train_y,test_size =0.10)


# In[559]:


train_x.shape


# In[560]:


train_y.shape


# In[561]:


vali_x.shape


# In[562]:


vali_y.shape


# In[563]:


# Hence: 
# Training Datas = train_x and train_y 
# Validation Datas =vali_x and vali_y 
# Test datas = test_x and test_y
# Thereafter, transposing all these datas


# In[564]:


train_x = train_x.T
vali_x = vali_x.T
train_y =train_y.T
vali_y = vali_y.T
test_x = test_x.T
test_y = test_y.T


# In[565]:


pf = np.array(pf)


# In[566]:


#  Generating all the hyperparameters in building the neural network (2 inputs, 1 hidden layer and 2 outputs)


# In[567]:


# Initializing parameters using random values between-1 and 1 where wt= weight, bs = bias, f = dot product, R=sigmoid


# In[595]:


def init_params():
    np.random.seed(4)
    wt1 = np.random.randn(2, 4)
    bs1 = np.random.randn(2)
    wt2 = np.random.randn(4, 2)
    bs2 = np.random.randn(2,)
    return wt1, bs1, wt2, bs2
lamda = 0.5
learning rate = 0.5
iterations = 10


# In[601]:


params = init_params()


# In[606]:


wt1= params[0]
bs1 = params[1]
wt2= params[2]
bs2= params[3]

params[0], params[1], params[2], params[3]


# In[573]:


#defining the sigmoid function and getting the derivative of it


# In[614]:


def sigmoid(f1):
        return 1/(1+np.exp(-f1))
    
def der_sigmoid(f1):
    return sigmd(f1)*(1- sigmoid(f1))


# In[578]:


#performing the forward propagation


# In[623]:


def forward_propagation(wt1, bs1, wt2, bs2, train_x):
    for i in range(0, len(train_x)):
        f1 = wt1.dot(train_x)+ bs1
        R1 = sigmoid(f1) 
        f2=  wt2.dot(f1) + bs2
        R2=  sigmoid(f2)
    return f1, R1, f2, R2


# In[571]:


#calculating the loss function


# In[598]:


def Loss_function(predict_output, train_y):
    return sqrt(((predict_output, train_y[])** 2).mean())

#where 
predict_output =[]
for i in range (0, len(f2)):
    predict_output_i = 1/1 + np.exp(-0.5 *[f2])


# In[580]:


#performing the backward propagation


# In[628]:


def backward_propagation(f1, R1, f2, R2, train_x, train_y):
    m = train_y.size
    der_f2 = R2 - (m * max() + 1)
    der_wt2 = 1 / m * der_f2.dot(R1.T)
    der_bs2 = 1 / np.sum(der_f2, 2)
    der_f1 = wt2.T.dot(der_f2) * der_sigmoid(f):
    der_wt1 = 1 / m * der_f1.dot(x.T)
    der_bs1 = 1 / np.sum(der_f1, 2)
    return der_wt1, der_bs1, der_wt2, der_bs2


# In[584]:


#updating to the newly generated parameters


# In[585]:


def update_parameters(wt1, bs1, wt2, bs2, der_wt1, der_bs1, der_wt2, der_bs2, alpha):
    wt1 = wt1 - alpha * der_wt1
    wt2 = wt2 - alpha * der_wt2
    bs1 = bs1 - alpha * der_bs1
    bs2 = bs2 - alpha * der_bs2
    return wt1, wt2, bs1, bs2


# In[587]:


def gradient_descent(x, y, alpha, iterations,):
    wt1, bs1, wt2, bs2, = init_params()
    for i in range (iterations):
        (f1, R1, f2, R2,) = forward_propagation(wt1, bs1, wt2, bs2, train_x)
        (der_wt1, der_bs1, der_wt2, der_bs2) = backward_propagation(f1, R1, f2, R2, train_x, train_y)
        (wt1,bs1, wt2,  bs2) = update_parameters(wt1, bs1, wt2, bs2, der_wt1, der_bs1, der_wt2, der_bs2, alpha)
    return wt1, wt2, bs1, bs2


# In[582]:


#indicating for predictions and accuracy


# In[583]:


def get_predictions(R2):
    return np.argmax(R2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


# In[588]:


#inputing the network into the training data set


# In[629]:


def fit ()
 def fit(train_x, train_y):
           train_x = x
            train_y = y


# In[630]:


wt1, wt2, bs1, bs2 = gradient_descent(train_x and train_y, 0.50, 100)


# In[ ]:




