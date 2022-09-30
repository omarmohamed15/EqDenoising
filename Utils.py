import numpy as np
import h5py
from keras.layers import Input, Dense, Dropout, Add, multiply, GlobalAvgPool1D, Reshape, concatenate, UpSampling1D, Flatten, MaxPooling1D, BatchNormalization, average, Conv1D
from keras.models import Model
import scipy.io
from keras import optimizers
from keras.callbacks import EarlyStopping
import h5py
from math import*
import numpy as np
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import backend as K
import tensorflow as tf
# To measure timing
#' http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")
        
from scipy.signal import butter, lfilter
from scipy import ndimage, misc


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
from scipy.signal import butter, lfilter, lfilter_zi

def butter_bandpass_filter_zi(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y,zo = lfilter(b, a, data, zi=zi*data[0])
    return y

from scipy.signal import butter, lfilter

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 1.
    return data[s<m]

def MASK_LOSS(y_true, y_pred):

    nois = y_true - y_pred
    maskbiS =  1 / (1 + (K.abs(nois)/K.abs(y_pred)))
    maskbiN = (K.abs(nois)/K.abs(y_pred)) / (1+(K.abs(nois)/K.abs(y_pred)))

    return   K.abs(1-maskbiS) + K.abs(maskbiN)   

import keras
def Attention(inp1):
    #filters = int(inp1._keras_shape[-1])
    filters = inp1._keras_shape[-1] 
    
    sa   = (1,filters)
    inp2 = GlobalAvgPool1D()(inp1)
    inp2 = Reshape(sa)(inp2)
    
    x = Dense(filters, activation='relu')(inp2)
    x = Dense(filters, activation='sigmoid')(x)
    x = multiply([inp1,x])
    x = Add()([x,inp1])
    
    return x

def Block(inp,D):  
    
    #inp1 = Flatten()(inp)
    #print(inp,inp1)
    x = Dense(D,activation='relu')(inp)
    #x  = LeakyReLU(0.1)(x)
    x = Dense(D, activation='relu')(x)
    #x  = LeakyReLU(0.)(x)
    #x = Dense(D)(x)
    #x  = LeakyReLU(0.1)(x)
    x = Reshape((x._keras_shape[-1],1))(x)
    
    return x

from keras.layers import Lambda

def compactlayer(y,D):
    #s0, s1, s2, s3 = tf.split(y, num_or_size_splits=4, axis=-1)
    s0 = Lambda(lambda x:x[:,:,0])(y)
    s1 = Lambda(lambda x:x[:,:,1])(y)
    #s2 = Lambda(lambda x:x[:,:,2])(y)
    #s3 = Lambda(lambda x:x[:,:,3])(y)
    #s4 = Lambda(lambda x:x[:,:,4])(y)
    #s5 = Lambda(lambda x:x[:,:,5])(y)
    
    #print(s0)
    B1 = Block(s0,D)
    B2 = Block(s1,D)
    #B3 = Block(s2,D)
    #B4 = Block(s3,D)
    #B5 = Block(s4,D)
    #B6 = Block(s5,D)
    B    = concatenate([B1,B2], axis=-1)
    Batt = Attention(B)
    
    return Batt

def yc_patch(A,l1,l2,o1,o2):

    n1,n2=np.shape(A);
    tmp=np.mod(n1-l1,o1)
    if tmp!=0:
        print(np.shape(A), o1-tmp, n2)
        A=np.concatenate([A,np.zeros((o1-tmp,n2))],axis=0)

    tmp=np.mod(n2-l2,o2);
    if tmp!=0:
        A=np.concatenate([A,np.zeros((A.shape[0],o2-tmp))],axis=-1); 


    N1,N2 = np.shape(A)
    X=[]
    for i1 in range (0,N1-l1+1, o1):
        for i2 in range (0,N2-l2+1,o2):
            tmp=np.reshape(A[i1:i1+l1,i2:i2+l2],(l1*l2,1));
            X.append(tmp);  
    X = np.array(X)
    return X[:,:,0]


def yc_snr(g,f):
    psnr = 20.*np.log10(np.linalg.norm(g)/np.linalg.norm(g-f))
    return psnr


def yc_patch_inv(X1,n1,n2,l1,l2,o1,o2):
    
    tmp1=np.mod(n1-l1,o1)
    tmp2=np.mod(n2-l2,o2)
    if (tmp1!=0) and (tmp2!=0):
        A     = np.zeros((n1+o1-tmp1,n2+o2-tmp2))
        mask  = np.zeros((n1+o1-tmp1,n2+o2-tmp2)) 

    if (tmp1!=0) and (tmp2==0): 
        A   = np.zeros((n1+o1-tmp1,n2))
        mask= np.zeros((n1+o1-tmp1,n2))


    if (tmp1==0) and (tmp2!=0):
        A    = np.zeros((n1,n2+o2-tmp2))   
        mask = np.zeros((n1,n2+o2-tmp2))   


    if (tmp1==0) and (tmp2==0):
        A    = np.zeros((n1,n2))
        mask = np.zeros((n1,n2))

    N1,N2= np.shape(A)
    ids=0
    for i1 in range(0,N1-l1+1,o1):
        for i2 in range(0,N2-l2+1,o2):
            #print(i1,i2)
    #       [i1,i2,ids]
            A[i1:i1+l1,i2:i2+l2]=A[i1:i1+l1,i2:i2+l2]+np.reshape(X1[:,ids],(l1,l2))
            mask[i1:i1+l1,i2:i2+l2]=mask[i1:i1+l1,i2:i2+l2]+ np.ones((l1,l2))
            ids=ids+1


    A=A/mask;  
    A=A[0:n1,0:n2]

    return A