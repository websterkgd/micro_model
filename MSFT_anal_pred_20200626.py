# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#clear environment
from IPython import get_ipython;   
get_ipython().magic('reset -sf')
from IPython import get_ipython; 

#import packages for data analysis 
import pandas as pd
import os
import numpy as np 
import matplotlib.pyplot as plt 
from collection import deque

#change directory to directory with data
os.chdir('D:\\a_Desktops_Git\\Current\\SpringBoard')

#import the microsoft stock data
MSFT_d = pd.read_csv('MSFT_20100622-20200626.csv')

#create a new column
MSFT_d['log_close'] = np.log10(MSFT_d['Adj Close'])

#quick plot to check 
plt.scatter(MSFT_d.index, MSFT_d['log_close'], 
            alpha = 0.7)
plt.title('MSFT trading price')
plt.xlabel('Session Number')
plt.ylabel('Log10(Closing Price ($USD))')
plt.show()

#create a column for the log of the daily change
ldc = [0]
for i in list(range(0,len(MSFT_d['Adj Close'])-1)):
    v = MSFT_d['log_close'][i+1]-MSFT_d['log_close'][i]
    ldc.append(v)

MSFT_d['ldc'] = ldc
MSFT_d['ldc'][0] = np.mean(MSFT_d.ldc)

#quick plot to check 
plt.scatter(MSFT_d.index, MSFT_d['ldc'], 
            alpha = 0.3)
plt.show()

#normalize the data 
MSFT_d['nldc'] = (MSFT_d['ldc']-np.mean(MSFT_d['ldc']))/np.std(MSFT_d['ldc'])

#pull in stock ratings
MSFT_r = pd.read_csv('MSFT_ratings.csv')

#pull out firm from firm_rating
Firm = []
for i in MSFT_r['Firm_Rating']:
    sep = ':'
    v = i.split(sep, 1)[0]
    Firm.append(v)
    
MSFT_r['Firm'] = Firm

#Rename Bank oferica to Bank of America
MSFT_r.Firm = MSFT_r.Firm.replace('Bank oferica', 'Bank of America')

#pull out rating from firm_rating
#change firm_rating 255 to Nuetral
MSFT_r['Firm_Rating'][255] = 'Nomura: to Neutral'
Rating = []
for i in MSFT_r['Firm_Rating']:
    sep = 'to '
    v = i.split(sep, 1)[1]
    Rating.append(v)
    
MSFT_r['Rating'] = Rating

#coding ratings, firms, up_down, into one hot encoders
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# integer encode firm
label_encoder = LabelEncoder()
firm_ie = label_encoder.fit_transform(MSFT_r.Firm) #integer encoding firm

#one hot encode firm
onehot_encoder = OneHotEncoder(sparse=False)
firm_ie = firm_ie.reshape(len(firm_ie), 1)
firm_oe = onehot_encoder.fit_transform(firm_ie)
firm_oe = pd.DataFrame(firm_oe, columns = label_encoder.classes_)

# integer encode Rating
label_encoder = LabelEncoder()
rating_ie = label_encoder.fit_transform(MSFT_r.Rating) 

#one hot encode Rating
onehot_encoder = OneHotEncoder(sparse=False)
rating_ie = rating_ie.reshape(len(rating_ie), 1)
rating_oe = onehot_encoder.fit_transform(rating_ie)
rating_oe = pd.DataFrame(rating_oe, columns = label_encoder.classes_)

# integer encode up_down
label_encoder = LabelEncoder()
up_down_ie = label_encoder.fit_transform(MSFT_r.Up_Down) 

#one hot encode Rating
onehot_encoder = OneHotEncoder(sparse=False)
up_down_ie = up_down_ie.reshape(len(up_down_ie), 1)
up_down_oe = onehot_encoder.fit_transform(up_down_ie)
up_down_oe = pd.DataFrame(up_down_oe, columns = label_encoder.classes_)

#add one hot encoders to MSFT_r
MSFT_woe = pd.concat([MSFT_r, firm_oe, rating_oe, up_down_oe], 
                     axis=1, join='outer')

#converting dates to datetime
MSFT_woe.Date = pd.to_datetime(MSFT_woe.Date)
MSFT_d.Date = pd.to_datetime(MSFT_d.Date)

#putting the data on the same timescales and indices
idx = pd.date_range(MSFT_woe.Date.iloc[0], MSFT_d.Date.iloc[-1])
MSFT_woe = MSFT_woe.reindex(idx, fill_value=0)

MSFT_d.index = MSFT_d.Date

#joing the dataframes into predictive dataframe
MSFT_x = pd.concat([MSFT_woe, MSFT_d], axis=1, join='inner')

#drop columns that are problematic
MSFT_x = MSFT_x.drop(['Date', 'Up_Down', 'Firm', 'Rating',
                      'Firm_Rating'], axis = 1) 

#building RNN
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, LSTM #for output and layers 
from keras.models import Sequential # for RNN
from keras.utils import to_categorical

#choosing interval to predict and sequence length
seq_len = 60
fut_pred = 3
#pred_var = "" #fill in

prev_days = deque(maxlen=seqlen)

#cutting last 10 % of data to predict on
L_10 = MSFT_x[int(0.9*len(MSFT_x.index)):]

test_x = MSFT_x[:int(0.9*len(MSFT_x.index))+1]

#create a future column
fut_class = []
for i in list(range(0,(len(MSFT_x.index)-3))):
    v = 0
    if MSFT_x['Adj Close'][i+3] > MSFT_x['Adj Close'][i]:
        v = 1
    else: v = 0
    fut_class.append(v)

#RNN 
model = Sequential()

model.add(LSTM(42, input_shape = x_train.shape[1:], activation = 'relu',
               return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(42, activation = 'relu'))
model.add(Dropout(0.2))

