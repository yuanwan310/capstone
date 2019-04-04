# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:39:47 2019

@author: Yuan Wan
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

## step 1 import data and data cleaning
all_I=pd.read_csv('D:/Yuan Wan/TDI/Milestone Project/Raw Data/all_inventors/all_inventors2017.csv', dtype={'application_number':str,'inventor_name_first':str,'inventor_name_middle':str,'inventor_name_last':str,'inventor_rank':int,'inventor_region_code':str,'inventor_country_name':str })
all_I=all_I[['application_number','inventor_name_first','inventor_name_middle','inventor_name_last','inventor_rank','inventor_region_code','inventor_country_name']]
all_I.inventor_name_middle = all_I.inventor_name_middle.fillna('   xxxx  x           ')
all_I['inventor_name_first'] = all_I['inventor_name_first'].str.upper()
all_I['inventor_name_last'] = all_I['inventor_name_last'].str.upper()
all_I['inventor_name_middle'] = all_I['inventor_name_middle'].str.upper()
all_I['inventor_name_first'] = all_I['inventor_name_first'].str.replace(" ","")
all_I['inventor_name_last'] = all_I['inventor_name_last'].str.replace(" ","")
all_I['inventor_name_middle'] = all_I['inventor_name_middle'].str.replace(" ","")

all_I.inventor_country_name = all_I.inventor_country_name.fillna('USA')
all_I['inventor_country_name'] = all_I['inventor_country_name'].str.replace(" ","")
all_I['inventor_name'] = all_I['inventor_name_first']+" "+ all_I['inventor_name_middle']+" "+all_I['inventor_name_last']

### only work with rank1 investor there are co-rank1 invetors case
all_I= all_I[all_I.inventor_rank==1]






all_I['application_number'].duplicated().sum()
all_I=all_I.sort_values(by=['application_number'])
all_I['dup']=all_I['application_number'].duplicated()









### lets work with smaller data set first 
all_I1=all_I.head(300000)
import gc
del [[all_I]]
gc.collect()
all_I=all_I1
del [[all_I1]]
##########################################


f1.tail(10)


def sockMerchant(n, ar):
    temp={}
    for i in ar:
        if temp.get(i)== None:
            temp[i]=0.5
        else:
            temp[i]=temp[i]+0.5
    v = [math.floor(float(x)) for x in temp.values()]
    
    return int(sum(v))
