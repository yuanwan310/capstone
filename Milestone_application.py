# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:59:38 2019

@author: Yuan Wan
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

## step 1 import data and data cleaning
app=pd.read_csv('D:/Yuan Wan/TDI/Milestone Project/Raw Data/application_data/application_data2017.csv', dtype={'application_number':str,'filing_date':str,'uspc_class':str,'uspc_subclass':str,'appl_status_code':str,'appl_status_date':str,'patent_number':str,'patent_issue_date':str,'invention_title':str, 'earliest_pgpub_number':str, 'earliest_pgpub_date':str })
app=app[['application_number','filing_date','uspc_class','uspc_subclass','appl_status_code','appl_status_date','patent_number','patent_issue_date','invention_title','earliest_pgpub_number', 'earliest_pgpub_date']]
app=app.sort_values(by=['application_number'])
app=app.dropna(subset=['filing_date'])
app=app.dropna(subset=['uspc_class'])
app=app.dropna(subset=['uspc_subclass'])
##filing_month and only look at filing starting from 1991 and ends at 2017
app['filing_month'] = app.filing_date.str[:7]
app['filing_year'] = app.filing_date.str[:4]
app['patent_issue_month'] = app.patent_issue_date.str[:7]
app['patent_issue_year'] = app.patent_issue_date.str[:4]
app= app.astype({"filing_year": float, "patent_issue_year": float})

app= app[app.filing_year>1990]
app= app[app.filing_year<2018]
app.drop(app[app.filing_year>app.patent_issue_year].index, inplace=True)

#Check Nan and duplicates 
app.isnull().sum()
app['application_number'].duplicated().sum()

## step 2 some visulization 
#here I plot number of patent granted vs filed
plt.figure(figsize=(14,4))
plt.hist([app.filing_year, app.patent_issue_year],bins=range(int(min(app.filing_year)), int(max(app.filing_year)) + 1, 1), label=['Patent filing', 'Patent issue'])
plt.legend(loc='upper left')
plt.show()



