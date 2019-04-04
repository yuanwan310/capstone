import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

### step 1 import training data
training_data=pd.read_csv('/Users/prcharming/Desktop/takehome/training.tsv', header = None, delimiter='\t', encoding='utf-8')
training_data.rename(columns = {0:'id',1:'date',2:'action'},inplace = True)

### check dimension
print('# of rows in the training data set: '+ str(len(training_data.index)))
#print(training_data.head())

### step2 get a general picture of data (visualization)
#print("training data action distribution")
#plt.figure()
#sns.countplot(x="action", data=training_data)

### step3 re-organize the data by id and create X and Y
### Y: each row is one customer id. y=o: the customer never purchased, y=1: the customer purchased 
purchased=pd.DataFrame(training_data.loc[training_data['action'] == 'Purchase']['id'].unique())
purchased.rename(columns={0:'id'}, inplace=True)
purchased['purchase']=1 
#print(purchased.head())

N_purchase=pd.DataFrame(training_data[~training_data['id'].isin(purchased['id'])]['id'].unique())
N_purchase['purchase'] = 0
N_purchase.rename(columns={0:'id'}, inplace=True) 
#print(N_purchase.head())

print('number of not purchased user:'+ str(N_purchase.shape[0]))
print('number of purchased user:'+ str(purchased.shape[0]))
print('numer of total user: '+str(len(training_data.id.unique())))
Y=purchased.append(N_purchase)
Y = Y[['id','purchase']]
#print(Y.head())
#print(Y.iloc[-10:-1,:])
#print(sum(Y['id'].isnull()))

#### X: each row is one customer if.  6 variables x1: % of actions that are email open; x2: % of actions that are form submit; x3: % of actions that are email click through.....  
### because I need to use actions to predict purchase, i am only looking at the observations that are not purchased for x data set
need=training_data.loc[training_data['action']!='Purchase'] 
#print(need.head())
data_by_id= need.groupby(['id','action']).count().reset_index().rename(columns={'date':'number'})
#print(data_by_id.head())

total_action=need.groupby('id').count().reset_index().drop('action', axis=1).rename(columns={'date':'total_actions'})
#print(total_action.head())

joined_data=pd.merge(data_by_id, total_action[['id','total_actions']], on='id')
joined_data['percentage']=joined_data.number/joined_data.total_actions
#print(joined_data.head())
x1=joined_data.loc[joined_data['action']=='EmailOpen'][['id','percentage']]
x1.rename(columns={'percentage':'EmailOpenPer'}, inplace=True)
#print(x1.head())
x2=joined_data.loc[joined_data['action']=='FormSubmit'][['id','percentage']]
x2.rename(columns={'percentage':'FormSubmitPer'}, inplace=True)
#print(x2.head())
x3=joined_data.loc[joined_data['action']=='EmailClickthrough'][['id','percentage']]
x3.rename(columns={'percentage':'EmailClickthroughPer'}, inplace=True)
#print(x3.head())
x4=joined_data.loc[joined_data['action']=='CustomerSupport'][['id','percentage']]
x4.rename(columns={'percentage':'CustomerSupportPer'}, inplace=True)
#print(x4.head())
x5=joined_data.loc[joined_data['action']=='PageView'][['id','percentage']]
x5.rename(columns={'percentage':'PageViewPer'}, inplace=True)
#print(x5.head())
x6=joined_data.loc[joined_data['action']=='WebVisit'][['id','percentage']]
x6.rename(columns={'percentage':'WebVisitPer'}, inplace=True)
#print(x6.head())

#### Here I re organized the final training data named Y: each row is a customer id, purchase column=1 if the customer purchsed column 3-8 are the action percentage 
Y=Y.merge(x1, how='left', on=['id'])
Y=Y.merge(x2, how='left', on=['id'])
Y=Y.merge(x3, how='left', on=['id'])
Y=Y.merge(x4, how='left', on=['id'])
Y=Y.merge(x5, how='left', on=['id'])
Y=Y.merge(x6, how='left', on=['id'])
#print(Y.head())
#print(Y.shape)
#### check and replace missing values. Here it is ok to replace the missing value with zero because each customer has to do some actions.
print('missing value report:')
print(Y.isnull().sum()/len(Y['id']))
Y.fillna(0, inplace=True)
print('missing value report after filling N/A:')
print(Y.isnull().sum()/len(Y['id']))

#### step4 pre-result analysis and visualization
###Now let's look at the distribution of Y
#plt.figure()
#sns.boxplot(x='purchase', y='EmailOpenPer', data=Y)
#plt.figure()
#sns.boxplot(x='purchase', y='FormSubmitPer', data=Y)
#plt.figure()
#sns.boxplot(x='purchase', y='EmailClickthroughPer', data=Y)
plt.figure()
sns.boxplot(x='purchase', y='CustomerSupportPer', data=Y)
#plt.figure()
#sns.boxplot(x='purchase', y='PageViewPer', data=Y)
#plt.figure()
#sns.boxplot(x='purchase', y='WebVisitPer', data=Y)


#### step 5 training data and training result evaluation:
x=Y.drop(["id","purchase"], axis=1)
#print(x.head())
y= Y["purchase"]
#print(y.head())
### split training and testing
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=1)
### here I want to train a logistic regression model and produce continuous ranking for my result
from sklearn.linear_model import LogisticRegression 
logmodel=LogisticRegression()
logmodel.fit(x_train, y_train)
### prediction
predictions=logmodel.predict(x_test)
### evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix
print("The confusion matrix is: ")
print(confusion_matrix(y_test, predictions))

### one important thing: during last step I found customer will only do customer support after they purchase so if we want to predict purchase we should not use the customer support variable
## train second model excluding customer support

x1=x.drop(["CustomerSupportPer"], axis=1)
#print(x1.head())

from sklearn.cross_validation import train_test_split
x1_train, x1_test, y1_train, y1_test=train_test_split(x1, y, test_size=0.3, random_state=1)
### here I want to train a logistic regression model and produce continuous ranking for my result
from sklearn.linear_model import LogisticRegression 
logmodel1=LogisticRegression()
logmodel1.fit(x1_train, y1_train)
### prediction
predictions1=logmodel1.predict(x1_test)
### evaluation
from sklearn.metrics import classification_report
print(classification_report(y1_test, predictions1))

from sklearn.metrics import confusion_matrix
print("The confusion matrix is: ")
print(confusion_matrix(y1_test, predictions1))
 
#### step 6 variable ranking
### there are many ways to do it but I choose to use recursive feature elimination to complement my logistic regression model

from sklearn.feature_selection import RFE
rfe = RFE(logmodel, 1)
# for model with customer support
fit = rfe.fit(x, y)
print("Model with customer service")
print(fit.ranking_)

#for model without customer support
fit = rfe.fit(x1, y)
print("Model without customer service")
print(fit.ranking_)

#### step 7 make prediction
### prepare the testing data for the trained model same as how I prepared for the training set
testing_data= pd.read_csv('/Users/prcharming/Desktop/takehome/test.tsv', header= None, delimiter='\t', encoding='utf-8')
testing_data.rename(columns={0:'id', 1:'date', 2:'action'}, inplace=True)
### check dimension
print('# of rows in the testing data set: '+ str(len(testing_data.index)))
print('numer of total user: '+str(len(testing_data.id.unique())))
Y_testing=pd.DataFrame(testing_data.id.unique())
Y_testing.rename(columns={0:'id'}, inplace=True)
#print(Y_testing.head())

#print(training_data.head())
#plt.figure()
#sns.countplot(x="action", data=testing_data)
#### testing X prepare: each row is one customer if.  6 variables x1: % of actions that are email open; x2: % of actions that are form submit; x3: % of actions that are email click through.....  
testing_by_id=testing_data.groupby(['id','action']).count().reset_index().rename(columns={'date':'number'})
#print(testing_by_id.head())

total_testing=testing_data.groupby('id').count().reset_index().drop('action', axis=1).rename(columns={'date':'total_actions'})
#print(total_action.head())
joined_testing=pd.merge(testing_by_id, total_testing[['id','total_actions']], on='id')
joined_testing['percentage']=joined_testing.number/joined_testing.total_actions
#print(joined_testing.head())

test1=joined_testing.loc[joined_testing['action']=='EmailOpen'][['id','percentage']]
test1.rename(columns={'percentage':'EmailOpenPer'}, inplace=True)
#print(test1.head())

test2=joined_testing.loc[joined_testing['action']=='FormSubmit'][['id','percentage']]
test2.rename(columns={'percentage':'FormSubmitPer'}, inplace=True)
#print(test2.head())

test3=joined_testing.loc[joined_testing['action']=='EmailClickthrough'][['id','percentage']]
test3.rename(columns={'percentage':'EmailClickthroughPer'}, inplace=True)
#print(test3.head())

test4=joined_testing.loc[joined_testing['action']=='PageView'][['id','percentage']]
test4.rename(columns={'percentage':'PageViewPer'}, inplace=True)
#print(test4.head())

test5=joined_testing.loc[joined_testing['action']=='WebVisit'][['id','percentage']]
test5.rename(columns={'percentage':'WebVisitPer'}, inplace=True)
#print(test5.head())

Y_testing=Y_testing.merge(test1, how='left', on=['id'])
Y_testing=Y_testing.merge(test2, how='left', on=['id'])
Y_testing=Y_testing.merge(test3, how='left', on=['id'])
Y_testing=Y_testing.merge(test4, how='left', on=['id'])
Y_testing=Y_testing.merge(test5, how='left', on=['id'])
#print(Y_testing.shape)

print('missing value report:')
print(Y_testing.isnull().sum()/len(Y_testing['id']))
Y_testing.fillna(0, inplace=True)
print('missing value report after filling N/A:')
print(Y_testing.isnull().sum()/len(Y_testing['id']))
#print(Y_testing.head())

##### now let's predict testing set outcome based on the model without customer service
testing1=Y_testing.drop('id', axis=1)
#print(testing1.head())
test_predictions1=logmodel1.predict_proba(testing1)
print("Purchase class: ")
print(logmodel1.classes_)
result_test=pd.DataFrame(test_predictions1)
result_test['id']=Y_testing['id']
#print(result_test.head())
#histogram for probability in nonpurchase class: 
plt.figure()
sns.distplot(result_test[0])
#histogram for probability in purchase class:
plt.figure()
sns.distplot(result_test[1])
### rank probability of being a future purchase customer
result_test.rename(columns={0:"N_purchase_prob", 1:"Purchase_prob"}, inplace=True)
top_list=result_test.sort_values(by=['Purchase_prob'], ascending=False)
top_list.reset_index(inplace=True)
print(top_list.head())
top=top_list[["N_purchase_prob", "Purchase_prob"]]
top.plot()
 
##### write file to txt
with open('/Users/prcharming/Desktop/takehome/top1000.txt', 'w') as top1000:  
    for i in range(1000):
        top1000.write(str(top_list['id'][i]))
        top1000.write("\n")
top1000.close() 








