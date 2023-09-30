#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
sns.set()
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[2]:


data=pd.read_csv(r'C:\Users\ASUS\Downloads\credit_score.csv')
data


# In[3]:


data.describe(include='all')


# In[4]:


data=data.drop('ID', axis=1)


# In[5]:


data=data.drop('Name', axis=1)


# In[6]:


data=data.drop('CustomerID', axis=1)


# In[7]:


data=data.drop('SSN', axis=1)


# In[8]:


data=data.drop('Occupation', axis=1)


# In[9]:


data=data.drop('TypeofLoan', axis=1)


# In[10]:


data.head()


# In[11]:


data['Score']=np.where(data['CreditScore']=='Poor', 1, 0)


# In[12]:


data.head(2)


# In[13]:


data=data.drop('CreditScore', axis=1)


# In[14]:


data.Score.value_counts()


# In[15]:


data.isnull().sum()


# In[16]:


data.dtypes


# In[17]:


data['MonthlyInhandSalary']=data['MonthlyInhandSalary'].fillna(value=data['MonthlyInhandSalary'].mean())
data['NumofDelayedPayment']=data['NumofDelayedPayment'].fillna(value=data['NumofDelayedPayment'].mean())
data['ChangedCreditLimit']=data['ChangedCreditLimit'].fillna(value=data['ChangedCreditLimit'].mean())
data['NumCreditInquiries']=data['NumCreditInquiries'].fillna(value=data['NumCreditInquiries'].mean())
data['Amountinvestedmonthly']=data['Amountinvestedmonthly'].fillna(value=data['Amountinvestedmonthly'].mean())
data['MonthlyBalance']=data['MonthlyBalance'].fillna(value=data['MonthlyBalance'].mean())


# In[18]:


data.isnull().sum()


# In[19]:


data.corr()['Score']


# In[20]:


data=data.drop('Age', axis=1)


# In[21]:


data=data.drop('AnnualIncome', axis=1)


# In[22]:


data=data.drop('NumBankAccounts', axis=1)


# In[23]:


data=data.drop('NumCreditCard', axis=1)


# In[24]:


data=data.drop('InterestRate', axis=1)


# In[25]:


data=data.drop('NumofLoan', axis=1)


# In[26]:


data=data.drop('NumofDelayedPayment', axis=1)


# In[27]:


data=data.drop('NumCreditInquiries', axis=1)


# In[28]:


data=data.drop('TotalEMIpermonth', axis=1)


# In[29]:


data.head(2)


# In[30]:


data.dtypes


# In[31]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables=data[['Delayfromduedate', 'ChangedCreditLimit', 'OutstandingDebt', 'Amountinvestedmonthly', 'MonthlyBalance']]
vif=pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns
vif


# In[32]:


data=data.drop('MonthlyInhandSalary', axis=1)


# In[33]:


data=data.drop('CreditUtilizationRatio', axis=1)


# In[34]:


data.head(2)


# In[35]:


data.dtypes


# In[36]:


for i in data[['Delayfromduedate', 'ChangedCreditLimit', 'OutstandingDebt', 'Amountinvestedmonthly', 'MonthlyBalance']]:
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[37]:


q1=data.quantile(0.25)
q3=data.quantile(0.75)
IQR=q3-q1
Lower=q1-1.5*IQR
Upper=q3+1.5*IQR


# In[38]:


for i in data[['Delayfromduedate', 'ChangedCreditLimit', 'OutstandingDebt', 'Amountinvestedmonthly', 'MonthlyBalance']]:
    data[i] = np.where(data[i] > Upper[i], Upper[i],data[i])
    data[i] = np.where(data[i] < Lower[i], Lower[i],data[i])    


# In[39]:


for i in data[['Delayfromduedate', 'ChangedCreditLimit', 'OutstandingDebt', 'Amountinvestedmonthly', 'MonthlyBalance']]:
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[40]:


data.head(3)


# In[41]:


data=data.reset_index(drop=True)


# In[42]:


data.describe(include='all')


# In[43]:


grouped=data.groupby(['Month', 'Score'])['Score'].count().unstack().reset_index()
grouped


# In[44]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['Month_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[45]:


data=data.merge(grouped[['Month', 'Month_woe']], how='left', on='Month')
data


# In[46]:


grouped=data.groupby(['PaymentofMinAmount', 'Score'])['Score'].count().unstack().reset_index()
grouped


# In[47]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['PaymentofMinAmount_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[48]:


data=data.merge(grouped[['PaymentofMinAmount', 'PaymentofMinAmount_woe']], how='left', on='PaymentofMinAmount')
data


# In[49]:


grouped=data.groupby(['PaymentBehaviour', 'Score'])['Score'].count().unstack().reset_index()
grouped


# In[50]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['PaymentBehaviour_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[51]:


data=data.merge(grouped[['PaymentBehaviour', 'PaymentBehaviour_woe']], how='left', on='PaymentBehaviour')
data


# In[52]:


ranges=[-np.inf, data['Delayfromduedate'].quantile(0.25), data['Delayfromduedate'].quantile(0.5), data['Delayfromduedate'].quantile(0.75), np.inf]
data['Delayfromduedate_category']=pd.cut(data['Delayfromduedate'], bins=ranges)
data


# In[53]:


grouped=data.groupby(['Delayfromduedate_category', 'Score'])['Score'].count().unstack().reset_index()
grouped


# In[54]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['Delayfromduedate_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[55]:


data=data.merge(grouped[['Delayfromduedate_category', 'Delayfromduedate_woe']], how='left', on='Delayfromduedate_category')
data


# In[56]:


ranges=[-np.inf, data['ChangedCreditLimit'].quantile(0.25), data['ChangedCreditLimit'].quantile(0.5), data['ChangedCreditLimit'].quantile(0.75), np.inf]
data['ChangedCreditLimit_category']=pd.cut(data['ChangedCreditLimit'], bins=ranges)
data


# In[57]:


grouped=data.groupby(['ChangedCreditLimit_category', 'Score'])['Score'].count().unstack().reset_index()
grouped


# In[58]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['ChangedCreditLimit_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[59]:


data=data.merge(grouped[['ChangedCreditLimit_category', 'ChangedCreditLimit_woe']], how='left', on='ChangedCreditLimit_category')
data


# In[60]:


ranges=[-np.inf, data['OutstandingDebt'].quantile(0.25), data['OutstandingDebt'].quantile(0.5), data['OutstandingDebt'].quantile(0.75), np.inf]
data['OutstandingDebt_category']=pd.cut(data['OutstandingDebt'], bins=ranges)
data


# In[61]:


grouped=data.groupby(['OutstandingDebt_category', 'Score'])['Score'].count().unstack().reset_index()
grouped


# In[62]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['OutstandingDebt_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[63]:


data=data.merge(grouped[['OutstandingDebt_category', 'OutstandingDebt_woe']], how='left', on='OutstandingDebt_category')
data


# In[64]:


ranges=[-np.inf, data['Amountinvestedmonthly'].quantile(0.25), data['Amountinvestedmonthly'].quantile(0.5), data['Amountinvestedmonthly'].quantile(0.75), np.inf]
data['Amountinvestedmonthly_category']=pd.cut(data['Amountinvestedmonthly'], bins=ranges)
data


# In[65]:


grouped=data.groupby(['Amountinvestedmonthly_category', 'Score'])['Score'].count().unstack().reset_index()
grouped


# In[66]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['Amountinvestedmonthly_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[67]:


data=data.merge(grouped[['Amountinvestedmonthly_category', 'Amountinvestedmonthly_woe']], how='left', on='Amountinvestedmonthly_category')
data


# In[68]:


ranges=[-np.inf, data['MonthlyBalance'].quantile(0.25), data['MonthlyBalance'].quantile(0.5), data['MonthlyBalance'].quantile(0.75), np.inf]
data['MonthlyBalance_category']=pd.cut(data['MonthlyBalance'], bins=ranges)
data


# In[69]:


grouped=data.groupby(['MonthlyBalance_category', 'Score'])['Score'].count().unstack().reset_index()
grouped


# In[70]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['MonthlyBalance_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[71]:


data=data.merge(grouped[['MonthlyBalance_category', 'MonthlyBalance_woe']], how='left', on='MonthlyBalance_category')
data


# In[72]:


data.columns


# In[73]:


data=data[['Month', 'Delayfromduedate', 'ChangedCreditLimit', 'OutstandingDebt',
       'PaymentofMinAmount', 'Amountinvestedmonthly', 'PaymentBehaviour',
       'MonthlyBalance', 'Month_woe', 'PaymentofMinAmount_woe',
       'PaymentBehaviour_woe', 'Delayfromduedate_category',
       'Delayfromduedate_woe', 'ChangedCreditLimit_category',
       'ChangedCreditLimit_woe', 'OutstandingDebt_category',
       'OutstandingDebt_woe', 'Amountinvestedmonthly_category',
       'Amountinvestedmonthly_woe', 'MonthlyBalance_category',
       'MonthlyBalance_woe', 'Score',]]


# In[74]:


data.head()


# In[75]:


X=data[['Month_woe', 'PaymentofMinAmount_woe', 'PaymentBehaviour_woe', 'Delayfromduedate_woe', 'ChangedCreditLimit_woe', 'OutstandingDebt_woe', 'Amountinvestedmonthly_woe','MonthlyBalance_woe']]
y=data['Score']


# In[76]:


X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42)


# In[77]:


def evaluate(model, X_test, y_test):
    y_pred=model.predict(X_test)
    y_prob=model.predict_proba(X_test)[:, 1]
    
    roc_prob=roc_auc_score(y_test, y_prob)
    gini_prob=2*roc_prob-1
    
    confusion_matrix=metrics.confusion_matrix(y_test, y_pred)
    report=classification_report(y_test, y_pred)
    
    print('Gini probability:', gini_prob*100)
    print('Confusion_matrix:', confusion_matrix)
    print('Classification report', report)


# In[78]:


lr=LogisticRegression()


# In[79]:


lr.fit(X_train, y_train)


# In[80]:


result=evaluate(lr, X_test, y_test)


# In[81]:


y_prob = lr.predict_proba(X_test)[:, 1]

roc_prob= roc_auc_score(y_test, y_prob)
gini = 2*roc_prob-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Auc = %0.2f)' % roc_prob)
plt.plot(fpr, tpr, label='(Gini = %0.2f)' % gini)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='lower right')
plt.show()


# In[82]:


variables= []
train_Gini=[]
test_Gini=[]

for i in X_train.columns:
    X_train_single=X_train[[i]]
    X_test_single=X_test[[i]]
    
    lr.fit(X_train_single, y_train)
    y_prob_train_single=lr.predict_proba(X_train_single)[:, 1]
    
    
    roc_prob_train=roc_auc_score(y_train, y_prob_train_single)
    gini_prob_train=2*roc_prob_train-1
    
    
    lr.fit(X_test_single, y_test)
    y_prob_test_single=lr.predict_proba(X_test_single)[:, 1]
    
    
    roc_prob_test=roc_auc_score(y_test, y_prob_test_single)
    gini_prob_test=2*roc_prob_test-1
    
    
    variables.append(i)
    train_Gini.append(gini_prob_train)
    test_Gini.append(gini_prob_test)
    

df = pd.DataFrame({'Variable': variables, 'Train Gini': train_Gini, 'Test Gini': test_Gini})

df= df.sort_values(by='Test Gini', ascending=False)

df   


# In[83]:


data.columns


# In[84]:


Input=data[['PaymentofMinAmount_woe', 'Delayfromduedate_woe','OutstandingDebt_woe', 'MonthlyBalance_woe']]
Output=data['Score']


# In[85]:


X_train, X_test, y_train, y_test = train_test_split (Input, Output, test_size=0.2, random_state=42)


# In[86]:


lr.fit(X_train, y_train)


# In[87]:


result=evaluate(lr, X_test, y_test)


# In[88]:


y_prob = lr.predict_proba(X_test)[:, 1]

roc_prob= roc_auc_score(y_test, y_prob)
gini = 2*roc_prob-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Auc = %0.2f)' % roc_prob)
plt.plot(fpr, tpr, label='(Gini = %0.2f)' % gini)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='lower right')
plt.show()


# ## Deployment

# In[89]:


test_data=pd.read_excel(r'C:\Users\ASUS\Downloads\test_data_LR.xlsx')
test_data


# In[90]:


test_data=test_data[['PaymentofMinAmount', 'Delayfromduedate','OutstandingDebt', 'MonthlyBalance']]
test_data


# In[91]:


grouped=data.groupby(['PaymentofMinAmount', 'Score'])['Score'].count().unstack().reset_index()
grouped


# In[92]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['PaymentofMinAmount_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[93]:


test_data=test_data.merge(grouped[['PaymentofMinAmount', 'PaymentofMinAmount_woe']], how='left', on='PaymentofMinAmount')
test_data


# In[94]:


ranges=[-np.inf, data['Delayfromduedate'].quantile(0.25), data['Delayfromduedate'].quantile(0.5), data['Delayfromduedate'].quantile(0.75), np.inf]
test_data['Delayfromduedate_category']=pd.cut(test_data['Delayfromduedate'], bins=ranges)
test_data


# In[95]:


grouped=data.groupby(['Delayfromduedate_category', 'Score'])['Score'].count().unstack().reset_index()
grouped


# In[96]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['Delayfromduedate_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[97]:


test_data=test_data.merge(grouped[['Delayfromduedate_category', 'Delayfromduedate_woe']], how='left', on='Delayfromduedate_category')
test_data


# In[98]:


ranges=[-np.inf, data['OutstandingDebt'].quantile(0.25), data['OutstandingDebt'].quantile(0.5), data['OutstandingDebt'].quantile(0.75), np.inf]
test_data['OutstandingDebt_category']=pd.cut(test_data['OutstandingDebt'], bins=ranges)
test_data


# In[99]:


grouped=data.groupby(['OutstandingDebt_category', 'Score'])['Score'].count().unstack().reset_index()
grouped


# In[100]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['OutstandingDebt_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[101]:


test_data=test_data.merge(grouped[['OutstandingDebt_category', 'OutstandingDebt_woe']], how='left', on='OutstandingDebt_category')
test_data


# In[102]:


ranges=[-np.inf, data['MonthlyBalance'].quantile(0.25), data['MonthlyBalance'].quantile(0.5), data['MonthlyBalance'].quantile(0.75), np.inf]
test_data['MonthlyBalance_category']=pd.cut(test_data['MonthlyBalance'], bins=ranges)
test_data


# In[103]:


grouped=data.groupby(['MonthlyBalance_category', 'Score'])['Score'].count().unstack().reset_index()
grouped


# In[104]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['MonthlyBalance_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[105]:


test_data=test_data.merge(grouped[['MonthlyBalance_category', 'MonthlyBalance_woe']], how='left', on='MonthlyBalance_category')
test_data


# In[106]:


test_data.columns


# In[107]:


final_test_data=test_data[['PaymentofMinAmount_woe', 'Delayfromduedate_woe', 'OutstandingDebt_woe', 'MonthlyBalance_woe']]
final_test_data


# In[108]:


final_test_data['Probability of Score']=lr.predict_proba(final_test_data)[:,1]


# In[109]:


final_test_data

