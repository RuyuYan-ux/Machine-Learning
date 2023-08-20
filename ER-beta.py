#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:16:32 2022

@author: demi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 19:57:54 2021

@author: demi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 10:12:08 2021

@author: demi
"""

from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from rdkit.Chem import Draw
from molvs.fragment import is_organic
from molvs.metal import MetalDisconnector
from rdkit.Chem.SaltRemover import SaltRemover
from molvs import standardize_smiles
from standardiser import process_smiles as ps



# In[2]:


a={}
mols=[mol for mol in Chem.SDMolSupplier('xxx.sdf')] #刚刚下载的sdf
a['Mol']=mols
a=pd.DataFrame(a)

pub=pd.read_csv('xxx.csv') #踢掉inconclusive的csv文件
data=pd.concat([pub,a],axis=1)

#处理空白值
blanklist=data[data['Mol'].isnull()].index.tolist()
data=data.drop(blanklist,axis=0)
data0=data
data['isosmiles']=data['Mol'].apply(Chem.MolToSmiles)

"""data['Mol']=data['isosmiles'].apply(Chem.MolFromSmiles)"""


# In[3]:




#筛选有机物

for i in data['Mol']:
    if is_organic(i)!=True:
        ix=data[data['Mol']==i].index
        data=data.drop(ix,axis=0) 
data1=data

metaorg=[]
for i in data['Mol']:
    ix1=data[data['Mol']==i].index
    i1,metals=ps.disconnect(i)
    if metals:
        metaorg.append(i)
        data=data.drop(ix1,axis=0) 
data2=data


salt=[]
for i in data['Mol']:
    remover = SaltRemover()
    res, deleted = remover.StripMolWithDeleted(i,dontRemoveEverything=True)
    if len(deleted) >= 1:
        ix2=data[data['Mol']==i].index
        data=data.drop(ix2,axis=0)
        salt.append(i)
data3=data
        
    
        
mix=[]
for a in data['isosmiles']:
    if '.' in a:
        mix.append(Chem.MolFromSmiles(a))
        ix4=data[data['isosmiles']==a].index
        data=data.drop(ix4,axis=0)  
data4=data

 
for i in data['Mol']:
     ix5=data[data['Mol']==i].index
     smi=Chem.MolToSmiles(i)
     smi1=standardize_smiles(smi)
     data.loc[ix5,'isosmiles']=smi1
     data.loc[ix5,'Mol']=Chem.MolFromSmiles(smi1)
data5=data

more=[]
for i in data['isosmiles']:
    if list(data['isosmiles']).count(i)!=1:
        ix6=data[data['isosmiles']==i].index
        more.append(ix6)
        data=data.drop(ix6[1:],axis=0)
        print(i)
data6=data

data.to_csv('训练.csv')
        


# In[ ]:





# In[4]:


# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 08:28:49 2022

@author: demi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 08:20:34 2022

@author: macbook
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 20:39:43 2022

@author: macbook
"""

from rdkit import Chem
import pandas as pd
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,balanced_accuracy_score
from rdkit.Chem import Descriptors
import numpy as np
from  sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold,StratifiedKFold
from sklearn.utils import compute_class_weight


# In[5]:



data=pd.read_csv(r'训练.csv') #清洗后的数据集
def get_md(mol):
    calc=MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    ds = np.asarray(calc.CalcDescriptors(mol))
    return ds
data['Mol']=data['isosmiles'].apply(Chem.MolFromSmiles)
data['descriptors']=data['Mol'].apply(get_md)
stra_data=StratifiedShuffleSplit(n_splits=1,test_size=0.2,train_size=0.8, random_state=42)
x=np.array(data['descriptors'])
y=data['Activity_Summary']
for train_index, test_index in stra_data.split(x, y):
   x_train, x_test = x[train_index], x[test_index]#训练集对应的值
   y_train, y_test = y[train_index], y[test_index]


des_list=[x[0] for x in Descriptors._descList]
des={}
for i in range(len(des_list)):
    a=[]
    for t in  x_train:
        a.append(t[i])
    des[des_list[i]]=a
des=pd.DataFrame(des)
zeros=des.describe().loc['mean'] #删除零值和零方差
zerostd=des.describe().loc['std']
delete=zeros[zeros==0].index
delete1=delete.append(zerostd[zerostd==0].index)
des1=des.drop(set(delete1),axis=1)

cormatrix=des1.corr(method='pearson')
highcor=[]
col=0
for t in cormatrix.columns:
    row=0
    for i in cormatrix[t]:
        if abs(i)>=0.95:
            if row>col:
                highcor.append([t,cormatrix[t].index[row],i])
        row+=1
    col+=1

delecor=[]
for i in highcor:
    delecor.append(i[0])
des2=des1.drop(set(delecor),axis=1)

r=RFC(random_state=42)
select=RFE(r,step=1)
x1=np.array(des2.values)
rfe=select.fit(x1,y_train)
x2=rfe.transform(x1)
scale=MinMaxScaler()
x_train=scale.fit_transform(x2)

des_t={}
for i in range(len(des_list)):
    a=[]
    for t in  x_test:
        a.append(t[i])
    des_t[des_list[i]]=a
des_t=pd.DataFrame(des_t)
des1_t=des_t.drop(set(delete1),axis=1)
des2_t=des1_t.drop(set(delecor),axis=1)
x1_t=np.array(des2_t.values)
x2_t=rfe.transform(x1_t)
x_test=scale.transform(x2_t)
   
weight=compute_class_weight(class_weight='balanced',classes=np.unique(y_train),y=y_train)
z={}
z[0]=weight[0]
z[1]=weight[1]
rf=RandomForestClassifier(class_weight=z,max_depth=10,random_state=42,n_estimators=200)


# In[8]:


'''
from sklearn.model_selection import GridSearchCV
para={'n_estimators':[5,10,25,50,75,100,200]}
weight=compute_class_weight(class_weight='balanced',classes=np.unique(y_train),y=y_train)
z={}
z[0]=weight[0]
z[1]=weight[1]
rf=RandomForestClassifier(class_weight=z,max_depth=10,random_state=42)
strKFold = StratifiedKFold(n_splits=5,shuffle=False)
scores = GridSearchCV(rf, para, cv=strKFold, scoring='roc_auc')
'''



def matthews(y_true,y_pred):
    score=matthews_corrcoef(y_true,y_pred)
    return score
def kappa(y_true,y_pred):
    score=cohen_kappa_score(y_true,y_pred)
    return score
def bala_accuracy(y_true,y_pred):
    score=balanced_accuracy_score(y_true,y_pred)
    return score

cohen_kappa=make_scorer(kappa,greater_is_better=True)
matthews_co=make_scorer(matthews,greater_is_better=True)
balance_accuracy=make_scorer(bala_accuracy,greater_is_better=True)
scoring={'ca':balance_accuracy,'f1_score':'f1','prec':'precision','rec':'recall','auc':'roc_auc','matt':matthews_co,'kappa':cohen_kappa}
k=RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=42)
scores=cross_validate(rf,x_train,y_train,scoring=scoring,cv=k,return_estimator=True)

rf.fit(x_train,y_train)
y2_pred=rf.predict(x_test)


print('training set:')
print('accuracy:%.3f(+/- %0.2f)'%(scores['test_ca'].mean(),scores['test_ca'].std()))
print('precision:%.3f(+/- %0.2f)'%(scores['test_prec'].mean(),scores['test_prec'].std()))
print('recall:%.3f(+/- %0.2f)'%(scores['test_rec'].mean(),scores['test_rec'].std()))
print('f1:%.3f(+/- %0.2f)'%(scores['test_f1_score'].mean(),scores['test_f1_score'].std()))
print('auc_score:%.3f(+/- %0.2f)'%(scores['test_auc'].mean(),scores['test_auc'].std()))
print('cohen_kappa:%.3f(+/- %0.2f)'%(scores['test_kappa'].mean(),scores['test_kappa'].std()))
print('matthews:%.3f(+/- %0.2f)'%(scores['test_matt'].mean(),scores['test_matt'].std()))

print('testing set:')
print('accuracy:%.3f'%balanced_accuracy_score(y_true=y_test,y_pred=y2_pred))
print('precision:%.3f'%precision_score(y_true=y_test,y_pred=y2_pred))
print('recall:%.3f'%recall_score(y_true=y_test,y_pred=y2_pred))
print('f1:%.3f'%f1_score(y_true=y_test,y_pred=y2_pred))
print('auc score:%.3f'%roc_auc_score(y_test,y2_pred))
print('cohen_kappa:%.3f'%cohen_kappa_score(y_test,y2_pred))
print('matthews:%.3f'%matthews_corrcoef(y_test,y2_pred))


# In[ ]:





# In[ ]:




