#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[2]:


df = pd.DataFrame({
    'x' : [4,12,33,20,54,39,11,46,60,71,78,8,24,30,34,64,73,16,29,41,75],
    'y' : [11,5,50,46,78,31,28,7,64,18,21,34,23,56,61,76,58,3,41,68,15]
})


# In[3]:


dendrogram = sch.dendrogram(sch.linkage(df,method='ward'))


# In[4]:


plt.scatter(df['x'],df['y'])


# In[5]:


hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')


# In[6]:


y_hc=hc.fit_predict(df)


# In[7]:


dendrogram = sch.dendrogram(sch.linkage(y_hc,method='ward'))


# In[8]:


print(hc.labels_)


# In[9]:


plt.scatter(df['x'],df['y'], c=hc.labels_, cmap='rainbow')


# In[ ]:




