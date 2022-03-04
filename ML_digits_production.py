#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_digits


# In[2]:


digits_data = load_digits()
df = pd.DataFrame(digits_data['data'],columns=digits_data['feature_names'])


# In[3]:


df['target'] = digits_data['target']


# In[4]:


df.head()


# In[17]:


print(digits_data['DESCR'])


# In[28]:


image = df.iloc[1000,:-1].values
image = image.reshape((8,8))


# In[29]:


import matplotlib.pyplot as plt
plt.imshow(image, cmap='gray')
plt.show()


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits_data['data'],
                                                     digits_data['target'],test_size=0.3,random_state=0)


# In[45]:


from sklearn.ensemble import RandomForestClassifier
accuracy = []
n_estimator_range = np.arange(50,180,1)
for estimator in n_estimator_range:
    clf = RandomForestClassifier(n_estimators=estimator, random_state = 0)
    clf.fit(X_train, y_train)
    accuracy.append(clf.score(X_test, y_test))


# In[48]:


fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10,5))
ax.grid(True)

ax.plot(n_estimator_range, accuracy)
ax.set_xlabel('n_estimators')
ax.set_ylabel('accuracy')
ax.set_title('No of randomforest estimators vs. accuracy')
fig.tight_layout()


# In[51]:


feature_importance = clf.feature_importances_
features = digits_data['feature_names']


# In[58]:


fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16,14))
plt.barh(features, feature_importance)
ax.set_xlabel('level of importance')
ax.set_ylabel('features')


# In[59]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 0.90, random_state = 0)
pca.fit(digits_data['data'])


# In[62]:


X_pca = pca.transform(digits_data['data'])

X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(X_pca, digits_data['target'],
                                                            test_size = 0.3, random_state=0)

clf_pca = RandomForestClassifier(n_estimators=130,random_state=0)
clf_pca.fit(X_p_train, y_p_train)
print(clf_pca.score(X_p_test, y_p_test))


# In[63]:


from sklearn.pipeline import Pipeline

pipe = Pipeline([
        ('pca', PCA(n_components=0.9, random_state=0)),
        ('rfclf', RandomForestClassifier(n_estimators=130, random_state=0))
    ])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)


# In[71]:


pipe.set_params(rfclf__n_estimators=500)


# In[72]:


pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

