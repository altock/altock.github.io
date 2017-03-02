Title: HR Post
Slug: hr-post
Date: 2017-03-02 20:00
Category: posts
Tags: python firsts
Author: Sam "Sven" Svenningsen
Summary: My first post, read it to find out.

```python
import pandas as pd
import numpy as np

from sklearn import  neighbors, svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA

%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import itertools
import pickle


DIR_DATA = "data"
DIR_PROCESSED = "processed"
CV_FOLDS = 5
```

# HR_comma_sep.csv

Employee satisfaction level
Last evaluation
Number of projects
Average monthly hours
Time spent at the company
Whether they have had a work accident
Whether they have had a promotion in the last 5 years
Department
Salary
Whether the employee has left


```python
hr = pd.read_csv(DIR_DATA + '/HR_comma_sep.csv')
hr.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>sales</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Percent who left: {:.2f}'.format(np.sum(hr.left) / len(hr.left) * 100))
```

    Percent who left: 23.81



```python
# Turn sales and salary into category types so that algorithms can handle them
hr.sales = hr.sales.astype('category').cat.codes
hr.salary = hr.salary.astype('category').cat.codes

```


```python
# Department metrics don't work

# hr["is_4"] = hr.sales.apply(lambda x: x== 4)

# depProperties = hr.groupby('sales').agg({'promotion_last_5years':np.mean}).values
# hr["avg_per_team"] = hr.sales.apply(lambda x: depProperties[x])
```


```python

```


```python
hr.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>sales</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
def predict_left(df, clf, test_size=0.2):
    X = df.drop(['left'],1)
    y = df.left 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    
    clf.fit(X_train, y_train)
    
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    print('Training Score: {:.3f}'.format(clf.score(X_train, y_train)))
    print('Testing Score: %.3f' % (clf.score(X_test, y_test)))
    
    print()
```


```python
classifiers = [RandomForestClassifier(n_jobs=-1), RandomForestClassifier(criterion='entropy', n_jobs=-1), svm.SVC(), LogisticRegressionCV(n_jobs=-1), AdaBoostClassifier(),GradientBoostingClassifier(), neighbors.KNeighborsClassifier(n_jobs=-1)] 

for i, clf in enumerate(classifiers):
    print('Classifier ', i)
    
    predict_left(hr, clf, test_size=0.4)
```

    Classifier  0
    Training Score: 0.998
    Testing Score: 0.985
    
    Classifier  1
    Training Score: 0.997
    Testing Score: 0.987
    
    Classifier  2
    Training Score: 0.961
    Testing Score: 0.949
    
    Classifier  3
    Training Score: 0.768
    Testing Score: 0.779
    
    Classifier  4
    Training Score: 0.961
    Testing Score: 0.958
    
    Classifier  5
    Training Score: 0.978
    Testing Score: 0.975
    
    Classifier  6
    Training Score: 0.949
    Testing Score: 0.935
    



```python

```


```python
def cross_val_left(hr, clf, cv_folds=CV_FOLDS, drop=['left']):
    X = hr.drop(drop, 1)
    y = hr.left 
    scores = cross_val_score(clf, X, y, cv=cv_folds, n_jobs=-1)#, scoring='roc_auc')
    
    
    
    print('Cross val score: ', sum(scores) / cv_folds )
    print(scores)
    
    print()
    
```


```python
classifiers = [RandomForestClassifier(n_estimators=500 ,n_jobs=-1), RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1)]

for i, clf in enumerate(classifiers):
    print('Classifier ', i)
    
    cross_val_left(hr, clf)
```

    Classifier  0
    Cross val score:  0.991999666496
    [ 0.99866711  0.98133333  0.98633333  0.99899967  0.99466489]
    
    Classifier  1
    Cross val score:  0.991666333163
    [ 0.99866711  0.981       0.985       0.99899967  0.99466489]
    



```python
clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
X = hr.drop(['left'],1)
y = hr.left 
train_sizes, train_scores, valid_scores = learning_curve(clf, X, y, train_sizes=np.linspace(.1, 1.0, 10), cv=5, n_jobs=-1)

```


```python
print(valid_scores)

```

    [[ 0.23825392  0.238       0.238       0.23807936  0.23807936]
     [ 0.97100966  0.97133333  0.96933333  0.9033011   0.83194398]
     [ 0.97767411  0.97466667  0.98066667  0.98966322  0.97965989]
     [ 0.97867378  0.97633333  0.98233333  0.99433144  0.98866289]
     [ 0.97867378  0.97666667  0.98333333  0.99633211  0.99166389]
     [ 0.979007    0.97766667  0.98333333  0.99733244  0.99266422]
     [ 0.97934022  0.97933333  0.983       0.99766589  0.993998  ]
     [ 0.99466844  0.979       0.98233333  0.99866622  0.993998  ]
     [ 0.99933356  0.982       0.983       0.99899967  0.99433144]
     [ 0.99900033  0.981       0.98633333  0.99899967  0.99466489]]



```python
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
test_scores_std = np.std(valid_scores, axis=1)
train_sizes = np.linspace(.1,1.0,10)
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

plt.xlabel("Training examples")
plt.ylabel("Score")

plt.grid()
plt.legend(loc="best")
plt.show()
```


![png](HR%20Post_files/HR%20Post_15_0.png)



```python
train_scores_mean = np.mean(train_scores, axis=1)[3:] # 0.4 on
train_scores_std = np.std(train_scores, axis=1)[3:]
test_scores_mean = np.mean(valid_scores, axis=1)[3:]
test_scores_std = np.std(valid_scores, axis=1)[3:]
train_sizes = np.linspace(.1,1.0,10)[3:]

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

plt.xlabel("Training examples")
plt.ylabel("Score")

plt.grid()
plt.legend(loc="best")
plt.show()
```


![png](HR%20Post_files/HR%20Post_16_0.png)



```python
clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
X = hr.drop(['left'],1)
y = hr.left 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    
clf.fit(X_train, y_train)
print(X.columns)
clf.feature_importances_
# Drop all lower than 0.01 relevance
```

    Index(['satisfaction_level', 'last_evaluation', 'number_project',
           'average_montly_hours', 'time_spend_company', 'Work_accident',
           'promotion_last_5years', 'sales', 'salary'],
          dtype='object')





    array([ 0.34702455,  0.11829727,  0.17513807,  0.14439736,  0.18723606,
            0.00589027,  0.00076228,  0.01269425,  0.00855989])




```python

```


```python
# Dropping all with <0.01 relevance seems to not affect score much (.9920->.9916)
drop = ['left', 'promotion_last_5years', 'Work_accident', 'sales', 'salary']
for i, clf in enumerate(classifiers):
    print('Classifier ', i)
    
    cross_val_left(hr, clf, drop=drop)
```

    Classifier  0
    Cross val score:  0.991599644237
    [ 0.99800067  0.981       0.98633333  0.99833278  0.99433144]
    
    Classifier  1
    Cross val score:  0.99166624423
    [ 0.99833389  0.98166667  0.98633333  0.99833278  0.99366455]
    



```python
hr.corr()["left"]
```




    satisfaction_level      -0.388375
    last_evaluation          0.006567
    number_project           0.023787
    average_montly_hours     0.071287
    time_spend_company       0.144822
    Work_accident           -0.154622
    left                     1.000000
    promotion_last_5years   -0.061788
    sales                    0.032105
    salary                  -0.001294
    Name: left, dtype: float64




```python
# Drop everything with corr to left of < 0.005
# Makes it worse
drop = ['left',  'sales', 'salary']
classifiers = [RandomForestClassifier(n_estimators=500 ,n_jobs=-1), RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1)]#, svm.SVC()]#xgb.XGBClassifier(n_estimators=500, nthread=-1)]# svm.SVC()]

for i, clf in enumerate(classifiers):
    print('Classifier ', i)
    
    cross_val_left(hr, clf, drop=drop)
```

    Classifier  0
    Cross val score:  0.991599666467
    [ 0.99800067  0.98066667  0.98633333  0.99866622  0.99433144]
    
    Classifier  1
    Cross val score:  0.991532977585
    [ 0.99833389  0.98066667  0.98566667  0.99866622  0.99433144]
    



```python
from sklearn.feature_selection import RFE
model = RandomForestClassifier(n_estimators=500, n_jobs=-1)
clf = RFE(model,7 )

X = hr.drop(['left'],1)
y = hr.left 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    
clf.fit(X_train, y_train)
print(clf.support_)
print(clf.ranking_)
# Gets same result as feature_importance, which makes sense
```

    [ True  True  True  True  True False False  True  True]
    [1 1 1 1 1 2 3 1 1]



```python
# from sklearn.feature_selection import RFE
# model = svm.SVC(kernel='linear')
# clf = RFE(model,5 )

# X = hr.drop(['left'],1)
# y = hr.left 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    
# clf.fit(X_train, y_train)
# print(clf.support_)
# print(clf.ranking_)
# # Gets same result as feature_importance
```


```python
# PCA on relevant features
drop = ['left',  'sales', 'salary']

y = np.array(hr.left)

X = np.array(hr.drop(drop,1))
pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)
```


```python
print(pca.components_)
```

    [[ -1.00023273e-04   1.16454622e-03   1.03016962e-02   9.99939260e-01
        3.73903652e-03  -7.14279278e-05  -1.02210933e-05]
     [ -2.15563235e-02   1.53851479e-02   2.72466874e-01  -6.42326612e-03
        9.61758360e-01   7.45692132e-04   6.28225738e-03]]



```python
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))
```

    [  9.98565340e-01   8.69246970e-04]
    0.999434587329



```python
# Still get good accuracy, ~97%
clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
y = np.array(hr.left) 

scores = cross_val_score(clf, X_pca, y, cv=CV_FOLDS, n_jobs=-1)
    
    
    
print('Cross val score: ', sum(scores) / CV_FOLDS )
print(scores)
```

    Cross val score:  0.970532643193
    [ 0.97700766  0.95133333  0.95766667  0.984995    0.98166055]



```python
colors = itertools.cycle('rb')
target_ids = range(2)
plt.figure()
for i, c, label in zip(target_ids, colors, ["stay","left"]):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                c=c, label=label)
plt.legend()
plt.show()
```


![png](HR%20Post_files/HR%20Post_28_0.png)



```python

```


```python

```


```python

```


```python

```


```python
# 3D PCA, Clear plane difference between them
drop = ['left',  'sales', 'salary']

y = np.array(hr.left)

X = np.array(hr.drop(drop,1))
pca = PCA(n_components=3).fit(X)
X_pca = pca.transform(X)

colors = itertools.cycle('rb')
target_ids = range(2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, c, label in zip(target_ids, colors, ["stay","left"]):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],X_pca[y == i, 2],
                c=c, label=label)
    
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
    
plt.legend()
plt.show()
```

    /home/altock/anaconda3/envs/py35/lib/python3.5/site-packages/matplotlib/collections.py:865: RuntimeWarning: invalid value encountered in sqrt
      scale = np.sqrt(self._sizes) * dpi / 72.0 * self._factor



![png](HR%20Post_files/HR%20Post_33_1.png)



```python
# 3D accuracy is the same as 2D
clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
y = hr.left 

scores = cross_val_score(clf, X_pca, y, cv=CV_FOLDS, n_jobs=-1)
    
    
    
print('Cross val score: ', sum(scores) / CV_FOLDS )
print(scores)
```

    Cross val score:  0.97146639883
    [ 0.97567478  0.95166667  0.95833333  0.98332778  0.98832944]



```python
# Attempt to find department level features, doesn't seem to matter
hr_corr = hr.corr()
hr_corr["sales"]
```




    satisfaction_level       0.003153
    last_evaluation          0.007772
    number_project           0.009268
    average_montly_hours     0.003913
    time_spend_company      -0.018010
    Work_accident            0.003425
    left                     0.032105
    promotion_last_5years   -0.027336
    sales                    1.000000
    salary                   0.000685
    Name: sales, dtype: float64




```python
hr.columns.values
```




    array(['satisfaction_level', 'last_evaluation', 'number_project',
           'average_montly_hours', 'time_spend_company', 'Work_accident',
           'left', 'promotion_last_5years', 'sales', 'salary'], dtype=object)




```python
for col in hr.columns:
    depProperties = hr.groupby('sales').agg({col:[np.size,np.mean]})
    print(depProperties)
```

          satisfaction_level          
                        size      mean
    sales                             
    0                 1227.0  0.618142
    1                  787.0  0.619822
    2                  767.0  0.582151
    3                  739.0  0.598809
    4                  630.0  0.621349
    5                  858.0  0.618601
    6                  902.0  0.619634
    7                 4140.0  0.614447
    8                 2229.0  0.618300
    9                 2720.0  0.607897
          last_evaluation          
                     size      mean
    sales                          
    0              1227.0  0.716830
    1               787.0  0.712122
    2               767.0  0.717718
    3               739.0  0.708850
    4               630.0  0.724000
    5               858.0  0.715886
    6               902.0  0.714756
    7              4140.0  0.709717
    8              2229.0  0.723109
    9              2720.0  0.721099
          number_project          
                    size      mean
    sales                         
    0               1227  3.816626
    1                787  3.853875
    2                767  3.825293
    3                739  3.654939
    4                630  3.860317
    5                858  3.687646
    6                902  3.807095
    7               4140  3.776329
    8               2229  3.803948
    9               2720  3.877941
          average_montly_hours            
                          size        mean
    sales                                 
    0                     1227  202.215974
    1                      787  200.800508
    2                      767  201.162973
    3                      739  198.684709
    4                      630  201.249206
    5                      858  199.385781
    6                      902  199.965632
    7                     4140  200.911353
    8                     2229  200.758188
    9                     2720  202.497426
          time_spend_company          
                        size      mean
    sales                             
    0                   1227  3.468623
    1                    787  3.367217
    2                    767  3.522816
    3                    739  3.355886
    4                    630  4.303175
    5                    858  3.569930
    6                    902  3.475610
    7                   4140  3.534058
    8                   2229  3.393001
    9                   2720  3.411397
          Work_accident          
                   size      mean
    sales                        
    0              1227  0.133659
    1               787  0.170267
    2               767  0.125163
    3               739  0.120433
    4               630  0.163492
    5               858  0.160839
    6               902  0.146341
    7              4140  0.141787
    8              2229  0.154778
    9              2720  0.140074
           left          
           size      mean
    sales                
    0      1227  0.222494
    1       787  0.153748
    2       767  0.265971
    3       739  0.290934
    4       630  0.144444
    5       858  0.236597
    6       902  0.219512
    7      4140  0.244928
    8      2229  0.248991
    9      2720  0.256250
          promotion_last_5years          
                           size      mean
    sales                                
    0                      1227  0.002445
    1                       787  0.034307
    2                       767  0.018253
    3                       739  0.020298
    4                       630  0.109524
    5                       858  0.050117
    6                       902  0.000000
    7                      4140  0.024155
    8                      2229  0.008973
    9                      2720  0.010294
          sales     
           size mean
    sales           
    0      1227    0
    1       787    1
    2       767    2
    3       739    3
    4       630    4
    5       858    5
    6       902    6
    7      4140    7
    8      2229    8
    9      2720    9
          salary          
            size      mean
    sales                 
    0       1227  1.368378
    1        787  1.407878
    2        767  1.340287
    3        739  1.424899
    4        630  1.000000
    5        858  1.344988
    6        902  1.349224
    7       4140  1.363043
    8       2229  1.359354
    9       2720  1.347794



```python
clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
X = hr.drop(drop,1)
y = hr.left 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```


```python
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=10)

class_names = ["Stay","Left"]

plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')


plt.show()
```


![png](HR%20Post_files/HR%20Post_39_0.png)



```python
print(classification_report(y_test,y_pred, target_names=class_names))
```

                 precision    recall  f1-score   support
    
           Stay       0.99      1.00      0.99      2296
           Left       0.99      0.96      0.98       704
    
    avg / total       0.99      0.99      0.99      3000
    



```python

```
