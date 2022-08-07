# Contents

__[1. Introduction](#Introduction)__

__[2. Loading modules and data](#Loading-modules-and-data)__

__[3. Preprocessing](#Preprocessing)__

__[4. Exploratory analysis and feature engineering](#Exploratory-analysis-and-feature-engineering)__  
    [4.1. Profiling variables](#Profiling-variables)  
    [4.1. Data transformation and Clustering variables](#Data-transformation-and-Clustering-variables)  

__[5. Model Selection](#Model-Selection)__

__[6. Analysis](#Analysis)__



# Introduction

Clustering is an unsupervised machine learning task, involving discovering groups in data. Clustering helps with pattern discovery. This project aims to use clustering approaches to perform customer segmentation on [Online Retail data](https://www.kaggle.com/datasets/vijayuv/onlineretail).

### Use cases:
- __data summarization__
    - clustering is a step for classification or outlier analysis
    - dimensionality reduction
- __collaborative filtering__
    - grouping of users with similar interests
- __customer segmentation__
    - grouping of customers
- __dynamic trend detection__
    - in social networks:  data is dynamically clustered in a streaming fashion and used to determine patterns of changes.
- __multimedia data analysis__
    - detecting similar areas in images, video, audio.
- __social network analysis__
    - detecting communities

### Validation
- use __case studies__ to illustrate the subjective quality of the clusters
- __measures of the clusters__ (cluster radius or density)
    - can be biased (measures could favor different algorithms in a different way)
- labels can be given to data points - then __correlations of the clusters with the labels__ can be used
    - class labels may not always align with the natural clusters


### Approach

1. Define goals: find users that are similar in important ways to the business (producs, usage, demographics, channels, etc) and:
    - discover how business metrics differ between them.
    - use that information to improve existing models.
    - tailor marketing strategy to each customer segment.


2. Data:
    1. Behavioural data (transactions):
        - visits, usage, penetration responses, engagement, lifestyle, preferences, channel preferences, etc.
        - number of times a user purchased, how much, what products and categories.
        - number of transactions over a period of time, number of units.
    1. Additional data:
        1. User side:
            - time between purchases, categories purchased, peaks and valleys of transactions, units and revenue, share of categories, number of units and transactions per user, percentage of discounts per user, top N categories purchased per user.
        1. Company side:
            - seasonality variables, featured categories, promotions in place.
        1. Third party data:
            - demographics, interests, attitudes, lifestyles.


3. Implement a model:
    - model should be multivariate, multivariable, probabilistic (e.g. LCA).
    - run model (e.g. linear regression) for each segment separately, thus taking into account different user profiles.


4. Analyse returned segments:
    - some segments could be price sensitive, prefer one channel, have high penetration of a particular product, prefer a certain way of communication.
    - we expect to find a segment that is penetrated in one category, but not another.
    - profiling:
        - profile - what is shown to managers as a proof that the segments are different:
            - KPIs.
            - indexes (e.g. take each segment's mean and divide by total mean to show how a segment is different from the rest in percentage).
    - name the segments (e.g. high revenue, low response, etc.)


5. Act based on learnt information:
    - e.g. if a segment is price sensitive, users should get a discount to motivate them to make a purchase.


__[🔼](#Contents)__


# Loading modules and data


```python
import pandas as pd
import numpy as np
from importlib import reload
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.cluster import SpectralClustering, OPTICS, MeanShift, KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import seaborn as sns

```


```python
import tools as t
reload(t)

from tools.preprocessing import eda
from tools.modeling import clustering

data = eda.Dataset(
    features=['StockCode', 'InvoiceDate', 'Country', 'Quantity', 'UnitPrice', 'CustomerID'],
    features_ohe=['StockCode', 'Country'],
)
print(data)
```

    Data transformation class.
    ---------------------------
    Inputted features: ['StockCode', 'InvoiceDate', 'Country', 'Quantity', 'UnitPrice', 'CustomerID'].
    ---------------------------
    Transformation steps:
    1. Correct data types
    2. Feature engineering: Revenue
    3. One Hot Encoding of ['StockCode', 'Country']


__[🔼](#Contents)__

# Preprocessing


```python
df0 = data.get_transformed()
```

__[🔼](#Contents)__

# Exploratory analysis and feature engineering

## Profiling variables




```python
df_profiling = data.get_profiling_df()
df_profiling
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#_stockCode</th>
      <th>#_InvoiceNo</th>
      <th>avg_Q</th>
      <th>avg_P</th>
      <th>avg_Revenue</th>
      <th>HighRevenueMonth</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17850</th>
      <td>312.0</td>
      <td>35.0</td>
      <td>5.426282</td>
      <td>3.924712</td>
      <td>16.950737</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>17850</th>
      <td>312.0</td>
      <td>35.0</td>
      <td>5.426282</td>
      <td>3.924712</td>
      <td>16.950737</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>17850</th>
      <td>312.0</td>
      <td>35.0</td>
      <td>5.426282</td>
      <td>3.924712</td>
      <td>16.950737</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>17850</th>
      <td>312.0</td>
      <td>35.0</td>
      <td>5.426282</td>
      <td>3.924712</td>
      <td>16.950737</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>17850</th>
      <td>312.0</td>
      <td>35.0</td>
      <td>5.426282</td>
      <td>3.924712</td>
      <td>16.950737</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12680</th>
      <td>52.0</td>
      <td>4.0</td>
      <td>8.519231</td>
      <td>3.637885</td>
      <td>16.592500</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>12680</th>
      <td>52.0</td>
      <td>4.0</td>
      <td>8.519231</td>
      <td>3.637885</td>
      <td>16.592500</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>12680</th>
      <td>52.0</td>
      <td>4.0</td>
      <td>8.519231</td>
      <td>3.637885</td>
      <td>16.592500</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>12680</th>
      <td>52.0</td>
      <td>4.0</td>
      <td>8.519231</td>
      <td>3.637885</td>
      <td>16.592500</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>12680</th>
      <td>52.0</td>
      <td>4.0</td>
      <td>8.519231</td>
      <td>3.637885</td>
      <td>16.592500</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
<p>541909 rows × 6 columns</p>
</div>



__[🔼](#Contents)__

## Data transformation and Clustering variables

These are variables that will be used in clustering algorithm.

The following transformations will be applied to them:
- Observations with missing values will be dropped.
- One Hot Encoding will be used to encode categorical variables (`'StockCode', 'Country'`).
- We will also break down `InvoiceDate` into Year, Month, Day.
- `Description` will be dropped since strings can't be used in clustering algorithms.




```python
df_clustering = data.get_clustering_df()
df_clustering
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quantity</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>InvoiceYear</th>
      <th>InvoiceMonth</th>
      <th>InvoiceDay</th>
      <th>StockCode_10002</th>
      <th>StockCode_10080</th>
      <th>StockCode_10120</th>
      <th>StockCode_10123C</th>
      <th>...</th>
      <th>Country_RSA</th>
      <th>Country_Saudi Arabia</th>
      <th>Country_Singapore</th>
      <th>Country_Spain</th>
      <th>Country_Sweden</th>
      <th>Country_Switzerland</th>
      <th>Country_USA</th>
      <th>Country_United Arab Emirates</th>
      <th>Country_United Kingdom</th>
      <th>Country_Unspecified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>2.55</td>
      <td>17850</td>
      <td>2010</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>3.39</td>
      <td>17850</td>
      <td>2010</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>2.75</td>
      <td>17850</td>
      <td>2010</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>3.39</td>
      <td>17850</td>
      <td>2010</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>3.39</td>
      <td>17850</td>
      <td>2010</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>541904</th>
      <td>12</td>
      <td>0.85</td>
      <td>12680</td>
      <td>2011</td>
      <td>9</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>541905</th>
      <td>6</td>
      <td>2.10</td>
      <td>12680</td>
      <td>2011</td>
      <td>9</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>541906</th>
      <td>4</td>
      <td>4.15</td>
      <td>12680</td>
      <td>2011</td>
      <td>9</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>541907</th>
      <td>4</td>
      <td>4.15</td>
      <td>12680</td>
      <td>2011</td>
      <td>9</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>541908</th>
      <td>3</td>
      <td>4.95</td>
      <td>12680</td>
      <td>2011</td>
      <td>9</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>541909 rows × 4114 columns</p>
</div>




```python
df_clustering.dropna(inplace=True)
```


```python
# df_clustering = df_clustering.iloc[:10000, :]
df_clustering
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quantity</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>InvoiceYear</th>
      <th>InvoiceMonth</th>
      <th>InvoiceDay</th>
      <th>StockCode_10002</th>
      <th>StockCode_10080</th>
      <th>StockCode_10120</th>
      <th>StockCode_10123C</th>
      <th>...</th>
      <th>Country_RSA</th>
      <th>Country_Saudi Arabia</th>
      <th>Country_Singapore</th>
      <th>Country_Spain</th>
      <th>Country_Sweden</th>
      <th>Country_Switzerland</th>
      <th>Country_USA</th>
      <th>Country_United Arab Emirates</th>
      <th>Country_United Kingdom</th>
      <th>Country_Unspecified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>2.55</td>
      <td>17850</td>
      <td>2010</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>3.39</td>
      <td>17850</td>
      <td>2010</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>2.75</td>
      <td>17850</td>
      <td>2010</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>3.39</td>
      <td>17850</td>
      <td>2010</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>3.39</td>
      <td>17850</td>
      <td>2010</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>541904</th>
      <td>12</td>
      <td>0.85</td>
      <td>12680</td>
      <td>2011</td>
      <td>9</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>541905</th>
      <td>6</td>
      <td>2.10</td>
      <td>12680</td>
      <td>2011</td>
      <td>9</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>541906</th>
      <td>4</td>
      <td>4.15</td>
      <td>12680</td>
      <td>2011</td>
      <td>9</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>541907</th>
      <td>4</td>
      <td>4.15</td>
      <td>12680</td>
      <td>2011</td>
      <td>9</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>541908</th>
      <td>3</td>
      <td>4.95</td>
      <td>12680</td>
      <td>2011</td>
      <td>9</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>406829 rows × 4114 columns</p>
</div>



__[🔼](#Contents)__

# Model Selection


```python
import tools as t
reload(t)
from tools.modeling import clustering

clustering = clustering.Clustering(df_clustering)
```


```python
df_clustering
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quantity</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>InvoiceYear</th>
      <th>InvoiceMonth</th>
      <th>InvoiceDay</th>
      <th>StockCode_10002</th>
      <th>StockCode_10080</th>
      <th>StockCode_10120</th>
      <th>StockCode_10123C</th>
      <th>...</th>
      <th>Country_RSA</th>
      <th>Country_Saudi Arabia</th>
      <th>Country_Singapore</th>
      <th>Country_Spain</th>
      <th>Country_Sweden</th>
      <th>Country_Switzerland</th>
      <th>Country_USA</th>
      <th>Country_United Arab Emirates</th>
      <th>Country_United Kingdom</th>
      <th>Country_Unspecified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>2.55</td>
      <td>17850</td>
      <td>2010</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>3.39</td>
      <td>17850</td>
      <td>2010</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>2.75</td>
      <td>17850</td>
      <td>2010</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>3.39</td>
      <td>17850</td>
      <td>2010</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>3.39</td>
      <td>17850</td>
      <td>2010</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>541904</th>
      <td>12</td>
      <td>0.85</td>
      <td>12680</td>
      <td>2011</td>
      <td>9</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>541905</th>
      <td>6</td>
      <td>2.10</td>
      <td>12680</td>
      <td>2011</td>
      <td>9</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>541906</th>
      <td>4</td>
      <td>4.15</td>
      <td>12680</td>
      <td>2011</td>
      <td>9</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>541907</th>
      <td>4</td>
      <td>4.15</td>
      <td>12680</td>
      <td>2011</td>
      <td>9</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>541908</th>
      <td>3</td>
      <td>4.95</td>
      <td>12680</td>
      <td>2011</td>
      <td>9</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>406829 rows × 4114 columns</p>
</div>




```python
name = 'Kmeans'
model = KMeans(n_clusters=3, random_state=42)
steps = [
#     ('scaler', StandardScaler())
]
plot=True

model_kmeans, ypred_kmeans = clustering.check_model(name, model, steps, plot)
```

    KMeans(n_clusters=3, random_state=42)




![png](README_files/README_16_1.png)




```python
df_clustering['clusters'] = ypred_kmeans
df_clustering_res = df_clustering[['CustomerID', 'clusters']].copy()
```

__[🔼](#Contents)__

# Analysis


```python
df_res = pd.merge(df_clustering_res, df_profiling.drop_duplicates(), how='left', left_on='CustomerID', right_index=True)
df_res
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>clusters</th>
      <th>#_stockCode</th>
      <th>#_InvoiceNo</th>
      <th>avg_Q</th>
      <th>avg_P</th>
      <th>avg_Revenue</th>
      <th>HighRevenueMonth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17850</td>
      <td>1</td>
      <td>312.0</td>
      <td>35.0</td>
      <td>5.426282</td>
      <td>3.924712</td>
      <td>16.950737</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17850</td>
      <td>1</td>
      <td>312.0</td>
      <td>35.0</td>
      <td>5.426282</td>
      <td>3.924712</td>
      <td>16.950737</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17850</td>
      <td>1</td>
      <td>312.0</td>
      <td>35.0</td>
      <td>5.426282</td>
      <td>3.924712</td>
      <td>16.950737</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17850</td>
      <td>1</td>
      <td>312.0</td>
      <td>35.0</td>
      <td>5.426282</td>
      <td>3.924712</td>
      <td>16.950737</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17850</td>
      <td>1</td>
      <td>312.0</td>
      <td>35.0</td>
      <td>5.426282</td>
      <td>3.924712</td>
      <td>16.950737</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>541904</th>
      <td>12680</td>
      <td>0</td>
      <td>52.0</td>
      <td>4.0</td>
      <td>8.519231</td>
      <td>3.637885</td>
      <td>16.592500</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>541905</th>
      <td>12680</td>
      <td>0</td>
      <td>52.0</td>
      <td>4.0</td>
      <td>8.519231</td>
      <td>3.637885</td>
      <td>16.592500</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>541906</th>
      <td>12680</td>
      <td>0</td>
      <td>52.0</td>
      <td>4.0</td>
      <td>8.519231</td>
      <td>3.637885</td>
      <td>16.592500</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>541907</th>
      <td>12680</td>
      <td>0</td>
      <td>52.0</td>
      <td>4.0</td>
      <td>8.519231</td>
      <td>3.637885</td>
      <td>16.592500</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>541908</th>
      <td>12680</td>
      <td>0</td>
      <td>52.0</td>
      <td>4.0</td>
      <td>8.519231</td>
      <td>3.637885</td>
      <td>16.592500</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
<p>406829 rows × 8 columns</p>
</div>




```python
sns.pairplot(df_res[['clusters', 'avg_Revenue', 'HighRevenueMonth', '#_InvoiceNo', '#_stockCode', 'avg_Q', 'avg_P']],
             hue="clusters",
            palette=sns.color_palette("hls", 3))

```




    <seaborn.axisgrid.PairGrid at 0x2212f18a490>





![png](README_files/README_20_1.png)


