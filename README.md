# Data Imputation Techniques


This notebook presents some **Data Imputation Techniques** using the training dataset provided by the WIDS Datathon 2021. We already performed an exploratory data analysis in this data. If you haven't seen it, you can find it <a href="https://github.com/chainao/EDA-WiDS_Datathon_2021">here</a>.

The main goal of this study was to investigate **how mean and standard deviation are affected by different techniques of data imputation**. 

We explore two types of data imputation, univariate and multivariate. 

The univariate techniques we explore are:

    - Constant
    - Mean
    - Median
    - Most Frequent
    
The multivariate methods are:

    - KNN
    - Iterative
    
    
We used <a href="https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer">SimpleImputer</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html#sklearn.impute.KNNImputer">KNNImputer</a> and  <a href="https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer">IterativeImputer</a> from Sklearn in the imputation process. For the visualization, we adopted <a href="http://seaborn.pydata.org/">seaborn</a> and <a href="https://matplotlib.org/">matplotlib</a>.

Hope you enjoy it ;)