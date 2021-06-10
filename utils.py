import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import kruskal



def counting_missing_values(df):
    percent_missing = df.isnull().sum(axis=0)/len(df)
    percent_missing = percent_missing.sort_values(ascending=False)
    percent_missing = percent_missing.mul(1)
    percent_missing = percent_missing.rename('percent').reset_index()
    return percent_missing

def heat_map_per_type(df, colum_type, y):
    df_aux = df.select_dtypes(include=colum_type)
    percent_missing = counting_missing_values(df_aux)
    percents = pd.DataFrame(percent_missing, columns= ['percent'])
    plt.figure(figsize = (5,y))
    g = sns.heatmap(percents, annot=True, cmap='Blues', yticklabels=percent_missing['index'].to_numpy(), vmin=0, vmax=1)
    

def divide_missing_by_range(df_missing):
    values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    ranges = []
    percents = []

    for i in range(0, len(values) - 1):

        if values[i] < 1:    
            percent = (len(df_missing.loc[(df_missing['percent'] >= values[i]) & 
                                      (df_missing['percent'] < values[i + 1])]))/len(df_missing)
        else:
            percent = (len(df_missing.loc[(df_missing['percent'] >= values[i]) & 
                                      (df_missing['percent'] <= values[i + 1])]))/len(df_missing)

        percent = float(f'{percent:.4f}')
        ranges.append(str(values[i]) +" - "+ str(values[i+1]))
        percents.append(percent)    

    df_percents = pd.DataFrame(data={'index' : ranges,
                                     'percent': percents})
    
    return df_percents

def truncate(num, n):
    num = int(num * (10**n))/(10**n)
    return float(num)

def horizontal_bar_plot(df, title='', palette="Blues"):
    
    g = sns.catplot(x='percent',y='index',kind='bar',data=df, palette=palette, alpha=1)
        
    for p in g.ax.patches:
        if math.isnan(p.get_width().round(2)):
            percentage = txt_y = 0
        else:
            percentage = round(truncate(float(p.get_width()), 3) * 100)
            txt_y = p.get_width()
        txt = str(percentage) + '%'
        txt_x = p.get_y() + 0.5
        g.ax.text(txt_y,txt_x,txt)
        
    sns.despine(left=True, bottom=True)
    g.ax.set_title(title,fontsize='medium', loc='left')    
    g.set_axis_labels("", "")
    g.set(xticklabels=[])
    
def bar_plot(df, x, y, labels, title='', fontsize='large', palette='Blues_r', alpha=1):
    
    g = sns.catplot(x=x, y=y, data=df, kind='bar', aspect=2, palette=palette, alpha=alpha)
    g.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    g.set_axis_labels(labels[0], labels[1])
    g.set(xticklabels=[])
    g.ax.set_ylim(0, 1)
    sns.despine()
    g.ax.set_title(title,fontsize='medium', loc='left')
    
    
def mini_heatmaps(df):
    lim_min = 0
    fig = plt.figure(figsize = (10,20))
    fig.subplots_adjust(hspace=0.1, wspace=1.5)
    for i in range(1, 5):    
        lim_max = lim_min + 45
        df_aux = df.iloc[lim_min:lim_max]
        percents = pd.DataFrame(df_aux, columns= ['percent'])    
        ax = fig.add_subplot(2, 2, i)
        g = sns.heatmap(percents, annot=True, cmap='Blues', yticklabels=df_aux['index'].to_numpy(), vmin=0, vmax=1)    
        lim_min = lim_max
        
def data_imputation(df, imputation_techinique, data_type):
    df_aux = df.select_dtypes(include=data_type)
    df_columns=list(df_aux.columns)
    mean_before = df_aux.mean(axis = 0)
    std_before = df_aux.std(axis = 0)
    
    if imputation_techinique == 'knn':
        imp = KNNImputer(n_neighbors=2, weights="uniform")
    elif imputation_techinique == 'iterative':
        imp = IterativeImputer(max_iter=10, random_state=0)
    else:
        imp = SimpleImputer(missing_values=np.nan, strategy=imputation_techinique)
    
    df_aux = imp.fit_transform(df_aux)
    df_aux = pd.DataFrame(df_aux, columns=df_columns)
     
    mean_after = df_aux.mean(axis = 0)
    std_after = df_aux.std(axis = 0)
    
    return mean_before, std_before, mean_after, std_after

def join_mean_and_std_series(serie1, serie2, df_percents_missing, name_serie1, name_serie2):
    df_join = pd.concat([serie1, serie2], axis=1)
    df_join = df_join.rename(columns={0: name_serie1, 1: name_serie2})
    
    # Transforming percent in the dataframe index
    df_join['percent'] = df_percents_missing['percent'].to_numpy()
    df_join = df_join.set_index('percent')
    df_join.index.name = None
    
    return df_join

def line_plot(df, title='', ylabel='Data Mean'):
    plt.figure(figsize=(12,5))
    g = sns.lineplot(data=df)    
    sns.despine()    
    
    plt.xlabel("Percentage of Missing Data (of the 24 selected features)")    
    plt.ylabel(ylabel)
    plt.title(title)

def box_plot(df, title='', ylabel='Mean'):
    plt.figure(figsize=(12,5))
    g = sns.boxplot(x='variable', y='value', data=pd.melt(df))  
    sns.despine()    
    plt.xlabel('Data Imputation Technique')    
    plt.ylabel(ylabel)
    
def corrlation(df):
    corr = df.corr(method='spearman')

    plt.figure(figsize = (8,6))
    g = sns.heatmap(corr, 
                    xticklabels=corr.columns,
                    yticklabels=corr.columns,
                    cmap='Blues',
                    annot = True)
    
def wallis(df, df_name, include_constant=False):
    data1 = df['Mean']
    data2 = df['Median']
    data3 = df['Most Frequent']
    data4 = df['Constant']
    data5 = df['KNN']
    data6 = df['Iterative']
    
    if not include_constant:
        stat, p = kruskal(data1, data2, data3, data5, data6, df['Without Imputation'])
    else:
        stat, p = kruskal(data1, data2, data3, data4, data5, data6, df['Without Imputation'])
    print(df_name)
    print('p=%.3f' % ( p))
    if p > 0.05:
        print('probably the same distribution\n')
    else:
        print('probably different distributions\n')