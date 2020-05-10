# Thank you P.Charin for the data exploration idea
# Refer to https://github.com/cstorm125/viztech

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import *
from mizani.breaks import *
from mizani.formatters import *
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from typing import Collection, Callable, Tuple

def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
        Check percent of missing value
    """

    per_missing = df.isnull().mean()
    missing_df = pd.DataFrame({'col_name': df.columns, 'per_missing': per_missing})
    missing_df = missing_df.sort_values('per_missing',ascending=False).reset_index(drop=True)
    missing_df['rnk'] = missing_df.index.map(lambda x: str(x).zfill(2)+'_') + missing_df.col_name
    missing_df['over90'] = missing_df.per_missing.map(lambda x: True if x>0.9 else False)
    return missing_df

def plot_percent_missing(missing: pd.DataFrame) -> ggplot:
    """
        Plot percent of missing value in each column

        Input
        --------
        missing: Must have 'over90' new column to fill the color
    """

    g = (ggplot(missing,aes(x='rnk', y='per_missing', fill='over90')) + #base plot
        geom_col() + #type of plot 
        geom_text(aes(x='rnk', y='per_missing+0.1', label='round(100*per_missing,2)')) + #annotate
        scale_y_continuous(labels=percent_format()) + #y-axis tick
        theme_minimal() + coord_flip() #theme and flipping plot
        )
    return g

def plot_hist_num_dist(train_df: pd.DataFrame, col_name: str) -> ggplot:
    """
        Plot histogram with density to show numerical distribution

        Input
        -------
        train_df: Can insert remove outlier function to drop outlier before plotting
    """

    g = (ggplot(train_df, aes(x=col_name)) +
        geom_histogram(aes(y='..density..'), fill='white', color='black') +
        geom_density(alpha=.2, fill='#FF6666') +
        theme_minimal())
    return g

def plot_box_num_dist(train_df: pd.DataFrame, col_name: str) -> ggplot:
    """
        Plot box plot to show numerical distribution

        Input
        -------
        train_df: Can insert remove outlier function to drop outlier before plotting
    """

    g = (ggplot(train_df, aes(x=1, y=col_name)) +
        geom_boxplot() +
        theme_minimal())
    return g

def value_dist(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
        Find percent value count
    """

    x = pd.DataFrame(df[col].value_counts()).reset_index()
    x.columns = ['value','cnt']
    x['per'] = x.cnt / x.cnt.sum()
    return x

def list_cat_value_dist(cat_df: pd.DataFrame) -> widgets.Dropdown:
    """
        Show list of DF for categorical value distribution
    """

    return interact(value_dist, df =fixed(cat_df),
                    col = widgets.Dropdown(options=list(cat_df.columns), value=cat_df.columns[0]))

def plot_cat_dist(df: pd.DataFrame, col: str) -> ggplot:
    """
        Plot categorical value distribution to bar plot

        Output
        ---------
        ggplot: Can change first function in list_cat_value_dist to show graph
    """
    g = (ggplot(df,aes(x=col)) + 
         geom_bar(stat='bin', #histogram
                  binwidth=0.5, #histogram binwidth
                  bins=len(df[col].unique())) + #how many bins
         coord_flip()
        )
    return g

def interact_plot_cat_dist(df: pd.DataFrame, cat_cols: list):

    interact(plot_cat_dist,
        df=fixed(df),
        col= widgets.Dropdown(options=cat_cols, values=cat_cols[0]))

def plot_num_cat(df: pd.DataFrame, num: str, cat: str, no_outliers: bool= True, geom: ggplot= geom_boxplot()) -> ggplot:
    """
        Plot numerical and categorical variables

        Input
        -----------
        no_outliers: Add remove_outliers function to drop outlier
    """

    if no_outliers:
        new_df = remove_outliers(df,num)
    else:
        new_df = df.copy()
    g = (ggplot(new_df, aes(x=cat,y=num)) +
         geom 
        )
    return g

def interact_plot_num_cat():
    """
        Create interactive to plot numerical and categorical variables

        Input
        ---------
        Use with plot_num_cat function
    """

    interact(plot_num_cat, 
            df=fixed(cat_df),
            num=fixed('sales_price'),
            no_outliers = widgets.Checkbox(value=True),
            geom=fixed(geom_boxplot()), #geom_violin, geom_jitter
            cat= widgets.Dropdown(options=list(cat_df.columns)[:-1],value='gen'))

def plot_num_cat_dist(df: pd.DataFrame, num: str, cat: str, geom: ggplot= geom_density(alpha=0.5), no_outliers: bool= True) -> ggplot:
    """
        Plot numerical with 2 categorical variables
        Be useful plotting the results of a binary classification

        Input
        -----------
        no_outliers: Add remove_outliers function to drop outlier
    """

    if no_outliers:
        new_df = remove_outliers(df,num)
    else:
        new_df = df.copy()
    g = (ggplot(new_df,aes(x=num, fill=cat)) +
          geom 
        )
    return g

def plot_cat_cat(df: pd.DataFrame, cat_dep: str, cat_ind: str) -> ggplot:
    """
        Plot categorical with categorical variable
        Target is category and Feature is category
    """
    g = (ggplot(df,aes(x=cat_dep,fill=cat_dep)) + geom_bar() + 
         theme(axis_text_x = element_blank()) +
         facet_wrap(f'~{cat_ind}',scales='free_y')) + theme(panel_spacing_x=0.5)
    return g

def interact_plot_cat_cat(df: pd.DataFrame, cat_cols: list):
    """
        Create interactive to plot categorical with categorical variables

        Input
        ---------
        Use with plot_cat_cat function
    """

    interact(plot_cat_cat, 
            df=fixed(df),
            cat_dep=widgets.Dropdown(options=cat_cols,value=cat_cols[0]),
            cat_ind= widgets.Dropdown(options=cat_cols,value=cat_cols[1]))

def check_mode(df:pd.DataFrame) -> pd.DataFrame:
    """
        Show mode for categorical columns
    """
    mode_df = []
    for col in df.columns:
        x = df[col].value_counts()
        mode_df.append({'col':col, 'value':x.index[0], 'per_mode': list(x)[0]/df.shape[0],
                       'num_unique_value':len(x)})
    mode_df = pd.DataFrame(mode_df)[['col','value','per_mode','num_unique_value']]\
                .sort_values('per_mode',ascending=False)
    mode_df['col'] = pd.Categorical(mode_df.col, categories=mode_df.col, ordered=True)
    return mode_df.reset_index(drop=True)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    print(unique_labels(y_true, y_pred))
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def top_feats_label(X: np.ndarray, features: Collection[str], label_idx: Collection[bool] = None,
                    min_val: float = 0.1, agg_func: Callable = np.mean)->pd.DataFrame:
    '''
    original code (Thomas Buhrman)[from https://buhrmann.github.io/tfidf-analysis.html]
    rank features of each label by their encoded values (CountVectorizer, TfidfVectorizer, etc.)
    aggregated with `agg_func`
    see example at https://github.com/vistec-AI/wangchan-analytica/blob/master/predict_price.ipynb
    :param X np.ndarray: document-value matrix
    :param features Collection[str]: feature names
    :param label_idx Collection[int]: position of rows with specified label
    :param min_val float: minimum value to take into account for each feature
    :param agg_func Callable: how to aggregate features such as `np.mean` or `np.sum`
    :return: a dataframe with `feature`, `score` and `ngram`
    '''
    res = X[label_idx] if label_idx is not None else X
    res[res < min_val] = 0
    res_agg = agg_func(res, axis=0)
    df = pd.DataFrame([(features[i], res_agg[i]) for i in np.argsort(res_agg)[::-1]])
    df.columns = ['feature','score']
    df['ngram'] = df.feature.map(lambda x: len(set(x.split(' '))))
    return df

def top_feats_all(X: np.ndarray, y: np.ndarray, features: Collection[str], min_val: float = 0.1, 
                  agg_func: Callable = np.mean)->Collection[pd.DataFrame]:
    '''
    original code (Thomas Buhrman)[from https://buhrmann.github.io/tfidf-analysis.html]
    for all labels, rank features of each label by their encoded values (CountVectorizer, TfidfVectorizer, etc.)
    see example at https://github.com/vistec-AI/wangchan-analytica/blob/master/predict_price.ipynb
    aggregated with `agg_func`
    :param X np.ndarray: document-value matrix
    :param y np.ndarray: labels
    :param features Collection[str]: feature names
    :param min_val float: minimum value to take into account for each feature
    :param agg_func Callable: how to aggregate features such as `np.mean` or `np.sum`
    :return: a list of dataframes with `rank` (rank within label), `feature`, `score`, `ngram` and `label`
    '''
    labels = np.unique(y)
    dfs = []
    for l in labels:
        label_idx = (y==l)
        df = top_feats_label(X,features,label_idx,min_val,agg_func).reset_index()
        df['label'] = l
        df.columns = ['rank','feature','score','ngram','label']
        dfs.append(df)
    return dfs

def plot_top_feats(dfs: Collection[pd.DataFrame], top_n: int = 25, ngram_range: Tuple[int,int]=(1,2),)-> None:
    '''
    original code (Thomas Buhrman)[from https://buhrmann.github.io/tfidf-analysis.html]
    plot top features from a collection of `top_feats_all` dataframes
    see example at https://github.com/vistec-AI/wangchan-analytica/blob/master/predict_price.ipynb
    :param dfs Collection[pd.DataFrame]: `top_feats_all` dataframes
    :param top_n int: number of top features to show
    :param ngram_range Tuple[int,int]: range of ngrams for features to show
    :return: nothing
    '''
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(top_n)
    for i, df in enumerate(dfs):
        df = df[(df.ngram>=ngram_range[0])&(df.ngram<=ngram_range[1])][:top_n]
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("score", labelpad=16, fontsize=14)
        ax.set_title(f"label = {str(df.label[0])}", fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.score, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        ax.invert_yaxis()
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()
