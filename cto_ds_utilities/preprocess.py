import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import *
from collections import defaultdict
from .explore import value_dist
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

agg_feats = [np.min, np.max, np.mean, np.std, np.sum, len, 'nunique', 'skew']

def get_datetime_feat(df: pd.DataFrame, col: str,feats: list=['year','month','day','hour','week','dayofweek','dayofyear','quarter',
                                    'is_month_start','is_month_end','is_quarter_start','is_quarter_end',
                                   'is_year_start','is_year_end'],drop: bool=True) -> pd.DataFrame:
    '''
        Extract numerical features from datetime column as specified in 'feats'.
        Datetime features are referenced from Datetime properties of Series.dt object
        see https://pandas.pydata.org/pandas-docs/stable/reference/series.html#time-series-related for all properties
    '''
    df = df.copy()
    # cast to datetime
    df[col] = pd.to_datetime(df[col])
    # create column of each datetime feature
    for feat in feats:
        df[feat] = getattr(df[col].dt,feat) # equivalent df[col].df.feat
        if 'is_' in feat: df[feat] = df[feat].astype(int)
    if drop: df = df.drop(col,axis=1)
    return df

def split_regression_validation(X: pd.DataFrame, y: pd.Series, num_bin: int = 50, test_size: float = 0.2, seed: int = 42) -> list:
    """
        Split regression dataset using bins for balancing continuous target variable
        
        Output
        --------
        X_train, X_test, y_train, y_test
    """
    bins = pd.cut(y, num_bin, labels = [str(i) for i in range(num_bin)])
    if X is None: return train_test_split(y, test_size=test_size, random_state=seed, stratify=bins)
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=bins)

def remove_outlier(df: pd.DataFrame, col:str, factor: float=1.5) -> pd.DataFrame:
    '''
        Remove outlier based one the specified feature column and factor

        Input
        ---------
        factor: factor when multiply IQR (default = 1.5IQR)
    '''
    df = df.copy()
    iqr = np.percentile(df[col],[75]) - np.percentile(df[col],[25])
    upper_thd = (np.percentile(df[col],[75]) + factor*iqr)[0]
    lower_thd = (np.percentile(df[col],[25]) - factor*iqr)[0]
    return df[(lower_thd<= df[col]) & (df[col] <= upper_thd)]

def drop_train_set_outlier(X_train: pd.DataFrame, y_train: pd.Series) -> list:
    """
    Drop training set outlier only

    Input
    ----------
    Use this function with remove_outlier function

    Output
    ----------
    X_train_drop_outlier, y_train_drop_outlier
    """
    y_train_df = pd.DataFrame(y_train).copy()
    y_train_drop_outlier = remove_outlier(y_train_df, y_train.name)
    print(f"Shape of y train after dropping outlier: {y_train_drop_outlier.shape}")
    X_train_drop_outlier = X_train.loc[X_train.index.isin(y_train_drop_outlier.index)]
    assert y_train_drop_outlier.shape[0] == X_train_drop_outlier.shape[0]
    return X_train_drop_outlier, y_train_drop_outlier


def fix_missing(df: pd.DataFrame, num_cols: list, cat_cols: list, cast_cat_to_str: bool=True) -> pd.DataFrame:
    '''
    ***IMPORTANT: fix missing of different sets of data seperately to avoid information leakage
    Fix missing columns for both numerical and categorical columns.
    Numeric columns: -> fill with medain + add a flag column
    Categorical columns: -> fill with -1 for already encoded one or 'xxna' for str/object column
    '''    
    # listtify input columns
    num_cols = listify(num_cols)
    cat_cols = listify(cat_cols)
    
    df_num = df[num_cols].copy()
    df_cat = df[cat_cols].copy()
    
    # fix numerical columns
    for col in df_num.columns:
        df_num[f'{col}_missing'] = df_num[col].isna().astype(int)
        df_num.loc[:,col] = df_num.loc[:,col].fillna(df_num[col].median())
    
    # fix categorical columns
    for col in df_cat.columns:

        if cast_cat_to_str:
            df_cat[col] = df_cat[col].fillna('xxna').astype(str)
            continue
        
        # categorical columns as numerical type
        if np.issubdtype(df_cat[col].dtype, np.number):
            df_cat[col] = df_cat[col].fillna(-1)
        # categorical columns as str type
        else:
            df_cat[col] = df_cat[col].fillna('xxna')

    return pd.concat([df_num,df_cat],axis=1)

def otherify(df: pd.DataFrame, cols: Collection, th: float= 0.01, retain: str= ['xxna']) -> pd.DataFrame:
    """
        Define number of categories less than 'th'% to other
    """
    for col in cols:
        counter = Counter(df[col].values)
        ratio = {k:v/len(df) for k,v in counter.items()}
        df[col] = [val if ratio[val] > th else 'others' for val in df[col].values]
    return df

def get_enc_dec_from_cols(df: pd.DataFrame,cols: list):
    '''
    Create encoder and decoder that map categoical feature value to and from integer index
    '''
    cols = listify(cols)
    encoder = dict()
    decoder = dict()
    for col in cols:
        encoder[col] = defaultdict(lambda:0,{val:i for i,val in enumerate(df[col].unique(),1)})
        dec = ['unk']
        dec.extend(list(df[col].unique()))
        decoder[col] = dec
    return encoder, decoder

def encode_df(df: pd.DataFrame, encs: defaultdict) -> pd.DataFrame:
    '''
    Encode categorical feature of Dataframe based on the encoder's key
    '''
    df = df.copy()
    for col in encs: df[col] = df[col].map(lambda x:encs[col][x])
    return df

def get_bin_idx(x,thds):
    for i, thd in enumerate(thds):
        if x<=thd: return i
    return len(thds)

def get_bin_from_cont(arr: Iterable, num_bins: int=4, thds: Iterable= None) -> Iterable:
    '''
    Return bin index of continous variable in form of array
    '''
    # convert to numpy array
    arr = np.array(arr)
    if thds is None:
        # get binning threshold
        thds = [interval.right for interval in pd.qcut(arr.reshape(-1), num_bins).categories][:-1]

    out = np.array([get_bin_idx(i[0],thds) for i in arr])
    return out, thds

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, Iterable): return list(o)
    return [o]

def split_data(cat_col: list, num_col: list, target_col: str, df: pd.DataFrame, 
                regression_flag: bool= True, test_size: float= .2, seed: int= 42) -> list:
    """
    Split data using categorical and numerical columns

    Input
    ---------
    regression_flag: If True using with split_regression_validation function

    Output
    ----------
    X_train, X_test, y_train, y_test
    """
    used_col = cat_col + num_col
    print(f"Used columns: {used_col}")
    if regression_flag:
        X_train, X_test, y_train, y_test = split_regression_validation(df[used_col], df[target_col], test_size=test_size, seed=seed)
    else:
        X_train, X_test, y_train, y_test = train_test_split(df[used_col], df[target_col], test_size=test_size, random_state=seed)
    print(f"Training size: {X_train.shape}")
    print(f"Testing size: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def get_one_hot_and_label_encode(df: pd.DataFrame) -> [np.array, np.array]:
    """
        Get One hot and label encoded of categorical features

        Output
        ---------
        oe_feature: Label encoded
        ohe_feature: One hot encoded
    """
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    oe = OrdinalEncoder()
    ohe_feature = ohe.fit_transform(df)
    oe_feature = oe.fit_transform(df)
    return oe_feature, ohe_feature

def norm_num_cols(df:pd.DataFrame ,cols:list, stats:dict=None):
    '''
    Alternative function to sklearn's StandardScaler
    Standard normalize DataFrame based on provided list of columns.
    Return normalized DataFrame and stat dict for normalizing valid/test data.
    '''
    if stats is None:
        stats = dict()
        for col in cols:
            stats[col] = [df[col].mean(),df[col].std()]
    for col in stats: df[col] = (df[col]-stats[col][0])/stats[col][1]
    return df, stats

def get_required_padding_len(toks, pct:int=75): return int(np.percentile([len(tok) for tok in toks],pct))

def zero_pad_tokens(enc_tokens:Collection,seq_length:int):
    '''
    zero pad tokens -> make sure that padding index is zero!!!
    function borrowed from written in in https://github.com/udacity/deep-learning-v2-pytorch/blob/master/sentiment-rnn/Sentiment_RNN_Solution.ipynb
    '''
    features = np.zeros((len(enc_tokens),seq_length),dtype=int)
    
    for i, toks in enumerate(enc_tokens):
        features[i, -len(toks):] = np.array(toks)[:seq_length]
        
    return features

def oversample_indices(indices_over:Collection, indices: Collection, p:float=9., size=20_000):
    # create sampling probability for indices
    probs = np.array([p if idx in indices_over else 1 for idx in indices])
    probs = probs/probs.sum()
    return np.random.choice(list(range(len(indices))),size, replace=False, p=probs)

# preallocate empty array and assign slice by chrisaycock
def shift_array(arr: np.array, num: int, fill_value: float=np.nan) -> np.array:
    """
        Shift array to corresponding axis
    """
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def use_past_label_to_feature(data: np.array, n_past: int=1, n_future: int=1) -> np.array:
    """
        Use past label to create feature engineering
    """
    result = []
    # input sequence in the past (t-1, ... t-n)  
    for i in range(n_past, 0, -1):
        result.append(shift_array(data, i))
    # forecast sequenct in the future (t, t+1,..., t+n)
    for i in range(0, n_future):
        result.append(shift_array(data, i)[:, -1].reshape(-1, 1))
    result = np.concatenate(result, axis=1)
    start_index = n_past + 1
    return result[start_index:]

def boxcox(ser: pd.Series, lamb=0) -> pd.Series:
    """
        Remove skewness by Box-Cox
    """

    ser+= 1 - ser.min()
    if lamb==0: 
        return np.log(ser)
    else:
        return (ser**lamb - 1)/lamb

def get_encoded_class_label(df: pd.DataFrame,col:str,col_name:str='label') -> tuple:
    '''
        Encode target column
        col: target column name
        col_name: output column name
    '''

    ctoi = {c:i for i,c in enumerate(df[col].unique())}
    itoc = [c for c in df[col].unique()]
    df[col_name] = df[col].apply(lambda x:ctoi[x])
    return df, ctoi, itoc

def save_objs(obj_list,name_list, dir):
    for obj,name in zip(obj_list,name_list): pickle.dump(obj,open(dir/f'{name}.pkl','wb'))