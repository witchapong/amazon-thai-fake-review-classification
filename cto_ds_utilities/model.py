from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def make_model_pipeline(model: object, cat_col: list=None, num_col: list=None) -> object:
    """
        Create pipeline of model

        Output
        ---------
        model_pipeline: Full pipeline of model (ready to fit)
    """
    if cat_col:
        cat_transformer = Pipeline(steps=[('impute', SimpleImputer(strategy='most_frequent')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    if num_col:
        num_transformer = Pipeline(steps=[('impute', SimpleImputer(strategy='mean')),
                                          ('scaler', StandardScaler())])
    if (cat_col != None) & (num_col != None):
        preprocessor = ColumnTransformer(transformers=[('cat', cat_transformer, cat_col),
                                                       ('num', num_transformer, num_col)])
    else:
        preprocessor = cat_transformer or num_transformer

    model_pipeline = Pipeline(steps=[('preprocess', preprocessor),
                                     ('model', model)])
    return model_pipeline