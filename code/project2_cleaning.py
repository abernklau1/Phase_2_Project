import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.formula.api import ols
import statsmodels.api as sm
import pandas as pd
import scipy.stats as stats

def splitting(data):
    y = data['price']
    X = data.drop(columns='price', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.75)

    return X_train, X_test, y_train, y_test

def dropped(X):
    
    X.drop(columns=['bathrooms', 'floors', 'yr_built', 'grade', 'view', 'waterfront', 
    'date', 'zipcode', 'yr_renovated', 'lat', 'long', 'sqft_lot', 'sqft_living15', 
    'sqft_lot15', 'sqft_above', 'sqft_basement'], inplace=True)
    
    dropped_feats = ['bathrooms', 'floors', 'yr_built', 'grade', 'view', 'waterfront', 
    'date', 'zipcode', 'yr_renovated', 'lat', 'long', 'sqft_lot', 'sqft_living15', 
    'sqft_lot15', 'sqft_above', 'sqft_basement']
    
    print('Features dropped:')
    print(dropped_feats)
    
    return X

def rem_outliers(X, y):
    
    df = pd.concat([X, y], axis=1)
    std_three = X['sqft_living'].std()*3
    threshold = X['sqft_living'].mean() + std_three
    df = df[df['sqft_living'] <= threshold]
    X = df.drop(columns='price', axis=1) 
    y = df['price']
    
    print(f'Houses with square feet of living greater than {round(threshold, 2)} removed.')
    
    return X, y

def ohe_train(X):
        
    #fit and transform the initiated OneHotEncoder
    o_h_e = OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore')
    X_condition = X[['condition']]
    X_fittransform = o_h_e.fit_transform(X_condition)
    #create a new dataframe
    condition_df = pd.DataFrame(X_fittransform, columns=o_h_e.categories_[0], 
                                index=X.index)
    #drop the old column
    X.drop(columns='condition', inplace=True)
    #add the new ones
    X = pd.concat([X, condition_df], axis=1)
    #strip white space from very good
    X.columns = X.columns.str.replace(" ", "")
    
    print('Categories encoded and added as columns:')
    print(o_h_e.categories_[0])
    
    return o_h_e, X

def ohe_test(o_h_e, X):
    X_condition = X[['condition']]
    X_transform = o_h_e.transform(X_condition)
    condition_df = pd.DataFrame(X_transform, columns=o_h_e.categories_[0], 
                                index=X.index)
    X.drop(columns='condition', inplace=True)
    X = pd.concat([X, condition_df], axis=1)
    X.columns = X.columns.str.replace(" ", "")
    
    print('Categories encoded and added as columns:')
    print(o_h_e.categories_[0])
    
    return X

def dummy_modeling(X_train, X_test, y_train, y_test):

    X_train = X_train['id']
    dr = DummyRegressor(strategy='mean')
    model = dr.fit(X_train, y_train)
    train_y_hat = dr.predict(X_train)
    train_score = model.score(X_train, y_train)
    #repeat with test data
    X_test = X_test['id']
    test_score = dr.score(X_test,y_test)
    test_y_hat = dr.predict(X_test)
    print('R^2 of train set:')
    print(train_score)

    return train_score, train_y_hat, test_score, test_y_hat

def regression_single(X_train, X_test, y_train, y_test):
    df_train = pd.concat([X_train, y_train], axis=1)
    f= 'price~sqft_living'
    model = ols(formula=f, data=df_train).fit()
    X_bathrooms_train = X_train[['sqft_living']]
    train_y_hat = model.predict(X_bathrooms_train)
    #repeat for test data 
    X_bathrooms_test = X_test[['sqft_living']]
    test_y_hat = model.predict(X_bathrooms_test)
    residuals = model.resid
    print(model.summary())
    
    return train_y_hat, test_y_hat, residuals
    

def regression_multiple(X_train, X_test, y_train, y_test):
    df_train = pd.concat([X_train, y_train], axis=1)
    f= 'price~sqft_living+bedrooms+VeryGood+Good'
    model = ols(formula=f, data=df_train).fit()
    train_y_hat = model.predict(X_train)
    test_y_hat = model.predict(X_test)
    residuals = model.resid
    print(model.summary())
    
    return train_y_hat, test_y_hat, residuals


def calc_rmse(y, y_hat):
    rmse = []
    rmse.append(np.sqrt(mean_squared_error(y, y_hat)))
    print('Root Mean Squared Error:')
    print(rmse)
    
    return rmse

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            #if verbose:
                #print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            #if verbose:
                #print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    included.append('id')
    print('resulting features:')
    print(included)
    
    return included

def resid_map(residuals):
    fig=sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
    return fig.show()