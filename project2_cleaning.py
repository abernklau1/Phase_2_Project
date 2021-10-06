def splitting(data):
    from sklearn.model_selection import train_test_split
    y=df['price']
    X=df.drop(columns='price', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.75)
    return X_train, X_test, y_train, y_test

def cleaning(X_train, X_test, y_train, y_test):
    from sklearn.preprocessing import OneHotEncoder

    X_train.drop(columns=['sqft_living', 'floors', 'yr_built', 'grade', 'view', 'waterfront', /
    'date', 'zipcode', 'yr_renovated', 'lat', 'long', 'sqft_lot', 'sqft_living15', /
    'sqft_lot15', 'sqft_above', 'sqft_basement'], inplace=True)

    X_test.drop(columns=['sqft_living', 'floors', 'yr_built', 'grade', 'view', 'waterfront', /
    'date', 'zipcode', 'yr_renovated', 'lat', 'long', 'sqft_lot', 'sqft_living15', /
    'sqft_lot15', 'sqft_above', 'sqft_basement'], inplace=True)

    #removing price outliers from train
    df_train = pd.concat([X_train,y_train], axis=1)
    std_three = y_train.std()*3
    threshold = y_train.mean() + std_three
    print(threshold)
    df_train = df_train[df_train['price'] <= threshold]
    X_train = df_train.drop(columns='price', axis=1) 
    y_train = df_train['price']

    #removing price outliers from test
    df_test = pd.concat([X_test,y_test], axis=1)
    std_three = y_test.std()*3
    threshold = y_test.mean() + std_three
    print(threshold)
    df_test = df_test[df_test['price'] <= threshold]
    X_test = df_test.drop(columns='price', axis=1) 
    y_test = df_test['price']

    #fit and transform the initiated OneHotEncoder on train
    ohe = OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore')
    X_train_condition = X_train[['condition']]
    X_train_fittransform = ohe.fit_transform(X_train_condition)
    ohe.categories_[0]
    #create a new dataframe
    condition_df = pd.DataFrame(X_train_fittransform, columns=ohe.categories_[0], /
                                index=X_train.index)
    #drop the old column
    X_train.drop(columns='condition', inplace=True)
    #add the new ones
    X_train = pd.concat([X_train, condition_df], axis=1)
    #strip white space from very good
    X_train.columns = X_train.columns.str.replace(" ", "")
    
    #OHE Condition on test
    X_test_condition = X_test[['condition']]
    X_test_transform = ohe.transform(X_test_condition)
    ohe.categories_[0]
    condition_df = pd.DataFrame(X_test_transform, columns=ohe.categories_[0], index=X_test.index)
    X_test.drop(columns='condition', inplace=True)
    X_test = pd.concat([X_test, condition_df], axis=1)
    # change Very Good to VeryGood
    X_test.columns = X_test.columns.str.replace(" ", "")

    return X_train, X_test, y_train, y_test

def dummy_modeling(X_train, X_test, y_train, y_test):
    from sklearn.dummy import DummyRegressor
    X_train = X_train['id']
    dr = DummyRegressor(strategy='mean')
    model = dr.fit(X_train, y_train)
    train_y_hat = dr.predict(X_test)
    train_score = model.score(X_train, y_train)
    #repeat with test data
    X_test = X_test['id']
    test_score = dr.score(X_test,y_test)
    test_y_hat = dr.predict(X_test)

    return train_score, train_y_hat, test_score, test_y_hat

def regression_single(X_train, X_test, y_train, y_test):
    from statsmodels.formula.api import ols
    df_train = pd.concat([X_train, y_train], axis=1)
    f= 'price~bathrooms'
    model = ols(formula=f, data=df_train).fit()
    model.summary()
    X_bathrooms_train = X_train[['bathrooms']]
    train_y_hat = model.predict(X_bathrooms_train)
    #repeat for test data 
    X_bathrooms_test = X_test[['bathrooms']]
    test_y_hat = model.predict(X_bathrooms_test)
    return train_y_hat, test_y_hat
    

def regression_multiple(X_train, X_test, y_train, y_test, included):
     from statsmodels.formula.api import ols
    df_train = pd.concat([X_train, y_train], axis=1)
    f= 'price~bathrooms+bedrooms+VeryGood+Good+Average'
    model = ols(formula=f, data=df_train).fit()
    model.summary()
    X_included_train = X_train[[included]]
    train_y_hat = model.predict(X_included_train)
    X_included_test = X_test[[included]]
    test_y_hat = model.predict(X_included_test)
    return train_y_hat, test_y_hat


def calculatingRMSE(y, y_hat):
    import numpy as np
    from sklearn.metrics import mean_squared_error
    rmse = []
    rmse.append(np.sqrt(mean_squared_error(y, y_hat)))
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
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    print('resulting features:')
    print(included)
    return included
