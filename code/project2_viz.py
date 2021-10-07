import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import pandas as pd

def correlations(X_train):
    
    new_df= X_train[['sqft_living', 'sqft_living15', 'sqft_above', 
                    'sqft_basement', 'bathrooms']]
    corr = new_df.corr()
    fig, ax = plt.subplots()
    ax = sns.heatmap(data=corr, annot=True)
    ax.set_title("Multicollinearity of Features");

def QQplot(residuals): 
    fig=sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
    return fig.show()

def scatter(residuals, y_hat):
    plt.scatter(residuals, y_hat);