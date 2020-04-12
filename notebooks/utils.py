"""
Utility functions for notebooks
"""
import re
import pandas as pd
import numpy as np
import scipy

import calendar

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from constants import *

##########
# General
##########


def get_cols(regex_pat: str, columns: list):
    """Get columns in a dataframe which matach a regex pattern
    
    Args:
        regex_pat (str): regex pattern
        columns (list): list of column names 
    """

    return [col for col in columns if re.match(regex_pat, col)]


#########
# Regression
#########


def save_model_results(model_results, model_name):
    """Saves regression results as an image"""

    plt.rc("figure", figsize=(12, 7))
    plt.text(
        0.01,
        0.05,
        str(model_results.summary()),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FOLDER}/{model_name}.png", bbox_inches="tight")
    plt.close()


def estimate_ols(target_col, feature_cols, df, model_name):
    """Estimate OLS model for a brand's unit sales

    Args:
        target_col (str): target column
        feature_cols (list): list of features to use in the model
        df (pd.DataFrame): dataframe to use for model (e.g. data, train or test)
        model_name (str): file name to save regression results to

    Returns:
        results (obj): Fitted regression model
    """

    X = sm.add_constant(df[feature_cols])
    y = df[target_col]

    model = sm.OLS(y, X)
    results = model.fit()

    # save regression results to folder
    save_model_results(results, model_name)

    return model.fit()


#########
# Plots
##########
def plot_cols(data, cols, ax, title, ylabel, xlabel=None):
    """Plot specified columns"""

    data[cols].plot(color=BRAND_COLORS, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))

    return ax


def plot_diagnostics(model, model_name):
    """Plot regression model diagnostics
    
    - residuals vs fitted
    - histogram of residuals
    - qqplot
    - cook's distance
    
    """

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    fig.suptitle(f"{model_name}: Regression Model Diagnostics", fontsize=20)

    # Resid vs fitted
    resid = model.resid
    fitted = model.fittedvalues
    smoothed = lowess(resid, fitted)

    axs[0, 0].scatter(y=resid, x=fitted, alpha=0.7)
    axs[0, 0].plot(smoothed[:, 0], smoothed[:, -1], color="r")
    axs[0, 0].hlines(
        y=0, xmin=min(fitted), xmax=max(fitted), color="red", linestyle=":", alpha=0.3
    )
    axs[0, 0].set(
        title="Residuals vs. Fitted", ylabel="Residuals", xlabel="Fitted Values"
    )

    # histogram of residuals
    sns.distplot(resid, ax=axs[0, 1])
    axs[0, 1].set(title="Distribution of residuals", xlabel="Residuals")

    # QQplot
    _, (__, ___, r) = scipy.stats.probplot(resid, plot=axs[1, 0], fit=True)
    axs[1, 0].text(
        -2,
        0.3,
        f"$R^2$ = {round(r**2,2)}",
        ha="left",
        va="center",
        bbox=dict(edgecolor="black", alpha=0.1),
    )

    # Cook's Distance

    infl = model.get_influence()
    cooks_distance = infl.cooks_distance[0]
    thres = 4 / len(cooks_distance)
    axs[1, 1].scatter(y=cooks_distance, x=range(len(cooks_distance)), alpha=0.7)
    axs[1, 1].hlines(
        y=thres, xmin=0, xmax=len(cooks_distance), color="r", linestyle=":"
    )
    axs[1, 1].set(
        title="Outlier Observations", xlabel="Datapoint index", ylabel="Cook's Distance"
    )

    plt.subplots_adjust(hspace=0.3, wspace=0.3)


def plot_coef_conf_int(model):
    """Plot coefficents and confidence intervals for main variables"""

    RED = "#f08080"
    GREEN = "#74c365"

    # calculate error
    err_series = model.params - model.conf_int()[0]

    # select subset of coeficients
    features = get_cols(".*?PRICE|.*?display|.*?FEATURE", model.params.index)
    coef_series = model.params[model.params.index.isin(features)]
    err_series = err_series[err_series.index.isin(features)]
    pvalues_series = model.pvalues[model.pvalues.index.isin(features)]

    # plot bar chart (green bar if p_value < 0.05 else red)
    plt.barh(
        y=range(len(features)),
        width=coef_series.values,
        xerr=err_series.values,
        tick_label=features,
        color=(pvalues_series < 0.05).map({True: GREEN, False: RED}),
    )
    plt.title("Coefficient significance of price, display and featured variables (w/95% confidence intervals)")
    plt.xlabel("Coefficient value")
    plt.xlim([-7,4])

    red_bars = mpl.patches.Patch(
        color=RED, label="Not statistically significant (5% level)"
    )
    green_bars = mpl.patches.Patch(
        color=GREEN, label="Statistically significant (5% level)"
    )
    plt.legend(handles=[red_bars, green_bars])
    plt.tight_layout()


def plot_preds_versus_actual(preds, actual, model_name):
    """Plot predicted versus actual on test data"""

    # diagonal line coordinates
    diag = np.linspace(start=min(actual), stop=max(preds))
    plt.plot(diag, diag, ls="--", c="r")

    plt.scatter(x=preds, y=actual)

    plt.title(f"{model_name}: Model validation actual vs. predicted")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
