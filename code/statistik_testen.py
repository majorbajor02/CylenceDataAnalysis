# Zu Testen:
# Wie kann ich zwei Variablen so programmieren, dass ihre Korrelation festgelegt ist
# Wie verhält sich logistische Regression bei stark vereinfachten Modell (also eine Variable plus eine zweite
# "unbekannte Variable"
# Wie verhält sich logistische Regression bei Modell mit zwei Variablen, die voneinander abhängen

import random
import pandas as pd
import correlation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import truncnorm


def get_truncated_normal(mean=0, sd=1, low=0, upp=10, amt=100):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(amt)


def generate_correlated_bin_list(list1, corr):
    list2 = []
    p_max = round((1 + corr) / 2, 4)  # Maximaler Ausschlag
    p_min = 1-p_max
    lst_min = min(list1)
    lst_max = max(list1)
    for item in list1:
        p = p_min + (p_max-p_min)*((item-lst_min)/(lst_max-lst_min))
        val = random.choices((0, 1), weights=(1-p, p))[0]
        list2.append(val)
        # print(f"Item: {item}, p: {p}, val: {val}")
    return list2


# Variable 1, 2 = bin
def func0(corr, size=100, m1=0):
    if m1 == 0:
        m1 = random.random()*7
    list1 = get_truncated_normal(m1, 2, 0, 7, size)
    list2 = generate_correlated_bin_list(list1, corr)
    df = pd.DataFrame({
        "Var1": list1,
        "Var2": list2
    })
    real_corr = correlation.spearman_rho(df, ["Var1", "Var2"])[0]
    # print(f'Mean: {m1}')
    # print(f'Correlation: {real_corr}')
    return df, real_corr


# Variable 1, 2 = bin
def func2(corr, size=100, m1=0, m2=0):
    if m1 == 0:
        m1 = random.random()*7
    if m2 == 0:
        m2 = random.random()*7
    list1 = get_truncated_normal(m1, 2, 0, 7, size)
    list2 = get_truncated_normal(m2, 2, 0, 7, size)
    list3 = generate_correlated_bin_list(list1, corr)
    df = pd.DataFrame({
        "Var1": list1,
        "Var2": list2,
        "Var3": list3
    })
    real_corr1 = correlation.spearman_rho(df, ["Var1", "Var3"])[0]
    real_corr2 = correlation.spearman_rho(df, ["Var2", "Var3"])[0]
    # print(f'Mean: {m1}')
    # print(f'Correlation: {real_corr}')
    return df, real_corr1, real_corr2


def test_func(func, step_size=0.05, test_size=10000):
    for i in np.arange(-1, 1+step_size, step_size):
        corr = round(i, 4)
        func(corr, size=test_size)


def test_1_func(func, corr=-1, test_size=100):
    func(corr, test_size)


def test_log(func, func_size=10000, test_size=10000):
    corr_diffs = []
    errors = []
    for i in range(test_size):
        print(f"Test No {i+1}/{test_size}")
        corr1 = random.random()*2-1  # Random between -1 and 1 # Test Correlation
        corr2 = random.random()*2-1  # Random between -1 and 1 # Train Correlation

        train_set, real_corr1 = func(corr1, func_size, 2)
        test_set, real_corr2 = func(corr2, func_size, 4)
        corr_diff = abs(real_corr2-real_corr1)
        corr_diffs.append(corr_diff)
        # Fit Model:
        logistic_model = LogisticRegression()
        logistic_model.fit(train_set[["Var1"]], np.ravel(train_set[["Var2"]]))

        # Test Model:
        pred2 = logistic_model.predict_proba(test_set[["Var1"]])
        pred2_sum = sum(pred2)[1]
        pred_error = abs(test_set["Var2"].sum()-pred2_sum)/func_size
        errors.append(pred_error)
    plt.scatter(corr_diffs, errors)
    plt.show()


def test_log_same_corr(func, func_size=10000, test_size=10000):
    corr_diffs = []
    errors = []
    for i in range(test_size):
        print(f"Test No {i+1}/{test_size}")
        corr1 = random.random()*2-1  # Random between -1 and 1 # Test Correlation

        train_set, real_corr1 = func(corr1, func_size, 2)
        test_set, real_corr2 = func(corr1, func_size, 4)
        corr_diff = abs(real_corr2-real_corr1)
        corr_diffs.append(corr_diff)
        # Fit Model:
        logistic_model = LogisticRegression()
        logistic_model.fit(train_set[["Var1"]], np.ravel(train_set[["Var2"]]))

        # Test Model:
        pred2 = logistic_model.predict_proba(test_set[["Var1"]])
        pred2_sum = sum(pred2)[1]
        pred_error = abs(test_set["Var2"].sum()-pred2_sum)/func_size
        errors.append(pred_error)
    plt.scatter(corr_diffs, errors)
    plt.show()


print("started")
test_log_same_corr(func0, 10000, 1000)

# print("Training Set:")
# df1 = func0(0.3)
# print(f"Actual Sum:{df1['Var2'].sum()}")
# logistic_model = LogisticRegression()
# logistic_model.fit(df1[["Var1"]], np.ravel(df1[["Var2"]]))
# pred1 = logistic_model.predict_proba(df1[["Var1"]])
# pred1_sum = sum(pred1)[1]
# print(f"Pred Sum:{pred1_sum}")
#
# print("Test Set:")
# df2 = func0(0.3)
# print(f"Actual Sum:{df2['Var2'].sum()}")
# pred2 = logistic_model.predict_proba(df2[["Var1"]])
# pred2_sum = sum(pred2)[1]
# print(f"Pred Sum:{pred2_sum}")
