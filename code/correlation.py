from scipy.stats import chi2_contingency  # Chi-Square and Cramer's V
from scipy.stats import kendalltau  # Kendall's tau
from scipy.stats import f_oneway  # ANOVA
from scipy.stats import spearmanr  # Spearman's rho (Ordinal)
from scipy.stats import pearsonr  # Pearson r
from scipy.stats import mannwhitneyu  # Mann Whitney U Test
import numpy as np
from scipy.stats import norm
import support
import plots


class ChiSquareResult:
    def __init__(self, chi2, p, df, cramer_v=None, effect_size=None):
        self.chi2 = chi2
        self.p = p
        self.df = df
        self.cramer_v = cramer_v
        self.effect_size = effect_size


# TODO: Raise warning when too little data (<5 in the contingency table)
# For 2 categorical variables (for example Gender + federal state)
def chi_square_columns(data, column_names, calculate_cramer_v=False, plot_if_significant=False):
    if len(column_names) != 2:
        raise Exception("Expected 2 column names, found %s" % len(column_names))
    columns = support.get_columns(data, column_names)

    crosstable = np.array(support.numerical_crosstable(*columns))
    result = chi2_contingency(crosstable)

    chi2 = round(result.statistic, 2)
    p = round(result.pvalue, 3)
    df = len(crosstable)-1  # degrees of freedom

    if calculate_cramer_v:
        v, effect_size = get_cramer_v(chi2, crosstable)
        effect_size_rating = support.get_effect_size_ratings(v=effect_size).r_v
        if plot_if_significant and p < support.sig_level and effect_size_rating >= support.min_effect_rating:
            plots.plot_averages(crosstable, labels=column_names)
    else:
        v, effect_size = None, None
        if plot_if_significant and p < support.sig_level:
            plots.plot_averages(crosstable, labels=column_names)
    return ChiSquareResult(chi2, p, df, v, effect_size)


def get_cramer_v(chi2, crosstable):
    n = np.sum(crosstable)
    min_dimension = min(crosstable.shape)
    v = np.sqrt(chi2 / (n * min_dimension))
    # https://real-statistics.com/chi-square-and-f-distributions/effect-size-chi-square/
    effect_size = v * np.sqrt(min_dimension)
    return round(v, 2), round(effect_size, 2)


# Each variable is ordinal, ratio or interval
# monotonic relationship (doesn't have to be linear)
def kendall_tau_b(data, column_names):
    if len(column_names) != 2:
        raise Exception("Expected 2 column names, found %s" % len(column_names))
    columns = support.get_columns(data, column_names)
    tau_b, p = kendalltau(*columns)
    return round(tau_b, 2), round(p, 3)


def anova(data, column_names):
    # No Exception, because more than 2 columns are allowed
    columns = support.get_columns(data, column_names)

    anova_f, p = f_oneway(*columns)
    return round(anova_f, 2), round(p, 3)


# Each variable is ordinal, ratio or interval
# monotonic relationship (doesn't have to be linear)
def spearman_rho(data, column_names):
    if len(column_names) != 2:
        raise Exception("Expected 2 column names, found %s" % len(column_names))
    columns = support.get_columns(data, column_names)

    rho, p = spearmanr(*columns, nan_policy='omit')
    return round(rho, 2), round(p, 3)


# Assumptions (https://pythonfordatascienceorg.wordpress.com/correlation-python/):
# - Each variable is ratio or interval
# - Each participant has data in each variable
# - No outliers
# - Linear relationship
# - Homoscedasticity (variability is constant)
def pearson_r(data, column_names):
    if len(column_names) != 2:
        raise Exception("Expected 2 column names, found %s" % len(column_names))
    columns = support.get_columns(data, column_names)

    r, p = pearsonr(*columns)
    return round(r, 2), round(p, 3)

def mann_whitney_u_columns(data1, data2):
    mwu_u1, p = mannwhitneyu(data1.dropna(), data2.dropna())
    n_col1 = len(data1)
    n_col2 = len(data2)
    mwu_u2 = n_col1 * n_col2 - mwu_u1
    mwu_u = min(mwu_u1, mwu_u2)
    z = norm.ppf(p)
    # effect_size = 1 - (2*mwu_u)/(n_col1*n_col2)  # Wikipedia
    effect_size = z / (n_col1 + n_col2) ** 0.5
    return mwu_u, round(p, 3), effect_size


# Unclear how to interpret
def mann_whitney_u(data, column_names):
    if len(column_names) != 2:
        raise Exception("Expected 2 column names, found %s" % len(column_names))
    columns = data[[*column_names]]
    data1, data2 = unpack_columns(columns)
    mwu_u1, p = mannwhitneyu(data1, data2)
    n_col1 = len(data1)
    n_col2 = len(data2)
    mwu_u2 = n_col1*n_col2-mwu_u1
    mwu_u = min(mwu_u1, mwu_u2)
    z = norm.ppf(p)
    # effect_size = 1 - (2*mwu_u)/(n_col1*n_col2)  # Wikipedia
    effect_size = z/(n_col1+n_col2)**0.5
    return mwu_u, round(p, 3), effect_size

def unpack_columns(data):
    binary_values = list(set(data.iloc[:, 0]))
    binary_name, other_name = data.columns


    data1 = np.array(data.loc[(data[binary_name] == binary_values[0])][other_name])
    data2 = np.array(data.loc[(data[binary_name] == binary_values[1])][other_name])
    print(f"1: {sum(data1)}")
    print(f"2: {sum(data2)}")

    return data1, data2