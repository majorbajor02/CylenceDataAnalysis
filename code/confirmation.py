import random

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import plots
import support
import correlation
import exploration
import math
from statsmodels.miscmodels.ordinal_model import OrderedModel

data_filename = "UmfrageRohdaten.csv"


def confirm_berlin_bremen(data):
    data['BerlinBremen'] = np.where((data['Q5'] == 3) | (data['Q5'] == 5), 1, 0)  # Q5: Bundesland
    exploration.explore_nominal(data, 'BerlinBremen', plot_if_significant=True, filter_question="Knows")


def confirm_berlin(data):
    data['Berlin'] = np.where((data['Q5'] == 3), 1, 0)  # Q5: Bundesland
    print(data['Berlin'])
    exploration.explore_nominal(data, 'Berlin', plot_if_significant=False, filter_question="Knows")


def confirm_past_future_victim_nominal(data):
    support.data_preprocess_2021(data)
    for item_past in data.columns.values:
        if not item_past.startswith("PastVictim"):
            continue
        data_copy = data.copy()

        item_future = item_past.replace("Past", "Future")

        data_copy = support.drop_idk(data_copy, item_past)
        # data_copy = support.drop_no_experience(data_copy, item_past)
        data_copy = support.drop_idk(data_copy, item_future)

        data_copy['PastVictim_YesNo'] = np.where((data_copy[item_past] == 2), 0, 1)

        result = correlation.chi_square_columns(data_copy, ['PastVictim_YesNo', item_future],
                                                plot_if_significant=True, calculate_cramer_v=True)

        effect_sizes = support.get_effect_size_ratings(v=result.effect_size)

        if result.p < support.sig_level and effect_sizes.max_r >= support.min_effect_rating:
            ansi = support.get_ansi(result.effect_size, "Cramer v")
            print(f"{ansi}Significant Correlation between {item_past} and {item_future}! "
                  f"(p={result.p}, Chi2={result.chi2}, ES={result.effect_size}){support.ansi_default_bg}")


def confirm_past_future_victim_ordinal(data):
    support.data_preprocess_2021(data)

    for item_past in data.columns.values:
        if not item_past.startswith("PastVictim"):
            continue
        data_copy = data.copy()

        item_future = item_past.replace("Past", "Future")

        data_copy = support.drop_idk(data_copy, item_past)
        data_copy = support.drop_no_experience(data_copy, item_past)
        data_copy = support.drop_idk(data_copy, item_future)

        rho, p_rho = correlation.spearman_rho(data_copy, [item_past, item_future])

        effect_sizes = support.get_effect_size_ratings(rho=rho)

        if p_rho < support.sig_level and effect_sizes.max_r >= support.min_effect_rating:
            statistic_string = ""
            if p_rho < support.sig_level:
                statistic_string += f"{support.effect_size_ansi[effect_sizes.r_rho]}" \
                                    f"p={p_rho}, rho={rho}{support.ansi_default_bg}; "

            statistic_string = statistic_string[:-2]

            print(f"Significant Correlation between {support.effect_size_ansi[effect_sizes.max_r]}{item_past} "
                  f"and {item_future}{support.ansi_default_bg}!"
                  f" ({statistic_string})")
            plots.plot_heatmap(data_copy[item_past], data_copy[item_future], False, True)


def more_direct_environment(data, year):
    support.data_preprocess(data, year)
    mod_prob = OrderedModel(data['FutureInfo_School'], data[['Age', 'Income', 'Degree']],
                            distr='probit')
    res_prob = mod_prob.fit(method='bfgs')
    print(res_prob.summary())


def multiple_logistic_regression(data2021, data2023, question):
    support.data_preprocess_2021(data2021)
    support.data_preprocess_2023(data2023)
    support.drop_rare_genders(data2021)
    support.drop_other_degrees(data2021)
    support.drop_rare_genders(data2023)
    support.drop_other_degrees(data2023)
    logistic_model = LogisticRegression()
    logistic_model.fit(data2021[["Age"]+["Gender"]+["Bundesland"]+["Income"]+["Degree"]], data2021[[question]])
    pred_2021 = logistic_model.predict_proba(data2021[["Age"]+["Gender"]+["Bundesland"]+["Income"]+["Degree"]])
    pred_2023 = logistic_model.predict_proba(data2023[["Age"]+["Gender"]+["Bundesland"]+["Income"]+["Degree"]])
    print(f"2021: Prediction: {sum(pred_2021)[1]}, Real: {data2021[question].sum()}")
    print(f"2023: Prediction: {sum(pred_2023)[1]}, Real: {data2023[question].sum()}")

def test_logistic():
    year = 2021
    data_for_confirmation = pd.read_csv(f"{year}{data_filename}", sep=";")  # sep necessary because of Order column

    splitting_coefs = [0]*math.floor(len(data_for_confirmation)/2) + [1] * math.ceil(len(data_for_confirmation)/2)
    random.shuffle(splitting_coefs)
    data_for_confirmation["Split"] = splitting_coefs
    train_data = data_for_confirmation[data_for_confirmation["Split"]==0]
    test_data = data_for_confirmation[data_for_confirmation["Split"]==1]
    data_for_confirmation.drop("Split", axis=1, inplace=True)
    print(train_data)
    multiple_logistic_regression(train_data, test_data, "FutureInfo_School")

def logistic_comparison():
    year = 2021
    data_for_confirmation = pd.read_csv(f"{year}{data_filename}", sep=";")  # sep necessary because of Order column

    year = 2023
    data_for_confirmation2023 = pd.read_csv(f"{year}{data_filename}", sep=";")  # sep necessary because of Order column

    multiple_logistic_regression(data_for_confirmation, data_for_confirmation2023, "FutureInfo_Fam")

def get_average_answer(question):
    year = 2021
    data_for_confirmation2021 = pd.read_csv(f"{year}{data_filename}", sep=";")  # sep necessary because of Order column

    year = 2023
    data_for_confirmation2023 = pd.read_csv(f"{year}{data_filename}", sep=";")  # sep necessary because of Order column


    support.data_preprocess_2021(data_for_confirmation2021)
    support.data_preprocess_2023(data_for_confirmation2023)

    if "Victim" in question:
        data_for_confirmation2021 = support.drop_idk(data_for_confirmation2021, question)
        data_for_confirmation2023 = support.drop_idk(data_for_confirmation2023, question)

    print(f"2021: {sum(data_for_confirmation2021[question]/len(data_for_confirmation2021[question]))}")
    print(f"2023: {sum(data_for_confirmation2023[question]/len(data_for_confirmation2023[question]))}")

if __name__ == "__main__":
    logistic_comparison()
# Folie 10: Leicht verbesserte Informationskompetenz: wegen Bildung und Einkommen?
# Folie 15: Deutlich gestiegenes Interesse im Austausch im direkten Umfeld (beruflich wie privat)
