from codes import *
from datetime import datetime
import numpy as np
import pandas as pd


def numerical_crosstable(set1, set2):
    np_set1 = np.array(set1)
    np_set2 = np.array(set2)

    range_1 = (min(np_set1), max(np_set1)+1)
    range_2 = (min(np_set2), max(np_set2)+1)

    for val_1 in range(*range_1):
        for val_2 in range(*range_2):
            np_set1 = np.append(np_set1, val_1)
            np_set2 = np.append(np_set2, val_2)

    crosstable = pd.crosstab(np_set1, np_set2)-1

    return crosstable


# Gets the average rating, based on a frequency table.
# The first entry is treated as a 0, the second entry as a 1, and so on.
# lst: (list of Integers) frequency table
# returns: (Integer) average of frequency table
def average_rating(lst):
    total_rating = 0
    total_frequency = 0
    for rating, frequency in enumerate(lst):
        total_rating += rating*frequency
        total_frequency += frequency
    return total_rating/total_frequency


# Gets the columns from a dataframe based on their names
# data: (dataframe) Dataset, from which columns are extracted
# column_names: (list of Strings) Names of the columns that should be extracted
# Returns: (list of dataframes) List of the columns that should be extracted
def get_columns(data, column_names):
    return [data[column_name].to_numpy() for column_name in column_names]


# Drops those rows where the gender is not male or female, because there are so few
# data: (dataframe) Dataset
# Returns: (dataframe) New Dataset without those rows
def drop_rare_genders(data):
    return data.drop(data[data["Gender"] > 2].index)


# Drops those rows where the degree is "Other" in order to make the scale ordinal
# data: (dataframe) Dataset
# Returns: (dataframe) New Dataset without those rows
def drop_other_degrees(data):
    return data.drop(data[data["Degree"] == 9].index)


def drop_idk(data, question):
    if "Victim" not in question:
        raise Exception("This question does not contain an 'I don't know' option.")
    return data.drop(data[data[question] == 1].index)


def drop_no_experience(data, question):
    if "Victim" not in question:
        raise Exception("This question does not contain an 'I don't know' option.")
    return data.drop(data[data[question] == 2].index)


def data_preprocess(data, data_id):
    if data_id == "2021Cylence":
        data_preprocess_2021(data)
    if data_id == "2023Cylence":
        data_preprocess_2023(data)
    if data_id == "Linsner":
        data_preprocess_linsner(data)

def data_preprocess_linsner(data):
    data.rename(columns=codes_linsner, inplace=True)

    data.drop(["ID", "TimeSent", "LastPage", "Language", "Seed", "TimeStart", "TimeEnd"], axis=1, inplace=True)
    for question in codes_linsner.values():
        if question.startswith("Meaning") or question.startswith("JaNein"):
            data.drop(question, axis=1, inplace=True)  # irrelevant questions

    data.replace(["männlich", "weiblich"], [1, 2], inplace=True)
    data.replace(["weitergeben", "nicht weitergeben"], [1, 2], inplace=True)
    data.replace(["Realschulabschluss", "Fachabitur", "Abitur", "Hochschulstudium (Bachelor)",
                  "Hochschulstudium (Master)", "Sonstiges"],
                 [4, 5, 6, 8, 8, 9], inplace=True)
    data.replace(["Sehr wichtig", "Eher wichtig", "Neutral", "Eher unwichtig", "Sehr unwichtig"],
                 [1,2,3,4,5], inplace=True)

    data["Protection_Sum"] = add_columns_that_start_with(data, "Protection")
    data["Give_Sum"] = add_columns_that_start_with(data, "Give")
    #print(data.to_string())

def data_preprocess_2021(data):
    """TODO Erklärung was in preprocess data passiert (z.B. nicht relevante Daten werden gelöscht)."""
    data.rename(columns=codes_2021, inplace=True)

    data.drop(data[data["OSINT2_QualityCheck"] != 5].index, inplace=True)
    data.drop("Interviewdauer [sec]", axis=1,  inplace=True)
    data.drop("TAN", axis=1, inplace=True)
    data.drop("START", axis=1, inplace=True)
    data.drop("END", axis=1, inplace=True)
    data.drop("Nummer", axis=1, inplace=True)  # TAN = Nummer, removing redundancy
    data.drop("STATUS_TEXT", axis=1, inplace=True)  # empty
    data.drop("tic", axis=1, inplace=True)
    data.drop("QUESTION_ORDER", axis=1, inplace=True)

    for question in codes_2021.values():
        if question.startswith("OSINT") or question.startswith("FutureHelp"):
            data.drop(question, axis=1, inplace=True)  # irrelevant questions

    data["Sum_institution_knowledge"] = add_columns_that_start_with(data, "Knows")
    data["Sum_Victim"] = add_columns_that_start_with(data, "PastVictim")
    data["Sum_Fear"] = add_columns_that_start_with(data, "FutureVictim")
    data["Sum_SecurityLevel"] = add_columns_that_start_with(data, "Uses")
    data["Sum_Info_Width"] = add_columns_that_start_with(data, "PastInfo")
    data["Sum_Malware"] = add_malware_cols(data)


def data_preprocess_2023(data):
    """TODO Erklärung was in preprocess data passiert (z.B. nicht relevante Daten werden gelöscht)."""
    data.rename(columns=codes_2023, inplace=True)

    data.drop(data[data["BEREINIGEN"] == 1].index, inplace=True)
    data.drop("BEREINIGEN", axis=1, inplace=True)
    data.drop("i_TIME", axis=1,  inplace=True)
    data.drop("i_TAN", axis=1, inplace=True)
    data.drop("i_START", axis=1, inplace=True)
    data.drop("i_END", axis=1, inplace=True)
    data.drop("i_NUMBER", axis=1, inplace=True)  # TAN = Nummer, removing redundancy
    data.drop("i_STATUS", axis=1, inplace=True)  # empty
    data.drop("i_ST_TXT", axis=1, inplace=True)
    data.drop("current_Q", axis=1, inplace=True)

    data["Sum_institution_knowledge"] = add_columns_that_start_with(data, "Knows")
    data["Sum_Victim"] = add_columns_that_start_with(data, "PastVictim")
    data["Sum_Fear"] = add_columns_that_start_with(data, "FutureVictim")
    data["Sum_SecurityLevel"] = add_columns_that_start_with(data, "Uses")
    data["Sum_Info_Width"] = add_columns_that_start_with(data, "PastInfo")
    data["Protection_Sum"] = add_columns_that_start_with(data, "Protection")
    data["Give_Sum"] = add_columns_that_start_with(data, "Give")
    data["Sum_Malware"] = add_malware_cols(data)


def add_malware_cols(data):
    added = 0
    malware_questions = ["Virus", "Ransomware", "Scareware", "Spyware"]
    # print(question_name)
    for col in malware_questions:
        added = added + data[f"PastVictim_{col}"]
    # print(len(added))
    return added


def add_columns_that_start_with(data, question_name):
    added = 0
    # print(question_name)
    for col in data.columns.values:
        if question_name in col:
            # print(f"{col}{data[col].shape}")
            added = added + data[col]
    # print(len(added))
    return added


def get_datetime_string():
    return datetime.now().strftime("%Y%m%d_%H%M_")


def in_set_list(set_instance, set_list):
    for listset in set_list:
        if set_instance == listset:
            return True
    return False


def get_es_thresholds(test):
    if test == "Cramer v":
        # https://real-statistics.com/chi-square-and-f-distributions/effect-size-chi-square/ (basierend auf Cohen 1988)
        return 0.1, 0.3, 0.5
    elif test == "Kendall tau":
        # https://www.researchgate.net/post/How-can-I-make-interpretation-of-kendalls-Tau-b-correlation-magnitude,
        # https://polisci.usca.edu/apls301/Text/Chapter%2012.%20Significance%20and%20Measures%20of%20Association.htm
        # (Botsch, 2011)
        return 0.1, 0.2, 0.3
    elif test == "Spearman rho":
        # https://peterstatistics.com/CrashCourse/3-TwoVarUnpair/OrdScale/OrdScale3.html (Rea & Parker, 1992)
        return 0.1, 0.2, 0.4
    elif test == "Mann Whitney U":
        # https://datatab.net/tutorial/mann-whitney-u-test
        return 0.1, 0.3, 0.5
    else:
        raise Exception(f"No thresholds known for test {test}")


class EffectSizeRatings:
    def __init__(self, r_tau, r_rho, r_mwu, r_v):
        self.r_tau = r_tau
        self.r_rho = r_rho
        self.r_mwu = r_mwu
        self.r_v = r_v
        self.max_r = max(abs(r_tau), abs(r_rho), abs(r_mwu), abs(r_v))


def get_effect_size_ratings(*, tau=0, rho=0, mwu=0, v=0):
    return EffectSizeRatings(judge_effect_size(tau, "Kendall tau"), judge_effect_size(rho, "Spearman rho"),
                             judge_effect_size(mwu, "Mann Whitney U"), judge_effect_size(v, "Cramer v"))


def judge_effect_size(effect_size, test):
    effect_size = abs(effect_size)
    weak, medium, strong = get_es_thresholds(test)
    if weak <= effect_size < medium:
        return 1  # Weak effect
    if medium <= effect_size < strong:
        return 2  # Moderate effect
    elif effect_size >= strong:
        return 3  # Strong effect
    else:
        return 0


def get_ansi(effect_size, test):
    return effect_size_ansi[judge_effect_size(effect_size, test)]


def filter_switcher(filter_question):
    if filter_question != "" and filter_question[0] == "!":
        switch_filtering = True
        filter_question = filter_question[1:]
    else:
        switch_filtering = False

    return switch_filtering, filter_question


ansi_default_bg = "\u001b[49m"

effect_size_ansi = {
    0: ansi_default_bg,  # No Effect
    1: "\u001b[43m",  # Weak
    2: "\u001b[44m",  # Moderate
    3: "\u001b[42m"  # Strong
}


sig_level = 0.05
# min_effect_size_cramer_v = 0.3
# min_effect_size_ord = 0.2
min_effect_rating = 0
