import pandas as pd
import support
import exploration
import correlation
import compare
import tkinter as tk
import numpy as np
from pathlib import Path

# Files to read
matrix_path = Path(__file__).absolute().parent.parent / 'corr_matrices'
matrix_filenames = f"{str(matrix_path)}\\2021 Correlation Matrix 2.csv", \
                   f"{str(matrix_path)}\\2023 Correlation Matrix.csv"
data_path = Path(__file__).absolute().parent.parent / 'raw_data'
data_filename = "Rohdaten.csv"

# Opens a window when the program is finished
# Useful for long calculations, so you can do something else
# and see when the program is finished
def notify_finished():
    root = tk.Tk()
    root.title("Program Finished")

    # Set the size of the window
    window_width = 1000
    window_height = 600
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = int((screen_width - window_width) / 2)
    y = int((screen_height - window_height) / 2)
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    label = tk.Label(root, text="The program has finished!")
    label.pack(padx=20, pady=20)
    root.mainloop()


# returns a matrix containing the differences in correlations between both years
def comparison():
    difference_matrix = compare.compare_correlation_matrices(*compare.load_matrix(matrix_filenames))
    return difference_matrix


# Explorative analysis, excluding correlations inside the same question group
def inter_question_exploration(data):
    exploration.huge_exploration(data, False, "CyberThreat", "!CyberThreat")
    exploration.huge_exploration(data, False, "Usage", "!Usage")
    exploration.huge_exploration(data, False, "Knows", "!Knows")
    exploration.huge_exploration(data, False, "PastVictim", "!PastVictim")
    exploration.huge_exploration(data, False, "PastHelp", "!PastHelp")
    exploration.huge_exploration(data, False, "FutureVictim", "!FutureVictim")
    exploration.huge_exploration(data, False, "Uses", "!Uses")
    exploration.huge_exploration(data, False, "PastInfo", "!PastInfo")
    exploration.huge_exploration(data, False, "FutureInfo", "!FutureInfo")


def demographic_explorations(filter_string, data):
    exploration.explore_demographic_items(data, "Age", False, filter_string)
    print("")
    exploration.explore_demographic_items(data, "Degree", False, filter_string)
    print("")
    exploration.explore_demographic_items(data, "Income", False, filter_string)
    print("")
    exploration.explore_nominal(data, "Bundesland", plot_if_significant=False, filter_question=filter_string)
    print("")
    exploration.explore_nominal(data, "Gender", plot_if_significant=False, filter_question=filter_string)


def male_female_comparison(comparison_data, year):
    male_data = comparison_data[comparison_data["Gender"] == 1]
    female_data = comparison_data[comparison_data["Gender"] == 2]
    support.sig_level = 1
    matrix_names = f"{year}_Male_CM.csv", f"{year}_Female_CM.csv"
    exploration.huge_exploration(male_data, False, "!Gender", "!Gender", matrix_names[0])
    exploration.huge_exploration(female_data, False, "!Gender", "!Gender", matrix_names[1])
    print(compare.find_big_differences(*matrix_names, filter_insignificant=True, threshold=0.3,
                                       df_filename=f"Difference_Male_Female_{year}"))


def east_west_variable(data):
    data['East'] = np.where((data['Bundesland'].isin((4, 8, 13, 14, 16))), 1, 0)
    return data


def east_west_comparison(comparison_data, year):
    east_data = comparison_data[comparison_data["Bundesland"].isin((4, 8, 13, 14, 16))]
    west_data = comparison_data[comparison_data["Bundesland"].isin((1, 2, 5, 6, 7, 9, 10, 11, 12, 15))]
    support.sig_level = 1
    matrix_names = f"{year}_East_CM.csv", f"{year}_West_CM.csv"
    exploration.huge_exploration(east_data, False, "!Bundesland", "!Bundesland", matrix_names[0])
    exploration.huge_exploration(west_data, False, "!Bundesland", "!Bundesland", matrix_names[1])
    print(compare.find_big_differences(*matrix_names, filter_insignificant=True, threshold=0.3,
                                       df_filename=f"Difference_East_West_{year}"))


def degree_comparison(comparison_data, year, cutoff=6):
    # Default Cutoff: Abitur
    low_degree_data = comparison_data[comparison_data["Degree"] <= cutoff]
    high_degree_data = comparison_data[comparison_data["Degree"] > cutoff]
    support.sig_level = 1
    matrix_names = f"{year}_DegLow_{cutoff}_CM.csv", f"{year}_DegHigh_{cutoff}_CM.csv"
    exploration.huge_exploration(low_degree_data, False, "!Degree", "!Degree", matrix_names[0])
    exploration.huge_exploration(high_degree_data, False, "!Degree", "!Degree", matrix_names[1])
    print(compare.find_big_differences(*matrix_names, filter_insignificant=True, threshold=0.3,
                                       df_filename=f"Difference_Degree_{year}_cf={cutoff}"))


def income_comparison(comparison_data, year, cutoff):
    low_income_data = comparison_data[comparison_data["Income"] <= cutoff]
    high_income_data = comparison_data[comparison_data["Income"] > cutoff]
    support.sig_level = 1
    matrix_names = f"{year}_IncLow_{cutoff}_CM.csv", f"{year}_IncHigh_{cutoff}_CM.csv"
    exploration.huge_exploration(low_income_data, False, "!Income", "!Income", matrix_names[0])
    exploration.huge_exploration(high_income_data, False, "!Income", "!Income", matrix_names[1])
    print(compare.find_big_differences(*matrix_names, filter_insignificant=True, threshold=0.3,
                                       df_filename=f"Difference_Income_{year}_cf={cutoff}"))


def year_comparison_consistency(year_1_data, year_2_data, filter1="", filter2=""):
    matrix_names = f"{2021}_CM.csv", f"2023_CM.csv"
    common_columns = year_1_data.columns.intersection(year_2_data.columns)
    data1_common = year_1_data[common_columns]
    data2_common = year_2_data[common_columns]
    exploration.huge_exploration(data1_common, False, filter1, filter2, matrix_names[0])
    exploration.huge_exploration(data2_common, False, filter1, filter2, matrix_names[1])
    print(compare.find_double_strong_corrs(*matrix_names))


def year_comparison(year_1_data, year_2_data):
    support.sig_level = 1
    matrix_names = f"{2021}_CM.csv", f"2023_CM.csv"
    exploration.huge_exploration(year_1_data, False, "", "", matrix_names[0])
    exploration.huge_exploration(year_2_data, False, "", "", matrix_names[1])
    print(compare.find_big_differences(*matrix_names, filter_insignificant=True, threshold=0.1,
                                       df_filename=f"Difference_2021_2023"))


def age_comparison(comparison_data, year, cutoff):
    young_data = comparison_data[comparison_data["Age"] <= cutoff]
    old_data = comparison_data[comparison_data["Age"] > cutoff]
    support.sig_level = 1
    matrix_names = f"{year}_Young_{cutoff}_CM.csv", f"{year}_Old_{cutoff}_CM.csv"
    exploration.huge_exploration(young_data, False, "!Age", "!Age", matrix_names[0])
    exploration.huge_exploration(old_data, False, "!Age", "!Age", matrix_names[1])
    print(compare.find_big_differences(*matrix_names, filter_insignificant=True, threshold=0.3,
                                       df_filename=f"Difference_Age_{year}_cf={cutoff}"))


def combine_difference_matrices():
    # Interessantes Ergebnis: Beim Ersten gibt es ein Paar sachen, sonst nirgendswo
    print(compare.combine_difference_dataframes("Difference_Age_2021_cf=2", "Difference_Age_2023_cf=2"))
    print(compare.combine_difference_dataframes("Difference_Age_2021_cf=3", "Difference_Age_2023_cf=3"))
    print(compare.combine_difference_dataframes("Difference_Degree_2021_cf=6", "Difference_Degree_2023_cf=6"))
    print(compare.combine_difference_dataframes("Difference_East_West_2021", "Difference_East_West_2023"))
    print(compare.combine_difference_dataframes("Difference_Income_2021_cf=5", "Difference_Income_2023_cf=5"))
    print(compare.combine_difference_dataframes("Difference_Male_Female_2021", "Difference_Male_Female_2023"))


# Creates a joint dataset out of two datasets
# Only uses columns that appear in both datasets
def combine_datasets(datasets):
    # Determine all columns that appear in every dataset
    common_columns = datasets[0].columns
    for dataset in datasets[1:]:
        common_columns = common_columns.intersection(dataset.columns)

    # Select only common columns for each dataset
    datasets_stripped = [dataset[common_columns] for dataset in datasets]

    # Combine all datasets into one joint set
    joint_dataset = pd.concat(datasets_stripped, axis=0)

    return joint_dataset

def fx_size(p, es):
    return_str = "Insignifikant"
    es = abs(es)
    if p<0.05:
        if es <= 0.10:
            return_str = "Sehr schwach"
        elif es <= 0.30:
            return_str = "Schwach"
        elif es <= 0.50:
            return_str = "Moderat"
        else:
            return_str = "Stark"
    return return_str

def main():
    data_id = "Linsner"

    data_linsner = pd.read_csv(f"{str(data_path)}\\{data_id}{data_filename}", sep=",")  # sep necessary because of Order column
    support.data_preprocess(data_linsner, data_id)
    #print(data_linsner)

    data_id = "2023Cylence"
    data_2023 = pd.read_csv(f"{str(data_path)}\\{data_id}{data_filename}", sep=";")  # sep necessary because of Order column
    support.data_preprocess(data_2023, data_id)
    #print(data_2023)
    data_2023_no_other_degrees = support.drop_other_degrees(data_2023)

    sample_A = data_linsner.filter(like="Give_")
    #print(sample_A)
    sample_B1 = data_2023[data_2023["Age"].isin([2,3,4])]
    sample_B2 = data_2023_no_other_degrees[(data_2023_no_other_degrees["Age"].isin([2,3,4])) & (data_2023_no_other_degrees["Degree"] >= 5)]
    sample_C1 = data_2023[~data_2023["Age"].isin([2,3,4])].filter(like="Give_")
    #print(sample_C)
    sample_C2 = data_2023_no_other_degrees[~((data_2023_no_other_degrees["Age"].isin([2,3,4])) & (data_2023_no_other_degrees["Degree"] >= 5))]

    print(correlation.spearman_rho(data_2023, ["Protection_Sum", "Give_Sum"]))
    print(correlation.spearman_rho(data_linsner, ["Protection_Sum", "Give_Sum"]))

    '''
    for question in sample_A.columns:
        u_ab1, p_ab1, es_ab1 = correlation.mann_whitney_u_columns(sample_A[question], sample_B1[question])
        u_b1c1, p_b1c1, es_b1c1 = correlation.mann_whitney_u_columns(sample_B1[question], sample_C1[question])
        u_ab2, p_ab2, es_ab2 = correlation.mann_whitney_u_columns(sample_A[question], sample_B2[question])
        u_b2c2, p_b2c2, es_b2c2 = correlation.mann_whitney_u_columns(sample_B2[question], sample_C2[question])
        avg_a = round(sample_A[question].mean(), 2)
        avg_b1 = round(sample_B1[question].mean(), 2)
        avg_c1 = round(sample_C1[question].mean(), 2)
        avg_b2 = round(sample_B2[question].mean(), 2)
        avg_c2 = round(sample_C2[question].mean(), 2)
        print(f"Question: {question}")
        print(f"A ({avg_a}) vs B1 ({avg_b1}): p={p_ab1}, ES={round(es_ab1, 2)}, U={u_ab1} ({fx_size(p_ab1, round(es_ab1, 2))})")
        print(f"B1 ({avg_b1}) vs C1 ({avg_c1}): p={p_b1c1}, ES={round(es_b1c1, 2)}, U={u_b1c1} ({fx_size(p_b1c1, round(es_b1c1, 2))})")
        print(f"A ({avg_a}) vs B2 ({avg_b2}): p={p_ab2}, ES={round(es_ab2, 2)}, U={u_ab2} ({fx_size(p_ab2, round(es_ab2, 2))})")
        print(f"B2 ({avg_b2}) vs C2 ({avg_c2}): p={p_b2c2}, ES={round(es_b2c2, 2)}, U={u_b2c2} ({fx_size(p_b2c2, round(es_b2c2, 2))})")
        print("")
    '''

if __name__ == "__main__":
    main()
