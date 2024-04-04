import pandas as pd
import numpy as np

def load_matrix(filename):
    if isinstance(filename, str):
        return pd.read_csv(filename, sep=",")
    else:  # Unpack
        # (this is a bit simplistic and does not cover other non-iterable datatypes, but it should be sufficient
        return [load_matrix(single_name) for single_name in filename]


def compare_correlation_matrices(matrix_1, matrix_2):
    matrix_1.set_index(matrix_1.columns[0], inplace=True)
    matrix_2.set_index(matrix_2.columns[0], inplace=True)
    matrix_1, matrix_2 = remove_non_double_questions(matrix_1, matrix_2)
    difference_matrix = matrix_2.subtract(matrix_1)
    difference_matrix = difference_matrix.round(2)
    difference_matrix.to_csv(f"Subtracted.csv")
    return difference_matrix


def remove_non_double_questions(matrix_1, matrix_2):
    for col in matrix_1.columns.values:
        if col not in matrix_2.columns.values:
            matrix_1.drop(columns=col, inplace=True)
    for col in matrix_2.columns.values:
        if col not in matrix_1.columns.values:
            matrix_2.drop(columns=col, inplace=True)
    for row in matrix_1.index.values:
        if row not in matrix_2.index.values:
            matrix_1.drop(index=row, inplace=True)
    for row in matrix_2.index.values:
        if row not in matrix_1.index.values:
            matrix_2.drop(index=row, inplace=True)
    return matrix_1, matrix_2


def intersection(lst1, lst2):
    return [value for value in lst1 if value in lst2]


def find_big_differences(matrix_1, matrix_2, threshold=0.15, filter_insignificant=False, df_filename="Big Differences"):
    if isinstance(matrix_1, str):
        matrix_1 = load_matrix(matrix_1)
    if isinstance(matrix_2, str):
        matrix_2 = load_matrix(matrix_2)

    difference_matrix = compare_correlation_matrices(matrix_1, matrix_2)
    big_differences = find_values_above_threshold(difference_matrix, threshold)
    if len(big_differences.index) == 0:
        big_differences['Value 1'] = None
        big_differences['Value 2'] = None
    else:
        big_differences['Value 1'] = big_differences.apply(lambda x: matrix_1.loc[x['Row'], x['Column']], axis=1)
        big_differences['Value 2'] = big_differences.apply(lambda x: matrix_2.loc[x['Row'], x['Column']], axis=1)

        if filter_insignificant:
            big_differences = big_differences.loc[((big_differences['Value 1'] != 0) &
                                                   (big_differences['Value 2'] != 0)) |
                                                  (abs(big_differences['Value 1'] - big_differences['Value 2']) >= 0.4)]

    big_differences.to_csv(df_filename)
    return big_differences


def find_double_strong_corrs(matrix_1, matrix_2):
    if isinstance(matrix_1, str):
        matrix_1 = load_matrix(matrix_1)
    if isinstance(matrix_2, str):
        matrix_2 = load_matrix(matrix_2)

    matrix_1.set_index(matrix_1.columns[0], inplace=True)
    matrix_2.set_index(matrix_2.columns[0], inplace=True)
    results = []
    for column in matrix_1.columns:
        print(column)
        for (index_1, value_1), (index_2, value_2) in zip(matrix_1[column].items(), matrix_2[column].items()):
            print(str(index_1) + " " + str(index_2))
            if column <= index_1 or column.startswith("Sum") or index_1.startswith("Sum"):
                continue

            if abs(value_1) > 0.4 and abs(value_2) > 0.4 \
                    and index_1.split("_")[0] != column.split("_")[0]: # Avoids obvious correlations within one question
                results.append((abs(value_1), value_1, value_2, column, index_1))

    results_df = pd.DataFrame(results, columns=['AbsValue', 'Value 1', 'Value 2', 'Column', 'Row'])
    sorted_df = results_df.sort_values(by='AbsValue', ascending=False)
    sorted_df.drop('AbsValue', axis=1, inplace=True)
    sorted_df.to_csv("Strong Correlations.csv")

    return sorted_df



def find_values_above_threshold(difference_matrix, threshold):
    # difference_matrix.set_index(difference_matrix.columns[0], inplace=True)

    # Create an empty list to store the results
    results = []

    # Iterate over each column in the DataFrame
    for column in difference_matrix.columns:
        # Iterate over each row in the column
        for index, value in difference_matrix[column].items():
            if column < index:
                continue  # Avoiding duplicates

            # Check if the absolute value is above the threshold
            if abs(value) > threshold:
                # Append the value, column name, and row index to the results list
                results.append([abs(value), value, column, index])

    # Create a new DataFrame from the results list
    results_df = pd.DataFrame(results, columns=['AbsValue', 'Value', 'Column', 'Row'])

    # Sort the DataFrame in descending order based on the 'Value' column
    sorted_df = results_df.sort_values(by='AbsValue', ascending=False)
    sorted_df.drop('AbsValue', axis=1, inplace=True)

    return sorted_df


def combine_difference_dataframes(*difference_dfs):
    difference_dfs = [load_matrix(item) if isinstance(item, str) else item for item in difference_dfs]

    # Merge all dataframes on "Column" and "Row"
    merged_df = pd.concat(difference_dfs, ignore_index=True)
    merged_df[['Column', 'Row']] = np.sort(merged_df[['Column', 'Row']], axis=1)
    grouped_df = merged_df.groupby(['Column', 'Row'])

    # Filter rows where combination appears in all dataframes
    filtered_df = grouped_df.filter(lambda x: len(x) == len(difference_dfs))

    # Drop "Value 1" and "Value 2" columns
    # filtered_df = filtered_df.drop(['Value 1', 'Value 2'], axis=1)

    # Create "Value" column containing the list of "Values" in other dataframes
    filtered_df = filtered_df.groupby(['Column', 'Row'])['Value'].apply(list).reset_index()

    return filtered_df
