import pandas as pd
import numpy as np

import plots
import support
import correlation


def explore_gender_mwu(data, explored_feature, filter_string=""):
    switch_filtering, filter_string = support.filter_switcher(filter_string)
    """TODO Erklärung was in Methode passiert, Parameter sollten erklärt werden."""
    if explored_feature == "Gender":
        data = support.drop_rare_genders(data)

    for item in data.columns.values:
        data_copy = data.copy()
        if not switch_filtering ^ (filter_string in item):
            continue
        if item == explored_feature:
            continue
        if item == "Degree":
            data_copy = support.drop_other_degrees(data_copy)
        elif item == "Gender":
            data_copy = support.drop_rare_genders(data_copy)
        elif "Victim" in item:
            data_copy = support.drop_idk(data, item)
        result = correlation.mann_whitney_u(data_copy, [explored_feature, item])

        effect_sizes = support.get_effect_size_ratings(mwu=result[2])

        if result[1] < support.sig_level and effect_sizes.max_r >= support.min_effect_rating:
            # ansi = support.get_ansi(result.effect_size, "Cramer v")
            statistic_string = ""
            if result[1] < support.sig_level:
                if result[1] < 0.001:
                    p_string = "p<.001"
                else:
                    p_string = f"p={str(result[1])[1:]}"

                statistic_string += f"{support.effect_size_ansi[effect_sizes.r_mwu]}" \
                                    f"{p_string}, r={result[2]}{support.ansi_default_bg}; "

            statistic_string = statistic_string[:-2]

            print(f"{support.effect_size_ansi[effect_sizes.max_r]}"
                  f"{explored_feature} and {item}{support.ansi_default_bg}!"
                  f" ({statistic_string})")


# Gender and Bundesland
def explore_nominal(data, explored_feature, plot_if_significant=False, filter_question=""):
    switch_filtering, filter_question = support.filter_switcher(filter_question)
    """TODO Erklärung was in Methode passiert, Parameter sollten erklärt werden."""

    # Since 2/1093 did not choose male or female as gender, they are removed for the exploration regarding Gender
    # The reason for this is that the data size for their groups is too small
    if explored_feature == "Gender":
        data = support.drop_rare_genders(data)

    for item in data.columns.values:
        if not switch_filtering ^ (filter_question in item):
            continue
        if item.startswith("Sum"):
            continue
        data_copy = data.copy()
        if item == "Degree":
            # Same Story bei allen Fragen mit "PastVictim", dort muss '1' (Weiß ich nicht) entfernt werden
            data_copy = support.drop_other_degrees(data_copy)
        elif item == "Gender":
            data_copy = support.drop_rare_genders(data_copy)
        elif "Victim" in item:
            data_copy = support.drop_idk(data_copy, item)

        result = correlation.chi_square_columns(data_copy, [explored_feature, item],
                                                plot_if_significant=plot_if_significant, calculate_cramer_v=True)
        correlation_size = data_copy.shape[0]
        effect_sizes = support.get_effect_size_ratings(v=result.effect_size)

        if result.p < support.sig_level and effect_sizes.max_r >= support.min_effect_rating:
            if result.p < 0.001:
                p_string = "p<.001"
            else:
                p_string = f"p={str(result.p)[1:]}"
            ansi = support.get_ansi(result.effect_size, "Cramer v")
            print(f"{ansi}{explored_feature} and {item}! ({p_string}, Chi2({result.df}, "
                  f"N={correlation_size})={result.chi2}, "
                  f"ES={result.effect_size}){support.ansi_default_bg}")


corr_list = []  # Sehr unelegant das global zu machen, tut mir leid


def explore_demographic_items(data, explored_feature, plot_if_significant=False, filter_string=""):
    switch_filtering, filter_string = support.filter_switcher(filter_string)
    """TODO Erklärung was in Methode passiert, Parameter sollten erklärt werden."""

    # The scale for "Degree" had a number 9 for the response "Other".
    # Since "Other" is not higher than the other degrees, though, it disturbs the ordinal scale level
    # Removing all subjects with the degree "Other" makes the scale ordinal again.
    if explored_feature == "Degree":
        data = support.drop_other_degrees(data)

    rhos = []

    for item in data.columns.values:
        if not switch_filtering ^ (filter_string in item):
            continue
        data_copy = data.copy()
        if item == "Degree":
            data_copy = support.drop_other_degrees(data_copy)
        elif item == "Gender":
            data_copy = support.drop_rare_genders(data_copy)
        elif "Victim" in item:
            data_copy = support.drop_idk(data_copy, item)
        rho, p_rho = correlation.spearman_rho(data_copy, [explored_feature, item])
        correlation_size = data_copy.shape[0]

        effect_sizes = support.get_effect_size_ratings(rho=rho)

        if p_rho < support.sig_level and effect_sizes.max_r >= support.min_effect_rating and item != explored_feature:
            statistic_string = ""
            if p_rho < support.sig_level:
                if p_rho < 0.001:
                    p_string = "p<.001"
                else:
                    p_string = f"p={str(p_rho)[1:]}"

                statistic_string += f"{support.effect_size_ansi[effect_sizes.r_rho]}" \
                                    f"{p_string}, rho({correlation_size-2})={rho}{support.ansi_default_bg}; "

            statistic_string = statistic_string[:-2]

            if plot_if_significant:
                plots.plot_heatmap(data_copy[explored_feature], data_copy[item])
                plots.plot_averages(np.array(support.numerical_crosstable(data_copy[explored_feature],
                                                                          data_copy[item])),
                                    labels=[explored_feature, item])
            if not support.in_set_list({explored_feature, item}, corr_list):
                print(f"{support.effect_size_ansi[effect_sizes.max_r]}"
                      f"{explored_feature} and {item}{support.ansi_default_bg}!"
                      f" ({statistic_string})")
                corr_list.append({explored_feature, item})

        if p_rho >= support.sig_level:
            rhos.append(0)
        else:
            rhos.append(round(rho, 2))
    return rhos


def huge_exploration(data, plot_if_significant=False, filter_question="", filter_question2="", matrix_filename=""):
    """TODO Erklärung was in Methode passiert, Parameter sollten erklärt werden."""
    # This means that instead of only showing question with the keyword,
    # no questions with this keyword are shown

    switch_filtering, filter_question = support.filter_switcher(filter_question)
    switch_filtering_items, filter_question_items = support.filter_switcher(filter_question2)

    dct = {}
    items = [item for item in data.columns.values if switch_filtering_items ^ (filter_question_items in item)]
    dct["Items"] = items
    for explorer in data.columns.values:
        if explorer in ["Consent", "Age", "Gender", "Degree", "Bundesland", "Income"]:
            continue
        if not switch_filtering ^ (filter_question in explorer):
            continue
        data_copy = data.copy()
        if explorer == "Degree":
            data_copy = support.drop_other_degrees(data_copy)
        elif explorer == "Gender":
            data_copy = support.drop_rare_genders(data_copy)
        elif "Victim" in explorer:
            data_copy = support.drop_idk(data_copy, explorer)
        dct[explorer] = explore_demographic_items(data_copy, explorer,
                                                  plot_if_significant=plot_if_significant,
                                                  filter_string=filter_question2)

    correlation_matrix = pd.DataFrame(data=dct)
    correlation_matrix.set_index("Items", inplace=True)
    if matrix_filename == "":
        matrix_filename = f"Correlation Matrix {filter_question} & {filter_question2}.csv"
    if not matrix_filename.endswith(".csv"):
        matrix_filename = matrix_filename + ".csv"
    correlation_matrix.to_csv(matrix_filename)
    print("")
    # plots.plot_correlation_matrix(correlation_matrix)
