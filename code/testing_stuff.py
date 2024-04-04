# from scipy.stats import f_oneway
# import numpy as np
import compare
import support
from correlation import *
# import matplotlib.pyplot as plt
# import plots

import numpy as np
# import pandas as pd

import pandas as pd
from scipy.stats import norm
from scipy.stats import mannwhitneyu
data_filename = "UmfrageRohdaten.csv"
year = 2021
data2021 = pd.read_csv(f"{year}{data_filename}", sep=";")  # sep necessary because of Order column
support.data_preprocess(data2021, year)
data2021 = support.drop_rare_genders(data2021)

result = mann_whitney_u(data2021, ["Gender", "PastVictim_CyStalk"])
print(result)