import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("water_potability.xlsx")
print(df.head())
sns.countplot(x="Potability", data=df)
sns.despine()
plt.show()
