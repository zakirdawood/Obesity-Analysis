import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# load dataset
data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# turn yes/no columns into 0/1 columns
binary_columns = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
for col in binary_columns:
    data[col] = data[col].map({'yes': 1, 'no': 0})

# label encoding to ensure cleaner correlation matrix, get_dummies used instead in main scripts for both binary and categorical columns
label_columns = ['Gender', 'CAEC', 'CALC', 'MTRANS', 'NObeyesdad']
label_encoder = LabelEncoder()
for col in label_columns:
    data[col] = label_encoder.fit_transform(data[col])

# plot correlation matrix
correlation_matrix = data.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
plt.tight_layout()
plt.show()