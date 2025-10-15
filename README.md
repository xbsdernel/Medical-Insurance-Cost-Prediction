# Medical-Insurance-Cost-Prediction
Predict Medical Insurance Cost using Classical Machine Learning (University project)

# Importing Libraries:
```python
# Importing libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
```

# Phase 1: Data understanding Process:

```python
# 1- Read Data
data = pd.read_csv("medical_insurance.csv", delimiter=',')
df = data.copy()
```

```python
# 2- Read first 5 samples of Data
print("Printing a Sample of Data\n")
df.head()
```

```python
# 3- Random Sample of Data
df.sample(5)
```

```python
# 3- Size of Data
df.shape
```

```python
# 4- Information of Data
df.info()
```

```python
# 5- Check Names Of columns
df.columns
```

```python
# 6- Check Description
df.describe().T
```

```python
# 7- Number of Unique Columns
df.nunique()
```

```python
# 8- Check Balance Of Data
df['bmi'].value_counts()
df['income'].value_counts(normalize=True) * 100
```

# Phase 2: Data Cleaning Process:

```python
# 1- Check Null (Missing Values)
df.isna().sum()
```

```python
# Check Null (Missing Values) By presentage %
(df.isna().sum() / len(df)) * 100
```

```python
# Dealing with NAN values throw Most Frequent Technique
imputer = SimpleImputer(strategy='most_frequent')
df[['alcohol_freq']] = imputer.fit_transform(df[['alcohol_freq']])

print("Missing values after imputation:", df['alcohol_freq'].isna().sum())
print(df['alcohol_freq'].value_counts())
```

```python
# Box plots for numerical columns
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
n_cols = 2
n_rows = math.ceil(len(num_cols) / n_cols)

plt.figure(figsize=(15, n_rows * 3))
for i, feature in enumerate(num_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.boxplot(x=df[feature])
    plt.title(f'Box Plot of {feature}')
plt.tight_layout()
plt.show()
```
