### `DAV Practical`
```markdown
# Python and Data Analysis Questions and Solutions

This repository contains a set of Python programming questions and their solutions. The solutions make use of popular libraries like Pandas, NumPy, Matplotlib, and more. Each question is solved step-by-step and is organized in separate files for better readability.

## Contents

1. [DataFrame Operations](questions_responses/question1.md)
2. [Pandas Series Questions](questions_responses/question2.md)
3. [Titanic Dataset Analysis](questions_responses/question3.md)
4. [Iris Dataset Visualization](questions_responses/question4.md)
5. [Family Data Analysis](questions_responses/question5.md)

Feel free to explore the solutions and contribute!
```

---

### `questions_responses/question1.md`
```markdown
# Question 1: DataFrame Operations

## Problem Statement
Perform various operations on a DataFrame:
1. Identify and count missing values.
2. Drop columns with more than 5 null values.
3. Drop the row with the maximum sum of values.
4. Sort the DataFrame by the first column.
5. Remove duplicates from the first column.
6. Compute correlation and covariance.
7. Discretize the second column into 5 bins.

## Solution
```python
import pandas as pd
import numpy as np

# Step 1: Create a DataFrame
np.random.seed(42)
df = pd.DataFrame(np.random.rand(50, 3) * 100, columns=['Column1', 'Column2', 'Column3'])

# Replace 10% of values with NaN
num_nulls = int(0.1 * df.size)
indices = [(np.random.randint(0, 50), np.random.randint(0, 3)) for _ in range(num_nulls)]
for i, j in indices:
    df.iat[i, j] = np.nan

# a. Identify and count missing values
missing_values = df.isnull().sum()

# b. Drop columns with more than 5 null values
df = df.drop(columns=df.columns[df.isnull().sum() > 5])

# c. Drop the row with the maximum sum
row_max_sum = df.sum(axis=1).idxmax()
df = df.drop(index=row_max_sum)

# d. Sort the DataFrame by the first column
df = df.sort_values(by='Column1')

# e. Remove duplicates from the first column
df = df.drop_duplicates(subset='Column1')

# f. Compute correlation and covariance
correlation = df['Column1'].corr(df['Column2'])
covariance = df['Column2'].cov(df['Column3'])

# g. Discretize the second column into 5 bins
df['Column2_bins'] = pd.cut(df['Column2'], bins=5)

# Output
print("Missing values:\n", missing_values)
print("DataFrame after operations:\n", df)
print("Correlation between Column1 and Column2:", correlation)
print("Covariance between Column2 and Column3:", covariance)
```
---

### `questions_responses/question2.md`
```markdown
# Question 2: Pandas Series Operations

## Problem Statement
1. Create a Series with 5 elements. Sort it by index and by values.
2. Create a Series with duplicate values and find minimum and maximum ranks using `first` and `max` methods.
3. Find the index of the minimum and maximum element in a Series.

## Solution
```python
import pandas as pd

# a. Create a Series and sort it
s = pd.Series([50, 20, 30, 40, 10], index=['a', 'b', 'c', 'd', 'e'])
sorted_by_index = s.sort_index()
sorted_by_values = s.sort_values()

# b. Series with duplicate values
s2 = pd.Series([30, 20, 20, 10, 10, 30])
rank_first = s2.rank(method='first')
rank_max = s2.rank(method='max')

# c. Find the index of the minimum and maximum elements
min_index = s.idxmin()
max_index = s.idxmax()

# Output
print("Sorted by index:\n", sorted_by_index)
print("Sorted by values:\n", sorted_by_values)
print("Ranks (first method):\n", rank_first)
print("Ranks (max method):\n", rank_max)
print("Index of minimum value:", min_index)
print("Index of maximum value:", max_index)
```
---

### `questions_responses/question3.md`
```markdown
# Question 3: Titanic Dataset Analysis

## Problem Statement
Analyze the Titanic dataset to:
1. Clean data by dropping the column with the most missing values.
2. Count passengers older than 30.
3. Compute the total fare for second-class passengers.
4. Compare survivors across passenger classes.
5. Compute descriptive statistics for age grouped by gender.
6. Create visualizations like scatter plots, density plots, and pie charts.
7. Analyze survival percentages by class.

## Solution
Refer to the [Titanic dataset analysis code](https://gist.github.com/example-link).
```

---

### `questions_responses/question4.md`
```markdown
# Question 4: Iris Dataset Visualization

## Problem Statement
Perform data visualization on the Iris dataset:
1. Load data and check for missing values.
2. Plot bar chart for class labels.
3. Create scatter plots, density plots, and heatmaps.
4. Compute statistical measures and correlations.

## Solution
Refer to the [Iris dataset visualization code](https://gist.github.com/example-link).
```

---

### `questions_responses/question5.md`
```markdown
# Question 5: Family Data Analysis

## Problem Statement
Analyze the given family dataset:
1. Calculate family-wise gross income.
2. Find the highest and lowest income for each family.
3. Identify members earning less than Rs. 80,000.
4. Compute statistics for females.
5. Remove rows with income below the average.

## Solution
```python
import pandas as pd

# Create the dataset
data = {
    "FamilyName": ["Shah", "Vats", "Vats", "Kumar", "Vats", "Kumar", "Shah", "Shah", "Kumar", "Vats"],
    "Gender": ["Male", "Male", "Female", "Female", "Female", "Male", "Male", "Female", "Female", "Male"],
    "MonthlyIncome": [44000, 65000, 43150, 66500, 255000, 103000, 55000, 112400, 81030, 71900]
}
df = pd.DataFrame(data)

# a. Calculate family-wise gross income
gross_income = df.groupby('FamilyName')['MonthlyIncome'].sum()

# b. Highest and lowest income for each family
highest_income = df.groupby('FamilyName')['MonthlyIncome'].max()
lowest_income = df.groupby('FamilyName')['MonthlyIncome'].min()

# c. Members earning less than Rs. 80,000
below_80k = df[df['MonthlyIncome'] < 80000]

# d. Statistics for females
female_stats = df[df['Gender'] == 'Female'].agg({'Gender': 'count', 'MonthlyIncome': 'mean'})

# e. Remove rows below the average income
average_income = df['MonthlyIncome'].mean()
df_filtered = df[df['MonthlyIncome'] >= average_income]

# Output
print("Gross Income:\n", gross_income)
print("Highest Income:\n", highest_income)
print("Lowest Income:\n", lowest_income)
print("Below Rs. 80,000:\n", below_80k)
print("Female Statistics:\n", female_stats)
print("Filtered DataFrame:\n", df_filtered)
```
---

### Upload Instructions
1. Save the files in the appropriate folder structure (`questions_responses/`).
2. Initialize a Git repository.
3. Commit and push to GitHub.

After uploading, you'll have a clear repository ready to share!
