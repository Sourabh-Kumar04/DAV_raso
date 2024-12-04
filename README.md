### `DAV Practical`
```markdown
# Python and Data Analysis Questions and Solutions

This repository contains a set of Python programming questions and their solutions. The solutions make use of popular libraries like Pandas, NumPy, Matplotlib, and more. Each question is solved step-by-step and is organized in separate files for better readability.

## Contents

1. [question_1](questions_responses/question1.md)
2. [question_2](questions_responses/question2.md)
3. [question_3](questions_responses/question3.md)
4. [question_4](questions_responses/question4.md)
5. [question_5](questions_responses/question5.md)
6. [question_6](questions_responses/question6.md)
7. [question_7](questions_responses/question7.md)


Feel free to explore the solutions and contribute!
```

---

### `questions_responses/question1.md`
```markdown
# Question 1: DataFrame Operations

## Problem Statement
Write programs in Python using NumPy library to do the following:
    a. Create  a two dimensional array, ARR1 having random values from 0 to 1. Compute the mean, standard deviation, and variance of ARR1 along the second axis.
    b.   Create a 2-dimensional array of size m x n integer elements, also print the shape, type and data type of the array and then   reshape it into an n x m array, where  n and m are user inputs given at the run time.
    c.  Test whether the elements of a given 1D array are zero, non-zero and NaN. Record the indices of these elements in three separate arrays.
    d. Create three random arrays of the same size: Array1, Array2 and Array3. Subtract Array 2 from Array3 and store in Array4. Create another array Array5 having two times the values  in Array1. Find Covariance and Correlation of  Array1 with  Array4 and Array5 respectively.
    e. Create two random arrays of the same size 10: Array1, and Array2. Find the sum of the first half of both the arrays and product of the second half of both the arrays.
    f. Create an array with random  values. Determine the size of the memory occupied by the array.     g. Create a 2-dimensional array of size m x n having integer elements in the range (10,100). Write statements to swap any two rows, reverse a specified column and store updated array in another variable 

## Solution
```python
import numpy as np

# a. Create a two-dimensional array, ARR1 having random values from 0 to 1.
ARR1 = np.random.rand(5, 5)  # Example size 5x5
mean_arr1 = np.mean(ARR1, axis=1)
std_arr1 = np.std(ARR1, axis=1)
var_arr1 = np.var(ARR1, axis=1)

print("ARR1:\n", ARR1)
print("Mean along second axis:", mean_arr1)
print("Standard Deviation along second axis:", std_arr1)
print("Variance along second axis:", var_arr1)

# b. Create a 2-dimensional array of size m x n integer elements.
m = int(input("Enter number of rows (m): "))
n = int(input("Enter number of columns (n): "))
array_2d = np.random.randint(0, 100, size=(m, n))
print("Array shape:", array_2d.shape)
print("Array type:", type(array_2d))
print("Data type of array elements:", array_2d.dtype)

# Reshape it into an n x m array
reshaped_array = array_2d.reshape(n, m)
print("Reshaped Array:\n", reshaped_array)

# c. Test whether the elements of a given 1D array are zero, non-zero and NaN.
array_1d = np.array([0, 1, 2, np.nan, 3, 0, np.nan])
zero_indices = np.where(array_1d == 0)[0]
non_zero_indices = np.where(array_1d != 0)[0]
nan_indices = np.where(np.isnan(array_1d))[0]

print("Indices of zeros:", zero_indices)
print("Indices of non-zeros:", non_zero_indices)
print("Indices of NaNs:", nan_indices)

# d. Create three random arrays of the same size.
Array1 = np.random.rand(5)
Array2 = np.random.rand(5)
Array3 = np.random.rand(5)

Array4 = Array3 - Array2
Array5 = 2 * Array1

covariance = np.cov(Array1, Array4)[0][1]
correlation = np.corrcoef(Array1, Array5)[0][1]

print("Array1:", Array1)
print("Array2:", Array2)
print("Array3:", Array3)
print("Array4 (Array3 - Array2):", Array4)
print("Array5 (2 * Array1):", Array5)
print("Covariance of Array1 and Array4:", covariance)
print("Correlation of Array1 and Array5:", correlation)

# e. Create two random arrays of the same size 10.
Array1 = np.random.rand(10)
Array2 = np.random.rand(10)

sum_first_half = np.sum(Array1[:5]) + np.sum(Array2[:5])
product_second_half = np.prod(Array1[5:]) * np.prod(Array2[5:])

print("Sum of first half of both arrays:", sum_first_half)
print("Product of second half of both arrays:", product_second_half)

# f. Create an array with random values and determine the size of the memory occupied by the array.
random_array = np.random.rand(1000)  # Example size
memory_size = random_array.nbytes  # Size in bytes
print("Memory size occupied by the array:", memory_size, "bytes")

# g. Create a 2-dimensional array of size m x n having integer elements in the range (10,100).
m = 5
n = 5
int_array = np.random.randint(10, 100, size=(m, n))
print("Original Array:\n", int_array)

# Swap any two rows (e.g., row 0 and row 1)
int_array[[0, 1]] = int_array[[1, 0]]
print("Array after swapping rows 0 and 1:\n", int_array)

# Reverse a specified column (e.g., column 2)
int_array[:, 2] = int_array[::-1, 2]
print("Array after reversing column 2:\n", int_array)
```
---

### `questions_responses/question2.md`
```markdown
# Question 2: Pandas Series Operations

## Problem Statement
Do the following using PANDAS Series:
    a. Create a series with 5 elements. Display the series sorted on index and also sorted on values seperately
    b. Create a  series with N elements with some duplicate values. Find  the minimum and maximum ranks  assigned to the values using ‘first’ and ‘max’ methods
    c. Display the index value of the minimum and maximum element of a Series 

## Solution
```python
import pandas as pd

# Task a
# Create a series with 5 elements
series_a = pd.Series([10, 5, 20, 15, 1], index=['a', 'b', 'c', 'd', 'e'])

# Display the series sorted by index
sorted_by_index = series_a.sort_index()

# Display the series sorted by values
sorted_by_values = series_a.sort_values()

# Task b
# Create a series with N elements and duplicate values
series_b = pd.Series([50, 10, 20, 50, 10, 30, 50])

# Rank the values using 'first' method
ranks_first = series_b.rank(method='first')

# Rank the values using 'max' method
ranks_max = series_b.rank(method='max')

# Minimum and maximum ranks
min_rank_first = ranks_first.min()
max_rank_first = ranks_first.max()

# Task c
# Index of the minimum and maximum elements of a Series
min_index = series_a.idxmin()
max_index = series_a.idxmax()

# Display results
print("Task a:")
print("Original Series:")
print(series_a)
print("\nSorted by Index:")
print(sorted_by_index)
print("\nSorted by Values:")
print(sorted_by_values)

print("\nTask b:")
print("Original Series:")
print(series_b)
print("\nRanks using 'first' method:")
print(ranks_first)
print("\nRanks using 'max' method:")
print(ranks_max)
print(f"\nMinimum Rank (first method): {min_rank_first}")
print(f"Maximum Rank (first method): {max_rank_first}")

print("\nTask c:")
print(f"Index of Minimum Value: {min_index}")
print(f"Index of Maximum Value: {max_index}")
```
---

### `questions_responses/question3.md`
```markdown
# Question 3: 

## Problem Statement
Create a data frame having at least 3 columns and 50 rows to store numeric data generated using a random function. Replace 10% of the values by null values whose index positions are generated using random function. Do the following:
    a.   Identify and count missing values in a data frame.
    b.   Drop the column having more than 5 null values.
    c.  Identify the row label having maximum of the sum of all values in a row and drop that row.      d.  Sort the data frame on the basis of the first column.
    e.  Remove all duplicates from the first column.
    f.  Find the correlation between first and second column and covariance between second and third column.
    g.  Discretize the second column and create 5 bins. 

## Solution
```python
import pandas as pd
import numpy as np

# Create a DataFrame with 3 columns and 50 rows of random numeric data
np.random.seed(0)  # For reproducibility
data = np.random.rand(50, 3) * 100  # Random values between 0 and 100
df = pd.DataFrame(data, columns=['Column1', 'Column2', 'Column3'])

# Replace 10% of the values with null values
num_nulls = int(0.1 * df.size)  # 10% of total values
null_indices = np.random.choice(df.size, num_nulls, replace=False)
df.values.ravel()[null_indices] = np.nan

print("DataFrame with NaN values:\n", df)

# Task a: Identify and count missing values in a DataFrame
missing_values_count = df.isnull().sum()
print("\nMissing values in each column:\n", missing_values_count)

# Task b: Drop the column having more than 5 null values
df_dropped_column = df.dropna(axis=1, thresh=len(df) - 5)
print("\nDataFrame after dropping columns with more than 5 null values:\n", df_dropped_column)

# Task c: Identify the row label having the maximum sum of all values in a row and drop that row
row_sums = df_dropped_column.sum(axis=1)
max_row_index = row_sums.idxmax()
df_dropped_row = df_dropped_column.drop(index=max_row_index)
print("\nDataFrame after dropping the row with maximum sum:\n", df_dropped_row)

# Task d: Sort the DataFrame on the basis of the first column
df_sorted = df_dropped_row.sort_values(by='Column1')
print("\nSorted DataFrame based on Column1:\n", df_sorted)

# Task e: Remove all duplicates from the first column
df_no_duplicates = df_sorted.drop_duplicates(subset='Column1')
print("\nDataFrame after removing duplicates from Column1:\n", df_no_duplicates)

# Task f: Find the correlation between the first and second column
correlation = df_no_duplicates['Column1'].corr(df_no_duplicates['Column2'])
print("\nCorrelation between Column1 and Column2:", correlation)

# Find the covariance between the second and third column
covariance = df_no_duplicates['Column2'].cov(df_no_duplicates['Column3'])
print("Covariance between Column2 and Column3:", covariance)

# Task g: Discretize the second column and create 5 bins
df_no_duplicates['Column2_Binned'] = pd.cut(df_no_duplicates['Column2'], bins=5, labels=False)
print("\nDataFrame with discretized Column2 into 5 bins:\n", df_no_duplicates)

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
```python
import pandas as pd
import numpy as np

df1 = pd.read_excel("Workshop_Attendance1.xlsx")
df2 = pd.read_excel("Workshop_Attendance2.xlsx")



# Task a: Merge the two data frames to find names of students who attended both workshops
common_attendees = pd.merge(df1, df2, on="Name", how="inner")["Name"].unique()

# Task b: Find names of all students who attended a single workshop only
all_names = set(df1["Name"]).union(df2["Name"])
single_workshop_names = all_names - set(common_attendees)

# Task c: Merge two data frames row-wise and find the total number of records
merged_rowwise = pd.concat([df1, df2], axis=0, ignore_index=True)
total_records = len(merged_rowwise)

# Task d: Merge two data frames row-wise and use 'Name' and 'Date' as multi-row indexes
hierarchical_df = merged_rowwise.set_index(["Name", "Date"])
descriptive_stats = hierarchical_df.describe()

# Display results
results = {
    "Task a": {
        "Common Attendees": list(common_attendees),
    },
    "Task b": {
        "Single Workshop Attendees": list(single_workshop_names),
    },
    "Task c": {
        "Total Records in Merged DataFrame": total_records,
    },
    "Task d": {
        "Hierarchical DataFrame Descriptive Statistics": descriptive_stats,
    },
}

for task, output in results.items():
    print(f"{task}:\n{output}\n")
```
---

### `questions_responses/question5.md`
```markdown
# Question 5: 

## Problem Statement


## Solution
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
import scipy.stats as stats

# Load the Iris dataset
iris_data = load_iris()
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
df['species'] = iris_data.target_names[iris_data.target]

# a. Display info on datatypes in the dataset
print("Iris Dataset Info:")
print(df.info())

# b. Find the number of missing values in each column
missing_values = df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# c. Plot bar chart to show the frequency of each class label in the data
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='species', palette='viridis')
plt.title('Frequency of Each Class Label in the Iris Dataset')
plt.xlabel('Species')
plt.ylabel('Frequency')
plt.legend(title='Species')
plt.show()

# d. Draw a scatter plot for Petal Length vs Sepal Length and fit a regression line
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x='sepal length (cm)', y='petal length (cm)', marker='o', color='blue')
plt.title('Petal Length vs Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# e. Plot density distribution for feature Petal width
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x='petal width (cm)', hue='species', fill=True, common_norm=False, palette='crest')
plt.title('Density Distribution of Petal Width')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Density')
plt.legend(title='Species')
plt.show()

# f. Use a pair plot to show pairwise bivariate distribution in the Iris Dataset
sns.pairplot(df, hue='species', palette='bright')
plt.suptitle('Pairwise Bivariate Distribution in the Iris Dataset', y=1.02)
plt.show()

# g. Draw heatmap for any two numeric attributes
plt.figure(figsize=(8, 5))
sns.heatmap(df[['sepal length (cm)', 'sepal width (cm)']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Sepal Length and Sepal Width Correlation')
plt.show()

# h. Compute mean, mode, median, standard deviation, confidence interval and standard error for each numeric feature
statistics = {}
for column in df.columns[:-1]:  # Exclude species column
    statistics[column] = {
        'Mean': df[column].mean(),
        'Median': df[column].median(),
        'Mode': df[column].mode()[0],
        'Standard Deviation': df[column].std(),
        'Standard Error': stats.sem(df[column]),
        'Confidence Interval (95%)': stats.t.interval(0.95, len(df[column])-1, loc=df[column].mean(), scale=stats.sem(df[column]))
    }

statistics_df = pd.DataFrame(statistics).T
print("\nStatistics for Numeric Features:")
print(statistics_df)

# i. Compute correlation coefficients between each pair of features and plot heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Coefficients Heatmap')
plt.show()
```
---

### `questions_responses/question6.md`
```markdown
# Question 6: Family Data Analysis

## Problem Statement


## Solution
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
titanic_data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# a. Clean the data by dropping the column which has the largest number of missing values
column_with_most_nan = titanic_data.isnull().sum().idxmax()
titanic_data.drop(columns=[column_with_most_nan], inplace=True)
print(f"Dropped column: {column_with_most_nan}")

# b. Find total number of passengers with age more than 30
passengers_over_30 = titanic_data[titanic_data['Age'] > 30].shape[0]
print(f"Total number of passengers with age more than 30: {passengers_over_30}")

# c. Find total fare paid by passengers of second class
total_fare_second_class = titanic_data[titanic_data['Pclass'] == 2]['Fare'].sum()
print(f"Total fare paid by passengers of second class: {total_fare_second_class}")

# d. Compare number of survivors of each passenger class
survivors_per_class = titanic_data.groupby('Pclass')['Survived'].sum()
print("\nNumber of survivors of each passenger class:")
print(survivors_per_class)

# e. Compute descriptive statistics for age attribute gender-wise
age_statistics_gender = titanic_data.groupby('Sex')['Age'].describe()
print("\nDescriptive statistics for age attribute gender-wise:")
print(age_statistics_gender)

# f. Draw a scatter plot for passenger fare paid by Female and Male passengers separately
plt.figure(figsize=(10, 6))
sns.scatterplot(data=titanic_data, x='Fare', y='Age', hue='Sex', alpha=0.6)
plt.title('Passenger Fare vs Age by Gender')
plt.xlabel('Fare')
plt.ylabel('Age')
plt.legend(title='Gender')
plt.show()

# g. Compare density distribution for features age and passenger fare
plt.figure(figsize=(10, 6))
sns.kdeplot(data=titanic_data, x='Age', fill=True, label='Age', color='blue', alpha=0.5)
sns.kdeplot(data=titanic_data, x='Fare', fill=True, label='Fare', color='orange', alpha=0.5)
plt.title('Density Distribution of Age and Fare')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()

# h. Draw the pie chart for three groups labelled as class 1, class 2, class 3 respectively
class_counts = titanic_data['Pclass'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(class_counts, labels=['Class 1', 'Class 2', 'Class 3'], autopct='%1.1f%%', colors=['gold', 'lightcoral', 'lightskyblue'])
plt.title('Passenger Distribution by Class')
plt.show()

# Find % of survived passengers for each class
survival_rate_per_class = titanic_data.groupby('Pclass')['Survived'].mean() * 100
print("\n% of survived passengers for each class:")
print(survival_rate_per_class)

# Answer the question: Did class play a role in survival?
print("\nDid class play a role in survival?")
if survival_rate_per_class.max() > survival_rate_per_class.min():
    print("Yes, class played a role in survival.")
else:
    print("No, class did not play a role in survival.")
```
---

### `questions_responses/question7.md`
```markdown
# Question 7: 

## Problem Statement


## Solution
```python
import pandas as pd

# Create the DataFrame
data = {
    'FamilyName': ['Shah', 'Vats', 'Vats', 'Kumar', 'Vats', 'Kumar', 'Shah', 'Shah', 'Kumar', 'Vats'],
    'Gender': ['Male', 'Male', 'Female', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male'],
    'MonthlyIncome': [44000.00, 65000.00, 43150.00, 66500.00, 255000.00, 103000.00, 55000.00, 112400.00, 81030.00, 71900.00]
}

df = pd.DataFrame(data)

# a. Calculate and display familywise gross monthly income
familywise_income = df.groupby('FamilyName')['MonthlyIncome'].sum()
print("Familywise Gross Monthly Income:")
print(familywise_income)

# b. Display the highest and lowest monthly income for each family name
highest_lowest_income = df.groupby('FamilyName')['MonthlyIncome'].agg(['max', 'min'])
print("\nHighest and Lowest Monthly Income for Each Family Name:")
print(highest_lowest_income)

# c. Calculate and display monthly income of all members earning income less than Rs. 80000.00
income_below_80000 = df[df['MonthlyIncome'] < 80000]
print("\nMonthly Income of Members Earning Less Than Rs. 80000.00:")
print(income_below_80000)

# d. Display total number of females along with their average monthly income
female_count = df[df['Gender'] == 'Female'].shape[0]
average_female_income = df[df['Gender'] == 'Female']['MonthlyIncome'].mean()
print(f"\nTotal Number of Females: {female_count}, Average Monthly Income: {average_female_income:.2f}")

# e. Delete rows with Monthly income less than the average income of all members
average_income = df['MonthlyIncome'].mean()
df_filtered = df[df['MonthlyIncome'] >= average_income]

print("\nDataFrame after deleting rows with Monthly Income less than the average income:")
print(df_filtered)
```
---


