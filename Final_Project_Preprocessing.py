# Databricks notebook source
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import length, col, count, upper, isnan, when, expr
from pyspark.sql.types import IntegerType
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

df = spark.read.table("hive_metastore.default.regularseason_1_csv")

# COMMAND ----------

df.show()

# COMMAND ----------

# Get the number of rows
num_rows = df.count()

# Get the number of columns
num_cols = len(df.columns)
print(f"The dataset has {num_rows} rows and {num_cols} columns.")

# COMMAND ----------

# Calculate summary statistics using Spark
summary = df.describe()
summary.toPandas()

# COMMAND ----------

# Get the count of duplicates in each column
duplicates_count = {}
for column in df.columns:
    duplicate_count = df.groupBy(column).count().filter(expr("count > 1")).count()
    if duplicate_count > 0:
        duplicates_count[column] = duplicate_count

# COMMAND ----------

# Display the count of duplicates in each column
for column, count in duplicates_count.items():
    print(f"Column '{column}' has {count} duplicates.")

# COMMAND ----------

from pyspark.sql.functions import count, when, col, isnan

# Calculate the number of missing values for each column
missing_values = df.select([count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in df.columns])

# Convert to Pandas DataFrame for easier manipulation
missing_values_df = missing_values.toPandas()

# Filter and print columns with missing values
columns_with_missing_values = missing_values_df.columns[missing_values_df.iloc[0] > 0].tolist()

print("Columns with missing values:")
for column in columns_with_missing_values:
    print(column, ":", missing_values_df[column].iloc[0])

# COMMAND ----------

from pyspark.sql.functions import col, length
# 1. Count occurrences of each abbreviation
print("Occurrences of each abbreviation:")
df.groupBy("TEAM_ABBREVIATION").count().orderBy("count", ascending=False).show()

# 2. Check for unexpected lengths or characters
# For example, if we expect all abbreviations to have a length of 3
df_wrong_length = df.filter(length(col("TEAM_ABBREVIATION")) != 3)
if df_wrong_length.count() > 0:
    print("Rows with unexpected abbreviation length:")
    df_wrong_length.show()

# Check if any abbreviation has characters other than uppercase letters
df_non_uppercase = df.filter(~col("TEAM_ABBREVIATION").rlike("^[A-Z]+$"))
if df_non_uppercase.count() > 0:
    print("Rows with non-uppercase letters in abbreviation:")
    df_non_uppercase.show()

# 3. If you have a predefined list of abbreviations
valid_abbreviations = ["ABC", "DEF", "GHI"]  # example list
df_invalid = df.filter(~col("TEAM_ABBREVIATION").isin(valid_abbreviations))
if df_invalid.count() > 0:
    print("Rows with invalid abbreviations:")
    df_invalid.show()

print("Unique TEAM_ABBREVIATION values:")
df.select("TEAM_ABBREVIATION").distinct().show()

# COMMAND ----------

# Function to detect outliers using IQR for a given column
def detect_outliers(df, col_name):
    # Compute Q1, Q3, and IQR
    quantiles = df.stat.approxQuantile(col_name, [0.25, 0.75], 0.01)
    Q1 = quantiles[0]
    Q3 = quantiles[1]
    IQR = Q3 - Q1

    # Lower and upper bounds to identify outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtering the outliers
    outliers = df.filter((col(col_name) < lower_bound) | (col(col_name) > upper_bound))

    return outliers

columns_with_outliers = []

# Detect outliers for each column
for column in df.columns:
    try:
        outliers = detect_outliers(df, column)
        if outliers.count() > 0:
            print(f"Outliers detected in column {column}!")
            columns_with_outliers.append(column)
    except:
        print(f"Failed to compute outliers for column {column}. Might be non-numeric.")

print("\nColumns with detected outliers:")
for col in columns_with_outliers:
    print(col)

# COMMAND ----------

# Function to detect outliers using IQR for a given column
def detect_outliers(df, col_name):
    # Compute Q1, Q3, and IQR
    quantiles = df.stat.approxQuantile(col_name, [0.25, 0.75], 0.01)
    Q1 = quantiles[0]
    Q3 = quantiles[1]
    IQR = Q3 - Q1

     # Lower and upper bounds to identify outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtering the outliers
    outliers = df.filter((col(col_name) < lower_bound) | (col(col_name) > upper_bound))

    return outliers

# Read data from CSV using Spark
df = spark.read.table("hive_metastore.default.regularseason_1_csv")

columns_with_outliers = []

# Detect outliers for each column
for column in df.columns:
    try:
        outliers = detect_outliers(df, column)
        if outliers.count() > 0:
            print(f"Outliers detected in column {column}!")
            columns_with_outliers.append(column)

            # Print rows with outliers
            print(f"Rows with outliers in column {column}:")
            outliers.show()
    except:
        print(f"Failed to compute outliers for column {column}. Might be non-numeric.")

print("\nColumns with detected outliers:")
for col in columns_with_outliers:
    print(col)

# COMMAND ----------

column_to_check = "PLAYER_AGE"
pandas_data = df.select(column_to_check).toPandas()

plt.figure(figsize=(6, 8))
pandas_data.boxplot(column=column_to_check)
plt.title("Box Plot of Player Age")
plt.ylabel("Age")
plt.show()


# COMMAND ----------

columns_to_check = ["FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
                    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
                    "AST", "STL", "BLK", "TOV", "PTS"]

for column in columns_to_check:
    stats = df.select(column).summary("25%", "75%").toPandas()
    q1 = float(stats.loc[0, column])
    q3 = float(stats.loc[1, column])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df.filter((col(column) < lower_bound) | (col(column) > upper_bound))

    print(f"Column: {column}")
    print(f"Number of outliers: {outliers.count()}")
    print("Outliers distribution:")
    outliers.groupBy(column).count().orderBy(column).show()

# COMMAND ----------

columns_to_check = ["FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
                    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
                    "AST", "STL", "BLK", "TOV", "PTS"]

pandas_data = df.select(columns_to_check).toPandas()

plt.figure(figsize=(12, 8))
pandas_data.boxplot(column=columns_to_check)
plt.xticks(rotation=45)
plt.title("Box Plot of WNBA Player Statistics")
plt.ylabel("Value")
plt.show()

# COMMAND ----------

# Checking columns for zero values
zero_values = {}
for col in df.columns:
    count = df.filter(df[col] == 0).count()
    if count > 0:
        zero_values[col] = count

# Print columns with zero values and their counts
for col, count in zero_values.items():
    print(f"Column {col} has {count} zero values")

total_rows = df.count()
zero_values = {}

for col in df.columns:
    count = df.filter(df[col] == 0).count()
    if count > 0:
        zero_values[col] = count / total_rows * 100  
 
for col, percentage in zero_values.items():
    print(f"Column {col} has {percentage:.2f}% zero values")

# COMMAND ----------

df = df.filter(df['TEAM_ID'] != 0)
df.show()