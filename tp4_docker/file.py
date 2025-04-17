# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shutil

# Step 1: Load the data and remove duplicates
file_path = "questions.csv"
"""data = pd.read_csv(file_path)"""
data = pd.read_csv(file_path, on_bad_lines='skip')  # تخطي السطور التالفة
print("\n البيانات قبل إزالة التكرارات:\n", data)

data.drop_duplicates(inplace=True)

print("\n بعد إزالة التكرارات:\n", data)

data.to_csv("questions_cleaned1.csv", index=False)

# Process in chunks to handle large datasets
chunk_size = 10000
temp_file = "questions_cleaned1.csv"
file_path = "questions.csv"

with open(temp_file, 'w') as f_out:
    first_chunk = True
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk_cleaned = chunk.drop_duplicates()
        chunk_cleaned.to_csv(f_out, index=False, header=first_chunk, mode='a')
        first_chunk = False

shutil.move(temp_file, file_path)
print(f" تم حذف القيم المكررة وحفظ التعديلات في نفس الملف: {file_path}")

# Step 2: Display the first chunk
chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size)
first_chunk = next(chunk_iterator)
print(first_chunk.head())

# Step 3: Convert 'Asks' to a datetime format (e.g., "asked 1 m" to a date)
# We'll extract the time difference and convert it to days since a reference date
def parse_asks(asks_str):
    if pd.isna(asks_str) or 'asked' not in asks_str:
        return None
    try:
        time_str = asks_str.replace("asked ", "").strip()
        if 'm' in time_str:
            months = int(time_str.replace(" m", ""))
            return pd.Timestamp("2025-04-02") - pd.Timedelta(days=months * 30)
        elif 'r' in time_str:
            years = int(time_str.replace(" r", ""))
            return pd.Timestamp("2025-04-02") - pd.Timedelta(days=years * 365)
        else:
            return None
    except:
        return None

temp_file = "questions_cleaned1.csv"
with open(temp_file, 'w') as f_out:
    first_chunk = True
    for chunk in pd.read_csv(file_path, chunksize=20000):
        if 'Asks' in chunk.columns:
            chunk['date'] = chunk['Asks'].apply(parse_asks)
        chunk.to_csv(f_out, index=False, header=first_chunk, mode='a')
        first_chunk = False

shutil.move(temp_file, file_path)
print(f" تم تحويل عمود 'Asks' إلى datetime وحفظ التعديلات في نفس الملف: {file_path}")

# Step 4: Convert 'Views' to numeric (e.g., "1,417 m" to 1417000)
def parse_views(views_str):
    if pd.isna(views_str):
        return None
    views_str = str(views_str).replace(",", "").strip()
    try:
        if 'm' in views_str:
            return float(views_str.replace(" m", "")) * 1_000_000
        elif 'k' in views_str:
            return float(views_str.replace(" k", "")) * 1_000
        else:
            return float(views_str)
    except:
        return None

temp_file = "questions_cleaned1.csv"
with open(temp_file, 'w') as f_out:
    first_chunk = True
    for chunk in pd.read_csv(file_path, chunksize=20000):
        if 'Views' in chunk.columns:
            chunk['Views'] = chunk['Views'].apply(parse_views)
        chunk.to_csv(f_out, index=False, header=first_chunk, mode='a')
        first_chunk = False

shutil.move(temp_file, file_path)
print(f" تم تحويل عمود 'Views' إلى قيم رقمية وحفظ التعديلات في: {file_path}")

# Step 5: Remove outliers from 'Reputation'
chunk_size = 20000
all_values = []
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    if 'Reputation' in chunk.columns:
        all_values.extend(chunk['Reputation'].dropna().tolist())

Q1 = pd.Series(all_values).quantile(0.25)
Q3 = pd.Series(all_values).quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

temp_file = "questions_cleaned1.csv"
with open(temp_file, 'w') as f_out:
    first_chunk = True
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        if 'Reputation' in chunk.columns:
            chunk = chunk[(chunk['Reputation'] >= lower_bound) & (chunk['Reputation'] <= upper_bound)]
        chunk.to_csv(f_out, index=False, header=first_chunk, mode='a')
        first_chunk = False

print(f" تم إزالة القيم المتطرفة من 'Reputation' وحفظ البيانات النظيفة في: {temp_file}")

# Step 6: Convert 'date' to numeric (days since 2000-01-01)
output_file = "questions_no_outliers1.csv"
with open(output_file, 'w') as f_out:
    first_chunk = True
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        if 'date' in chunk.columns:
            chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
            chunk['date_numeric'] = (chunk['date'] - pd.Timestamp("2000-01-01")) // pd.Timedelta('1D')
        chunk.to_csv(f_out, index=False, header=first_chunk, mode='a')
        first_chunk = False

print(f" تم تحويل عمود 'date' إلى عدد أيام وحفظ البيانات في '{output_file}'")

# Step 7: Summary statistics and visualization
file_path = "questions.csv"
summary_stats = []
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    summary_stats.append(chunk.describe())

print(pd.concat(summary_stats).groupby(level=0).mean())

# Correlation heatmap
corr_matrices = []
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    numeric_chunk = chunk.select_dtypes(include=['number'])
    corr_matrices.append(numeric_chunk.corr())

mean_corr_matrix = pd.concat(corr_matrices).groupby(level=0).mean()
sns.heatmap(mean_corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Trend of Views over time
views_trend = {}
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    if 'date' in chunk.columns and 'Views' in chunk.columns:
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
        daily_views = chunk.groupby('date')['Views'].sum()
        for date, views in daily_views.items():
            views_trend[date] = views_trend.get(date, 0) + views

views_df = pd.DataFrame(list(views_trend.items()), columns=['date', 'Views'])
views_df.sort_values('date', inplace=True)
views_df.plot(x='date', y='Views', kind='line', title="Total Views Over Time")
plt.xlabel("Date")
plt.ylabel("Total Views")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 8: Predictive modeling (Predict Reputation based on Votes, Answers, Views, and date_numeric)
use_cols = ['Votes', 'Answers', 'Views', 'Reputation', 'date_numeric']
data_chunks = []
for chunk in pd.read_csv(file_path, usecols=use_cols, chunksize=8000):
    chunk['date_numeric'] = chunk['date_numeric'].astype(float)
    data_chunks.append(chunk)

df = pd.concat(data_chunks, ignore_index=True)

features = ['Votes', 'Answers', 'Views', 'date_numeric']
target = 'Reputation'

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

print(f"عدد العينات في التدريب: {len(X_train)}, وعدد العينات في الاختبار: {len(X_test)}")

dt_model = DecisionTreeRegressor(max_depth=10, min_samples_split=10, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print(f"\n **أداء النموذج:**")
print(f" MAE: {mae_dt:.2f}")
print(f" MSE: {mse_dt:.2f}")
print(f" R² Score: {r2_dt:.2f}")