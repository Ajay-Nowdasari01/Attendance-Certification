import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------
# 1. DATA COLLECTION & PREPROCESSING (Real Raw Data)
# ----------------------------------------------------

def preprocess_sheet(file_path):
    """Loads a Zoom sheet, filters for ML internship, and removes duplicates."""
    df = pd.read_csv(file_path)
    
    # Filter only for Machine Learning sessions
    ml_df = df[df['Topic'].str.contains('Machine Learning', case=False, na=False)].copy()
    
    # Clean up emails (lowercase, strip whitespace)
    ml_df['Email'] = ml_df['Email'].astype(str).str.strip().str.lower()
    
    # Drop rows without valid emails or names
    ml_df = ml_df.dropna(subset=['Email', 'Name (original name)'])
    
    # Deduplicate: If a student reconnected, keep their maximum duration or just one record
    # For daily presence, we just need the unique list of emails per day
    unique_students = ml_df.drop_duplicates(subset=['Email']).copy()
    
    return unique_students[['Name (original name)', 'Email']]

# Processing the uploaded sheets
day1_present = preprocess_sheet('meetinglistdetails_2026_05_29_2026_05_29.csv')
day2_present = preprocess_sheet('meetinglistdetails_2026_05_30_2026_05_30.csv')

# Combine to extract our Master List of ~980 unique student emails
master_students = pd.concat([day1_present, day2_present]).drop_duplicates(subset=['Email']).reset_index(drop=True)

# If the unique list is larger than 980, we limit it to exactly 980 for your project scope
if len(master_students) > 980:
    master_students = master_students.sample(n=980, random_state=42).reset_index(drop=True)
elif len(master_students) < 980:
    # Fallback padding if unique names are fewer than 980
    missing = 980 - len(master_students)
    extra_records = pd.DataFrame({
        'Name (original name)': [f'Student_{i}' for i in range(missing)],
        'Email': [f'student_{i}@example.com' for i in range(missing)]
    })
    master_students = pd.concat([master_students, extra_records]).reset_index(drop=True)

# ----------------------------------------------------
# 2. SIMULATING 40 SESSIONS (Randomized Attendance)
# ----------------------------------------------------
# We mark Day 1 and Day 2 based on actual files, and randomly generate Days 3 to 40.
np.random.seed(42)  # For reproducibility

attendance_matrix = pd.DataFrame(index=master_students['Email'])

# Day 1 & Day 2 from files
attendance_matrix['Session_1'] = attendance_matrix.index.isin(day1_present['Email']).astype(int)
attendance_matrix['Session_2'] = attendance_matrix.index.isin(day2_present['Email']).astype(int)

# Simulating Sessions 3 to 40 (Random attendance probability between 60% and 95% per student)
for i in range(3, 41):
    # Generating realistic randomized attendance rates per student
    attendance_matrix[f'Session_{i}'] = np.random.choice([1, 0], size=980, p=[0.82, 0.18])

# ----------------------------------------------------
# 3. EDA & CERTIFICATION LOGIC
# ----------------------------------------------------

# Calculate Total Attendance per student
master_students['Total_Attendance'] = attendance_matrix.sum(axis=1).values
master_students['Attendance_Percentage'] = (master_students['Total_Attendance'] / 40) * 100

# Certification Condition: Total Attendance >= 32 sessions (80%)
master_students['Status'] = np.where(master_students['Total_Attendance'] >= 32, 'Certified', 'Not Certified')

# Quick Summary Statistics
summary = master_students['Status'].value_counts()
print("--- Certification Summary ---")
print(summary)
print("\nPercentage Certified: {:.2f}%".format((summary['Certified'] / 980) * 100))

# ----------------------------------------------------
# 4. VISUALIZATION (Scatter Plot)
# ----------------------------------------------------
plt.figure(figsize=(10, 6))

# Custom noise jitter added to Y-axis to prevent overlapping data points on identical integers
jitter = np.random.uniform(-0.3, 0.3, size=980)

sns.scatterplot(
    x=master_students['Attendance_Percentage'],
    y=master_students['Total_Attendance'] + jitter,
    hue=master_students['Status'],
    palette={'Certified': '#2ecc71', 'Not Certified': '#e74c3c'},
    alpha=0.7,
    edgecolor='w'
)

# Benchmark line for qualification (32 classes / 80%)
plt.axvline(x=80, color='blue', linestyle='--', linewidth=1.5, label='80% Attendance Threshold')
plt.axhline(y=32, color='blue', linestyle='--', linewidth=1.5)

plt.title('ML Internship Milestone Project: Attendance vs Certification', fontsize=14, pad=15)
plt.xlabel('Attendance Percentage (%)', fontsize=12)
plt.ylabel('Total Sessions Attended (Out of 40)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# 5. EXPORTING CERTIFICATION LISTS
download = input("Do you want to download the certification list as CSV? (yes/no): ").strip().lower()
if download == 'yes':
    # 1. Filter out certified and uncertified students separately
    certified_df = master_students[master_students['Status'] == 'Certified']
    not_certified_df = master_students[master_students['Status'] == 'Not Certified']

    # 2. Keep only important columns for clean roster lists
    columns_to_keep = ['Name (original name)', 'Email', 'Total_Attendance', 'Attendance_Percentage']

    # 3. Save directly to CSVs on your computer
    certified_df[columns_to_keep].to_csv('certified_students.csv', index=False)
    not_certified_df[columns_to_keep].to_csv('non_certified_students.csv', index=False)

    print("Lists exported successfully!")
else:
    print("Export skipped. You can run the script again to export later.")

# Show the visualization plot
plt.show()

