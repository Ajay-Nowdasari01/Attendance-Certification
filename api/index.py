from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import os
from io import StringIO

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "message": "ML Attendance & Certification System",
        "status": "running",
        "endpoints": {
            "GET /": "This message",
            "POST /api/process": "Process attendance data (upload CSVs)"
        }
    }), 200

@app.route('/api/process', methods=['POST'])
def process_attendance():
    """Process attendance from uploaded CSV files"""
    try:
        # Check if files are provided
        if 'file1' not in request.files or 'file2' not in request.files:
            return jsonify({"error": "Please provide two CSV files (file1, file2)"}), 400
        
        file1 = request.files['file1']
        file2 = request.files['file2']
        
        if file1.filename == '' or file2.filename == '':
            return jsonify({"error": "Files must have names"}), 400
        
        # Read CSV files
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # Preprocess
        def preprocess_sheet(df):
            ml_df = df[df['Topic'].str.contains('Machine Learning', case=False, na=False)].copy()
            ml_df['Email'] = ml_df['Email'].astype(str).str.strip().str.lower()
            ml_df = ml_df.dropna(subset=['Email', 'Name (original name)'])
            return ml_df.drop_duplicates(subset=['Email'])[['Name (original name)', 'Email']]
        
        day1_present = preprocess_sheet(df1)
        day2_present = preprocess_sheet(df2)
        
        # Create master student list
        master_students = pd.concat([day1_present, day2_present]).drop_duplicates(subset=['Email']).reset_index(drop=True)
        
        if len(master_students) > 980:
            master_students = master_students.sample(n=980, random_state=42).reset_index(drop=True)
        elif len(master_students) < 980:
            missing = 980 - len(master_students)
            extra_records = pd.DataFrame({
                'Name (original name)': [f'Student_{i}' for i in range(missing)],
                'Email': [f'student_{i}@example.com' for i in range(missing)]
            })
            master_students = pd.concat([master_students, extra_records]).reset_index(drop=True)
        
        # Simulate attendance
        np.random.seed(42)
        attendance_matrix = pd.DataFrame(index=master_students['Email'])
        attendance_matrix['Session_1'] = attendance_matrix.index.isin(day1_present['Email']).astype(int)
        attendance_matrix['Session_2'] = attendance_matrix.index.isin(day2_present['Email']).astype(int)
        
        for i in range(3, 41):
            attendance_matrix[f'Session_{i}'] = np.random.choice([1, 0], size=len(master_students), p=[0.82, 0.18])
        
        # Calculate certification
        master_students['Total_Attendance'] = attendance_matrix.sum(axis=1).values
        master_students['Attendance_Percentage'] = (master_students['Total_Attendance'] / 40) * 100
        master_students['Status'] = np.where(master_students['Total_Attendance'] >= 32, 'Certified', 'Not Certified')
        
        # Statistics
        certified_count = len(master_students[master_students['Status'] == 'Certified'])
        not_certified_count = len(master_students[master_students['Status'] == 'Not Certified'])
        
        return jsonify({
            "status": "success",
            "total_students": len(master_students),
            "certified": certified_count,
            "not_certified": not_certified_count,
            "certification_rate": f"{(certified_count / len(master_students)) * 100:.2f}%",
            "data": master_students.to_dict(orient='records')[:10]  # Return first 10 for preview
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check for Vercel"""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=False)
