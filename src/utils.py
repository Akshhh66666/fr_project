# utils.py
import csv
import os
from datetime import datetime

ATTENDANCE_FILE = "attendance.csv"

def ensure_csv():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

def mark_attendance(name):
    ensure_csv()
    today = datetime.now().strftime("%Y-%m-%d")

    with open(ATTENDANCE_FILE, "r") as f:
        rows = list(csv.reader(f))

    existing = {(r[0], r[1]) for r in rows[1:]} if len(rows) > 1 else set()
    if (name, today) not in existing:
        with open(ATTENDANCE_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, today, datetime.now().strftime("%H:%M:%S")])
        print(f"[INFO] Attendance marked for {name}")
