"""Synthetic university classroom & laboratory dataset generator.

Reproduces the exact data-generation logic from
AI_Classroom_Lab_Utilization_System.ipynb so all four CSV files
(rooms.csv, timetable.csv, utilization_history.csv, academic_calendar.csv)
match the notebook's schema, random seed, distributions, and classification rules.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def classify_slot_status(is_scheduled: int, actual_occupancy: int, capacity: int, utilization: float) -> str:
    """Classify an individual time-slot record using exact notebook logic."""
    if not is_scheduled:
        return "Available"
    if actual_occupancy > capacity:
        return "Overloaded"
    if utilization < 30:
        return "Underutilized"
    return "Normal"


def classify_room_utilization(utilization: float) -> str:
    """Classify room-level average utilization using exact notebook thresholds."""
    if utilization < 30:
        return "Underutilized"
    if utilization <= 80:
        return "Normal"
    return "Overloaded"


def generate_datasets(data_dir: Path | str = DATA_DIR, force: bool = False) -> dict[str, pd.DataFrame]:
    """Generate and persist all four CSV datasets if missing or when force=True."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    required_files = [
        data_path / "rooms.csv",
        data_path / "timetable.csv",
        data_path / "utilization_history.csv",
        data_path / "academic_calendar.csv",
    ]
    if not force and all(p.exists() for p in required_files):
        return load_datasets(data_path)

    rng = np.random.default_rng(42)

    rooms: list[dict] = []
    for i in range(1, 15):
        rooms.append({
            "room_id": f"C{i:03d}",
            "room_type": "Classroom",
            "capacity": int(rng.choice([40, 50, 60, 70, 80, 100])),
            "building": str(rng.choice(["Block A", "Block B", "Block C"])),
            "equipment": str(rng.choice(["Projector", "Smart Board", "Projector, Smart Board"])),
        })

    for i in range(1, 11):
        rooms.append({
            "room_id": f"L{i:03d}",
            "room_type": "Laboratory",
            "capacity": int(rng.choice([25, 30, 35, 40, 45])),
            "building": str(rng.choice(["Block A", "Block B", "Block C"])),
            "equipment": str(rng.choice(["Computers", "Electronics Kits", "Computers, Projector"])),
        })

    rooms_df = pd.DataFrame(rooms)

    dates = pd.date_range("2025-07-01", "2026-06-30", freq="D")
    working_dates = dates[dates.dayofweek != 6]

    holiday_candidates = working_dates[working_dates.month.isin([8, 10, 11, 1, 3])]
    holiday_count = min(15, len(holiday_candidates))
    holidays = set(rng.choice(holiday_candidates, size=holiday_count, replace=False))

    calendar_df = pd.DataFrame({
        "date": dates,
        "day_of_week": dates.day_name(),
        "is_holiday": [int(d in holidays) for d in dates],
        "is_exam_period": [int(d.month in [11, 12, 4, 5]) for d in dates],
        "semester": [1 if d.month >= 7 else 2 for d in dates],
    })

    time_slots = [
        "09:00-10:00",
        "10:00-11:00",
        "11:00-12:00",
        "13:00-14:00",
        "14:00-15:00",
        "15:00-16:00",
    ]
    courses = {
        "Classroom": ["CSE101", "MAT201", "PHY101", "ENG102", "AI301"],
        "Laboratory": ["CSE_LAB", "AI_LAB", "DBMS_LAB", "NETWORK_LAB"],
    }
    departments = ["CSE", "ECE", "ME", "IT"]

    timetable: list[dict] = []
    history: list[dict] = []

    calendar_lookup = calendar_df.set_index("date").to_dict(orient="index")

    for date in working_dates:
        calendar_row = calendar_lookup[date]
        is_holiday = int(calendar_row["is_holiday"])
        is_exam_period = int(calendar_row["is_exam_period"])
        day_name = date.day_name()

        for _, room in rooms_df.iterrows():
            room_id = room["room_id"]
            room_type = room["room_type"]
            capacity = int(room["capacity"])

            for slot in time_slots:
                probability = 0.55 if room_type == "Classroom" else 0.40
                is_scheduled = 0 if is_holiday else int(rng.random() < probability)

                course_id = None
                department = None
                semester = None
                scheduled_students = 0
                actual_occupancy = 0

                if is_scheduled:
                    course_id = str(rng.choice(courses[room_type]))
                    department = str(rng.choice(departments))
                    semester = int(rng.integers(1, 9))

                    max_students = max(16, int(capacity * 1.20))
                    scheduled_students = int(rng.integers(15, max_students + 1))
                    attendance_rate = float(rng.uniform(0.65, 1.0))
                    actual_occupancy = int(round(scheduled_students * attendance_rate))

                utilization = round(actual_occupancy / capacity * 100, 2)
                status = classify_slot_status(is_scheduled, actual_occupancy, capacity, utilization)

                record = {
                    "date": date,
                    "day_of_week": day_name,
                    "room_id": room_id,
                    "room_type": room_type,
                    "capacity": capacity,
                    "time_slot": slot,
                    "course_id": course_id,
                    "department": department,
                    "semester": semester,
                    "scheduled_students": scheduled_students,
                    "actual_occupancy": actual_occupancy,
                    "utilization_pct": utilization,
                    "is_scheduled": is_scheduled,
                    "is_holiday": is_holiday,
                    "is_exam_period": is_exam_period,
                    "status": status,
                }
                history.append(record)
                if is_scheduled:
                    timetable.append(record.copy())

    timetable_df = pd.DataFrame(timetable)
    history_df = pd.DataFrame(history)

    rooms_df.to_csv(data_path / "rooms.csv", index=False)
    timetable_df.to_csv(data_path / "timetable.csv", index=False)
    history_df.to_csv(data_path / "utilization_history.csv", index=False)
    calendar_df.to_csv(data_path / "academic_calendar.csv", index=False)

    return {
        "rooms": rooms_df,
        "timetable": timetable_df,
        "history": history_df,
        "calendar": calendar_df,
    }


def load_datasets(data_dir: Path | str = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the four CSV files from disk with proper date parsing."""
    data_path = Path(data_dir)
    rooms_df = pd.read_csv(data_path / "rooms.csv")
    timetable_df = pd.read_csv(data_path / "timetable.csv", parse_dates=["date"])
    history_df = pd.read_csv(data_path / "utilization_history.csv", parse_dates=["date"])
    calendar_df = pd.read_csv(data_path / "academic_calendar.csv", parse_dates=["date"])
    return {
        "rooms": rooms_df,
        "timetable": timetable_df,
        "history": history_df,
        "calendar": calendar_df,
    }


if __name__ == "__main__":
    dfs = generate_datasets(force=True)
    print("DATASET CREATION SUCCESSFUL!")
    print("Rooms:", len(dfs["rooms"]))
    print("Timetable records:", len(dfs["timetable"]))
    print("Historical records:", len(dfs["history"]))
    print("Academic calendar records:", len(dfs["calendar"]))
    print("Average utilization (%):", round(float(dfs["history"]["utilization_pct"].mean()), 2))
