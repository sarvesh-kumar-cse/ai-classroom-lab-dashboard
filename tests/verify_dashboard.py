"""End-to-end verification of datasets, ML models, and live HTTP server endpoints."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request

from src.data_generator import load_datasets


def main() -> None:
    dfs = load_datasets()
    assert len(dfs["rooms"]) == 24, f"Expected 24 rooms, got {len(dfs['rooms'])}"
    assert len(dfs["timetable"]) == 21036, f"Expected 21036 timetable rows, got {len(dfs['timetable'])}"
    assert len(dfs["history"]) == 45072, f"Expected 45072 history rows, got {len(dfs['history'])}"
    assert len(dfs["calendar"]) == 365, f"Expected 365 calendar rows, got {len(dfs['calendar'])}"

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8051"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        base = "http://127.0.0.1:8051"
        for _ in range(30):
            try:
                with urllib.request.urlopen(f"{base}/api/overview", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("Server failed to become ready in time")

        with urllib.request.urlopen(f"{base}/") as r:
            html = r.read().decode("utf-8")
            assert r.status == 200 and "UniSpace AI" in html

        with urllib.request.urlopen(f"{base}/api/meta") as r:
            meta = json.loads(r.read().decode("utf-8"))
            assert len(meta["rooms"]) == 24
            assert len(meta["time_slots"]) == 6

        with urllib.request.urlopen(f"{base}/api/overview") as r:
            ov = json.loads(r.read().decode("utf-8"))
            k = ov["kpis"]
            assert k["total_classrooms"] == 14
            assert k["total_laboratories"] == 10
            assert k["total_available_resources"] == 24
            assert k["average_utilization_pct"] == 28.95
            assert k["underutilized_rooms"] == 13
            assert k["normal_rooms"] == 11
            assert k["overloaded_rooms"] == 0

        with urllib.request.urlopen(f"{base}/api/rooms?room_type=Laboratory&building=Block%20A") as r:
            rm = json.loads(r.read().decode("utf-8"))
            assert all(x["room_type"] == "Laboratory" and x["building"] == "Block A" for x in rm["rooms"])
            assert len(rm["heatmap"]["rows"]) == len(rm["rooms"])

        with urllib.request.urlopen(f"{base}/api/demand?department=CSE&room_type=Classroom") as r:
            dm = json.loads(r.read().decode("utf-8"))
            assert not dm["empty"]
            assert len(dm["day_wise"]) == 6
            assert len(dm["slot_wise"]) == 6

        for model_name in ("xgboost", "random_forest"):
            req = urllib.request.Request(
                f"{base}/api/predict",
                data=json.dumps({
                    "model_name": model_name,
                    "room_id": "C001",
                    "room_type": "Classroom",
                    "date": "2026-05-12",
                    "day_of_week": "Tuesday",
                    "time_slot": "09:00-10:00",
                    "course_id": "CSE101",
                    "department": "CSE",
                    "capacity": 40,
                    "semester": 2,
                    "scheduled_students": 36,
                    "is_scheduled": 1,
                    "is_holiday": 0,
                    "is_exam_period": 1,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as r:
                pred = json.loads(r.read().decode("utf-8"))
                assert pred["selected_model"] == model_name
                assert 0.0 <= pred["predicted_utilization_pct"] <= 130.0
                print(f"Predict ({model_name}): {pred['predicted_utilization_pct']}% -> {pred['predicted_resource_status']}")

        with urllib.request.urlopen(f"{base}/api/model-performance") as r:
            mp = json.loads(r.read().decode("utf-8"))
            assert mp["split_info"]["train_records"] == 36000
            assert mp["split_info"]["test_records"] == 9072
            assert mp["leakage_audit"]["passed"] is True
            print("All HTTP & ML assertions PASSED!")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
