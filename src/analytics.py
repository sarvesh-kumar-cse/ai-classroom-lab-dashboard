"""Analytics aggregations computed strictly from the 4 CSV datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_generator import classify_room_utilization


DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
TIME_ORDER = [
    "09:00-10:00",
    "10:00-11:00",
    "11:00-12:00",
    "13:00-14:00",
    "14:00-15:00",
    "15:00-16:00",
]


def build_room_summary(rooms_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce notebook room_summary enriched with building & equipment from rooms.csv."""
    summary = (
        history_df.groupby(["room_id", "room_type", "capacity"], as_index=False)
        .agg(
            average_utilization=("utilization_pct", "mean"),
            scheduled_utilization=("utilization_pct", lambda s: s[history_df.loc[s.index, "is_scheduled"] == 1].mean()),
            total_slots=("utilization_pct", "count"),
            scheduled_slots=("is_scheduled", "sum"),
            overloaded_slots=("status", lambda x: int((x == "Overloaded").sum())),
            underutilized_slots=("status", lambda x: int((x == "Underutilized").sum())),
            normal_slots=("status", lambda x: int((x == "Normal").sum())),
            available_slots=("status", lambda x: int((x == "Available").sum())),
        )
    )
    summary["resource_status"] = summary["average_utilization"].apply(classify_room_utilization)
    summary["average_utilization"] = summary["average_utilization"].round(2)
    summary["scheduled_utilization"] = summary["scheduled_utilization"].fillna(0).round(2)

    enriched = summary.merge(
        rooms_df[["room_id", "building", "equipment"]],
        on="room_id",
        how="left",
    )
    return enriched.sort_values("room_id").reset_index(drop=True)


def get_overview_data(dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Compute Page 1 Overview KPIs and charts directly from the datasets."""
    rooms_df = dfs["rooms"]
    history_df = dfs["history"]
    timetable_df = dfs["timetable"]
    calendar_df = dfs["calendar"]

    room_summary = build_room_summary(rooms_df, history_df)

    total_classrooms = int((rooms_df["room_type"] == "Classroom").sum())
    total_labs = int((rooms_df["room_type"] == "Laboratory").sum())
    total_rooms = int(len(rooms_df))
    avg_utilization = round(float(history_df["utilization_pct"].mean()), 2)
    scheduled_avg_utilization = round(float(timetable_df["utilization_pct"].mean()), 2)

    room_status_counts = (
        room_summary["resource_status"]
        .value_counts()
        .reindex(["Underutilized", "Normal", "Overloaded"], fill_value=0)
    )
    slot_status_counts = (
        history_df["status"]
        .value_counts()
        .reindex(["Available", "Normal", "Underutilized", "Overloaded"], fill_value=0)
    )

    # Monthly overall utilization trend
    hist_with_month = history_df.copy()
    hist_with_month["month_key"] = hist_with_month["date"].dt.to_period("M").astype(str)
    hist_with_month["month_label"] = hist_with_month["date"].dt.strftime("%b %Y")
    monthly = (
        hist_with_month.groupby(["month_key", "month_label"], as_index=False)
        .agg(
            avg_utilization=("utilization_pct", "mean"),
            classroom_util=("utilization_pct", lambda s: s[hist_with_month.loc[s.index, "room_type"] == "Classroom"].mean()),
            lab_util=("utilization_pct", lambda s: s[hist_with_month.loc[s.index, "room_type"] == "Laboratory"].mean()),
        )
        .sort_values("month_key")
    )

    # Classroom vs Laboratory comparison
    by_room_type = (
        history_df.groupby("room_type", as_index=False)
        .agg(
            avg_utilization=("utilization_pct", "mean"),
            scheduled_avg_utilization=("utilization_pct", lambda s: s[history_df.loc[s.index, "is_scheduled"] == 1].mean()),
            total_slots=("utilization_pct", "count"),
            scheduled_slots=("is_scheduled", "sum"),
        )
    )

    # Weekly utilization (Monday to Saturday)
    weekly = (
        history_df.groupby("day_of_week")["utilization_pct"]
        .mean()
        .reindex(DAY_ORDER)
        .round(2)
    )

    # Time-slot utilization
    timeslot = (
        history_df.groupby("time_slot")["utilization_pct"]
        .mean()
        .reindex(TIME_ORDER)
        .round(2)
    )

    return {
        "kpis": {
            "total_classrooms": total_classrooms,
            "total_laboratories": total_labs,
            "total_available_resources": total_rooms,
            "total_capacity_seats": int(rooms_df["capacity"].sum()),
            "average_utilization_pct": avg_utilization,
            "scheduled_average_utilization_pct": scheduled_avg_utilization,
            "underutilized_rooms": int(room_status_counts["Underutilized"]),
            "normal_rooms": int(room_status_counts["Normal"]),
            "overloaded_rooms": int(room_status_counts["Overloaded"]),
            "total_historical_records": int(len(history_df)),
            "total_timetable_records": int(len(timetable_df)),
            "total_calendar_days": int(len(calendar_df)),
            "available_slot_records": int(slot_status_counts["Available"]),
            "overloaded_slot_records": int(slot_status_counts["Overloaded"]),
        },
        "overall_trend": [
            {
                "month": row["month_label"],
                "avg_utilization": round(float(row["avg_utilization"]), 2),
                "classroom_util": round(float(row["classroom_util"]), 2),
                "lab_util": round(float(row["lab_util"]), 2),
            }
            for _, row in monthly.iterrows()
        ],
        "room_type_comparison": [
            {
                "room_type": row["room_type"],
                "avg_utilization": round(float(row["avg_utilization"]), 2),
                "scheduled_avg_utilization": round(float(row["scheduled_avg_utilization"]), 2),
                "scheduled_rate_pct": round(float(row["scheduled_slots"] / row["total_slots"] * 100), 2),
            }
            for _, row in by_room_type.iterrows()
        ],
        "room_status_distribution": [
            {"status": k, "count": int(v)}
            for k, v in room_status_counts.items()
        ],
        "slot_status_distribution": [
            {"status": k, "count": int(v)}
            for k, v in slot_status_counts.items()
        ],
        "weekly_utilization": [
            {"day": day, "avg_utilization": float(val)}
            for day, val in weekly.items()
        ],
        "timeslot_utilization": [
            {"time_slot": slot, "avg_utilization": float(val)}
            for slot, val in timeslot.items()
        ],
    }


def get_room_analytics(
    dfs: dict[str, pd.DataFrame],
    room_type: str = "All",
    building: str = "All",
    resource_status: str = "All",
    min_util: float = 0.0,
    max_util: float = 100.0,
) -> dict[str, Any]:
    """Compute Page 2 Room Analytics table & Day x Room heatmap with filters."""
    rooms_df = dfs["rooms"]
    history_df = dfs["history"]

    room_summary = build_room_summary(rooms_df, history_df)

    filtered = room_summary.copy()
    if room_type != "All":
        filtered = filtered[filtered["room_type"] == room_type]
    if building != "All":
        filtered = filtered[filtered["building"] == building]
    if resource_status != "All":
        filtered = filtered[filtered["resource_status"] == resource_status]
    filtered = filtered[
        (filtered["average_utilization"] >= min_util)
        & (filtered["average_utilization"] <= max_util)
    ]

    filtered_ids = filtered["room_id"].tolist()
    heat_source = history_df[history_df["room_id"].isin(filtered_ids)]

    if len(filtered_ids) > 0:
        pivot = (
            heat_source.pivot_table(
                index="room_id",
                columns="day_of_week",
                values="utilization_pct",
                aggfunc="mean",
            )
            .reindex(index=filtered_ids, columns=DAY_ORDER)
            .round(1)
        )
        heatmap_rows = []
        for room_id, row in pivot.iterrows():
            room_meta = filtered[filtered["room_id"] == room_id].iloc[0]
            heatmap_rows.append({
                "room_id": room_id,
                "room_type": room_meta["room_type"],
                "building": room_meta["building"],
                "values": {d: (None if pd.isna(row[d]) else float(row[d])) for d in DAY_ORDER},
            })
    else:
        heatmap_rows = []

    return {
        "rooms": filtered.to_dict(orient="records"),
        "heatmap": {
            "days": DAY_ORDER,
            "rows": heatmap_rows,
        },
        "summary": {
            "matching_rooms": int(len(filtered)),
            "total_rooms": int(len(room_summary)),
            "avg_utilization": round(float(filtered["average_utilization"].mean()), 2) if len(filtered) else 0.0,
            "underutilized_count": int((filtered["resource_status"] == "Underutilized").sum()),
            "normal_count": int((filtered["resource_status"] == "Normal").sum()),
            "overloaded_count": int((filtered["resource_status"] == "Overloaded").sum()),
        },
        "filter_options": {
            "room_types": ["All"] + sorted(rooms_df["room_type"].unique().tolist()),
            "buildings": ["All"] + sorted(rooms_df["building"].unique().tolist()),
            "statuses": ["All", "Underutilized", "Normal", "Overloaded"],
        },
    }


def get_demand_analysis(
    dfs: dict[str, pd.DataFrame],
    start_date: str | None = None,
    end_date: str | None = None,
    department: str = "All",
    room_type: str = "All",
    time_slot: str = "All",
) -> dict[str, Any]:
    """Compute Page 3 Demand & Utilization Analysis with interactive filters."""
    history_df = dfs["history"].copy()

    if start_date:
        history_df = history_df[history_df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        history_df = history_df[history_df["date"] <= pd.to_datetime(end_date)]
    if room_type != "All":
        history_df = history_df[history_df["room_type"] == room_type]
    if time_slot != "All":
        history_df = history_df[history_df["time_slot"] == time_slot]
    if department != "All":
        history_df = history_df[history_df["department"] == department]

    if len(history_df) == 0:
        return {
            "empty": True,
            "record_count": 0,
            "day_wise": [],
            "slot_wise": [],
            "monthly_trends": [],
            "room_type_demand": [],
            "exam_period": [],
            "holiday_comparison": [],
            "department_demand": [],
        }

    # Day-wise utilization
    day_wise = (
        history_df.groupby("day_of_week", as_index=False)
        .agg(
            avg_utilization=("utilization_pct", "mean"),
            avg_occupancy=("actual_occupancy", "mean"),
            scheduled_sessions=("is_scheduled", "sum"),
        )
    )
    day_wise["day_of_week"] = pd.Categorical(day_wise["day_of_week"], categories=DAY_ORDER, ordered=True)
    day_wise = day_wise.sort_values("day_of_week")

    # Time-slot-wise utilization
    slot_wise = (
        history_df.groupby("time_slot", as_index=False)
        .agg(
            avg_utilization=("utilization_pct", "mean"),
            avg_occupancy=("actual_occupancy", "mean"),
            scheduled_sessions=("is_scheduled", "sum"),
        )
    )
    slot_wise["time_slot"] = pd.Categorical(slot_wise["time_slot"], categories=TIME_ORDER, ordered=True)
    slot_wise = slot_wise.sort_values("time_slot")

    # Monthly utilization trends (overall + Classroom vs Laboratory)
    history_df["month_key"] = history_df["date"].dt.to_period("M").astype(str)
    history_df["month_label"] = history_df["date"].dt.strftime("%b %Y")
    monthly = (
        history_df.groupby(["month_key", "month_label"], as_index=False)
        .agg(
            avg_utilization=("utilization_pct", "mean"),
            scheduled_sessions=("is_scheduled", "sum"),
            total_students=("actual_occupancy", "sum"),
        )
        .sort_values("month_key")
    )

    # Classroom vs Laboratory demand by month
    monthly_rt = (
        history_df.groupby(["month_key", "month_label", "room_type"], as_index=False)["utilization_pct"]
        .mean()
        .pivot(index=["month_key", "month_label"], columns="room_type", values="utilization_pct")
        .reset_index()
        .sort_values("month_key")
    )

    # Classroom & Laboratory aggregate demand
    rt_demand = (
        history_df.groupby("room_type", as_index=False)
        .agg(
            avg_utilization=("utilization_pct", "mean"),
            total_scheduled=("is_scheduled", "sum"),
            total_records=("is_scheduled", "count"),
            avg_scheduled_students=("scheduled_students", "mean"),
            avg_actual_occupancy=("actual_occupancy", "mean"),
        )
    )

    # Exam-period utilization
    exam_df = (
        history_df.groupby("is_exam_period", as_index=False)
        .agg(
            avg_utilization=("utilization_pct", "mean"),
            scheduled_sessions=("is_scheduled", "sum"),
            records=("is_scheduled", "count"),
        )
    )

    # Holiday vs regular academic days
    holiday_df = (
        history_df.groupby("is_holiday", as_index=False)
        .agg(
            avg_utilization=("utilization_pct", "mean"),
            scheduled_sessions=("is_scheduled", "sum"),
            records=("is_scheduled", "count"),
        )
    )

    # Department breakdown (scheduled records)
    sched_df = history_df[history_df["is_scheduled"] == 1]
    dept_df = (
        sched_df.groupby("department", as_index=False)
        .agg(
            avg_utilization=("utilization_pct", "mean"),
            sessions=("is_scheduled", "count"),
            avg_students=("actual_occupancy", "mean"),
        )
        .sort_values("sessions", ascending=False)
    )

    return {
        "empty": False,
        "record_count": int(len(history_df)),
        "scheduled_count": int(history_df["is_scheduled"].sum()),
        "avg_utilization_pct": round(float(history_df["utilization_pct"].mean()), 2),
        "department_filter_note": (
            "Filtered to scheduled sessions for the selected department (unscheduled slots have no department)."
            if department != "All"
            else None
        ),
        "day_wise": [
            {
                "day": str(r["day_of_week"]),
                "avg_utilization": round(float(r["avg_utilization"]), 2),
                "avg_occupancy": round(float(r["avg_occupancy"]), 2),
                "scheduled_sessions": int(r["scheduled_sessions"]),
            }
            for _, r in day_wise.iterrows()
        ],
        "slot_wise": [
            {
                "time_slot": str(r["time_slot"]),
                "avg_utilization": round(float(r["avg_utilization"]), 2),
                "avg_occupancy": round(float(r["avg_occupancy"]), 2),
                "scheduled_sessions": int(r["scheduled_sessions"]),
            }
            for _, r in slot_wise.iterrows()
        ],
        "monthly_trends": [
            {
                "month": r["month_label"],
                "avg_utilization": round(float(r["avg_utilization"]), 2),
                "scheduled_sessions": int(r["scheduled_sessions"]),
                "total_students": int(r["total_students"]),
            }
            for _, r in monthly.iterrows()
        ],
        "monthly_room_type_demand": [
            {
                "month": r["month_label"],
                "classroom": round(float(r["Classroom"]), 2) if "Classroom" in r and not pd.isna(r["Classroom"]) else None,
                "laboratory": round(float(r["Laboratory"]), 2) if "Laboratory" in r and not pd.isna(r["Laboratory"]) else None,
            }
            for _, r in monthly_rt.iterrows()
        ],
        "room_type_demand": [
            {
                "room_type": r["room_type"],
                "avg_utilization": round(float(r["avg_utilization"]), 2),
                "total_scheduled": int(r["total_scheduled"]),
                "scheduling_rate_pct": round(float(r["total_scheduled"] / r["total_records"] * 100), 2),
                "avg_scheduled_students": round(float(r["avg_scheduled_students"]), 2),
                "avg_actual_occupancy": round(float(r["avg_actual_occupancy"]), 2),
            }
            for _, r in rt_demand.iterrows()
        ],
        "exam_period": [
            {
                "label": "Exam Period" if int(r["is_exam_period"]) == 1 else "Regular Period",
                "avg_utilization": round(float(r["avg_utilization"]), 2),
                "scheduled_sessions": int(r["scheduled_sessions"]),
                "records": int(r["records"]),
            }
            for _, r in exam_df.iterrows()
        ],
        "holiday_comparison": [
            {
                "label": "Holiday" if int(r["is_holiday"]) == 1 else "Regular Academic Day",
                "avg_utilization": round(float(r["avg_utilization"]), 2),
                "scheduled_sessions": int(r["scheduled_sessions"]),
                "records": int(r["records"]),
            }
            for _, r in holiday_df.iterrows()
        ],
        "department_demand": [
            {
                "department": r["department"],
                "avg_utilization": round(float(r["avg_utilization"]), 2),
                "sessions": int(r["sessions"]),
                "avg_students": round(float(r["avg_students"]), 2),
            }
            for _, r in dept_df.iterrows()
        ],
    }
