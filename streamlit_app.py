"""Streamlit Community Cloud entrypoint for UniSpace AI Dashboard."""

from __future__ import annotations

import datetime as dt
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from src.analytics import (
    DAY_ORDER,
    TIME_ORDER,
    get_demand_analysis,
    get_overview_data,
    get_room_analytics,
)
from src.data_generator import generate_datasets
from src.ml_pipeline import ModelRegistry


COURSES_BY_TYPE: dict[str, list[str]] = {
    "Classroom": ["CSE101", "MAT201", "PHY101", "ENG102", "AI301"],
    "Laboratory": ["CSE_LAB", "AI_LAB", "DBMS_LAB", "NETWORK_LAB"],
}
DEPARTMENTS: list[str] = ["CSE", "ECE", "ME", "IT"]

st.set_page_config(
    page_title="UniSpace AI — Classroom & Lab Utilization",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Initializing synthetic datasets and ML models...")
def load_app_state() -> tuple[dict[str, pd.DataFrame], ModelRegistry]:
    """Load/generate the 4 CSV datasets and load/train RF & XGBoost models."""
    dfs = generate_datasets()
    registry = ModelRegistry()
    registry.ensure_loaded(history_df=dfs["history"])
    return dfs, registry


def _status_badge_color(status: str) -> str:
    return {
        "Normal": "#10b981",
        "Underutilized": "#f59e0b",
        "Overloaded": "#ef4444",
        "Available": "#0891b2",
    }.get(status, "#64748b")


def render_overview(dfs: dict[str, pd.DataFrame]) -> None:
    """Render Page 1: Overview Dashboard."""
    data = get_overview_data(dfs)
    k = data["kpis"]

    st.subheader("Overview Dashboard")
    st.caption("University-wide classroom & laboratory utilization telemetry")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Classrooms", k["total_classrooms"], "C001–C014 (Blocks A–C)")
    c2.metric("Total Laboratories", k["total_laboratories"], "L001–L010 specialized labs")
    c3.metric("Total Resources", k["total_available_resources"], f"{k['total_capacity_seats']:,} total seats")
    c4.metric(
        "Average Utilization",
        f"{k['average_utilization_pct']:.2f}%",
        f"Scheduled only: {k['scheduled_average_utilization_pct']:.2f}%",
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Underutilized Rooms (<30%)", k["underutilized_rooms"])
    c6.metric("Normal Rooms (30–80%)", k["normal_rooms"])
    c7.metric("Overloaded Rooms (>80%)", k["overloaded_rooms"])
    c8.metric("Historical Records", f"{k['total_historical_records']:,}", f"{k['total_calendar_days']} calendar days")

    st.divider()

    col_left, col_right = st.columns((1.4, 1.0))
    with col_left:
        st.markdown("#### Monthly Utilization Trend")
        trend_df = pd.DataFrame(data["overall_trend"])
        trend_df["order"] = range(len(trend_df))
        melted = trend_df.melt(
            id_vars=["month", "order"],
            value_vars=["avg_utilization", "classroom_util", "lab_util"],
            var_name="Series",
            value_name="Utilization (%)",
        )
        melted["Series"] = melted["Series"].replace({
            "avg_utilization": "Overall",
            "classroom_util": "Classroom",
            "lab_util": "Laboratory",
        })
        chart = (
            alt.Chart(melted)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("month:N", sort=alt.SortField("order"), title="Month"),
                y=alt.Y("Utilization (%):Q", title="Utilization (%)"),
                color=alt.Color("Series:N", scale=alt.Scale(range=["#0f172a", "#2563eb", "#7c3aed"])),
                tooltip=["month", "Series", "Utilization (%)"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, width="stretch")

    with col_right:
        st.markdown("#### Classroom vs Laboratory")
        rt_df = pd.DataFrame(data["room_type_comparison"])
        rt_melt = rt_df.melt(
            id_vars=["room_type"],
            value_vars=["avg_utilization", "scheduled_avg_utilization", "scheduled_rate_pct"],
            var_name="Metric",
            value_name="Percentage (%)",
        )
        rt_melt["Metric"] = rt_melt["Metric"].replace({
            "avg_utilization": "Overall Avg Util %",
            "scheduled_avg_utilization": "Scheduled Avg Util %",
            "scheduled_rate_pct": "Scheduling Rate %",
        })
        rt_chart = (
            alt.Chart(rt_melt)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("room_type:N", title="Room Type"),
                y=alt.Y("Percentage (%):Q"),
                color="Metric:N",
                xOffset="Metric:N",
                tooltip=["room_type", "Metric", "Percentage (%)"],
            )
            .properties(height=320)
        )
        st.altair_chart(rt_chart, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Weekly Utilization (Mon–Sat)")
        wk_df = pd.DataFrame(data["weekly_utilization"])
        wk_chart = (
            alt.Chart(wk_df)
            .mark_bar(color="#2563eb", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("day:N", sort=DAY_ORDER, title="Day of Week"),
                y=alt.Y("avg_utilization:Q", title="Avg Utilization (%)"),
                tooltip=["day", "avg_utilization"],
            )
            .properties(height=270)
        )
        st.altair_chart(wk_chart, width="stretch")

    with col_b:
        st.markdown("#### Time-Slot Utilization")
        ts_df = pd.DataFrame(data["timeslot_utilization"])
        ts_chart = (
            alt.Chart(ts_df)
            .mark_bar(color="#7c3aed", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("time_slot:N", sort=TIME_ORDER, title="Time Slot"),
                y=alt.Y("avg_utilization:Q", title="Avg Utilization (%)"),
                tooltip=["time_slot", "avg_utilization"],
            )
            .properties(height=270)
        )
        st.altair_chart(ts_chart, width="stretch")

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### Room Status Distribution")
        rs_df = pd.DataFrame(data["room_status_distribution"])
        rs_chart = (
            alt.Chart(rs_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("status:N", title="Room Status"),
                y=alt.Y("count:Q", title="Room Count"),
                color=alt.Color(
                    "status:N",
                    scale=alt.Scale(
                        domain=["Underutilized", "Normal", "Overloaded"],
                        range=["#f59e0b", "#10b981", "#ef4444"],
                    ),
                    legend=None,
                ),
                tooltip=["status", "count"],
            )
            .properties(height=240)
        )
        st.altair_chart(rs_chart, width="stretch")

    with col_d:
        st.markdown("#### Time-Slot Status Distribution")
        ss_df = pd.DataFrame(data["slot_status_distribution"])
        ss_chart = (
            alt.Chart(ss_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("status:N", title="Slot Status"),
                y=alt.Y("count:Q", title="Slot Records"),
                color=alt.Color(
                    "status:N",
                    scale=alt.Scale(
                        domain=["Available", "Normal", "Underutilized", "Overloaded"],
                        range=["#0891b2", "#10b981", "#f59e0b", "#ef4444"],
                    ),
                    legend=None,
                ),
                tooltip=["status", "count"],
            )
            .properties(height=240)
        )
        st.altair_chart(ss_chart, width="stretch")


def render_rooms(dfs: dict[str, pd.DataFrame]) -> None:
    """Render Page 2: Room Analytics."""
    st.subheader("Room Analytics")
    st.caption("Room-level occupancy inventory, filters, and Day × Room heatmap")

    base = get_room_analytics(dfs)
    opts = base["filter_options"]

    f1, f2, f3, f4 = st.columns(4)
    room_type = f1.selectbox("Room Type", opts["room_types"])
    building = f2.selectbox("Building", opts["buildings"])
    status = f3.selectbox("Resource Status", opts["statuses"])
    util_range = f4.slider("Average Utilization (%)", 0.0, 100.0, (0.0, 100.0), step=1.0)

    data = get_room_analytics(
        dfs,
        room_type=room_type,
        building=building,
        resource_status=status,
        min_util=util_range[0],
        max_util=util_range[1],
    )
    summary = data["summary"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Matching Rooms", f"{summary['matching_rooms']} / {summary['total_rooms']}")
    m2.metric("Mean Utilization", f"{summary['avg_utilization']:.2f}%")
    m3.metric("Underutilized", summary["underutilized_count"])
    m4.metric("Normal", summary["normal_count"])
    m5.metric("Overloaded", summary["overloaded_count"])

    if not data["rooms"]:
        st.info("No rooms match the selected filters.")
        return

    st.markdown("#### Room Inventory Summary")
    rooms_df = pd.DataFrame(data["rooms"])
    display_cols = [
        "room_id",
        "room_type",
        "building",
        "capacity",
        "equipment",
        "average_utilization",
        "scheduled_utilization",
        "resource_status",
        "scheduled_slots",
        "overloaded_slots",
        "underutilized_slots",
    ]
    st.dataframe(rooms_df[display_cols], width="stretch", hide_index=True)

    st.markdown("#### Day × Room Utilization Heatmap (%)")
    heat_records = []
    for row in data["heatmap"]["rows"]:
        for day in data["heatmap"]["days"]:
            val = row["values"].get(day)
            if val is not None:
                heat_records.append({
                    "room_id": row["room_id"],
                    "day": day,
                    "utilization": val,
                })
    heat_df = pd.DataFrame(heat_records)
    base_heat = alt.Chart(heat_df).encode(
        x=alt.X("day:N", sort=DAY_ORDER, title="Day of Week"),
        y=alt.Y("room_id:N", title="Room ID"),
    )
    rects = base_heat.mark_rect().encode(
        color=alt.Color("utilization:Q", scale=alt.Scale(scheme="blues"), title="Util %"),
        tooltip=["room_id", "day", "utilization"],
    )
    text = base_heat.mark_text(fontSize=11).encode(
        text=alt.Text("utilization:Q", format=".1f"),
        color=alt.condition(
            alt.datum.utilization > 40,
            alt.value("white"),
            alt.value("#0f172a"),
        ),
    )
    chart_height = max(240, len(data["heatmap"]["rows"]) * 26)
    st.altair_chart((rects + text).properties(height=chart_height), width="stretch")


def render_demand(dfs: dict[str, pd.DataFrame]) -> None:
    """Render Page 3: Demand & Utilization Analysis."""
    st.subheader("Demand & Utilization Analysis")
    st.caption("Temporal, departmental, exam-period, and holiday demand patterns")

    min_date = dfs["history"]["date"].min().date()
    max_date = dfs["history"]["date"].max().date()

    c1, c2, c3, c4 = st.columns(4)
    date_vals = c1.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    department = c2.selectbox("Department", ["All"] + DEPARTMENTS)
    room_type = c3.selectbox("Room Type", ["All", "Classroom", "Laboratory"])
    time_slot = c4.selectbox("Time Slot", ["All"] + TIME_ORDER)

    if isinstance(date_vals, tuple) and len(date_vals) == 2:
        start_d, end_d = date_vals
    else:
        start_d = end_d = date_vals if isinstance(date_vals, dt.date) else min_date

    data = get_demand_analysis(
        dfs,
        start_date=start_d.isoformat(),
        end_date=end_d.isoformat(),
        department=department,
        room_type=room_type,
        time_slot=time_slot,
    )

    if data["empty"]:
        st.warning("No utilization records found for the selected filter combination.")
        return

    if data.get("department_filter_note"):
        st.info(data["department_filter_note"])

    k1, k2, k3 = st.columns(3)
    k1.metric("Matching Slot Records", f"{data['record_count']:,}")
    k2.metric("Scheduled Sessions", f"{data['scheduled_count']:,}")
    k3.metric("Mean Utilization", f"{data['avg_utilization_pct']:.2f}%")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Day-wise Utilization")
        dw_df = pd.DataFrame(data["day_wise"])
        st.altair_chart(
            alt.Chart(dw_df)
            .mark_bar(color="#2563eb", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("day:N", sort=DAY_ORDER, title="Day"),
                y=alt.Y("avg_utilization:Q", title="Avg Utilization (%)"),
                tooltip=["day", "avg_utilization", "avg_occupancy", "scheduled_sessions"],
            )
            .properties(height=270),
            width="stretch",
        )

    with col_b:
        st.markdown("#### Time-Slot Demand")
        sw_df = pd.DataFrame(data["slot_wise"])
        st.altair_chart(
            alt.Chart(sw_df)
            .mark_bar(color="#0891b2", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("time_slot:N", sort=TIME_ORDER, title="Time Slot"),
                y=alt.Y("avg_utilization:Q", title="Avg Utilization (%)"),
                tooltip=["time_slot", "avg_utilization", "avg_occupancy", "scheduled_sessions"],
            )
            .properties(height=270),
            width="stretch",
        )

    st.markdown("#### Monthly Demand Trends (Classroom vs Laboratory)")
    mrt_df = pd.DataFrame(data["monthly_room_type_demand"])
    mrt_df["order"] = range(len(mrt_df))
    value_cols = [c for c in ["classroom", "laboratory"] if c in mrt_df.columns and mrt_df[c].notna().any()]
    if value_cols:
        mrt_melt = mrt_df.melt(
            id_vars=["month", "order"],
            value_vars=value_cols,
            var_name="Room Type",
            value_name="Avg Utilization (%)",
        ).dropna()
        mrt_melt["Room Type"] = mrt_melt["Room Type"].str.title()
        st.altair_chart(
            alt.Chart(mrt_melt)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("month:N", sort=alt.SortField("order"), title="Month"),
                y=alt.Y("Avg Utilization (%):Q"),
                color=alt.Color("Room Type:N", scale=alt.Scale(range=["#2563eb", "#7c3aed"])),
                tooltip=["month", "Room Type", "Avg Utilization (%)"],
            )
            .properties(height=280),
            width="stretch",
        )

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### Exam Period vs Regular Period")
        ex_df = pd.DataFrame(data["exam_period"])
        st.dataframe(ex_df, width="stretch", hide_index=True)

    with col_d:
        st.markdown("#### Holiday vs Regular Academic Day")
        hol_df = pd.DataFrame(data["holiday_comparison"])
        st.dataframe(hol_df, width="stretch", hide_index=True)

    if data["department_demand"]:
        st.markdown("#### Department Demand Breakdown (Scheduled Sessions)")
        st.dataframe(pd.DataFrame(data["department_demand"]), width="stretch", hide_index=True)


def render_predict(dfs: dict[str, pd.DataFrame], registry: ModelRegistry) -> None:
    """Render Page 4: AI Prediction Centre."""
    st.subheader("AI Prediction Centre")
    st.caption("Live Random Forest & XGBoost utilization inference using the 15-feature pipeline")

    rooms_df = dfs["rooms"].sort_values("room_id")
    calendar_df = dfs["calendar"]
    cal_map = {
        row["date"].strftime("%Y-%m-%d"): {
            "day_of_week": row["day_of_week"],
            "is_holiday": int(row["is_holiday"]),
            "is_exam_period": int(row["is_exam_period"]),
            "semester": int(row["semester"]),
        }
        for _, row in calendar_df.iterrows()
    }

    room_records = rooms_df.set_index("room_id").to_dict(orient="index")
    room_ids = list(room_records.keys())

    col_left, col_right = st.columns((1.0, 1.1))
    with col_left:
        selected_room = st.selectbox("Room ID", room_ids, index=0)
        room_meta = room_records[selected_room]
        room_type = room_meta["room_type"]
        default_cap = int(room_meta["capacity"])

        date_val = st.date_input(
            "Target Date",
            value=dt.date(2026, 5, 12),
            min_value=dt.date(2025, 7, 1),
            max_value=dt.date(2027, 12, 31),
        )
        cal_info = cal_map.get(date_val.strftime("%Y-%m-%d"))
        default_day = cal_info["day_of_week"] if cal_info else date_val.strftime("%A")
        if default_day not in DAY_ORDER:
            default_day = "Monday"

        with st.form("prediction_form"):
            model_label = st.radio(
                "Primary Model",
                ["XGBoost Regressor (Recommended)", "Random Forest Regressor"],
                horizontal=True,
            )
            model_name = "xgboost" if model_label.startswith("XGBoost") else "random_forest"

            p1, p2 = st.columns(2)
            day_of_week = p1.selectbox("Day of Week", DAY_ORDER, index=DAY_ORDER.index(default_day))
            time_slot = p2.selectbox("Time Slot", TIME_ORDER, index=0)

            p3, p4 = st.columns(2)
            capacity = p3.number_input("Room Capacity", min_value=1, max_value=500, value=default_cap)
            semester = p4.selectbox("Semester", [1, 2], index=(cal_info["semester"] - 1) if cal_info else 1)

            is_scheduled = st.checkbox("Slot is Scheduled", value=True)
            course_choices = COURSES_BY_TYPE.get(room_type, COURSES_BY_TYPE["Classroom"])
            p5, p6 = st.columns(2)
            course_id = p5.selectbox("Course ID", course_choices, disabled=not is_scheduled)
            department = p6.selectbox("Department", DEPARTMENTS, disabled=not is_scheduled)

            scheduled_students = st.slider(
                "Scheduled Students",
                min_value=0,
                max_value=200,
                value=min(36, default_cap) if is_scheduled else 0,
                disabled=not is_scheduled,
            )

            p7, p8 = st.columns(2)
            is_holiday = p7.checkbox(
                "Academic Holiday",
                value=bool(cal_info["is_holiday"]) if cal_info else False,
            )
            is_exam_period = p8.checkbox(
                "Exam Period",
                value=bool(cal_info["is_exam_period"]) if cal_info else False,
            )

            submitted = st.form_submit_button("Predict Utilization", type="primary", width="stretch")

    payload: dict[str, Any] = {
        "model_name": model_name,
        "room_id": selected_room,
        "room_type": room_type,
        "date": date_val.strftime("%Y-%m-%d"),
        "day_of_week": day_of_week,
        "time_slot": time_slot,
        "course_id": course_id if is_scheduled else "Unscheduled",
        "department": department if is_scheduled else "Unscheduled",
        "capacity": int(capacity),
        "semester": int(semester) if is_scheduled else None,
        "scheduled_students": int(scheduled_students) if is_scheduled else 0,
        "is_scheduled": int(is_scheduled),
        "is_holiday": int(is_holiday),
        "is_exam_period": int(is_exam_period),
    }

    with col_right:
        if submitted or "last_prediction" not in st.session_state:
            st.session_state["last_prediction"] = registry.predict(payload)

        res = st.session_state["last_prediction"]
        pred_pct = res["predicted_utilization_pct"]
        res_status = res["predicted_resource_status"]
        slot_status = res["predicted_slot_status"]

        st.markdown("#### Prediction Output")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Utilization", f"{pred_pct:.2f}%", res["selected_model_name"])
        m2.metric("Resource Band", res_status)
        m3.metric("Slot Classification", slot_status)

        badge_color = _status_badge_color(res_status)
        st.markdown(
            f'<div style="padding:10px 14px;border-radius:8px;background:{badge_color}18;'
            f'border-left:4px solid {badge_color};margin:8px 0 16px 0;">'
            f'<strong>Status Summary:</strong> {selected_room} ({room_type}, {room_meta["building"]}) '
            f'is projected at <strong>{pred_pct:.2f}%</strong> ({res_status}).</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Model Comparison (Same Inputs)")
        comp_rows = []
        for mkey in ("xgboost", "random_forest"):
            mdata = res["predictions"][mkey]
            comp_rows.append({
                "Model": mdata["name"],
                "Predicted Utilization (%)": mdata["predicted_utilization_pct"],
                "Predicted Status": mdata["predicted_status"],
                "Test MAE": mdata["metrics"]["mae"],
                "Test RMSE": mdata["metrics"]["rmse"],
                "Test R²": mdata["metrics"]["r2"],
            })
        st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

        with st.expander("Inspect 15 Model Input Features"):
            feat_items = [
                {"Feature": k, "Value": str(v)}
                for k, v in res["features_sent"].items()
            ]
            st.dataframe(pd.DataFrame(feat_items), width="stretch", hide_index=True)


def render_model_performance(registry: ModelRegistry) -> None:
    """Render Page 5: Model Performance & Leakage Audit."""
    assert registry.artifacts is not None
    art = registry.artifacts
    split = art["split_info"]
    rec = art["recommendation"]
    rf_m = art["models"]["random_forest"]["metrics"]
    xgb_m = art["models"]["xgboost"]["metrics"]

    st.subheader("Model Performance")
    st.caption("Chronological 80:20 evaluation, residual diagnostics, feature importance & leakage audit")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recommended Model", rec["recommended_model_name"])
    c2.metric("Train Records", f"{split['train_records']:,}", f"{split['train_start']} → {split['train_end']}")
    c3.metric("Test Records", f"{split['test_records']:,}", f"{split['test_start']} → {split['test_end']}")
    c4.metric(
        "XGBoost R² / RMSE",
        f"{xgb_m['r2']:.4f}",
        f"RMSE {xgb_m['rmse']:.4f} (Δ {rec['rmse_improvement']:+.4f} vs RF)",
    )

    st.markdown("#### Evaluation Metrics Comparison")
    metrics_df = pd.DataFrame([
        {"Model": "Random Forest Regressor", **rf_m},
        {"Model": "XGBoost Regressor", **xgb_m},
    ])
    st.dataframe(metrics_df, width="stretch", hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Top Feature Importances (%)")
        fi_df = pd.DataFrame(art["feature_importance"][:12])
        fi_melt = fi_df.melt(
            id_vars=["feature"],
            value_vars=["xgb_importance", "rf_importance"],
            var_name="Model",
            value_name="Importance (%)",
        )
        fi_melt["Model"] = fi_melt["Model"].replace({
            "xgb_importance": "XGBoost",
            "rf_importance": "Random Forest",
        })
        st.altair_chart(
            alt.Chart(fi_melt)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                y=alt.Y("feature:N", sort="-x", title="Feature"),
                x=alt.X("Importance (%):Q"),
                color=alt.Color("Model:N", scale=alt.Scale(range=["#2563eb", "#7c3aed"])),
                yOffset="Model:N",
                tooltip=["feature", "Model", "Importance (%)"],
            )
            .properties(height=360),
            width="stretch",
        )

    with col_b:
        st.markdown("#### Actual vs Predicted Utilization (Test Sample)")
        sc_df = pd.DataFrame(art["visualizations"]["scatter_points"])
        diag_max = max(float(sc_df["actual"].max()), float(sc_df["xgb_pred"].max()), 100.0)
        diag_df = pd.DataFrame({"x": [0.0, diag_max], "y": [0.0, diag_max]})
        scatter = (
            alt.Chart(sc_df)
            .mark_circle(size=28, opacity=0.45, color="#2563eb")
            .encode(
                x=alt.X("actual:Q", title="Actual Utilization (%)"),
                y=alt.Y("xgb_pred:Q", title="XGBoost Predicted (%)"),
                tooltip=["date", "actual", "xgb_pred", "rf_pred"],
            )
        )
        line = alt.Chart(diag_df).mark_line(color="#ef4444", strokeDash=[5, 4]).encode(x="x:Q", y="y:Q")
        st.altair_chart((scatter + line).properties(height=360), width="stretch")

    st.markdown("#### Test Residual Distribution (Full Test Set)")
    hist_df = pd.DataFrame(art["visualizations"]["error_histogram"])
    hist_melt = hist_df.melt(
        id_vars=["bin_label", "bin_mid"],
        value_vars=["xgb_count", "rf_count"],
        var_name="Model",
        value_name="Count",
    )
    hist_melt["Model"] = hist_melt["Model"].replace({
        "xgb_count": "XGBoost",
        "rf_count": "Random Forest",
    })
    st.altair_chart(
        alt.Chart(hist_melt)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("bin_mid:Q", title="Residual Error (Actual - Predicted %)"),
            y=alt.Y("Count:Q", title="Test Records"),
            color=alt.Color("Model:N", scale=alt.Scale(range=["#2563eb", "#7c3aed"])),
            tooltip=["bin_label", "Model", "Count"],
        )
        .properties(height=260),
        width="stretch",
    )

    st.markdown("#### Target Leakage Audit")
    audit = art["leakage_audit"]
    if audit["passed"]:
        st.success(f"Leakage audit PASSED — all {audit['total_input_features']} inputs verified clean.")
    else:
        st.error("Leakage audit detected forbidden target columns!")
    st.dataframe(pd.DataFrame(audit["checks"]), width="stretch", hide_index=True)


def main() -> None:
    dfs, registry = load_app_state()

    with st.sidebar:
        st.title("🏫 UniSpace AI")
        st.caption("Classroom & Lab Utilization Analytics")
        page = st.radio(
            "Navigation",
            [
                "Overview Dashboard",
                "Room Analytics",
                "Demand & Utilization Analysis",
                "AI Prediction Centre",
                "Model Performance",
            ],
        )
        st.divider()
        st.caption(
            "Powered by reproducible synthetic academic data (`rng=42`), "
            "Random Forest, and XGBoost."
        )

    if page == "Overview Dashboard":
        render_overview(dfs)
    elif page == "Room Analytics":
        render_rooms(dfs)
    elif page == "Demand & Utilization Analysis":
        render_demand(dfs)
    elif page == "AI Prediction Centre":
        render_predict(dfs, registry)
    else:
        render_model_performance(registry)


if __name__ == "__main__":
    main()
