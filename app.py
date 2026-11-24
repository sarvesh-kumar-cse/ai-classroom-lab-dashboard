"""FastAPI backend serving analytics, live ML predictions, and the dashboard UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.analytics import (
    DAY_ORDER,
    TIME_ORDER,
    get_demand_analysis,
    get_overview_data,
    get_room_analytics,
)
from src.data_generator import PROJECT_ROOT, generate_datasets
from src.ml_pipeline import ModelRegistry


STATIC_DIR = PROJECT_ROOT / "static"

state: dict[str, Any] = {
    "dfs": None,
    "registry": ModelRegistry(),
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    dfs = generate_datasets()
    state["dfs"] = dfs
    state["registry"].ensure_loaded(history_df=dfs["history"])
    yield


app = FastAPI(
    title="AI Classroom & Lab Utilization Dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class PredictionRequest(BaseModel):
    model_name: str = Field(default="xgboost", pattern="^(random_forest|xgboost)$")
    room_id: str
    room_type: str
    date: str
    day_of_week: str
    time_slot: str
    course_id: Optional[str] = "Unscheduled"
    department: Optional[str] = "Unscheduled"
    capacity: int = Field(ge=1, le=500)
    semester: Optional[int] = None
    scheduled_students: int = Field(ge=0, le=500)
    is_scheduled: int = Field(ge=0, le=1)
    is_holiday: int = Field(ge=0, le=1)
    is_exam_period: int = Field(ge=0, le=1)


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/meta")
def api_meta() -> dict[str, Any]:
    dfs = state["dfs"]
    rooms_df = dfs["rooms"]
    calendar_df = dfs["calendar"]
    history_df = dfs["history"]

    calendar_map = {
        row["date"].strftime("%Y-%m-%d"): {
            "day_of_week": row["day_of_week"],
            "is_holiday": int(row["is_holiday"]),
            "is_exam_period": int(row["is_exam_period"]),
            "semester": int(row["semester"]),
        }
        for _, row in calendar_df.iterrows()
    }

    return {
        "rooms": rooms_df.sort_values("room_id").to_dict(orient="records"),
        "days": DAY_ORDER,
        "time_slots": TIME_ORDER,
        "courses": {
            "Classroom": ["CSE101", "MAT201", "PHY101", "ENG102", "AI301"],
            "Laboratory": ["CSE_LAB", "AI_LAB", "DBMS_LAB", "NETWORK_LAB"],
        },
        "departments": ["CSE", "ECE", "ME", "IT"],
        "date_range": {
            "min": history_df["date"].min().strftime("%Y-%m-%d"),
            "max": history_df["date"].max().strftime("%Y-%m-%d"),
        },
        "calendar": calendar_map,
        "synthetic_notice": (
            "Dataset & ML predictions are powered by synthetic academic data "
            "generated using the notebook's reproducible seed (rng=42)."
        ),
    }


@app.get("/api/overview")
def api_overview() -> dict[str, Any]:
    return get_overview_data(state["dfs"])


@app.get("/api/rooms")
def api_rooms(
    room_type: str = Query(default="All"),
    building: str = Query(default="All"),
    resource_status: str = Query(default="All"),
    min_util: float = Query(default=0.0, ge=0.0, le=200.0),
    max_util: float = Query(default=100.0, ge=0.0, le=200.0),
) -> dict[str, Any]:
    return get_room_analytics(
        state["dfs"],
        room_type=room_type,
        building=building,
        resource_status=resource_status,
        min_util=min_util,
        max_util=max_util,
    )


@app.get("/api/demand")
def api_demand(
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
    department: str = Query(default="All"),
    room_type: str = Query(default="All"),
    time_slot: str = Query(default="All"),
) -> dict[str, Any]:
    return get_demand_analysis(
        state["dfs"],
        start_date=start_date,
        end_date=end_date,
        department=department,
        room_type=room_type,
        time_slot=time_slot,
    )


@app.post("/api/predict")
def api_predict(req: PredictionRequest) -> dict[str, Any]:
    return state["registry"].predict(req.model_dump())


@app.get("/api/model-performance")
def api_model_performance() -> dict[str, Any]:
    registry: ModelRegistry = state["registry"]
    registry.ensure_loaded(history_df=state["dfs"]["history"])
    assert registry.artifacts is not None
    return registry.artifacts


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8050, reload=False)
