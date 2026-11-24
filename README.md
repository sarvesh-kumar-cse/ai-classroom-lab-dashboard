# UniSpace AI — Classroom & Laboratory Utilization Dashboard

Enterprise-style university analytics and ML forecasting dashboard built directly from `AI_Classroom_Lab_Utilization_System.ipynb`.

## Architecture
- `src/data_generator.py`: Reproduces the synthetic university dataset (`rooms.csv`, `timetable.csv`, `utilization_history.csv`, `academic_calendar.csv`) with `np.random.default_rng(42)`.
- `src/analytics.py`: Computes all KPIs, room-wise summaries, day-of-week heatmaps, and demand breakdowns directly from the CSV datasets.
- `src/ml_pipeline.py`: Implements the notebook's 15-feature preprocessing pipeline (`ColumnTransformer` with `OneHotEncoder` + `SimpleImputer`), chronological 80:20 split, target leakage audit, `RandomForestRegressor`, and `XGBRegressor`.
- `app.py` & `static/`: FastAPI backend serving REST endpoints and a dark-navy + crisp-white SaaS analytics frontend with Chart.js visualizations.

## Run Locally
```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8050
```
Then open `http://127.0.0.1:8050` in your browser.
