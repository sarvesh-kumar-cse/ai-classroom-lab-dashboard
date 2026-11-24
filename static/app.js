/* UniSpace AI — Interactive Frontend Controller */

const PAGE_META = {
  overview: {
    title: "Overview Dashboard",
    subtitle: "University-wide classroom & laboratory utilization telemetry",
  },
  rooms: {
    title: "Room Analytics",
    subtitle: "Room-level occupancy inventory, filters, and weekly heatmap",
  },
  demand: {
    title: "Demand & Utilization Analysis",
    subtitle: "Temporal, departmental, exam-period, and holiday demand patterns",
  },
  predict: {
    title: "AI Prediction Centre",
    subtitle: "Live Random Forest & XGBoost utilization inference",
  },
  models: {
    title: "Model Performance",
    subtitle: "Chronological 80:20 evaluation, residual diagnostics & leakage audit",
  },
};

const PALETTE = {
  navy: "#0f172a",
  blue: "#2563eb",
  blueSoft: "rgba(37, 99, 235, 0.14)",
  violet: "#7c3aed",
  violetSoft: "rgba(124, 58, 237, 0.14)",
  cyan: "#0891b2",
  emerald: "#10b981",
  amber: "#f59e0b",
  rose: "#ef4444",
  slate: "#64748b",
  grid: "#e2e8f0",
};

Chart.defaults.font.family = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
Chart.defaults.font.size = 12;
Chart.defaults.color = "#475569";

const charts = {};
let appMeta = null;
let modelArtifacts = null;
let selectedModelKey = "xgboost";

function upsertChart(id, config) {
  if (charts[id]) {
    charts[id].destroy();
  }
  const canvas = document.getElementById(id);
  if (!canvas) return;
  charts[id] = new Chart(canvas.getContext("2d"), config);
}

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Request failed (${res.status}): ${errText}`);
  }
  return res.json();
}

function statusBadgeHtml(status) {
  const map = {
    Normal: "badge-normal",
    Underutilized: "badge-under",
    Overloaded: "badge-over",
    Available: "badge-avail",
  };
  return `<span class="badge ${map[status] || "badge-avail"}">${status}</span>`;
}

function kpiCardHtml(label, value, sub, badgeHtml = "") {
  return `
    <div class="kpi-card">
      <div class="kpi-label"><span>${label}</span>${badgeHtml}</div>
      <div class="kpi-value">${value}</div>
      <div class="kpi-sub">${sub}</div>
    </div>
  `;
}

/* -------------------- NAVIGATION -------------------- */
function setupNavigation() {
  const sidebar = document.getElementById("sidebar");
  const mobileBtn = document.getElementById("mobileMenuBtn");

  mobileBtn.addEventListener("click", () => {
    sidebar.classList.toggle("open");
  });

  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      const page = btn.dataset.page;
      document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(".page-section").forEach((sec) => sec.classList.remove("active"));
      document.getElementById(`page-${page}`).classList.add("active");
      document.getElementById("topbarTitle").textContent = PAGE_META[page].title;
      document.getElementById("topbarSubtitle").textContent = PAGE_META[page].subtitle;
      sidebar.classList.remove("open");
    });
  });
}

/* -------------------- PAGE 1: OVERVIEW -------------------- */
async function loadOverview() {
  const data = await fetchJson("/api/overview");
  const k = data.kpis;

  document.getElementById("overviewKpis").innerHTML = [
    kpiCardHtml("Total Classrooms", k.total_classrooms, "C001–C014 across Blocks A–C", '<span class="badge badge-blue">Classroom</span>'),
    kpiCardHtml("Total Laboratories", k.total_laboratories, "L001–L010 specialized labs", '<span class="badge badge-violet">Lab</span>'),
    kpiCardHtml("Total Available Resources", k.total_available_resources, `${k.total_capacity_seats.toLocaleString()} total seats`, '<span class="badge badge-avail">Rooms</span>'),
    kpiCardHtml("Average Utilization", `${k.average_utilization_pct.toFixed(2)}%`, `Scheduled only: ${k.scheduled_average_utilization_pct.toFixed(2)}%`),
    kpiCardHtml("Underutilized Rooms", k.underutilized_rooms, "Average utilization < 30%", '<span class="badge badge-under">&lt;30%</span>'),
    kpiCardHtml("Normal Utilization Rooms", k.normal_rooms, "Average utilization 30–80%", '<span class="badge badge-normal">30–80%</span>'),
    kpiCardHtml("Overloaded Rooms", k.overloaded_rooms, `${k.overloaded_slot_records.toLocaleString()} overloaded slot events`, '<span class="badge badge-over">&gt;80%</span>'),
  ].join("");

  upsertChart("chartOverviewTrend", {
    type: "line",
    data: {
      labels: data.overall_trend.map((d) => d.month),
      datasets: [
        {
          label: "Overall Utilization (%)",
          data: data.overall_trend.map((d) => d.avg_utilization),
          borderColor: PALETTE.navy,
          backgroundColor: "rgba(15, 23, 42, 0.06)",
          borderWidth: 2.5,
          tension: 0.3,
          fill: true,
          pointRadius: 3,
        },
        {
          label: "Classrooms (%)",
          data: data.overall_trend.map((d) => d.classroom_util),
          borderColor: PALETTE.blue,
          borderWidth: 2,
          borderDash: [4, 3],
          tension: 0.3,
          pointRadius: 2.5,
        },
        {
          label: "Laboratories (%)",
          data: data.overall_trend.map((d) => d.lab_util),
          borderColor: PALETTE.violet,
          borderWidth: 2,
          borderDash: [4, 3],
          tension: 0.3,
          pointRadius: 2.5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, max: 50, grid: { color: PALETTE.grid }, title: { display: true, text: "Utilization (%)" } },
        x: { grid: { display: false } },
      },
      plugins: { legend: { position: "bottom" } },
    },
  });

  upsertChart("chartRoomTypeComp", {
    type: "bar",
    data: {
      labels: data.room_type_comparison.map((d) => d.room_type),
      datasets: [
        {
          label: "Overall Avg Utilization (%)",
          data: data.room_type_comparison.map((d) => d.avg_utilization),
          backgroundColor: [PALETTE.blue, PALETTE.violet],
          borderRadius: 8,
          barPercentage: 0.55,
        },
        {
          label: "Scheduled-Only Avg (%)",
          data: data.room_type_comparison.map((d) => d.scheduled_avg_utilization),
          backgroundColor: ["rgba(37, 99, 235, 0.25)", "rgba(124, 58, 237, 0.25)"],
          borderColor: [PALETTE.blue, PALETTE.violet],
          borderWidth: 1.5,
          borderRadius: 8,
          barPercentage: 0.55,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, max: 100, grid: { color: PALETTE.grid } },
        x: { grid: { display: false } },
      },
      plugins: { legend: { position: "bottom" } },
    },
  });

  upsertChart("chartStatusDist", {
    type: "doughnut",
    data: {
      labels: data.room_status_distribution.map((d) => d.status),
      datasets: [
        {
          data: data.room_status_distribution.map((d) => d.count),
          backgroundColor: [PALETTE.amber, PALETTE.emerald, PALETTE.rose],
          borderWidth: 2,
          borderColor: "#ffffff",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "64%",
      plugins: { legend: { position: "bottom" } },
    },
  });

  upsertChart("chartWeeklyOverview", {
    type: "bar",
    data: {
      labels: data.weekly_utilization.map((d) => d.day.slice(0, 3)),
      datasets: [
        {
          label: "Avg Utilization (%)",
          data: data.weekly_utilization.map((d) => d.avg_utilization),
          backgroundColor: PALETTE.blue,
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, suggestedMax: 40, grid: { color: PALETTE.grid } },
        x: { grid: { display: false } },
      },
      plugins: { legend: { display: false } },
    },
  });

  upsertChart("chartTimeslotOverview", {
    type: "bar",
    data: {
      labels: data.timeslot_utilization.map((d) => d.time_slot),
      datasets: [
        {
          label: "Avg Utilization (%)",
          data: data.timeslot_utilization.map((d) => d.avg_utilization),
          backgroundColor: PALETTE.violet,
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, suggestedMax: 40, grid: { color: PALETTE.grid } },
        x: { grid: { display: false }, ticks: { font: { size: 10.5 } } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

/* -------------------- PAGE 2: ROOM ANALYTICS -------------------- */
function heatmapCellStyle(val) {
  if (val === null || val === undefined) {
    return "background:#f1f5f9;color:#94a3b8;";
  }
  if (val < 25) {
    return "background:#fef3c7;color:#92400e;";
  }
  if (val < 30) {
    return "background:#dbeafe;color:#1e40af;";
  }
  if (val <= 35) {
    return "background:#6366f1;color:#ffffff;";
  }
  return "background:#312e81;color:#ffffff;";
}

async function loadRoomAnalytics() {
  const roomType = document.getElementById("filterRoomType").value;
  const building = document.getElementById("filterBuilding").value;
  const status = document.getElementById("filterStatus").value;
  const minUtil = document.getElementById("filterMinUtil").value || 0;
  const maxUtil = document.getElementById("filterMaxUtil").value || 100;

  const params = new URLSearchParams({
    room_type: roomType,
    building,
    resource_status: status,
    min_util: minUtil,
    max_util: maxUtil,
  });

  const data = await fetchJson(`/api/rooms?${params.toString()}`);
  const s = data.summary;

  document.getElementById("roomSummaryKpis").innerHTML = [
    kpiCardHtml("Matching Rooms", `${s.matching_rooms} / ${s.total_rooms}`, "Filtered room inventory"),
    kpiCardHtml("Filtered Mean Utilization", `${s.avg_utilization.toFixed(2)}%`, "Average across matching rooms"),
    kpiCardHtml("Underutilized (<30%)", s.underutilized_count, "Rooms below 30% utilization", '<span class="badge badge-under">Under</span>'),
    kpiCardHtml("Normal (30–80%)", s.normal_count, "Rooms within target band", '<span class="badge badge-normal">Normal</span>'),
    kpiCardHtml("Overloaded (>80%)", s.overloaded_count, "Rooms exceeding 80% mean", '<span class="badge badge-over">Over</span>'),
  ].join("");

  // Heatmap
  const hmContainer = document.getElementById("roomHeatmapContainer");
  if (data.heatmap.rows.length === 0) {
    hmContainer.innerHTML = `<p style="color:#64748b;padding:16px 4px;">No rooms match the current filter selection.</p>`;
  } else {
    const headerCells = data.heatmap.days.map((d) => `<th>${d}</th>`).join("");
    const bodyRows = data.heatmap.rows
      .map((row) => {
        const dayCells = data.heatmap.days
          .map((d) => {
            const v = row.values[d];
            const txt = v === null ? "—" : `${v.toFixed(1)}%`;
            return `<td><div class="heatmap-cell" style="${heatmapCellStyle(v)}" title="${row.room_id} · ${d}: ${txt}">${txt}</div></td>`;
          })
          .join("");
        return `
          <tr>
            <td class="heatmap-room-label">${row.room_id}<span class="heatmap-room-meta">${row.room_type === "Classroom" ? "Cls" : "Lab"} · ${row.building}</span></td>
            ${dayCells}
          </tr>
        `;
      })
      .join("");

    hmContainer.innerHTML = `
      <table class="heatmap-table">
        <thead><tr><th>Room ID</th>${headerCells}</tr></thead>
        <tbody>${bodyRows}</tbody>
      </table>
    `;
  }

  // Table
  const tbody = document.getElementById("roomsTableBody");
  if (data.rooms.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" style="text-align:center;color:#64748b;padding:24px;">No rooms match these filter criteria.</td></tr>`;
  } else {
    tbody.innerHTML = data.rooms
      .map(
        (r) => `
      <tr>
        <td><strong>${r.room_id}</strong></td>
        <td>${r.room_type}</td>
        <td>${r.building}</td>
        <td>${r.equipment}</td>
        <td>${r.capacity}</td>
        <td><strong>${r.average_utilization.toFixed(2)}%</strong></td>
        <td>${r.scheduled_utilization.toFixed(2)}%</td>
        <td>${r.underutilized_slots}</td>
        <td>${r.overloaded_slots}</td>
        <td>${statusBadgeHtml(r.resource_status)}</td>
      </tr>
    `
      )
      .join("");
  }
}

function setupRoomFilters() {
  ["filterRoomType", "filterBuilding", "filterStatus", "filterMinUtil", "filterMaxUtil"].forEach((id) => {
    document.getElementById(id).addEventListener("change", loadRoomAnalytics);
  });
  document.getElementById("btnResetRoomFilters").addEventListener("click", () => {
    document.getElementById("filterRoomType").value = "All";
    document.getElementById("filterBuilding").value = "All";
    document.getElementById("filterStatus").value = "All";
    document.getElementById("filterMinUtil").value = 0;
    document.getElementById("filterMaxUtil").value = 100;
    loadRoomAnalytics();
  });
}

/* -------------------- PAGE 3: DEMAND & UTILIZATION -------------------- */
async function loadDemandAnalysis() {
  const startDate = document.getElementById("demandStartDate").value;
  const endDate = document.getElementById("demandEndDate").value;
  const department = document.getElementById("demandDept").value;
  const roomType = document.getElementById("demandRoomType").value;
  const timeSlot = document.getElementById("demandTimeSlot").value;

  const params = new URLSearchParams({
    start_date: startDate,
    end_date: endDate,
    department,
    room_type: roomType,
    time_slot: timeSlot,
  });

  const data = await fetchJson(`/api/demand?${params.toString()}`);

  const banner = document.getElementById("demandFilterBanner");
  if (data.department_filter_note) {
    banner.style.display = "flex";
    banner.innerHTML = `<span>${data.department_filter_note}</span>`;
  } else {
    banner.style.display = "none";
  }

  if (data.empty) {
    document.getElementById("demandKpis").innerHTML = kpiCardHtml("Matching Records", "0", "Adjust filters to inspect demand");
    return;
  }

  const clsRow = data.room_type_demand.find((d) => d.room_type === "Classroom");
  const labRow = data.room_type_demand.find((d) => d.room_type === "Laboratory");
  const examRow = data.exam_period.find((d) => d.label === "Exam Period");
  const holRow = data.holiday_comparison.find((d) => d.label === "Holiday");

  document.getElementById("demandKpis").innerHTML = [
    kpiCardHtml("Filtered Records", data.record_count.toLocaleString(), `${data.scheduled_count.toLocaleString()} scheduled sessions`),
    kpiCardHtml("Mean Utilization", `${data.avg_utilization_pct.toFixed(2)}%`, "Across filtered selection"),
    kpiCardHtml("Classroom Demand", clsRow ? `${clsRow.avg_utilization.toFixed(2)}%` : "—", clsRow ? `Scheduling rate: ${clsRow.scheduling_rate_pct}%` : "Not in filter"),
    kpiCardHtml("Laboratory Demand", labRow ? `${labRow.avg_utilization.toFixed(2)}%` : "—", labRow ? `Scheduling rate: ${labRow.scheduling_rate_pct}%` : "Not in filter"),
    kpiCardHtml("Exam Period Mean", examRow ? `${examRow.avg_utilization.toFixed(2)}%` : "—", examRow ? `${examRow.scheduled_sessions.toLocaleString()} exam-period slots` : "No exam days in range"),
    kpiCardHtml("Holiday Mean", holRow ? `${holRow.avg_utilization.toFixed(2)}%` : "—", holRow ? `${holRow.records.toLocaleString()} holiday slot records` : "No holidays in range"),
  ].join("");

  upsertChart("chartDemandDay", {
    type: "bar",
    data: {
      labels: data.day_wise.map((d) => d.day),
      datasets: [
        {
          label: "Avg Utilization (%)",
          data: data.day_wise.map((d) => d.avg_utilization),
          backgroundColor: PALETTE.blue,
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true, grid: { color: PALETTE.grid } }, x: { grid: { display: false } } },
      plugins: { legend: { display: false } },
    },
  });

  upsertChart("chartDemandSlot", {
    type: "bar",
    data: {
      labels: data.slot_wise.map((d) => d.time_slot),
      datasets: [
        {
          label: "Avg Utilization (%)",
          data: data.slot_wise.map((d) => d.avg_utilization),
          backgroundColor: PALETTE.violet,
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true, grid: { color: PALETTE.grid } }, x: { grid: { display: false } } },
      plugins: { legend: { display: false } },
    },
  });

  upsertChart("chartDemandMonthly", {
    type: "line",
    data: {
      labels: data.monthly_room_type_demand.map((d) => d.month),
      datasets: [
        {
          label: "Classroom (%)",
          data: data.monthly_room_type_demand.map((d) => d.classroom),
          borderColor: PALETTE.blue,
          backgroundColor: PALETTE.blueSoft,
          borderWidth: 2.2,
          tension: 0.3,
        },
        {
          label: "Laboratory (%)",
          data: data.monthly_room_type_demand.map((d) => d.laboratory),
          borderColor: PALETTE.violet,
          backgroundColor: PALETTE.violetSoft,
          borderWidth: 2.2,
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true, grid: { color: PALETTE.grid } }, x: { grid: { display: false } } },
      plugins: { legend: { position: "bottom" } },
    },
  });

  upsertChart("chartDemandRoomType", {
    type: "bar",
    data: {
      labels: data.room_type_demand.map((d) => d.room_type),
      datasets: [
        {
          label: "Avg Utilization (%)",
          data: data.room_type_demand.map((d) => d.avg_utilization),
          backgroundColor: PALETTE.blue,
          borderRadius: 6,
        },
        {
          label: "Scheduling Rate (%)",
          data: data.room_type_demand.map((d) => d.scheduling_rate_pct),
          backgroundColor: PALETTE.cyan,
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true, max: 100, grid: { color: PALETTE.grid } }, x: { grid: { display: false } } },
      plugins: { legend: { position: "bottom" } },
    },
  });

  upsertChart("chartDemandExam", {
    type: "bar",
    data: {
      labels: data.exam_period.map((d) => d.label),
      datasets: [
        {
          label: "Avg Utilization (%)",
          data: data.exam_period.map((d) => d.avg_utilization),
          backgroundColor: [PALETTE.slate, PALETTE.violet],
          borderRadius: 6,
          barPercentage: 0.5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true, grid: { color: PALETTE.grid } }, x: { grid: { display: false } } },
      plugins: { legend: { display: false } },
    },
  });

  upsertChart("chartDemandHoliday", {
    type: "bar",
    data: {
      labels: data.holiday_comparison.map((d) => d.label),
      datasets: [
        {
          label: "Avg Utilization (%)",
          data: data.holiday_comparison.map((d) => d.avg_utilization),
          backgroundColor: [PALETTE.blue, PALETTE.amber],
          borderRadius: 6,
          barPercentage: 0.5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true, grid: { color: PALETTE.grid } }, x: { grid: { display: false } } },
      plugins: { legend: { display: false } },
    },
  });

  upsertChart("chartDemandDept", {
    type: "bar",
    data: {
      labels: data.department_demand.map((d) => d.department),
      datasets: [
        {
          label: "Avg Utilization (%)",
          data: data.department_demand.map((d) => d.avg_utilization),
          backgroundColor: PALETTE.emerald,
          borderRadius: 6,
          barPercentage: 0.55,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true, max: 100, grid: { color: PALETTE.grid } }, x: { grid: { display: false } } },
      plugins: { legend: { display: false } },
    },
  });
}

function setupDemandFilters() {
  ["demandStartDate", "demandEndDate", "demandDept", "demandRoomType", "demandTimeSlot"].forEach((id) => {
    document.getElementById(id).addEventListener("change", loadDemandAnalysis);
  });
  document.getElementById("btnResetDemandFilters").addEventListener("click", () => {
    document.getElementById("demandStartDate").value = appMeta.date_range.min;
    document.getElementById("demandEndDate").value = appMeta.date_range.max;
    document.getElementById("demandDept").value = "All";
    document.getElementById("demandRoomType").value = "All";
    document.getElementById("demandTimeSlot").value = "All";
    loadDemandAnalysis();
  });
}

/* -------------------- PAGE 4: AI PREDICTION CENTRE -------------------- */
function populatePredictionDropdowns() {
  const roomSelect = document.getElementById("predRoomId");
  roomSelect.innerHTML = appMeta.rooms
    .map((r) => `<option value="${r.room_id}">${r.room_id} (${r.room_type} · Cap ${r.capacity} · ${r.building})</option>`)
    .join("");

  const slotSelect = document.getElementById("predTimeSlot");
  slotSelect.innerHTML = appMeta.time_slots.map((s) => `<option value="${s}">${s}</option>`).join("");

  const demandSlotSelect = document.getElementById("demandTimeSlot");
  demandSlotSelect.innerHTML =
    `<option value="All">All Time Slots</option>` +
    appMeta.time_slots.map((s) => `<option value="${s}">${s}</option>`).join("");

  const syncCourseOptions = (roomType) => {
    const courseSel = document.getElementById("predCourseId");
    const list = appMeta.courses[roomType] || [];
    courseSel.innerHTML =
      list.map((c) => `<option value="${c}">${c}</option>`).join("") +
      `<option value="Unscheduled">Unscheduled</option>`;
  };

  const syncRoomSelection = () => {
    const selectedId = roomSelect.value;
    const room = appMeta.rooms.find((r) => r.room_id === selectedId);
    if (!room) return;
    document.getElementById("predRoomType").value = room.room_type;
    document.getElementById("predCapacity").value = room.capacity;
    syncCourseOptions(room.room_type);
  };

  roomSelect.addEventListener("change", syncRoomSelection);
  document.getElementById("predRoomType").addEventListener("change", (e) => {
    syncCourseOptions(e.target.value);
  });

  document.getElementById("predDate").addEventListener("change", (e) => {
    const cal = appMeta.calendar[e.target.value];
    if (cal) {
      if (appMeta.days.includes(cal.day_of_week)) {
        document.getElementById("predDay").value = cal.day_of_week;
      }
      document.getElementById("predHoliday").value = String(cal.is_holiday);
      document.getElementById("predExam").value = String(cal.is_exam_period);
      document.getElementById("predSemester").value = String(cal.semester);
    } else {
      const dt = new Date(e.target.value + "T00:00:00");
      const names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
      const dayName = names[dt.getDay()];
      if (appMeta.days.includes(dayName)) {
        document.getElementById("predDay").value = dayName;
      }
    }
  });

  document.getElementById("predIsScheduled").addEventListener("change", (e) => {
    if (e.target.value === "0") {
      document.getElementById("predCourseId").value = "Unscheduled";
      document.getElementById("predDepartment").value = "Unscheduled";
      document.getElementById("predSemester").value = "Unscheduled";
      document.getElementById("predScheduledStudents").value = "0";
    } else {
      const rt = document.getElementById("predRoomType").value;
      const firstCourse = (appMeta.courses[rt] || [])[0];
      if (firstCourse) document.getElementById("predCourseId").value = firstCourse;
      if (document.getElementById("predDepartment").value === "Unscheduled") {
        document.getElementById("predDepartment").value = "CSE";
      }
      if (document.getElementById("predSemester").value === "Unscheduled") {
        document.getElementById("predSemester").value = "2";
      }
      if (Number(document.getElementById("predScheduledStudents").value) === 0) {
        document.getElementById("predScheduledStudents").value = "32";
      }
    }
  });

  document.querySelectorAll("#modelSelectorGroup .model-option").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#modelSelectorGroup .model-option").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      selectedModelKey = btn.dataset.model;
      runPrediction();
    });
  });

  document.getElementById("predictionForm").addEventListener("submit", (e) => {
    e.preventDefault();
    runPrediction();
  });

  syncRoomSelection();
  document.getElementById("predDate").dispatchEvent(new Event("change"));
}

async function runPrediction() {
  const semRaw = document.getElementById("predSemester").value;
  const payload = {
    model_name: selectedModelKey,
    room_id: document.getElementById("predRoomId").value,
    room_type: document.getElementById("predRoomType").value,
    date: document.getElementById("predDate").value,
    day_of_week: document.getElementById("predDay").value,
    time_slot: document.getElementById("predTimeSlot").value,
    course_id: document.getElementById("predCourseId").value,
    department: document.getElementById("predDepartment").value,
    capacity: Number(document.getElementById("predCapacity").value),
    semester: semRaw === "Unscheduled" ? null : Number(semRaw),
    scheduled_students: Number(document.getElementById("predScheduledStudents").value),
    is_scheduled: Number(document.getElementById("predIsScheduled").value),
    is_holiday: Number(document.getElementById("predHoliday").value),
    is_exam_period: Number(document.getElementById("predExam").value),
  };

  const res = await fetchJson("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  document.getElementById("predModelUsedLabel").textContent = `${res.selected_model_name} · Live Prediction`;
  document.getElementById("predUtilValue").textContent = `${res.predicted_utilization_pct.toFixed(2)}%`;
  document.getElementById("predSlotStatus").textContent = res.predicted_slot_status;

  const badge = document.getElementById("predStatusBadge");
  badge.textContent = res.predicted_resource_status;
  badge.className = "badge " + (
    res.predicted_resource_status === "Normal"
      ? "badge-normal"
      : res.predicted_resource_status === "Underutilized"
      ? "badge-under"
      : "badge-over"
  );

  const metrics = res.predictions[res.selected_model].metrics;
  document.getElementById("predMetricMae").textContent = metrics.mae.toFixed(2);
  document.getElementById("predMetricRmse").textContent = metrics.rmse.toFixed(2);
  document.getElementById("predMetricR2").textContent = `${metrics.r2.toFixed(4)} (${metrics.r2_pct.toFixed(2)}%)`;

  const rfVal = res.predictions.random_forest.predicted_utilization_pct;
  const xgbVal = res.predictions.xgboost.predicted_utilization_pct;

  upsertChart("chartPredictComparison", {
    type: "bar",
    data: {
      labels: ["Random Forest Regressor", "XGBoost Regressor"],
      datasets: [
        {
          label: "Predicted Utilization (%)",
          data: [rfVal, xgbVal],
          backgroundColor: [PALETTE.blue, PALETTE.violet],
          borderRadius: 8,
          barPercentage: 0.48,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, suggestedMax: 100, grid: { color: PALETTE.grid }, title: { display: true, text: "Utilization (%)" } },
        x: { grid: { display: false } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

/* -------------------- PAGE 5: MODEL PERFORMANCE -------------------- */
function renderActualVsPredScatter(mode = "both") {
  const viz = modelArtifacts.visualizations;
  const pts = viz.scatter_points;
  const minVal = viz.test_min;
  const maxVal = viz.test_max;

  const datasets = [
    {
      type: "line",
      label: "Ideal Fit (y = x)",
      data: [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }],
      borderColor: PALETTE.rose,
      borderDash: [6, 4],
      borderWidth: 2,
      pointRadius: 0,
    },
  ];

  if (mode === "both" || mode === "random_forest") {
    datasets.push({
      type: "scatter",
      label: "Random Forest",
      data: pts.map((p) => ({ x: p.actual, y: p.rf_pred })),
      backgroundColor: "rgba(37, 99, 235, 0.45)",
      pointRadius: 2.7,
    });
  }
  if (mode === "both" || mode === "xgboost") {
    datasets.push({
      type: "scatter",
      label: "XGBoost",
      data: pts.map((p) => ({ x: p.actual, y: p.xgb_pred })),
      backgroundColor: "rgba(124, 58, 237, 0.45)",
      pointRadius: 2.7,
    });
  }

  upsertChart("chartActualVsPred", {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { type: "linear", title: { display: true, text: "Actual Utilization (%)" }, grid: { color: PALETTE.grid } },
        y: { type: "linear", title: { display: true, text: "Predicted Utilization (%)" }, grid: { color: PALETTE.grid } },
      },
      plugins: { legend: { position: "bottom" } },
    },
  });
}

async function loadModelPerformance() {
  modelArtifacts = await fetchJson("/api/model-performance");
  const rf = modelArtifacts.models.random_forest.metrics;
  const xgb = modelArtifacts.models.xgboost.metrics;
  const split = modelArtifacts.split_info;
  const rec = modelArtifacts.recommendation;

  document.getElementById("modelPerfKpis").innerHTML = [
    kpiCardHtml("Random Forest MAE", rf.mae.toFixed(4), "Mean Absolute Error (test)", '<span class="badge badge-blue">RF</span>'),
    kpiCardHtml("Random Forest RMSE", rf.rmse.toFixed(4), "Root Mean Squared Error", '<span class="badge badge-blue">RF</span>'),
    kpiCardHtml("Random Forest R²", rf.r2.toFixed(4), `Variance explained: ${rf.r2_pct.toFixed(2)}%`, '<span class="badge badge-blue">RF</span>'),
    kpiCardHtml("XGBoost MAE", xgb.mae.toFixed(4), "Mean Absolute Error (test)", '<span class="badge badge-violet">XGB</span>'),
    kpiCardHtml("XGBoost RMSE", xgb.rmse.toFixed(4), "Root Mean Squared Error", '<span class="badge badge-violet">XGB</span>'),
    kpiCardHtml("XGBoost R²", xgb.r2.toFixed(4), `Variance explained: ${xgb.r2_pct.toFixed(2)}%`, '<span class="badge badge-violet">XGB</span>'),
  ].join("");

  upsertChart("chartModelMetricsComp", {
    type: "bar",
    data: {
      labels: ["MAE (Lower Better)", "RMSE (Lower Better)", "R² × 100 (Higher Better)"],
      datasets: [
        {
          label: "Random Forest",
          data: [rf.mae, rf.rmse, rf.r2_pct],
          backgroundColor: PALETTE.blue,
          borderRadius: 6,
        },
        {
          label: "XGBoost",
          data: [xgb.mae, xgb.rmse, xgb.r2_pct],
          backgroundColor: PALETTE.violet,
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true, grid: { color: PALETTE.grid } }, x: { grid: { display: false } } },
      plugins: { legend: { position: "bottom" } },
    },
  });

  renderActualVsPredScatter("both");
  document.getElementById("scatterModelToggle").addEventListener("change", (e) => {
    renderActualVsPredScatter(e.target.value);
  });

  const hist = modelArtifacts.visualizations.error_histogram;
  upsertChart("chartResidualDist", {
    type: "bar",
    data: {
      labels: hist.map((b) => b.bin_mid),
      datasets: [
        {
          label: "Random Forest Errors",
          data: hist.map((b) => b.rf_count),
          backgroundColor: "rgba(37, 99, 235, 0.55)",
          borderRadius: 3,
        },
        {
          label: "XGBoost Errors",
          data: hist.map((b) => b.xgb_count),
          backgroundColor: "rgba(124, 58, 237, 0.55)",
          borderRadius: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "Prediction Error (Actual − Predicted %)" }, grid: { display: false } },
        y: { title: { display: true, text: "Test Records Count" }, grid: { color: PALETTE.grid } },
      },
      plugins: { legend: { position: "bottom" } },
    },
  });

  const topFeats = modelArtifacts.feature_importance.slice(0, 8);
  upsertChart("chartFeatureImportance", {
    type: "bar",
    data: {
      labels: topFeats.map((f) => f.feature),
      datasets: [
        {
          label: "Random Forest (%)",
          data: topFeats.map((f) => f.rf_importance),
          backgroundColor: PALETTE.blue,
          borderRadius: 4,
        },
        {
          label: "XGBoost (%)",
          data: topFeats.map((f) => f.xgb_importance),
          backgroundColor: PALETTE.violet,
          borderRadius: 4,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: { x: { beginAtZero: true, grid: { color: PALETTE.grid } }, y: { grid: { display: false } } },
      plugins: { legend: { position: "bottom" } },
    },
  });

  // Leakage Audit + Split Panel
  const audit = modelArtifacts.leakage_audit;
  document.getElementById("leakageAuditContainer").innerHTML = `
    <div style="margin-bottom:12px;font-size:12.5px;color:#475569;">
      <strong>Chronological 80:20 Split:</strong> Train ${split.train_records.toLocaleString()} rows (${split.train_start} → ${split.train_end}) · Test ${split.test_records.toLocaleString()} rows (${split.test_start} → ${split.test_end})
    </div>
    <div class="audit-list">
      ${audit.checks
        .map(
          (c) => `
        <div class="audit-row">
          <span><code>${c.feature}</code></span>
          <span class="badge ${c.present_in_inputs ? "badge-over" : "badge-normal"}">${c.status}</span>
        </div>
      `
        )
        .join("")}
    </div>
    <div style="margin-top:14px;font-size:12px;font-weight:600;color:#475569;">
      Verified Input Features (${audit.total_input_features}):
    </div>
    <div class="feature-chips">
      ${audit.input_features.map((f) => `<span class="feature-chip">${f}</span>`).join("")}
    </div>
  `;

  // Recommendation Card
  const viz = modelArtifacts.visualizations;
  document.getElementById("modelRecommendationCard").innerHTML = `
    <span class="badge badge-violet" style="margin-bottom:10px;">Recommended Production Model</span>
    <h3>${rec.recommended_model_name}</h3>
    <p>
      Based on chronological out-of-time evaluation across <strong>${split.test_records.toLocaleString()}</strong> test records,
      <strong>${rec.recommended_model_name}</strong> achieves lower prediction error and higher explained variance than Random Forest.
    </p>
    <div class="audit-list" style="margin-top:14px;">
      <div class="audit-row" style="background:rgba(255,255,255,0.06);border-color:rgba(255,255,255,0.12);color:#e2e8f0;">
        <span>MAE Reduction (RF → XGBoost)</span>
        <strong>${rec.mae_improvement >= 0 ? "-" : "+"}${Math.abs(rec.mae_improvement).toFixed(4)} pct pts</strong>
      </div>
      <div class="audit-row" style="background:rgba(255,255,255,0.06);border-color:rgba(255,255,255,0.12);color:#e2e8f0;">
        <span>RMSE Reduction (RF → XGBoost)</span>
        <strong>${rec.rmse_improvement >= 0 ? "-" : "+"}${Math.abs(rec.rmse_improvement).toFixed(4)} pct pts</strong>
      </div>
      <div class="audit-row" style="background:rgba(255,255,255,0.06);border-color:rgba(255,255,255,0.12);color:#e2e8f0;">
        <span>R² Gain</span>
        <strong>+${rec.r2_improvement.toFixed(4)} (${(rec.r2_improvement * 100).toFixed(2)}%)</strong>
      </div>
      <div class="audit-row" style="background:rgba(255,255,255,0.06);border-color:rgba(255,255,255,0.12);color:#e2e8f0;">
        <span>Scheduled-Slot MAE (XGB vs RF)</span>
        <strong>${viz.scheduled_xgb_mae.toFixed(4)} vs ${viz.scheduled_rf_mae.toFixed(4)}</strong>
      </div>
    </div>
    <p style="margin-top:14px;font-size:12px;color:#94a3b8;">
      Note: Both models exceed 97% R² on the notebook's synthetic dataset because attendance rate varies uniformly between 65% and 100% of scheduled_students / capacity.
    </p>
  `;
}

/* -------------------- BOOTSTRAP -------------------- */
async function init() {
  setupNavigation();
  setupRoomFilters();
  setupDemandFilters();

  appMeta = await fetchJson("/api/meta");
  populatePredictionDropdowns();

  await Promise.all([
    loadOverview(),
    loadRoomAnalytics(),
    loadDemandAnalysis(),
    loadModelPerformance(),
  ]);

  await runPrediction();
}

init().catch((err) => {
  console.error("Dashboard initialization failed:", err);
});
