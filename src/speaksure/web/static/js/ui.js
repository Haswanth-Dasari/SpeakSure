/**
 * UI manipulation module for SpeakSure web application.
 * Enforces textContent usage for all server-supplied strings and manages accessible status updates.
 */

/**
 * Update an aria-live region with a status message.
 *
 * @param {HTMLElement} element
 * @param {string} text
 */
export function announce(element, text) {
  if (element) {
    element.textContent = text;
  }
}

/**
 * Format elapsed seconds into mm:ss display.
 *
 * @param {number} seconds
 * @returns {string}
 */
export function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

/**
 * Switch active visible screen.
 *
 * @param {"setup"|"interview"|"results"} screenId
 */
export function showScreen(screenId) {
  const screens = {
    setup: document.getElementById("screen-setup"),
    interview: document.getElementById("screen-interview"),
    results: document.getElementById("screen-results"),
  };

  Object.entries(screens).forEach(([id, element]) => {
    if (element) {
      if (id === screenId) {
        element.classList.remove("hidden");
        element.removeAttribute("aria-hidden");
      } else {
        element.classList.add("hidden");
        element.setAttribute("aria-hidden", "true");
      }
    }
  });

  window.scrollTo({ top: 0, behavior: "smooth" });
}

/**
 * Render past sessions into table container.
 *
 * @param {HTMLElement} container
 * @param {Array<{ id: string, started: string, number_of_turns: number, report_available: boolean }>} sessions
 */
export function renderPastSessions(container, sessions) {
  if (!container) return;
  container.textContent = "";

  if (!sessions || sessions.length === 0) {
    const emptyMsg = document.createElement("p");
    emptyMsg.className = "text-muted";
    emptyMsg.textContent = "No previous sessions found.";
    container.appendChild(emptyMsg);
    return;
  }

  const table = document.createElement("table");
  table.className = "sessions-table";

  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  ["Session", "Started", "Turns", "Report"].forEach((title) => {
    const th = document.createElement("th");
    th.textContent = title;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  sessions.forEach((s) => {
    const row = document.createElement("tr");

    const idCell = document.createElement("td");
    idCell.className = "monospace";
    idCell.textContent = s.id.substring(0, 8);
    row.appendChild(idCell);

    const startedCell = document.createElement("td");
    const dateStr = s.started ? new Date(s.started).toLocaleString() : "Unknown";
    startedCell.textContent = dateStr;
    row.appendChild(startedCell);

    const turnsCell = document.createElement("td");
    turnsCell.textContent = String(s.number_of_turns ?? 0);
    row.appendChild(turnsCell);

    const actionCell = document.createElement("td");
    if (s.report_available) {
      const link = document.createElement("a");
      link.href = `/report/${encodeURIComponent(s.id)}`;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.className = "link-button";
      link.textContent = "Open report";
      actionCell.appendChild(link);
    } else {
      const pending = document.createElement("span");
      pending.className = "text-muted";
      pending.textContent = "Pending";
      actionCell.appendChild(pending);
    }
    row.appendChild(actionCell);

    tbody.appendChild(row);
  });
  table.appendChild(tbody);
  container.appendChild(table);
}

/**
 * Update audio level meter visualization.
 *
 * @param {HTMLElement} meterBar
 * @param {number} rmsLevel - RMS level between 0.0 and 1.0.
 */
export function updateLevelMeter(meterBar, rmsLevel) {
  if (!meterBar) return;
  // Boost lower levels visually using square root curve
  const percentage = Math.min(100, Math.max(0, Math.round(Math.sqrt(rmsLevel) * 100)));
  meterBar.style.width = `${percentage}%`;

  if (rmsLevel > 0.25) {
    meterBar.classList.add("meter-danger");
    meterBar.classList.remove("meter-normal");
  } else {
    meterBar.classList.add("meter-normal");
    meterBar.classList.remove("meter-danger");
  }
}

