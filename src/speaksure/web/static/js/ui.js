export function announce(element, text) {
  if (element) element.textContent = text;
}

export function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

export function renderPastSessions(container, sessions) {
  if (!container) return;
  container.textContent = "";
  if (!sessions.length) {
    const empty = document.createElement("p");
    empty.textContent = "No recent sessions yet.";
    container.appendChild(empty);
    return;
  }
  sessions.slice(-5).reverse().forEach((session) => {
    const row = document.createElement("div");
    row.className = "recent-item";
    const details = document.createElement("span");
    const started = session.started ? new Date(session.started).toLocaleString() : "Date unavailable";
    details.textContent = `${started} · ${session.number_of_turns} answers`;
    row.appendChild(details);
    if (session.report_available) {
      const link = document.createElement("a");
      link.href = `/report/${encodeURIComponent(session.id)}`;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = "Open report";
      row.appendChild(link);
    } else {
      const pending = document.createElement("span");
      pending.textContent = "In progress";
      row.appendChild(pending);
    }
    container.appendChild(row);
  });
}

export function updateLevelMeter(meter, level) {
  if (!meter) return;
  const bars = meter.querySelectorAll("span");
  bars.forEach((bar, index) => {
    const shape = 1 - Math.abs(index - (bars.length - 1) / 2) / Math.max(1, bars.length / 2);
    const height = 4 + Math.max(0, Math.min(1, level * 10)) * 18 * shape;
    bar.style.height = `${height}px`;
  });
}
