const SESSION_ID_PATTERN = /^[0-9a-f]{32}$/;

export function isValidSessionId(id) {
  return typeof id === "string" && SESSION_ID_PATTERN.test(id);
}

export async function getHealth() {
  const response = await fetch("/api/health");
  if (!response.ok) throw new Error(`Health check failed with status ${response.status}`);
  return response.json();
}

export async function listSessions() {
  const response = await fetch("/api/sessions");
  if (!response.ok) throw new Error(`Failed to list sessions (status ${response.status})`);
  const data = await response.json();
  return Array.isArray(data.sessions) ? data.sessions : [];
}

export async function createSession({
  role,
  length,
  candidateName,
  vocabularyHints,
  inputDeviceLabel,
  constraints,
}) {
  const payload = {
    role,
    length,
    candidate_name: candidateName?.trim() || null,
    client: {
      input_device_label: inputDeviceLabel || "Unknown microphone",
      constraints: constraints || {},
    },
  };
  if (vocabularyHints) payload.vocabulary_hints = vocabularyHints;

  const response = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.message || `Failed to create session (${response.status})`);
  if (!isValidSessionId(data.session_id)) throw new Error("Invalid session identifier received from server.");
  return data;
}

function uploadWav(sessionId, wavData) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    const startedAt = performance.now();
    let uploadMs = 0;
    request.open("POST", `/api/sessions/${sessionId}/answer`);
    request.setRequestHeader("Content-Type", "audio/wav");
    request.upload.addEventListener("loadend", () => {
      uploadMs = performance.now() - startedAt;
    });
    request.addEventListener("load", () => {
      let data;
      try {
        data = JSON.parse(request.responseText);
      } catch {
        reject(new Error("The server returned an invalid answer response."));
        return;
      }
      if (request.status < 200 || request.status >= 300) {
        const error = new Error(data.message || `Upload failed with status ${request.status}`);
        error.statusCode = request.status;
        error.errorCode = data.error;
        reject(error);
        return;
      }
      resolve({ data, uploadMs });
    });
    request.addEventListener("error", () => reject(new Error("The answer could not be uploaded.")));
    request.send(wavData);
  });
}

async function saveClientTiming(sessionId, turnIndex, uploadMs, endWaitMs) {
  const response = await fetch(`/api/sessions/${sessionId}/turns/${turnIndex}/timing`, {
    method: "PATCH",
    headers: {
      "X-Client-Upload-Ms": String(uploadMs),
      "X-Client-End-Wait-Ms": String(endWaitMs),
    },
  });
  if (!response.ok) throw new Error(`Timing update failed with status ${response.status}`);
}

export async function submitAnswer(sessionId, wavData, endWaitMs = 0) {
  if (!isValidSessionId(sessionId)) throw new Error("Invalid session identifier.");
  const { data, uploadMs } = await uploadWav(sessionId, wavData);
  if (data.turn_recorded) {
    try {
      await saveClientTiming(sessionId, data.turn_index, uploadMs, endWaitMs);
    } catch {
      data.timing_warning = "Turn timing could not be saved.";
    }
  }
  return data;
}

export async function finishSession(sessionId, includeContentScoring = false) {
  if (!isValidSessionId(sessionId)) throw new Error("Invalid session identifier.");
  const response = await fetch(`/api/sessions/${sessionId}/finish`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content: Boolean(includeContentScoring) }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.message || `Failed to finish session (${response.status})`);
  return data;
}

export async function getSessionStatus(sessionId) {
  if (!isValidSessionId(sessionId)) throw new Error("Invalid session identifier.");
  const response = await fetch(`/api/sessions/${sessionId}/status`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.message || `Failed to fetch session status (${response.status})`);
  return data;
}
