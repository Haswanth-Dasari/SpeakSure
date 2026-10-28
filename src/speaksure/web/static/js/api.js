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

export async function uploadResume(file) {
  const response = await fetch("/api/resume", {
    method: "POST",
    headers: {
      "Content-Type": "application/octet-stream",
      "X-Filename": encodeURIComponent(file.name),
    },
    body: file,
  });
  let data = {};
  try {
    data = await response.json();
  } catch {
    // Fall through to the generic message below.
  }
  if (!response.ok) throw new Error(data.message || `The resume could not be read (${response.status}).`);
  return data;
}

export async function createSession({
  role,
  length,
  candidateName,
  vocabularyHints,
  resumeProfile,
  inputDeviceLabel,
  constraints,
  onDiagnostic,
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
  if (resumeProfile) payload.resume_profile = resumeProfile;

  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 60000);
  let response;
  try {
    response = await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error("The server did not respond within 60 seconds. Check the server window for errors.");
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
  onDiagnostic?.("session_creation_response", {
    status: "received",
    http_status: response.status,
  });
  onDiagnostic?.("session_response_body", { status: "parsing_started" });
  const data = await response.json();
  onDiagnostic?.("session_creation_response_fields", {
    status: "checked",
    session_id_present: typeof data?.session_id === "string",
    question_index_present: Number.isFinite(data?.question_index),
    total_main_questions_present: Number.isFinite(data?.total_main_questions),
    first_question_present: Boolean(data?.first_question && typeof data.first_question === "object"),
    first_question_id_present: typeof data?.first_question?.id === "string",
    first_question_text_present: typeof data?.first_question?.text === "string",
    first_question_audio_url_present: typeof data?.first_question?.audio_url === "string",
  });
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
