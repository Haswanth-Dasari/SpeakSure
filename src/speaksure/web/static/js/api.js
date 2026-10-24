/**
 * API client module for SpeakSure local interview backend.
 * Validates session identifiers and formats requests according to contract.
 */

const SESSION_ID_PATTERN = /^[0-9a-f]{32}$/;

/**
 * Validate that a session ID consists of exactly 32 lowercase hexadecimal characters.
 *
 * @param {string} id
 * @returns {boolean}
 */
export function isValidSessionId(id) {
  return typeof id === "string" && SESSION_ID_PATTERN.test(id);
}

/**
 * Fetch health status and model configuration.
 *
 * @returns {Promise<{ status: string, stt_model: string, stt_device_setting: string, tts_voice: string, llm_configured: boolean }>}
 */
export async function getHealth() {
  const response = await fetch("/api/health");
  if (!response.ok) {
    throw new Error(`Health check failed with status ${response.status}`);
  }
  return response.json();
}

/**
 * List past sessions.
 *
 * @returns {Promise<Array<{ id: string, started: string, number_of_turns: number, report_available: boolean }>>}
 */
export async function listSessions() {
  const response = await fetch("/api/sessions");
  if (!response.ok) {
    throw new Error(`Failed to list sessions (status ${response.status})`);
  }
  const data = await response.json();
  return Array.isArray(data.sessions) ? data.sessions : [];
}

/**
 * Create a new interview session.
 *
 * @param {object} params
 * @param {number} params.questions - Number of questions (3-10).
 * @param {string} [params.candidateName] - Optional candidate name.
 * @param {string[]|string} [params.vocabularyHints] - Optional vocabulary hints.
 * @param {string} params.inputDeviceLabel - Label of microphone used.
 * @param {object} params.constraints - Track constraints applied.
 * @returns {Promise<{ session_id: string, first_question: { id: string, text: string, audio_url: string }, question_index: number, total_main_questions: number }>}
 */
export async function createSession({
  questions,
  candidateName,
  vocabularyHints,
  inputDeviceLabel,
  constraints,
}) {
  const payload = {
    questions,
    candidate_name: candidateName?.trim() || null,
    client: {
      input_device_label: inputDeviceLabel || "Unknown microphone",
      constraints: constraints || {},
    },
  };

  if (vocabularyHints) {
    payload.vocabulary_hints = vocabularyHints;
  }

  const response = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.message || `Failed to create session (${response.status})`);
  }

  if (!isValidSessionId(data.session_id)) {
    throw new Error("Invalid session identifier received from server.");
  }

  return data;
}

/**
 * Upload one answer recording as raw audio/wav.
 *
 * @param {string} sessionId
 * @param {ArrayBuffer|Blob} wavData
 * @returns {Promise<{ turn_index: number, transcript: string, words_count: number, flags: object, interviewer: { action: string, text: string, audio_url: string }, finished: boolean }>}
 */
export async function submitAnswer(sessionId, wavData) {
  if (!isValidSessionId(sessionId)) {
    throw new Error("Invalid session identifier.");
  }

  const response = await fetch(`/api/sessions/${sessionId}/answer`, {
    method: "POST",
    headers: { "Content-Type": "audio/wav" },
    body: wavData,
  });

  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.message || `Upload failed with status ${response.status}`);
    error.statusCode = response.status;
    error.errorCode = data.error;
    throw error;
  }

  return data;
}

/**
 * Finish the interview session and request report generation.
 *
 * @param {string} sessionId
 * @param {boolean} includeContentScoring
 * @returns {Promise<{ status: string }>}
 */
export async function finishSession(sessionId, includeContentScoring = false) {
  if (!isValidSessionId(sessionId)) {
    throw new Error("Invalid session identifier.");
  }

  const response = await fetch(`/api/sessions/${sessionId}/finish`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content: Boolean(includeContentScoring) }),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.message || `Failed to finish session (${response.status})`);
  }
  return data;
}

/**
 * Poll session status during analysis.
 *
 * @param {string} sessionId
 * @returns {Promise<{ state: string, message: string, report_url?: string }>}
 */
export async function getSessionStatus(sessionId) {
  if (!isValidSessionId(sessionId)) {
    throw new Error("Invalid session identifier.");
  }

  const response = await fetch(`/api/sessions/${sessionId}/status`);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.message || `Failed to fetch session status (${response.status})`);
  }
  return data;
}

