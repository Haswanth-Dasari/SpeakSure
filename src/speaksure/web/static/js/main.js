import { createSession, finishSession, getHealth, getSessionStatus, listSessions, submitAnswer } from "./api.js";
import { AudioCaptureSession, isBluetoothOrHeadset, requestMicrophonePermission, runMicrophoneTest } from "./audio.js";
import { InterviewerAudioPlayer } from "./interviewer-audio.js";
import { VadDetector } from "./vad.js";
import { announce, formatTime, renderPastSessions, updateLevelMeter } from "./ui.js";

const el = {
  setup: document.getElementById("setup"),
  form: document.getElementById("setup-form"),
  name: document.getElementById("candidate-name"),
  role: document.getElementById("role-select"),
  otherRoleWrap: document.getElementById("other-role-wrap"),
  otherRole: document.getElementById("other-role"),
  hints: document.getElementById("vocabulary-hints"),
  micSelect: document.getElementById("mic-select"),
  grantMic: document.getElementById("grant-mic"),
  testMic: document.getElementById("test-mic"),
  micMeter: document.getElementById("mic-meter"),
  micVerdict: document.getElementById("mic-verdict"),
  bluetoothWarning: document.getElementById("bluetooth-warning"),
  contentScoring: document.getElementById("content-scoring"),
  contentScoringNote: document.getElementById("content-scoring-note"),
  cameraPreview: document.getElementById("camera-preview"),
  join: document.getElementById("join-room"),
  joinWait: document.getElementById("join-wait"),
  recentSessions: document.getElementById("recent-sessions"),
  liveAnnouncer: document.getElementById("live-announcer"),
  room: document.getElementById("room"),
  roomRole: document.getElementById("room-role"),
  roomStatus: document.getElementById("room-status"),
  questionProgress: document.getElementById("question-progress"),
  questionCount: document.getElementById("question-count"),
  clock: document.getElementById("clock"),
  orb: document.getElementById("orb"),
  caption: document.getElementById("caption"),
  avatar: document.getElementById("avatar"),
  avatarRing: document.getElementById("avatar-ring"),
  roomMeter: document.getElementById("room-meter"),
  camera: document.getElementById("camera"),
  timingWarning: document.getElementById("timing-warning"),
  captions: document.getElementById("captions-toggle"),
  repeat: document.getElementById("repeat-question"),
  end: document.getElementById("end-interview"),
  dialog: document.getElementById("end-dialog"),
  confirmEnd: document.getElementById("confirm-end"),
  callStage: document.getElementById("call-stage"),
  footer: document.getElementById("room-footer"),
  preparing: document.getElementById("preparing"),
  analysisStatus: document.getElementById("analysis-status"),
  analysisProgress: document.getElementById("analysis-progress"),
  analysisElapsed: document.getElementById("analysis-elapsed"),
  analysisError: document.getElementById("analysis-error"),
  retryAnalysis: document.getElementById("retry-analysis"),
  reportView: document.getElementById("report-view"),
  reportFrame: document.getElementById("report-frame"),
  openReport: document.getElementById("open-report"),
  downloadReport: document.getElementById("download-report"),
  newInterview: document.getElementById("new-interview"),
};

const app = {
  health: null,
  modelsReady: false,
  microphoneGranted: false,
  deviceId: "",
  deviceLabel: "",
  sessionId: null,
  questionIndex: 1,
  totalQuestions: 5,
  currentLine: "",
  captionsOn: false,
  interviewer: new InterviewerAudioPlayer(),
  capture: null,
  cameraStream: null,
  vad: null,
  heardSpeech: false,
  hintShown: false,
  firstSpeechAt: null,
  answerStartedAt: 0,
  endWaitMs: 0,
  roomStartedAt: 0,
  roomTimer: null,
  answerTimer: null,
  hintTimer: null,
  analysisStartedAt: 0,
  active: false,
  orbLevel: 0,
};

const sleep = (milliseconds) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

const JOIN_DIAGNOSTICS_FLAG = "speaksure.joinDiagnostics";
let joinDiagnosticAttempt = null;

function joinDiagnosticsEnabled() {
  if (!["localhost", "127.0.0.1", "::1"].includes(window.location.hostname)) return false;
  try {
    return window.localStorage.getItem(JOIN_DIAGNOSTICS_FLAG) === "1";
  } catch {
    return false;
  }
}

function sanitizedError(error) {
  const errorName = typeof error?.name === "string" ? error.name.slice(0, 80) : "Error";
  let errorMessage = typeof error?.message === "string" ? error.message : "Unknown error";
  for (const sensitiveValue of [el.name.value.trim(), app.deviceLabel]) {
    if (sensitiveValue) errorMessage = errorMessage.split(sensitiveValue).join("[redacted]");
  }
  errorMessage = errorMessage
    .replace(/[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}/g, "[email]")
    .replace(/\b[0-9a-f]{8}-[0-9a-f-]{27,}\b/gi, "[identifier]")
    .replace(/\b[0-9a-f]{32}\b/gi, "[identifier]")
    .replace(/\bBearer\s+\S+/gi, "Bearer [redacted]")
    .replace(/https?:\/\/\S+/gi, "[url]")
    .replace(/([?&][^=\s]+)=([^&\s]+)/g, "$1=[redacted]")
    .replace(/(["']).*?\1/g, "[quoted text]")
    .replace(/[\r\n\t]/g, " ")
    .slice(0, 240);
  return { error_name: errorName, error_message: errorMessage };
}

function logJoinDiagnostic(stage, details = {}) {
  const attempt = joinDiagnosticAttempt;
  if (!attempt?.enabled) return;
  attempt.currentStage = stage;
  console.info("[SpeakSure Join diagnostic]", {
    stage,
    elapsed_ms: Math.round(performance.now() - attempt.startedAt),
    ...details,
  });
}

function setRoomState(state, message) {
  el.room.dataset.state = state;
  el.roomStatus.textContent = message;
  announce(el.liveAnnouncer, message);
  el.end.disabled = state === "thinking";
  el.confirmEnd.disabled = state === "thinking";
}

function updateCaption() {
  el.caption.textContent = app.captionsOn ? app.currentLine : "";
  el.caption.classList.toggle("hidden", !app.captionsOn || !app.currentLine);
}

function initials(name) {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "S";
  return parts.slice(0, 2).map((part) => part[0]).join("").toUpperCase();
}

function selectedLength() {
  return document.querySelector('input[name="length"]:checked')?.value || "standard";
}

function selectedRole() {
  return el.role.value === "Other" ? el.otherRole.value.trim() : el.role.value;
}

function updateJoinAvailability() {
  const roleOk = selectedRole().length > 0 && selectedRole().length <= 60;
  el.join.disabled = !app.microphoneGranted || !app.modelsReady || !roleOk || app.active;
  const pending = [];
  if (!app.microphoneGranted) pending.push("microphone access");
  if (!app.modelsReady) pending.push("speech and feedback models");
  if (!roleOk) pending.push("a role");
  if (!el.joinWait.textContent.startsWith("Models could not be loaded:")) {
    el.joinWait.textContent = pending.length
      ? `Waiting for ${pending.join(" and ")}.`
      : "Microphone and models are ready.";
  }
}

function renderModelHealth(health) {
  app.health = health;
  const components = Array.isArray(health.components) ? health.components : [];
  app.modelsReady = health.status === "ok"
    && ["stt", "tts", "spacy"].every((name) =>
      components.some((component) => component.name === name && component.status === "ready"));
  const failedComponents = components
    .filter((component) => component.status === "error")
    .map((component) => component.name);
  if (failedComponents.length) {
    el.joinWait.textContent = `Models could not be loaded: ${failedComponents.join(", ")}. Restart the server after checking its model setup.`;
  }
  if (health.llm_configured) {
    el.contentScoring.disabled = false;
    el.contentScoringNote.textContent = "";
    el.contentScoringNote.classList.add("hidden");
  } else {
    el.contentScoring.disabled = true;
    el.contentScoring.checked = false;
    el.contentScoringNote.textContent = "AI content scoring is unavailable because the server has no scoring model configured.";
    el.contentScoringNote.classList.remove("hidden");
  }
  updateJoinAvailability();
}

async function pollHealth() {
  try {
    renderModelHealth(await getHealth());
  } catch {
    app.modelsReady = false;
    el.joinWait.textContent = "The server’s model status is unavailable. Reconnecting…";
    updateJoinAvailability();
  }
}

async function loadRecentSessions() {
  try {
    renderPastSessions(el.recentSessions, await listSessions());
  } catch {
    el.recentSessions.textContent = "Recent sessions could not be loaded.";
  }
}

function populateMicrophones(devices) {
  el.micSelect.textContent = "";
  if (!devices.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No microphone found";
    el.micSelect.appendChild(option);
    el.micSelect.disabled = true;
    app.microphoneGranted = false;
    app.deviceId = "";
    app.deviceLabel = "";
    el.micVerdict.textContent = "No microphone found";
    updateJoinAvailability();
    return;
  }
  devices.forEach((device, index) => {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || `Microphone ${index + 1}`;
    if (device.deviceId === app.deviceId) option.selected = true;
    el.micSelect.appendChild(option);
  });
  el.micSelect.disabled = false;
  app.microphoneGranted = true;
  app.deviceId = el.micSelect.value;
  app.deviceLabel = el.micSelect.selectedOptions[0]?.textContent || "Unknown microphone";
  el.micVerdict.textContent = "Ready to test";
  el.testMic.disabled = false;
  el.grantMic.textContent = "Permission granted";
  el.grantMic.disabled = true;
  el.bluetoothWarning.classList.toggle("hidden", !isBluetoothOrHeadset(app.deviceLabel));
  updateJoinAvailability();
}

async function grantMicrophone() {
  el.grantMic.disabled = true;
  el.grantMic.textContent = "Requesting access…";
  try {
    const devices = await requestMicrophonePermission();
    populateMicrophones(devices);
    if (app.microphoneGranted) el.micVerdict.textContent = "Ready to test";
  } catch {
    app.microphoneGranted = false;
    el.grantMic.disabled = false;
    el.grantMic.textContent = "Allow microphone";
    el.micVerdict.textContent = "Microphone access was not granted";
    updateJoinAvailability();
  }
}

async function testMicrophone() {
  el.testMic.disabled = true;
  el.micVerdict.textContent = "Speak now…";
  try {
    const result = await runMicrophoneTest(app.deviceId, (level) => updateLevelMeter(el.micMeter, level));
    el.micVerdict.textContent = result.assessment;
  } catch {
    el.micVerdict.textContent = "Microphone test failed";
  } finally {
    el.testMic.disabled = !app.microphoneGranted;
  }
}

function updateQuestionProgress() {
  el.questionProgress.textContent = "";
  for (let index = 1; index <= app.totalQuestions; index += 1) {
    const segment = document.createElement("span");
    segment.className = "progress-segment";
    if (index < app.questionIndex) segment.classList.add("done");
    if (index === app.questionIndex) segment.classList.add("current");
    el.questionProgress.appendChild(segment);
  }
  const label = `Question ${app.questionIndex} of ${app.totalQuestions}`;
  el.questionCount.textContent = label;
  el.questionProgress.setAttribute("aria-label", label);
}

function refreshRepeatButton() {
  const canRepeat = Boolean(app.interviewer.lastBuffer)
    && app.room.dataset.state !== "speaking"
    && app.room.dataset.state !== "thinking"
    && (!app.heardSpeech || app.room.dataset.state !== "listening");
  el.repeat.disabled = !canRepeat;
}

async function startCameraPreview() {
  if (!el.cameraPreview.checked) {
    logJoinDiagnostic("camera_initialization", { status: "skipped" });
    return;
  }
  logJoinDiagnostic("camera_initialization", { status: "started" });
  try {
    app.cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    el.camera.srcObject = app.cameraStream;
    el.camera.classList.remove("hidden");
    logJoinDiagnostic("camera_initialization", { status: "completed" });
  } catch (error) {
    app.cameraStream = null;
    logJoinDiagnostic("camera_initialization", { status: "failed", ...sanitizedError(error) });
  }
}

function stopCameraPreview() {
  if (app.cameraStream) app.cameraStream.getTracks().forEach((track) => track.stop());
  app.cameraStream = null;
  el.camera.srcObject = null;
  el.camera.classList.add("hidden");
}

function releaseCapture() {
  if (app.capture) {
    app.capture.release();
    app.capture = null;
  }
  if (app.answerTimer) window.clearInterval(app.answerTimer);
  if (app.hintTimer) window.clearInterval(app.hintTimer);
  app.answerTimer = null;
  app.hintTimer = null;
}

function stopClock() {
  if (app.roomTimer) window.clearInterval(app.roomTimer);
  app.roomTimer = null;
}

async function joinRoom(event) {
  event.preventDefault();
  if (el.join.disabled) return;
  joinDiagnosticAttempt = { enabled: joinDiagnosticsEnabled(), startedAt: performance.now(), currentStage: "join_submission" };
  logJoinDiagnostic("join_submission", { status: "received" });
  logJoinDiagnostic("audio_unlock", { status: "started" });
  const unlockPromise = app.interviewer.unlock();
  app.active = true;
  updateJoinAvailability();
  el.join.textContent = "Joining…";
  try {
    await unlockPromise;
    logJoinDiagnostic("audio_unlock", { status: "completed" });
    logJoinDiagnostic("session_creation_request", { status: "started" });
    const session = await createSession({
      role: selectedRole(),
      length: selectedLength(),
      candidateName: el.name.value,
      vocabularyHints: el.hints.value.trim() || undefined,
      inputDeviceLabel: app.deviceLabel,
      constraints: {},
      onDiagnostic: (stage, details) => logJoinDiagnostic(stage, details),
    });
    app.sessionId = session.session_id;
    logJoinDiagnostic("session_id_assigned", { status: "completed", session_id_present: true });
    app.questionIndex = session.question_index;
    app.totalQuestions = session.total_main_questions;
    app.roomRole.textContent = selectedRole();
    app.avatar.textContent = initials(el.name.value);
    updateQuestionProgress();
    logJoinDiagnostic("room_view_transition", { status: "started" });
    el.setup.classList.add("hidden");
    el.room.classList.remove("hidden");
    logJoinDiagnostic("room_view_transition", { status: "completed" });
    app.roomStartedAt = performance.now();
    app.roomTimer = window.setInterval(() => {
      el.clock.textContent = formatTime((performance.now() - app.roomStartedAt) / 1000);
    }, 500);
    await startCameraPreview();
    logJoinDiagnostic("interviewer_audio_playback", { status: "started" });
    await speakInterviewerLine(session.first_question.text, session.first_question.audio_url, (stage, details) => logJoinDiagnostic(stage, details));
    logJoinDiagnostic("interviewer_audio_playback", { status: "completed" });
    await sleep(250);
    await startListening();
    logJoinDiagnostic("join_handler", {
      status: el.room.dataset.state === "error" ? "completed_with_handled_error" : "completed",
    });
  } catch (error) {
    const failedStage = joinDiagnosticAttempt?.currentStage;
    if (failedStage) logJoinDiagnostic(failedStage, { status: "failed", ...sanitizedError(error) });
    logJoinDiagnostic("join_handler", {
      status: "caught_exception",
      ...sanitizedError(error),
      failed_stage: failedStage,
    });
    if (!app.sessionId) {
      app.active = false;
      el.join.textContent = "Join the interview room";
      updateJoinAvailability();
      el.joinWait.textContent = error.message || "The interview could not be started.";
      announce(el.liveAnnouncer, el.joinWait.textContent);
    } else {
      setRoomState("error", error.message || "The interview could not be started.");
    }
  } finally {
    joinDiagnosticAttempt = null;
  }
}

async function speakInterviewerLine(text, audioUrl, onDiagnostic) {
  app.currentLine = text;
  updateCaption();
  if (app.questionIndex === 1 && !app.heardSpeech) logJoinDiagnostic("first_question_presented", { status: "presented" });
  setRoomState("speaking", "Interviewer is speaking");
  el.repeat.disabled = true;
  await app.interviewer.playUrl(audioUrl, onDiagnostic);
}

async function startListening() {
  releaseCapture();
  app.heardSpeech = false;
  app.hintShown = false;
  app.firstSpeechAt = null;
  app.endWaitMs = 0;
  app.vad = new VadDetector(16000, { noSpeechTimeout: 20 });
  app.capture = new AudioCaptureSession();
  app.answerStartedAt = performance.now();
  setRoomState("listening", "Your turn · listening");
  logJoinDiagnostic("first_listening_state", { status: "entered" });
  updateLevelMeter(el.roomMeter, 0);
  el.avatarRing.classList.remove("active");
  refreshRepeatButton();
  try {
    logJoinDiagnostic("microphone_capture", { status: "started" });
    await app.capture.start(
      app.deviceId,
      (frame) => {
        const result = app.vad.processFrame(frame);
        if (result.isSpeech && app.firstSpeechAt === null) {
          app.firstSpeechAt = performance.now();
          app.endWaitMs = app.firstSpeechAt - app.answerStartedAt;
          app.heardSpeech = true;
          el.avatarRing.classList.add("active");
          el.repeat.classList.remove("repeat-hint");
          setRoomState("listening", "Your turn · listening");
          refreshRepeatButton();
        }
        if (result.shouldStop) void finishListening(result.stopReason !== "timeout");
      },
      (level) => {
        updateLevelMeter(el.roomMeter, level);
        el.avatarRing.style.boxShadow = `0 0 0 ${Math.min(14, level * 34)}px color-mix(in srgb, var(--teal) 30%, transparent)`;
      },
    );
    logJoinDiagnostic("microphone_capture", { status: "completed" });
    app.hintTimer = window.setInterval(() => {
      const elapsed = (performance.now() - app.answerStartedAt) / 1000;
      if (!app.heardSpeech && elapsed >= 8 && !app.hintShown) {
        app.hintShown = true;
        setRoomState("listening", "Take your time. You can repeat the question.");
        el.repeat.classList.add("repeat-hint");
        refreshRepeatButton();
      }
    }, 250);
  } catch (error) {
    logJoinDiagnostic("microphone_capture", { status: "failed", ...sanitizedError(error) });
    releaseCapture();
    setRoomState("error", "Microphone access ended. Allow microphone access and try again.");
  }
}

async function finishListening(trimSilence) {
  if (!app.capture?.isRecording) return;
  if (app.answerTimer) window.clearInterval(app.answerTimer);
  if (app.hintTimer) window.clearInterval(app.hintTimer);
  app.answerTimer = null;
  app.hintTimer = null;
  el.repeat.classList.remove("repeat-hint");
  setRoomState("thinking", "Thinking about your answer…");
  el.avatarRing.classList.remove("active");
  refreshRepeatButton();
  const thinkingStartedAt = performance.now();
  try {
    const capture = app.capture;
    const recording = await capture.stop({ trimSilence });
    capture.release();
    app.capture = null;
    const result = await submitAnswer(app.sessionId, recording.wavBlob, app.endWaitMs);
    const thinkingRemaining = 600 - (performance.now() - thinkingStartedAt);
    if (thinkingRemaining > 0) await sleep(thinkingRemaining);
    if (result.timing_warning) {
      el.timingWarning.textContent = result.timing_warning;
      el.timingWarning.classList.remove("hidden");
    } else {
      el.timingWarning.classList.add("hidden");
    }

    const reply = result.interviewer;
    if (reply.action === "next_question") {
      app.questionIndex = Math.min(app.questionIndex + 1, app.totalQuestions);
      updateQuestionProgress();
    }
    await speakInterviewerLine(reply.text, reply.audio_url);
    if (result.finished || reply.action === "finish") {
      await prepareFeedback();
      return;
    }
    await sleep(250);
    await startListening();
  } catch (error) {
    releaseCapture();
    setRoomState("error", error.message || "The answer could not be sent. Please try again.");
    el.repeat.classList.remove("repeat-hint");
  }
}

async function repeatQuestion() {
  if (el.repeat.disabled || !app.interviewer.lastBuffer) return;
  if (app.room.dataset.state === "listening") {
    if (app.heardSpeech) return;
    releaseCapture();
  }
  el.repeat.classList.remove("repeat-hint");
  setRoomState("speaking", "Interviewer is repeating the question");
  el.repeat.disabled = true;
  try {
    await app.interviewer.replay();
    await sleep(250);
    await startListening();
  } catch {
    setRoomState("error", "The question could not be repeated.");
  }
}

function openEndDialog() {
  if (!el.dialog.open) el.dialog.showModal();
}

async function confirmEndInterview(event) {
  if (event) event.preventDefault();
  if (el.dialog.open) el.dialog.close();
  await prepareFeedback();
}

async function prepareFeedback() {
  releaseCapture();
  stopCameraPreview();
  stopClock();
  await app.interviewer.release();
  el.callStage.classList.add("hidden");
  el.footer.classList.add("hidden");
  el.preparing.classList.remove("hidden");
  setRoomState("thinking", "Preparing your feedback");
  el.analysisStartedAt = performance.now();
  el.analysisError.classList.add("hidden");
  el.retryAnalysis.classList.add("hidden");
  el.analysisProgress.style.width = "0%";
  el.analysisStatus.textContent = "Getting your interview ready…";
  announce(el.liveAnnouncer, "Preparing your feedback.");
  await requestAnalysis();
}

async function requestAnalysis() {
  try {
    await finishSession(app.sessionId, el.contentScoring.checked && !el.contentScoring.disabled);
    await pollAnalysis();
  } catch (error) {
    showAnalysisError(error.message || "Feedback could not be prepared.");
  }
}

const stageLabels = {
  transcribing: "Transcribing",
  delivery: "Checking delivery",
  grammar: "Checking grammar",
  content: "Scoring answer content",
  report: "Building your report",
};

async function pollAnalysis() {
  const deadline = performance.now() + 300_000;
  while (performance.now() < deadline) {
    try {
      const status = await getSessionStatus(app.sessionId);
      const elapsed = (performance.now() - app.analysisStartedAt) / 1000;
      el.analysisElapsed.textContent = `Elapsed · ${formatTime(elapsed)}`;
      if (status.stage && stageLabels[status.stage]) {
        const percent = Number.isFinite(status.progress) ? Math.round(status.progress * 100) : null;
        el.analysisStatus.textContent = percent === null
          ? stageLabels[status.stage]
          : `${stageLabels[status.stage]} · ${percent}%`;
        if (percent !== null) el.analysisProgress.style.width = `${Math.max(0, Math.min(100, percent))}%`;
      } else {
        el.analysisStatus.textContent = status.message || `Preparing your feedback · ${Math.floor(elapsed)} seconds`;
      }
      if (status.state === "ready" && status.report_url) {
        showReport(status.report_url);
        return;
      }
      if (status.state === "error") {
        showAnalysisError(status.message || "Feedback could not be prepared.");
        return;
      }
    } catch (error) {
      showAnalysisError(error.message || "The feedback status could not be checked.");
      return;
    }
    await sleep(1200);
  }
  showAnalysisError("Preparing feedback took longer than expected.");
}

function showAnalysisError(message) {
  el.analysisError.textContent = message;
  el.analysisError.classList.remove("hidden");
  el.retryAnalysis.classList.remove("hidden");
  announce(el.liveAnnouncer, message);
}

function showReport(reportUrl) {
  el.room.classList.add("hidden");
  el.reportView.classList.remove("hidden");
  el.reportFrame.src = reportUrl;
  el.openReport.href = reportUrl;
  el.downloadReport.href = reportUrl;
  announce(el.liveAnnouncer, "Your feedback report is ready.");
}

async function resetForNewInterview() {
  releaseCapture();
  stopCameraPreview();
  stopClock();
  await app.interviewer.release();
  app.interviewer = new InterviewerAudioPlayer();
  app.sessionId = null;
  app.active = false;
  app.questionIndex = 1;
  app.totalQuestions = 5;
  app.currentLine = "";
  app.room.dataset.state = "idle";
  el.room.classList.add("hidden");
  el.reportView.classList.add("hidden");
  el.callStage.classList.remove("hidden");
  el.footer.classList.remove("hidden");
  el.preparing.classList.add("hidden");
  el.reportFrame.src = "about:blank";
  el.join.textContent = "Join the interview room";
  updateCaption();
  updateJoinAvailability();
  await loadRecentSessions();
  el.setup.classList.remove("hidden");
  el.name.focus();
}

function bindEvents() {
  el.role.addEventListener("change", () => {
    const other = el.role.value === "Other";
    el.otherRoleWrap.classList.toggle("hidden", !other);
    el.otherRole.required = other;
    if (other) el.otherRole.focus();
    updateJoinAvailability();
  });
  el.otherRole.addEventListener("input", updateJoinAvailability);
  el.form.addEventListener("submit", (event) => void joinRoom(event));
  el.grantMic.addEventListener("click", () => void grantMicrophone());
  el.micSelect.addEventListener("change", () => {
    app.deviceId = el.micSelect.value;
    app.deviceLabel = el.micSelect.selectedOptions[0]?.textContent || "";
    el.bluetoothWarning.classList.toggle("hidden", !isBluetoothOrHeadset(app.deviceLabel));
    el.micVerdict.textContent = "Ready to test";
  });
  el.testMic.addEventListener("click", () => void testMicrophone());
  el.captions.addEventListener("click", () => {
    app.captionsOn = !app.captionsOn;
    el.captions.setAttribute("aria-pressed", String(app.captionsOn));
    el.captions.textContent = app.captionsOn ? "Captions on" : "Captions off";
    updateCaption();
    announce(el.liveAnnouncer, app.captionsOn ? "Captions on." : "Captions off.");
  });
  el.repeat.addEventListener("click", () => void repeatQuestion());
  el.end.addEventListener("click", openEndDialog);
  el.confirmEnd.addEventListener("click", (event) => void confirmEndInterview(event));
  el.newInterview.addEventListener("click", () => void resetForNewInterview());
  el.retryAnalysis.addEventListener("click", () => {
    el.analysisError.classList.add("hidden");
    el.retryAnalysis.classList.add("hidden");
    el.analysisStartedAt = performance.now();
    void requestAnalysis();
  });
  document.addEventListener("keydown", (event) => {
    const target = event.target;
    if (target instanceof HTMLElement && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
    if (event.key.toLowerCase() === "r" && !el.room.classList.contains("hidden")) {
      event.preventDefault();
      void repeatQuestion();
    }
    if (event.key === "Escape" && !el.room.classList.contains("hidden") && !el.preparing.classList.contains("hidden")) {
      event.preventDefault();
    } else if (event.key === "Escape" && !el.room.classList.contains("hidden")) {
      event.preventDefault();
      openEndDialog();
    }
  });
}

async function init() {
  bindEvents();
  updateJoinAvailability();
  await Promise.all([pollHealth(), loadRecentSessions()]);
  window.setInterval(() => {
    if (!app.modelsReady) void pollHealth();
  }, 1200);
}

function animateCallRoom() {
  app.orbLevel = app.orbLevel * 0.78 + Math.min(1, app.interviewer.getLevel() * 8) * 0.22;
  el.orb.querySelectorAll("i").forEach((layer, index) => {
    const scale = 0.62 + app.orbLevel * 0.38 * (1 - index * 0.16);
    layer.style.transform = `scale(${scale})`;
  });
  window.requestAnimationFrame(animateCallRoom);
}

window.addEventListener("DOMContentLoaded", () => void init());
window.requestAnimationFrame(animateCallRoom);
