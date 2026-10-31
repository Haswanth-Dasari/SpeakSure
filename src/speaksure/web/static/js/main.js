import { createSession, finishSession, getHealth, getSessionStatus, listSessions, submitAnswer, uploadResume } from "./api.js";
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
  resumeFile: document.getElementById("resume-file"),
  resumeAnalyze: document.getElementById("resume-analyze"),
  resumeStatus: document.getElementById("resume-status"),
  resumeReview: document.getElementById("resume-review"),
  resumeHeadline: document.getElementById("resume-headline"),
  resumeSkills: document.getElementById("resume-skills"),
  resumeProjects: document.getElementById("resume-projects"),
  resumeExperience: document.getElementById("resume-experience"),
  resumeEducation: document.getElementById("resume-education"),
  resumeAchievements: document.getElementById("resume-achievements"),
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
    && el.room.dataset.state !== "speaking"
    && el.room.dataset.state !== "thinking"
    && (!app.heardSpeech || el.room.dataset.state !== "listening");
  el.repeat.disabled = !canRepeat;
}

async function startCameraPreview() {
  if (!el.cameraPreview.checked) return;
  try {
    app.cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    el.camera.srcObject = app.cameraStream;
    el.camera.classList.remove("hidden");
  } catch {
    app.cameraStream = null;
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

function splitLines(text) {
  return text.split("\n").map((line) => line.trim()).filter(Boolean);
}

function fillResumeForm(profile) {
  el.resumeHeadline.value = profile.headline || "";
  el.resumeSkills.value = (profile.skills || []).join("\n");
  el.resumeProjects.value = (profile.projects || [])
    .map((project) => {
      const tools = (project.technologies || []).length ? ` (uses ${project.technologies.join(", ")})` : "";
      return `${project.name}${project.summary ? " - " + project.summary : ""}${tools}`;
    })
    .join("\n");
  el.resumeExperience.value = (profile.experience || []).join("\n");
  el.resumeEducation.value = (profile.education || []).join("\n");
  el.resumeAchievements.value = (profile.achievements || []).join("\n");
  el.resumeReview.classList.remove("hidden");
}

function readResumeProfile() {
  const projects = splitLines(el.resumeProjects.value).map((line) => {
    const [name, ...rest] = line.split(" - ");
    return { name: name.trim(), summary: rest.join(" - ").trim(), technologies: [] };
  });
  const profile = {
    headline: el.resumeHeadline.value.trim(),
    skills: splitLines(el.resumeSkills.value),
    projects,
    experience: splitLines(el.resumeExperience.value),
    education: splitLines(el.resumeEducation.value),
    achievements: splitLines(el.resumeAchievements.value),
  };
  const hasContent = profile.headline || projects.length || Object.values(profile).some((v) => Array.isArray(v) && v.length);
  return hasContent ? profile : undefined;
}

async function analyseResume() {
  const file = el.resumeFile.files?.[0];
  if (!file) return;
  el.resumeAnalyze.disabled = true;
  el.resumeStatus.textContent = "Reading your resume. This takes a few seconds…";
  try {
    const data = await uploadResume(file);
    fillResumeForm(data.profile);
    const removed = data.info ? (data.info.emails || 0) + (data.info.phones || 0) + (data.info.links || 0) : 0;
    el.resumeStatus.textContent =
      data.warning ||
      `Done. Please check the details below${removed ? ` (${removed} contact detail${removed === 1 ? "" : "s"} removed)` : ""}.`;
  } catch (error) {
    el.resumeStatus.textContent = error.message || "The resume could not be read.";
  } finally {
    el.resumeAnalyze.disabled = false;
  }
}

el.resumeFile.addEventListener("change", () => {
  el.resumeAnalyze.disabled = !el.resumeFile.files?.length;
  el.resumeStatus.textContent = "";
});
el.resumeAnalyze.addEventListener("click", analyseResume);

async function joinRoom(event) {
  event.preventDefault();
  if (el.join.disabled) return;
  const unlockPromise = app.interviewer.unlock();
  app.active = true;
  updateJoinAvailability();
  el.join.textContent = "Joining…";
  try {
    // Start the session request immediately; audio unlock is awaited afterwards
    // so a slow audio device cannot freeze the join button.
    const sessionPromise = createSession({
      role: selectedRole(),
      length: selectedLength(),
      candidateName: el.name.value,
      // The candidate's own name is a likely mis-heard word, so add it as a hint.
      resumeProfile: readResumeProfile(),
      vocabularyHints: [el.name.value.trim().slice(0, 40), el.hints.value.trim()]
        .filter(Boolean)
        .join(", ") || undefined,
      inputDeviceLabel: app.deviceLabel,
      constraints: {},
    });
    unlockPromise.catch(() => {});
    const session = await sessionPromise;
    await unlockPromise;
    app.sessionId = session.session_id;
    app.questionIndex = session.question_index;
    app.totalQuestions = session.total_main_questions;
    el.roomRole.textContent = selectedRole();
    el.avatar.textContent = initials(el.name.value);
    updateQuestionProgress();
    el.setup.classList.add("hidden");
    el.room.classList.remove("hidden");
    app.roomStartedAt = performance.now();
    app.roomTimer = window.setInterval(() => {
      el.clock.textContent = formatTime((performance.now() - app.roomStartedAt) / 1000);
    }, 500);
    await startCameraPreview();
    await speakInterviewerLine(session.first_question.text, session.first_question.audio_url);
    await sleep(250);
    await startListening();
  } catch (error) {
    if (!app.sessionId) {
      app.active = false;
      el.join.textContent = "Join the interview room";
      updateJoinAvailability();
      el.joinWait.textContent = error.message || "The interview could not be started.";
      announce(el.liveAnnouncer, el.joinWait.textContent);
    } else if (el.room.classList.contains("hidden")) {
      // Failed after the session was created but before the room was shown:
      // the room's status line is invisible, so report on the setup screen.
      app.sessionId = null;
      app.active = false;
      el.join.textContent = "Join the interview room";
      updateJoinAvailability();
      el.joinWait.textContent = error.message || "The interview could not be started.";
      announce(el.liveAnnouncer, el.joinWait.textContent);
    } else {
      setRoomState("error", error.message || "The interview could not be started.");
    }
  }
}

async function speakInterviewerLine(text, audioUrl) {
  app.currentLine = text;
  updateCaption();
  setRoomState("speaking", "Interviewer is speaking");
  el.repeat.disabled = true;
  await app.interviewer.playUrl(audioUrl);
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
  updateLevelMeter(el.roomMeter, 0);
  el.avatarRing.classList.remove("active");
  refreshRepeatButton();
  try {
    await app.capture.start(
      app.deviceId,
      (frame) => {
        // Frames arrive at the browser's audio rate (often 48 kHz), not 16 kHz.
        // The detector converts samples to seconds, so it must know the real rate.
        const contextRate = app.capture?.audioContext?.sampleRate;
        if (contextRate && app.vad.sampleRate !== contextRate) app.vad.sampleRate = contextRate;
        const result = app.vad.processFrame(frame);
        // Thinking time = first delivered audio frame to the start of speech that
        // the detector later confirmed (not the first noisy frame, and not the
        // time the microphone took to open).
        if (app.vad.speechDetected && app.endWaitMs === 0 && app.vad.speechOnsetSec !== null) {
          app.endWaitMs = Math.max(1, app.vad.speechOnsetSec * 1000);
        }
        if (result.isSpeech && app.firstSpeechAt === null) {
          app.firstSpeechAt = performance.now();
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
    app.hintTimer = window.setInterval(() => {
      const elapsed = (performance.now() - app.answerStartedAt) / 1000;
      if (!app.heardSpeech && elapsed >= 8 && !app.hintShown) {
        app.hintShown = true;
        setRoomState("listening", "Take your time. You can repeat the question.");
        el.repeat.classList.add("repeat-hint");
        refreshRepeatButton();
      }
    }, 250);
  } catch {
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
  if (el.room.dataset.state === "listening") {
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
  app.analysisStartedAt = performance.now();
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
  el.room.dataset.state = "idle";
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
    app.analysisStartedAt = performance.now();
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
