/**
 * Main application orchestrator for SpeakSure interview client.
 * Coordinates Setup, Interview, and Results screens with an accessible state machine.
 */

import {
  getHealth,
  listSessions,
  createSession,
  submitAnswer,
  finishSession,
  getSessionStatus,
} from "./api.js";
import {
  AudioCaptureSession,
  requestMicrophonePermission,
  enumerateMicrophones,
  isBluetoothOrHeadset,
  runMicrophoneTest,
} from "./audio.js";
import { VadDetector } from "./vad.js";
import {
  showScreen,
  announce,
  formatTime,
  renderPastSessions,
  updateLevelMeter,
} from "./ui.js";

// Application State
const appState = {
  health: null,
  selectedDeviceId: "",
  selectedDeviceLabel: "",
  activeSessionId: null,
  questionIndex: 1,
  totalQuestions: 5,
  isFollowUp: false,
  includeContentScoring: false,
  reviewBeforeSending: false,
  candidateName: "",
  vocabularyHints: null,
  captureSession: null,
  timerInterval: null,
  recordingStartTime: 0,
  lastRecordedWav: null,
  lastRecordedBlob: null,
  interviewerAudio: new Audio(),
  pollTimer: null,
};

// DOM Elements
const el = {
  liveAnnouncer: document.getElementById("live-announcer"),

  // Setup Screen
  setupForm: document.getElementById("setup-form"),
  questionsInput: document.getElementById("questions-count"),
  nameInput: document.getElementById("candidate-name"),
  hintsInput: document.getElementById("vocabulary-hints"),
  contentScoringCheck: document.getElementById("content-scoring-check"),
  contentScoringNote: document.getElementById("content-scoring-note"),
  reviewCheck: document.getElementById("review-check"),
  grantMicBtn: document.getElementById("grant-mic-btn"),
  micSelect: document.getElementById("mic-select"),
  bluetoothWarning: document.getElementById("bluetooth-warning"),
  testMicBtn: document.getElementById("test-mic-btn"),
  testMeterContainer: document.getElementById("test-meter-container"),
  testMeterBar: document.getElementById("test-meter-bar"),
  testAssessment: document.getElementById("test-assessment"),
  testAudioPlayback: document.getElementById("test-audio-playback"),
  sttStatusBadge: document.getElementById("stt-status-badge"),
  startInterviewBtn: document.getElementById("start-interview-btn"),
  pastSessionsContainer: document.getElementById("past-sessions-container"),

  // Interview Screen
  turnIndicator: document.getElementById("turn-indicator"),
  questionText: document.getElementById("question-text"),
  statusBadge: document.getElementById("interview-status-badge"),
  playQuestionBtn: document.getElementById("play-question-btn"),
  listeningPanel: document.getElementById("listening-panel"),
  listeningMeterBar: document.getElementById("listening-meter-bar"),
  timerDisplay: document.getElementById("timer-display"),
  stopAnsweringBtn: document.getElementById("stop-answering-btn"),
  endEarlyBtn: document.getElementById("end-early-btn"),
  noSpeechNotice: document.getElementById("no-speech-notice"),
  retryNoSpeechBtn: document.getElementById("retry-no-speech-btn"),
  reviewPanel: document.getElementById("review-panel"),
  reviewAudio: document.getElementById("review-audio"),
  sendAnswerBtn: document.getElementById("send-answer-btn"),
  rerecordBtn: document.getElementById("rerecord-btn"),
  uploadErrorNotice: document.getElementById("upload-error-notice"),
  uploadErrorMessage: document.getElementById("upload-error-message"),
  retryUploadBtn: document.getElementById("retry-upload-btn"),
  transcriptCard: document.getElementById("transcript-card"),
  transcriptText: document.getElementById("transcript-text"),
  transcriptFlags: document.getElementById("transcript-flags"),

  // Results Screen
  resultsStatus: document.getElementById("results-status"),
  resultsProgressBar: document.getElementById("results-progress-bar"),
  reportContainer: document.getElementById("report-container"),
  reportFrame: document.getElementById("report-frame"),
  resultsActions: document.getElementById("results-actions"),
  openNewTabBtn: document.getElementById("open-new-tab-btn"),
  newInterviewBtn: document.getElementById("new-interview-btn"),
  resultsErrorNotice: document.getElementById("results-error-notice"),
  resultsErrorMessage: document.getElementById("results-error-message"),
  retryResultsBtn: document.getElementById("retry-results-btn"),
};

// -------------------------------------------------------------
// INITIALIZATION
// -------------------------------------------------------------
async function initApp() {
  showScreen("setup");
  announce(el.liveAnnouncer, "SpeakSure setup loaded.");

  // Check remembered mic device
  try {
    const savedDeviceId = localStorage.getItem("speaksure_selected_mic");
    if (savedDeviceId) {
      appState.selectedDeviceId = savedDeviceId;
    }
  } catch {
    // LocalStorage might be restricted
  }

  // Fetch health diagnostics
  try {
    const health = await getHealth();
    appState.health = health;
    el.sttStatusBadge.textContent = `STT: ${health.stt_model} (${health.status === "ok" ? "Ready" : "Loading…"})`;
    if (!health.llm_configured) {
      el.contentScoringCheck.disabled = true;
      el.contentScoringCheck.checked = false;
      el.contentScoringNote.textContent = "AI content scoring requires Groq API key configuration on the server.";
      el.contentScoringNote.classList.remove("hidden");
    } else {
      el.contentScoringCheck.disabled = false;
      el.contentScoringNote.classList.add("hidden");
    }
  } catch (err) {
    el.sttStatusBadge.textContent = "STT: Offline";
  }

  // Load past sessions
  loadPastSessionsList();

  // Attach event handlers
  setupEventHandlers();
}

async function loadPastSessionsList() {
  try {
    const sessions = await listSessions();
    renderPastSessions(el.pastSessionsContainer, sessions);
  } catch {
    renderPastSessions(el.pastSessionsContainer, []);
  }
}

// -------------------------------------------------------------
// SETUP HANDLERS
// -------------------------------------------------------------
function setupEventHandlers() {
  // Microphone permission & selection
  el.grantMicBtn.addEventListener("click", async () => {
    try {
      el.grantMicBtn.disabled = true;
      el.grantMicBtn.textContent = "Requesting permission…";
      const devices = await requestMicrophonePermission();
      populateMicrophoneList(devices);
      el.grantMicBtn.textContent = "Permission granted";
    } catch (err) {
      el.grantMicBtn.disabled = false;
      el.grantMicBtn.textContent = "Grant microphone";
      alert("Microphone permission was denied. Please allow microphone access in your browser settings.");
    }
  });

  el.micSelect.addEventListener("change", () => {
    const selectedOption = el.micSelect.options[el.micSelect.selectedIndex];
    appState.selectedDeviceId = el.micSelect.value;
    appState.selectedDeviceLabel = selectedOption ? selectedOption.text : "";
    try {
      localStorage.setItem("speaksure_selected_mic", appState.selectedDeviceId);
    } catch {}
    checkBluetoothWarning(appState.selectedDeviceLabel);
    checkCanStart();
  });

  // Test microphone button (3 seconds)
  el.testMicBtn.addEventListener("click", async () => {
    el.testMicBtn.disabled = true;
    el.testAssessment.textContent = "Recording 3 seconds… please speak clearly.";
    el.testAssessment.className = "test-assessment-badge";
    el.testMeterContainer.classList.remove("hidden");
    el.testAudioPlayback.classList.add("hidden");

    try {
      const testResult = await runMicrophoneTest(appState.selectedDeviceId, (rms) => {
        updateLevelMeter(el.testMeterBar, rms);
      });

      el.testAssessment.textContent = testResult.assessment;
      if (testResult.assessment === "Good level") {
        el.testAssessment.classList.add("assessment-good");
      } else {
        el.testAssessment.classList.add("assessment-warn");
      }

      el.testAudioPlayback.src = URL.createObjectURL(testResult.wavBlob);
      el.testAudioPlayback.classList.remove("hidden");
    } catch (err) {
      el.testAssessment.textContent = "Test failed: " + (err.message || "Microphone error");
      el.testAssessment.classList.add("assessment-warn");
    } finally {
      el.testMicBtn.disabled = false;
      updateLevelMeter(el.testMeterBar, 0);
    }
  });

  // Start interview form submit
  el.setupForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    await handleStartInterview();
  });

  // Interview actions
  el.playQuestionBtn.addEventListener("click", () => {
    el.playQuestionBtn.classList.add("hidden");
    appState.interviewerAudio.play().catch(() => {
      // If still blocked, advance directly to listening
      startListeningTurn();
    });
  });

  el.stopAnsweringBtn.addEventListener("click", () => {
    handleStopAnswering();
  });

  el.retryNoSpeechBtn.addEventListener("click", () => {
    el.noSpeechNotice.classList.add("hidden");
    startListeningTurn();
  });

  el.retryUploadBtn.addEventListener("click", async () => {
    el.uploadErrorNotice.classList.add("hidden");
    if (appState.lastRecordedBlob) {
      await uploadAnswerBlob(appState.lastRecordedBlob);
    }
  });

  el.sendAnswerBtn.addEventListener("click", async () => {
    el.reviewPanel.classList.add("hidden");
    if (appState.lastRecordedBlob) {
      await uploadAnswerBlob(appState.lastRecordedBlob);
    }
  });

  el.rerecordBtn.addEventListener("click", () => {
    el.reviewPanel.classList.add("hidden");
    startListeningTurn();
  });

  el.endEarlyBtn.addEventListener("click", () => {
    const confirmEnd = window.confirm(
      "Are you sure you want to end this interview early? Your answers so far will be saved and analyzed."
    );
    if (confirmEnd) {
      handleEndInterviewEarly();
    }
  });

  // Results actions
  el.newInterviewBtn.addEventListener("click", () => {
    resetToSetup();
  });

  el.retryResultsBtn.addEventListener("click", () => {
    el.resultsErrorNotice.classList.add("hidden");
    pollSessionStatus(appState.activeSessionId);
  });
}

function populateMicrophoneList(devices) {
  el.micSelect.innerHTML = "";
  if (devices.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No microphone found";
    el.micSelect.appendChild(opt);
    el.micSelect.disabled = true;
    el.startInterviewBtn.disabled = true;
    return;
  }

  devices.forEach((device, index) => {
    const opt = document.createElement("option");
    opt.value = device.deviceId;
    opt.textContent = device.label || `Microphone ${index + 1}`;
    if (device.deviceId === appState.selectedDeviceId) {
      opt.selected = true;
    }
    el.micSelect.appendChild(opt);
  });

  el.micSelect.disabled = false;
  const currentOption = el.micSelect.options[el.micSelect.selectedIndex];
  appState.selectedDeviceId = el.micSelect.value;
  appState.selectedDeviceLabel = currentOption ? currentOption.text : "";
  checkBluetoothWarning(appState.selectedDeviceLabel);
  checkCanStart();
}

function checkBluetoothWarning(label) {
  if (isBluetoothOrHeadset(label)) {
    el.bluetoothWarning.classList.remove("hidden");
  } else {
    el.bluetoothWarning.classList.add("hidden");
  }
}

function checkCanStart() {
  const hasMic = Boolean(appState.selectedDeviceId || el.micSelect.value);
  el.startInterviewBtn.disabled = !hasMic;
  el.testMicBtn.disabled = !hasMic;
}

// -------------------------------------------------------------
// INTERVIEW FLOW
// -------------------------------------------------------------
async function handleStartInterview() {
  const questionsCount = parseInt(el.questionsInput.value, 10) || 5;
  const candidateName = el.nameInput.value.trim();
  const rawHints = el.hintsInput.value.trim();
  const includeContent = el.contentScoringCheck.checked && !el.contentScoringCheck.disabled;
  const reviewBeforeSending = el.reviewCheck.checked;

  appState.totalQuestions = questionsCount;
  appState.candidateName = candidateName;
  appState.includeContentScoring = includeContent;
  appState.reviewBeforeSending = reviewBeforeSending;
  appState.vocabularyHints = rawHints ? rawHints : null;

  el.startInterviewBtn.disabled = true;
  el.startInterviewBtn.textContent = "Creating session…";

  try {
    const sessionData = await createSession({
      questions: questionsCount,
      candidateName: candidateName || undefined,
      vocabularyHints: appState.vocabularyHints || undefined,
      inputDeviceLabel: appState.selectedDeviceLabel,
      constraints: {},
    });

    appState.activeSessionId = sessionData.session_id;
    appState.questionIndex = sessionData.question_index;
    appState.totalQuestions = sessionData.total_main_questions;
    appState.isFollowUp = false;

    // Transition to interview screen
    showScreen("interview");
    presentQuestion(
      sessionData.first_question.text,
      sessionData.first_question.audio_url,
      false
    );
  } catch (err) {
    alert("Could not start interview session: " + (err.message || "Unknown error"));
    el.startInterviewBtn.disabled = false;
    el.startInterviewBtn.textContent = "Start interview";
  }
}

function presentQuestion(text, audioUrl, isFollowUp) {
  appState.isFollowUp = isFollowUp;
  const turnLabel = isFollowUp
    ? `Question ${appState.questionIndex} of ${appState.totalQuestions} (Follow-up)`
    : `Question ${appState.questionIndex} of ${appState.totalQuestions}`;

  el.turnIndicator.textContent = turnLabel;
  el.questionText.textContent = text;
  el.statusBadge.textContent = "Interviewer is speaking";
  el.statusBadge.className = "status-badge status-speaking";
  announce(el.liveAnnouncer, `${turnLabel}: ${text}`);

  // Hide panels
  el.listeningPanel.classList.add("hidden");
  el.reviewPanel.classList.add("hidden");
  el.noSpeechNotice.classList.add("hidden");
  el.uploadErrorNotice.classList.add("hidden");

  // Play interviewer audio
  appState.interviewerAudio.pause();
  appState.interviewerAudio.src = audioUrl;

  appState.interviewerAudio.onended = () => {
    startListeningTurn();
  };

  appState.interviewerAudio.onerror = () => {
    // If TTS audio fails, start listening after a brief pause
    setTimeout(() => {
      startListeningTurn();
    }, 1000);
  };

  appState.interviewerAudio.play().catch(() => {
    // Autoplay blocked: show explicit play button
    el.playQuestionBtn.classList.remove("hidden");
  });
}

async function startListeningTurn() {
  el.statusBadge.textContent = "Your turn – listening";
  el.statusBadge.className = "status-badge status-listening";
  el.listeningPanel.classList.remove("hidden");
  el.reviewPanel.classList.add("hidden");
  el.noSpeechNotice.classList.add("hidden");
  el.uploadErrorNotice.classList.add("hidden");
  el.playQuestionBtn.classList.add("hidden");
  announce(el.liveAnnouncer, "Interviewer finished speaking. Your turn to answer.");

  // Start timer
  appState.recordingStartTime = Date.now();
  updateTimerDisplay();
  clearInterval(appState.timerInterval);
  appState.timerInterval = setInterval(updateTimerDisplay, 500);

  // Initialize VAD detector
  const vad = new VadDetector(16000);

  if (appState.captureSession) {
    appState.captureSession.release();
  }
  appState.captureSession = new AudioCaptureSession();

  try {
    await appState.captureSession.start(
      appState.selectedDeviceId,
      (frame) => {
        // Feed frame to VAD
        const vadResult = vad.processFrame(frame);
        if (vadResult.shouldStop) {
          if (vadResult.stopReason === "timeout") {
            handleNoSpeechDetected();
          } else {
            handleStopAnswering();
          }
        }
      },
      (rms) => {
        updateLevelMeter(el.listeningMeterBar, rms);
      }
    );
  } catch (err) {
    alert("Could not access microphone: " + (err.message || "Microphone capture error"));
    clearInterval(appState.timerInterval);
  }
}

function updateTimerDisplay() {
  const elapsed = (Date.now() - appState.recordingStartTime) / 1000;
  el.timerDisplay.textContent = formatTime(elapsed);
}

async function handleStopAnswering() {
  clearInterval(appState.timerInterval);
  updateLevelMeter(el.listeningMeterBar, 0);

  if (!appState.captureSession || !appState.captureSession.isRecording) {
    return;
  }

  el.statusBadge.textContent = "Processing your answer…";
  el.statusBadge.className = "status-badge status-processing";
  announce(el.liveAnnouncer, "Recording stopped. Processing answer.");

  try {
    const recordResult = await appState.captureSession.stop({ trimSilence: true });
    appState.lastRecordedWav = recordResult.wavBuffer;
    appState.lastRecordedBlob = recordResult.wavBlob;

    if (appState.reviewBeforeSending) {
      // Offer review before uploading
      el.listeningPanel.classList.add("hidden");
      el.reviewAudio.src = URL.createObjectURL(recordResult.wavBlob);
      el.reviewPanel.classList.remove("hidden");
      announce(el.liveAnnouncer, "Review your answer recording before submitting.");
    } else {
      await uploadAnswerBlob(recordResult.wavBlob);
    }
  } catch (err) {
    el.uploadErrorMessage.textContent = err.message || "Failed to process audio recording.";
    el.uploadErrorNotice.classList.remove("hidden");
  }
}

async function handleNoSpeechDetected() {
  clearInterval(appState.timerInterval);
  updateLevelMeter(el.listeningMeterBar, 0);

  if (appState.captureSession && appState.captureSession.isRecording) {
    await appState.captureSession.stop({ trimSilence: false });
  }

  el.listeningPanel.classList.add("hidden");
  el.noSpeechNotice.classList.remove("hidden");
  el.statusBadge.textContent = "I did not hear you";
  el.statusBadge.className = "status-badge status-warn";
  announce(el.liveAnnouncer, "I did not hear you. Please check your microphone and try again.");
}

async function uploadAnswerBlob(wavBlob) {
  el.listeningPanel.classList.add("hidden");
  el.reviewPanel.classList.add("hidden");
  el.uploadErrorNotice.classList.add("hidden");
  el.statusBadge.textContent = "Processing your answer…";
  el.statusBadge.className = "status-badge status-processing";

  try {
    const answerResult = await submitAnswer(appState.activeSessionId, wavBlob);
    renderTurnFeedback(answerResult);

    if (answerResult.finished || answerResult.interviewer.action === "finish") {
      transitionToResults();
      return;
    }

    const interviewerAction = answerResult.interviewer.action;
    const nextText = answerResult.interviewer.text;
    const nextAudioUrl = answerResult.interviewer.audio_url;

    if (interviewerAction === "next_question") {
      appState.questionIndex++;
      presentQuestion(nextText, nextAudioUrl, false);
    } else if (interviewerAction === "follow_up") {
      presentQuestion(nextText, nextAudioUrl, true);
    } else if (interviewerAction === "repeat" || interviewerAction === "clarify") {
      // Repeat or clarify question without advancing question index
      presentQuestion(nextText, nextAudioUrl, appState.isFollowUp);
    }
  } catch (err) {
    el.uploadErrorMessage.textContent = err.message || "Failed to upload answer to server.";
    el.uploadErrorNotice.classList.remove("hidden");
    el.statusBadge.textContent = "Upload failed";
    el.statusBadge.className = "status-badge status-warn";
  }
}

function renderTurnFeedback(result) {
  el.transcriptCard.classList.remove("hidden");
  el.transcriptText.textContent = result.transcript || "(No words recognized)";

  el.transcriptFlags.innerHTML = "";
  const flags = result.flags || {};
  if (flags.too_quiet) {
    addFlagNotice("Notice: Your voice was recorded at a low volume.");
  }
  if (flags.very_short || flags.too_few_words) {
    addFlagNotice("Notice: The response was very short.");
  }
  if (flags.transcription_error) {
    addFlagNotice("Notice: Speech transcription encountered an error. Please repeat.");
  }
}

function addFlagNotice(text) {
  const p = document.createElement("p");
  p.className = "flag-notice";
  p.textContent = text;
  el.transcriptFlags.appendChild(p);
}

async function handleEndInterviewEarly() {
  clearInterval(appState.timerInterval);
  if (appState.captureSession) {
    appState.captureSession.release();
  }
  transitionToResults();
}

// -------------------------------------------------------------
// RESULTS & REPORT POLLING
// -------------------------------------------------------------
async function transitionToResults() {
  if (appState.captureSession) {
    appState.captureSession.release();
  }
  showScreen("results");
  announce(el.liveAnnouncer, "Interview complete. Starting analysis.");

  el.resultsStatus.textContent = "Finishing interview and generating feedback report…";
  el.resultsProgressBar.classList.remove("hidden");
  el.reportContainer.classList.add("hidden");
  el.resultsActions.classList.add("hidden");
  el.resultsErrorNotice.classList.add("hidden");

  try {
    await finishSession(appState.activeSessionId, appState.includeContentScoring);
    pollSessionStatus(appState.activeSessionId);
  } catch (err) {
    el.resultsErrorMessage.textContent = err.message || "Failed to finalize session.";
    el.resultsErrorNotice.classList.remove("hidden");
    el.resultsProgressBar.classList.add("hidden");
  }
}

function pollSessionStatus(sessionId) {
  let attempts = 0;
  const maxAttempts = 200; // 200 * 1.5s = 300s (5 minutes)

  if (appState.pollTimer) {
    clearInterval(appState.pollTimer);
  }

  appState.pollTimer = setInterval(async () => {
    attempts++;
    try {
      const status = await getSessionStatus(sessionId);
      if (status.state === "ready" && status.report_url) {
        clearInterval(appState.pollTimer);
        displayReport(status.report_url);
      } else if (status.state === "error") {
        clearInterval(appState.pollTimer);
        el.resultsErrorMessage.textContent = status.message || "Analysis could not be completed.";
        el.resultsErrorNotice.classList.remove("hidden");
        el.resultsProgressBar.classList.add("hidden");
      } else {
        el.resultsStatus.textContent = status.message || "Analysis is running…";
      }
    } catch (err) {
      if (attempts >= maxAttempts) {
        clearInterval(appState.pollTimer);
        el.resultsErrorMessage.textContent = "Analysis timed out after 5 minutes.";
        el.resultsErrorNotice.classList.remove("hidden");
        el.resultsProgressBar.classList.add("hidden");
      }
    }

    if (attempts >= maxAttempts) {
      clearInterval(appState.pollTimer);
      el.resultsErrorMessage.textContent = "Analysis timed out after 5 minutes.";
      el.resultsErrorNotice.classList.remove("hidden");
      el.resultsProgressBar.classList.add("hidden");
    }
  }, 1500);
}

function displayReport(reportUrl) {
  el.resultsStatus.textContent = "Report ready!";
  el.resultsProgressBar.classList.add("hidden");
  el.reportFrame.src = reportUrl;
  el.reportContainer.classList.remove("hidden");
  el.resultsActions.classList.remove("hidden");
  announce(el.liveAnnouncer, "Report generated and loaded.");

  el.openNewTabBtn.onclick = () => {
    window.open(reportUrl, "_blank", "noopener,noreferrer");
  };
}

function resetToSetup() {
  if (appState.pollTimer) {
    clearInterval(appState.pollTimer);
  }
  if (appState.captureSession) {
    appState.captureSession.release();
  }
  appState.activeSessionId = null;
  appState.questionIndex = 1;
  appState.lastRecordedWav = null;
  appState.lastRecordedBlob = null;
  el.transcriptCard.classList.add("hidden");
  el.reportFrame.src = "about:blank";
  showScreen("setup");
  loadPastSessionsList();
}

// Start app on DOMContentLoaded
window.addEventListener("DOMContentLoaded", initApp);

