/**
 * Pure functions for Voice Activity Detection (VAD) and audio level calculation.
 * Contains no DOM or browser-specific dependencies.
 */

export const SILENCE_DURATION_SEC = 2.5;
export const MIN_SPEECH_DURATION_SEC = 0.5;
export const NOISE_CALIBRATION_SEC = 0.4;
export const MAX_RECORDING_SEC = 120.0;
export const NO_SPEECH_TIMEOUT_SEC = 15.0;

// Mic test RMS threshold constants
export const RMS_TOO_QUIET_THRESHOLD = 0.01;
export const RMS_NOISY_THRESHOLD = 0.20;

/**
 * Calculate Root-Mean-Square (RMS) amplitude of audio samples.
 *
 * @param {Float32Array} samples - Audio samples.
 * @returns {number} - RMS value between 0.0 and 1.0.
 */
export function calculateRms(samples) {
  if (!samples || samples.length === 0) {
    return 0.0;
  }
  let sumSq = 0.0;
  for (let i = 0; i < samples.length; i++) {
    sumSq += samples[i] * samples[i];
  }
  return Math.sqrt(sumSq / samples.length);
}

/**
 * State machine for voice activity detection over incoming audio frames.
 */
export class VadDetector {
  /**
   * @param {number} sampleRate - Audio sample rate in Hz.
   * @param {object} [options] - Optional custom durations.
   */
  constructor(sampleRate = 16000, options = {}) {
    this.sampleRate = sampleRate;
    this.silenceDuration = options.silenceDuration ?? SILENCE_DURATION_SEC;
    this.minSpeechDuration = options.minSpeechDuration ?? MIN_SPEECH_DURATION_SEC;
    this.noiseCalibrationDuration = options.noiseCalibrationDuration ?? NOISE_CALIBRATION_SEC;
    this.maxDuration = options.maxDuration ?? MAX_RECORDING_SEC;
    this.noSpeechTimeout = options.noSpeechTimeout ?? NO_SPEECH_TIMEOUT_SEC;

    this.totalSamples = 0;
    this.noiseSamples = 0;
    this.noiseSumSq = 0.0;
    this.noiseRms = 0.005;
    this.threshold = 0.015;

    this.speechDetected = false;
    this.totalSpeechSamples = 0;
    this.silenceSamplesAfterSpeech = 0;
    this.isCalibrated = false;

    this.shouldStop = false;
    this.stopReason = null; // 'silence' | 'timeout' | 'max_duration'
  }

  /**
   * Process a single audio frame (typically 20ms to 100ms of samples).
   *
   * @param {Float32Array} frame - Audio frame samples.
   * @returns {{ rms: number, threshold: number, isSpeech: boolean, shouldStop: boolean, stopReason: string | null }}
   */
  processFrame(frame) {
    if (this.shouldStop) {
      return {
        rms: 0,
        threshold: this.threshold,
        isSpeech: false,
        shouldStop: true,
        stopReason: this.stopReason,
      };
    }

    const frameLen = frame.length;
    const currentSec = this.totalSamples / this.sampleRate;
    const frameRms = calculateRms(frame);

    // Adaptive noise floor estimation during first 0.4 s
    if (currentSec < this.noiseCalibrationDuration) {
      for (let i = 0; i < frameLen; i++) {
        this.noiseSumSq += frame[i] * frame[i];
      }
      this.noiseSamples += frameLen;
      this.noiseRms = Math.sqrt(this.noiseSumSq / this.noiseSamples);
      this.threshold = Math.max(0.015, this.noiseRms * 2.5);
    } else {
      this.isCalibrated = true;
    }

    this.totalSamples += frameLen;
    const totalElapsedSec = this.totalSamples / this.sampleRate;

    // Hard limit check (120 s)
    if (totalElapsedSec >= this.maxDuration) {
      this.shouldStop = true;
      this.stopReason = "max_duration";
      return {
        rms: frameRms,
        threshold: this.threshold,
        isSpeech: frameRms > this.threshold,
        shouldStop: true,
        stopReason: "max_duration",
      };
    }

    const isSpeechFrame = frameRms > this.threshold;

    if (isSpeechFrame) {
      this.totalSpeechSamples += frameLen;
      this.silenceSamplesAfterSpeech = 0;
      if (this.totalSpeechSamples / this.sampleRate >= this.minSpeechDuration) {
        this.speechDetected = true;
      }
    } else {
      if (this.speechDetected) {
        this.silenceSamplesAfterSpeech += frameLen;
        const silenceSec = this.silenceSamplesAfterSpeech / this.sampleRate;
        if (silenceSec >= this.silenceDuration) {
          this.shouldStop = true;
          this.stopReason = "silence";
          return {
            rms: frameRms,
            threshold: this.threshold,
            isSpeech: false,
            shouldStop: true,
            stopReason: "silence",
          };
        }
      } else {
        // No speech detected yet, check 15 s timeout
        if (totalElapsedSec >= this.noSpeechTimeout) {
          this.shouldStop = true;
          this.stopReason = "timeout";
          return {
            rms: frameRms,
            threshold: this.threshold,
            isSpeech: false,
            shouldStop: true,
            stopReason: "timeout",
          };
        }
      }
    }

    return {
      rms: frameRms,
      threshold: this.threshold,
      isSpeech: isSpeechFrame,
      shouldStop: false,
      stopReason: null,
    };
  }
}

