/**
 * Browser audio capture, microphone enumeration, AudioWorklet processing,
 * and test recording utilities.
 */

import { encodeWav, resampleAudio, trimTrailingSilence } from "./wav.js";
import { calculateRms, RMS_TOO_QUIET_THRESHOLD, RMS_NOISY_THRESHOLD } from "./vad.js";

export const BLUETOOTH_REGEX = /hands.?free|bluetooth|airdopes|buds|headset/i;

/**
 * Check if a microphone device label looks like a Bluetooth or hands-free device.
 *
 * @param {string} label
 * @returns {boolean}
 */
export function isBluetoothOrHeadset(label) {
  return typeof label === "string" && BLUETOOTH_REGEX.test(label);
}

/**
 * Request user permission for microphone access and list available audio inputs.
 *
 * @returns {Promise<MediaDeviceInfo[]>}
 */
export async function requestMicrophonePermission() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  // Stop initial permission probe stream
  stream.getTracks().forEach((track) => track.stop());
  return enumerateMicrophones();
}

/**
 * List all available audio input devices.
 *
 * @returns {Promise<MediaDeviceInfo[]>}
 */
export async function enumerateMicrophones() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((device) => device.kind === "audioinput");
}

/**
 * Audio recorder handling stream capture via AudioWorklet (fallback to ScriptProcessor).
 */
export class AudioCaptureSession {
  constructor() {
    this.audioContext = null;
    this.mediaStream = null;
    this.sourceNode = null;
    this.workletNode = null;
    this.scriptNode = null;
    this.chunks = [];
    this.isRecording = false;
    this.inputDeviceLabel = "Default microphone";
    this.appliedConstraints = {};
    this.blobUrl = null;
  }

  /**
   * Start capturing audio from selected microphone.
   *
   * @param {string} [deviceId] - Target device ID.
   * @param {function(Float32Array): void} [onAudioFrame] - Callback for each audio buffer.
   * @param {function(number): void} [onLevel] - Callback for live RMS level (0.0 to 1.0).
   * @returns {Promise<{ inputDeviceLabel: string, constraints: object }>}
   */
  async start(deviceId, onAudioFrame, onLevel) {
    this.chunks = [];
    this.isRecording = true;

    const audioConstraints = {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    };
    if (deviceId) {
      audioConstraints.deviceId = { exact: deviceId };
    }

    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: audioConstraints,
    });

    const track = this.mediaStream.getAudioTracks()[0];
    this.inputDeviceLabel = track?.label || "Unknown microphone";
    this.appliedConstraints = track?.getSettings ? track.getSettings() : {};

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    this.audioContext = new AudioContextClass();
    if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }

    this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

    // Try AudioWorklet first, fall back to ScriptProcessorNode
    let workletSupported = false;
    if (this.audioContext.audioWorklet && typeof Blob !== "undefined") {
      try {
        const workletCode = `
          class RecorderProcessor extends AudioWorkletProcessor {
            process(inputs) {
              const input = inputs[0];
              if (input && input.length > 0 && input[0].length > 0) {
                this.port.postMessage(input[0]);
              }
              return true;
            }
          }
          registerProcessor('recorder-processor', RecorderProcessor);
        `;
        const blob = new Blob([workletCode], { type: "application/javascript" });
        this.blobUrl = URL.createObjectURL(blob);
        await this.audioContext.audioWorklet.addModule(this.blobUrl);
        this.workletNode = new AudioWorkletNode(this.audioContext, "recorder-processor");
        this.workletNode.port.onmessage = (event) => {
          if (!this.isRecording) return;
          const frame = new Float32Array(event.data);
          this.chunks.push(frame);
          const rms = calculateRms(frame);
          if (onLevel) onLevel(rms);
          if (onAudioFrame) onAudioFrame(frame);
        };
        this.sourceNode.connect(this.workletNode);
        // Connect to destination through a silent gain node to keep worklet alive
        const silentGain = this.audioContext.createGain();
        silentGain.gain.value = 0;
        this.workletNode.connect(silentGain);
        silentGain.connect(this.audioContext.destination);
        workletSupported = true;
      } catch {
        workletSupported = false;
      }
    }

    if (!workletSupported) {
      // Fallback: ScriptProcessorNode
      const bufferSize = 2048;
      this.scriptNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
      this.scriptNode.onaudioprocess = (event) => {
        if (!this.isRecording) return;
        const channelData = event.inputBuffer.getChannelData(0);
        const frame = new Float32Array(channelData);
        this.chunks.push(frame);
        const rms = calculateRms(frame);
        if (onLevel) onLevel(rms);
        if (onAudioFrame) onAudioFrame(frame);
      };
      this.sourceNode.connect(this.scriptNode);
      this.scriptNode.connect(this.audioContext.destination);
    }

    return {
      inputDeviceLabel: this.inputDeviceLabel,
      constraints: this.appliedConstraints,
    };
  }

  /**
   * Stop capturing and return encoded 16 kHz mono WAV.
   *
   * @param {object} [options]
   * @param {boolean} [options.trimSilence=true] - Whether to trim trailing silence.
   * @param {number} [options.threshold=0.015] - Threshold for silence trimming.
   * @returns {Promise<{ wavBuffer: ArrayBuffer, wavBlob: Blob, durationSeconds: number, rawRms: number }>}
   */
  async stop(options = {}) {
    this.isRecording = false;

    // Disconnect processing nodes
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }
    if (this.scriptNode) {
      this.scriptNode.disconnect();
      this.scriptNode = null;
    }
    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }

    const sourceRate = this.audioContext?.sampleRate || 48000;

    // Combine all chunks
    let totalSamplesCount = 0;
    for (const chunk of this.chunks) {
      totalSamplesCount += chunk.length;
    }
    const combined = new Float32Array(totalSamplesCount);
    let offset = 0;
    for (const chunk of this.chunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }

    const rawRms = calculateRms(combined);

    // Resample to 16 kHz mono
    let samples16k = resampleAudio(combined, sourceRate, 16000);

    // Trim trailing silence if requested
    if (options.trimSilence !== false && samples16k.length > 0) {
      samples16k = trimTrailingSilence(
        samples16k,
        16000,
        options.threshold || 0.015,
        0.4
      );
    }

    const durationSeconds = samples16k.length / 16000;
    const wavBuffer = encodeWav(samples16k, 16000);
    const wavBlob = new Blob([wavBuffer], { type: "audio/wav" });

    return {
      wavBuffer,
      wavBlob,
      durationSeconds,
      rawRms,
    };
  }

  /**
   * Release media stream, audio context, and object URLs completely.
   */
  release() {
    this.isRecording = false;
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }
    if (this.audioContext && this.audioContext.state !== "closed") {
      this.audioContext.close().catch(() => {});
      this.audioContext = null;
    }
    if (this.blobUrl) {
      URL.revokeObjectURL(this.blobUrl);
      this.blobUrl = null;
    }
    this.chunks = [];
  }
}

/**
 * Record a 3-second test clip from microphone and assess signal level.
 *
 * @param {string} [deviceId]
 * @param {function(number): void} [onLevel]
 * @returns {Promise<{ assessment: string, rms: number, wavBlob: Blob }>}
 */
export async function runMicrophoneTest(deviceId, onLevel) {
  const session = new AudioCaptureSession();
  try {
    await session.start(deviceId, null, onLevel);
    await new Promise((resolve) => setTimeout(resolve, 3000));
    const result = await session.stop({ trimSilence: false });

    let assessment = "Good level";
    if (result.rawRms < RMS_TOO_QUIET_THRESHOLD) {
      assessment = "Too quiet - move closer or raise input volume";
    } else if (result.rawRms > RMS_NOISY_THRESHOLD) {
      assessment = "Very noisy";
    }

    return {
      assessment,
      rms: result.rawRms,
      wavBlob: result.wavBlob,
    };
  } finally {
    session.release();
  }
}

