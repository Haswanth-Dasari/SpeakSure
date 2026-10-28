/**
 * Pure functions for anti-aliased audio resampling and 16-bit PCM WAV encoding.
 * Contains no DOM or browser-specific dependencies.
 */

/**
 * Resample mono Float32Array audio to targetSampleRate using windowed averaging.
 * Provides anti-aliasing when downsampling.
 *
 * @param {Float32Array} input - Input audio samples in mono (-1.0 to 1.0).
 * @param {number} sourceRate - Sample rate of input audio (e.g. 48000, 44100).
 * @param {number} targetRate - Target sample rate (e.g. 16000).
 * @returns {Float32Array} - Resampled mono audio samples.
 */
export function resampleAudio(input, sourceRate, targetRate) {
  if (sourceRate === targetRate) {
    return new Float32Array(input);
  }
  if (!input || input.length === 0) {
    return new Float32Array(0);
  }

  const ratio = sourceRate / targetRate;
  const targetLength = Math.round(input.length / ratio);
  const output = new Float32Array(targetLength);

  for (let i = 0; i < targetLength; i++) {
    const center = i * ratio;
    // Window over source samples corresponding to this target sample
    const start = Math.max(0, Math.floor(center - ratio / 2));
    const end = Math.min(input.length, Math.ceil(center + ratio / 2));
    let sum = 0;
    let weightSum = 0;

    for (let j = start; j < end; j++) {
      // Triangle / Hann-like weighting around center for smoother anti-aliasing
      const dist = Math.abs(j - center);
      const weight = Math.max(0, 1 - dist / ratio);
      sum += input[j] * weight;
      weightSum += weight;
    }

    output[i] = weightSum > 0 ? sum / weightSum : 0;
  }

  return output;
}

/**
 * Trim trailing silence from audio samples, leaving a tail of tailSec seconds.
 *
 * @param {Float32Array} samples - Mono audio samples.
 * @param {number} sampleRate - Sample rate in Hz.
 * @param {number} threshold - RMS or amplitude threshold for speech (default 0.015).
 * @param {number} tailSec - Tail duration to preserve after last speech (default 0.4).
 * @returns {Float32Array} - Trimmed samples.
 */
export function trimTrailingSilence(samples, sampleRate, threshold = 0.015, tailSec = 0.4) {
  if (!samples || samples.length === 0) {
    return new Float32Array(0);
  }

  const tailSamples = Math.round(tailSec * sampleRate);
  const frameSize = Math.round(0.02 * sampleRate); // 20ms frames
  let lastSpeechSample = -1;

  for (let i = samples.length - 1; i >= 0; i -= frameSize) {
    const start = Math.max(0, i - frameSize);
    let sumSq = 0;
    for (let j = start; j <= i; j++) {
      sumSq += samples[j] * samples[j];
    }
    const rms = Math.sqrt(sumSq / (i - start + 1));
    if (rms > threshold) {
      lastSpeechSample = i;
      break;
    }
  }

  if (lastSpeechSample === -1) {
    return new Float32Array(samples);
  }

  const keepEnd = Math.min(samples.length, lastSpeechSample + tailSamples);
  return samples.slice(0, keepEnd);
}

/**
 * Encode mono Float32Array audio into a standard 16-bit PCM WAV ArrayBuffer.
 *
 * @param {Float32Array} samples - Mono audio samples (-1.0 to 1.0).
 * @param {number} sampleRate - Sample rate in Hz (e.g. 16000).
 * @returns {ArrayBuffer} - Valid RIFF/WAVE 16-bit PCM buffer.
 */
export function encodeWav(samples, sampleRate) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // Helper to write ASCII string
  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  // RIFF header
  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true); // ChunkSize
  writeString(8, "WAVE");

  // fmt chunk
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true);  // AudioFormat (1 = PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // data chunk
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  // Write PCM 16-bit samples
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1.0, Math.min(1.0, samples[i]));
    const intSample = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
    view.setInt16(offset, intSample, true);
    offset += 2;
  }

  return buffer;
}

