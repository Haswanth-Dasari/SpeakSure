import test from "node:test";
import assert from "node:assert/strict";
import { encodeWav, resampleAudio, trimTrailingSilence } from "../src/speaksure/web/static/js/wav.js";
import { VadDetector, calculateRms, SILENCE_DURATION_SEC } from "../src/speaksure/web/static/js/vad.js";

test("WAV encoder: verifies header fields, endianness, and buffer length", () => {
  const sampleRate = 16000;
  const numSamples = 3200; // 0.2 seconds
  const samples = new Float32Array(numSamples);
  for (let i = 0; i < numSamples; i++) {
    samples[i] = Math.sin((2 * Math.PI * 440 * i) / sampleRate) * 0.5;
  }

  const buffer = encodeWav(samples, sampleRate);
  assert.equal(buffer.byteLength, 44 + numSamples * 2, "Total buffer length must be 44 + numSamples * 2");

  const view = new DataView(buffer);
  const textDecoder = new TextDecoder("ascii");

  // ChunkID: "RIFF"
  assert.equal(textDecoder.decode(new Uint8Array(buffer, 0, 4)), "RIFF");
  // ChunkSize: 36 + dataSize
  assert.equal(view.getUint32(4, true), 36 + numSamples * 2);
  // Format: "WAVE"
  assert.equal(textDecoder.decode(new Uint8Array(buffer, 8, 4)), "WAVE");

  // Subchunk1ID: "fmt "
  assert.equal(textDecoder.decode(new Uint8Array(buffer, 12, 4)), "fmt ");
  // Subchunk1Size: 16
  assert.equal(view.getUint32(16, true), 16);
  // AudioFormat: 1 (PCM)
  assert.equal(view.getUint16(20, true), 1);
  // NumChannels: 1 (mono)
  assert.equal(view.getUint16(22, true), 1);
  // SampleRate: 16000
  assert.equal(view.getUint32(24, true), 16000);
  // ByteRate: 32000
  assert.equal(view.getUint32(28, true), 32000);
  // BlockAlign: 2
  assert.equal(view.getUint16(32, true), 2);
  // BitsPerSample: 16
  assert.equal(view.getUint16(34, true), 16);

  // Subchunk2ID: "data"
  assert.equal(textDecoder.decode(new Uint8Array(buffer, 36, 4)), "data");
  // Subchunk2Size: numSamples * 2
  assert.equal(view.getUint32(40, true), numSamples * 2);

  // Verify first sample value
  const firstSample = view.getInt16(44, true);
  assert.equal(typeof firstSample, "number");
});

test("Resampler: 440 Hz tone keeps frequency and amplitude within 5%", () => {
  const sourceRate = 48000;
  const targetRate = 16000;
  const durationSec = 1.0;
  const frequency = 440;
  const amplitude = 0.8;

  const numSourceSamples = Math.round(sourceRate * durationSec);
  const sourceSamples = new Float32Array(numSourceSamples);
  for (let i = 0; i < numSourceSamples; i++) {
    sourceSamples[i] = amplitude * Math.sin((2 * Math.PI * frequency * i) / sourceRate);
  }

  const resampled = resampleAudio(sourceSamples, sourceRate, targetRate);
  assert.equal(resampled.length, targetRate * durationSec);

  // 1. Check amplitude (RMS of sine wave is A / sqrt(2))
  const expectedRms = amplitude / Math.SQRT2;
  const actualRms = calculateRms(resampled);
  const amplitudeDiffRatio = Math.abs(actualRms - expectedRms) / expectedRms;
  assert.ok(
    amplitudeDiffRatio <= 0.05,
    `Amplitude difference ratio ${amplitudeDiffRatio.toFixed(4)} exceeds 5% (expected RMS ${expectedRms.toFixed(4)}, got ${actualRms.toFixed(4)})`
  );

  // 2. Check frequency by counting positive zero crossings
  let positiveZeroCrossings = 0;
  for (let i = 1; i < resampled.length; i++) {
    if (resampled[i - 1] <= 0 && resampled[i] > 0) {
      positiveZeroCrossings++;
    }
  }
  const measuredFreq = positiveZeroCrossings / durationSec;
  const freqDiffRatio = Math.abs(measuredFreq - frequency) / frequency;
  assert.ok(
    freqDiffRatio <= 0.05,
    `Frequency difference ratio ${freqDiffRatio.toFixed(4)} exceeds 5% (expected ${frequency} Hz, got ${measuredFreq} Hz)`
  );
});

test("VAD: silence then speech then silence segments correctly; stop fires 2.5s after speech ends", () => {
  const sampleRate = 16000;
  const frameDurationSec = 0.05; // 50ms frames
  const frameSamplesCount = Math.round(sampleRate * frameDurationSec);
  const vad = new VadDetector(sampleRate);

  const silenceFrame = new Float32Array(frameSamplesCount); // zeros
  const speechFrame = new Float32Array(frameSamplesCount);
  for (let i = 0; i < frameSamplesCount; i++) {
    speechFrame[i] = 0.5 * Math.sin((2 * Math.PI * 440 * i) / sampleRate);
  }

  // 1. Initial 0.4s silence (noise calibration)
  const calibrationFrames = Math.round(0.4 / frameDurationSec); // 8 frames
  for (let f = 0; f < calibrationFrames; f++) {
    const res = vad.processFrame(silenceFrame);
    assert.equal(res.shouldStop, false);
    assert.equal(res.isSpeech, false);
  }
  assert.equal(vad.speechDetected, false);

  // 2. 1.0s speech (speechDetected should become true after 0.5s of speech)
  const speechFrames = Math.round(1.0 / frameDurationSec); // 20 frames
  for (let f = 0; f < speechFrames; f++) {
    const res = vad.processFrame(speechFrame);
    assert.equal(res.shouldStop, false);
    assert.equal(res.isSpeech, true);
  }
  assert.equal(vad.speechDetected, true, "Speech should be confirmed detected after 1.0s of speech");

  // 3. Post-speech silence: stop should fire after 2.5s of silence
  const silenceSecRequired = SILENCE_DURATION_SEC; // 2.5s
  const silenceFramesNeeded = Math.round(silenceSecRequired / frameDurationSec); // 50 frames

  let stoppedAtFrame = -1;
  let finalReason = null;
  for (let f = 0; f < silenceFramesNeeded + 10; f++) {
    const res = vad.processFrame(silenceFrame);
    if (res.shouldStop) {
      stoppedAtFrame = f;
      finalReason = res.stopReason;
      break;
    }
  }

  assert.equal(finalReason, "silence", "Stop reason must be 'silence'");
  const silenceElapsedBeforeStop = (stoppedAtFrame * frameDurationSec);
  assert.ok(
    Math.abs(silenceElapsedBeforeStop - 2.5) <= frameDurationSec,
    `Stop should fire 2.5s after speech ends (fired at ~${silenceElapsedBeforeStop.toFixed(2)}s)`
  );
});

test("VAD: 15s timeout fires if no speech detected", () => {
  const sampleRate = 16000;
  const frameDurationSec = 0.1; // 100ms
  const frameSamples = Math.round(sampleRate * frameDurationSec);
  const vad = new VadDetector(sampleRate);
  const silence = new Float32Array(frameSamples);

  let stopped = false;
  let stopReason = null;
  let frameCount = 0;

  while (frameCount < 200) {
    frameCount++;
    const res = vad.processFrame(silence);
    if (res.shouldStop) {
      stopped = true;
      stopReason = res.stopReason;
      break;
    }
  }

  assert.equal(stopped, true);
  assert.equal(stopReason, "timeout");
  assert.equal(frameCount * frameDurationSec, 15.0);
});
