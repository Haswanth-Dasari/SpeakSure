function withTimeout(promise, milliseconds, message) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = window.setTimeout(() => reject(new Error(message)), milliseconds);
  });
  return Promise.race([promise, timeout]).finally(() => window.clearTimeout(timer));
}

export class InterviewerAudioPlayer {
  constructor() {
    this.context = null;
    this.analyser = null;
    this.currentSource = null;
    this.lastBuffer = null;
    this.levelSamples = null;
  }

  async unlock() {
    if (!this.context) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) throw new Error("Audio playback is not supported in this browser.");
      this.context = new AudioContextClass();
      this.analyser = this.context.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.connect(this.context.destination);
      this.levelSamples = new Float32Array(this.analyser.fftSize);
    }
    if (this.context.state === "suspended") {
      await withTimeout(
        this.context.resume(),
        6000,
        "Audio output did not start. Check your Windows sound output device " +
          "(a Bluetooth headset in hands-free mode can block it), then try again.",
      );
    }
  }

  async playUrl(url, onDiagnostic) {
    onDiagnostic?.("interviewer_audio_unlock", { status: "started" });
    await this.unlock();
    onDiagnostic?.("interviewer_audio_unlock", { status: "completed" });
    onDiagnostic?.("interviewer_audio_fetch", { status: "started" });
    const response = await fetch(url);
    if (!response.ok) {
      onDiagnostic?.("interviewer_audio_fetch", { status: "failed", http_status: response.status });
      throw new Error(`Interviewer audio failed to load (${response.status}).`);
    }
    onDiagnostic?.("interviewer_audio_fetch", { status: "completed", http_status: response.status });
    onDiagnostic?.("interviewer_audio_response_body", { status: "reading_started" });
    const bytes = await response.arrayBuffer();
    onDiagnostic?.("interviewer_audio_response_body", { status: "read_completed" });
    onDiagnostic?.("interviewer_audio_decode", { status: "started" });
    this.lastBuffer = await this.context.decodeAudioData(bytes);
    onDiagnostic?.("interviewer_audio_decode", { status: "completed" });
    onDiagnostic?.("interviewer_audio_playback", { status: "audio_started" });
    await this.playBuffer(this.lastBuffer);
    onDiagnostic?.("interviewer_audio_playback", { status: "audio_ended" });
  }

  async replay() {
    if (!this.lastBuffer) throw new Error("There is no interviewer audio to repeat yet.");
    await this.unlock();
    await this.playBuffer(this.lastBuffer);
  }

  playBuffer(buffer) {
    this.stop();
    return new Promise((resolve) => {
      // Safety net: never wait forever if the browser drops the "ended" event.
      const safety = window.setTimeout(resolve, (buffer.duration + 5) * 1000);
      const source = this.context.createBufferSource();
      this.currentSource = source;
      source.buffer = buffer;
      source.connect(this.analyser);
      source.addEventListener("ended", () => {
        window.clearTimeout(safety);
        if (this.currentSource === source) this.currentSource = null;
        resolve();
      }, { once: true });
      source.start();
    });
  }

  getLevel() {
    if (!this.analyser || !this.levelSamples || !this.currentSource) return 0;
    this.analyser.getFloatTimeDomainData(this.levelSamples);
    let total = 0;
    for (const sample of this.levelSamples) total += sample * sample;
    return Math.sqrt(total / this.levelSamples.length);
  }

  stop() {
    if (!this.currentSource) return;
    const source = this.currentSource;
    this.currentSource = null;
    try {
      source.stop();
    } catch {
      // The source may have ended between the state check and stop.
    }
  }

  async release() {
    this.stop();
    this.lastBuffer = null;
    if (this.context && this.context.state !== "closed") await this.context.close();
    this.context = null;
    this.analyser = null;
    this.levelSamples = null;
  }
}
