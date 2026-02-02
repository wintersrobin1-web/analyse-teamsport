// docs/app.js
import { FilesetResolver, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ---------- DOM ----------
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const statusEl = document.getElementById("status");
const btnWebcam = document.getElementById("btnWebcam");
const btnStop = document.getElementById("btnStop");
const fileVideo = document.getElementById("fileVideo");

// ---------- State ----------
let landmarker = null;
let stream = null;

let running = false;
let rafId = null;

let lastFrameMs = 0;
const fpsLimit = 30;

// Simple track IDs (nearest neighbor on ankle midpoint)
let nextId = 1;
const tracks = new Map(); // id -> {x,y,lastSeen}
const MAX_AGE_MS = 700;
const MAX_DIST_PX = 80;

// ---------- Helpers ----------
function setStatus(msg) {
  statusEl.textContent = msg;
  console.log(msg);
}

function hardResetVideoElement() {
  // Stops playback and resets src/srcObject safely
  video.pause();
  video.srcObject = null;

  // revoke old objectUrl
  if (video.dataset.objectUrl) {
    try { URL.revokeObjectURL(video.dataset.objectUrl); } catch {}
    delete video.dataset.objectUrl;
  }

  // Reset src
  video.removeAttribute("src");
  video.load();
}

function resizeCanvasToVideo() {
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 360;
  canvas.width = w;
  canvas.height = h;
}

function dist(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function updateTracks(dets, tMs) {
  // Greedy nearest-neighbor assignment
  const ids = Array.from(tracks.keys());
  const used = new Set();
  const assigned = new Map(); // detIdx -> id

  for (let i = 0; i < dets.length; i++) {
    let bestId = null, bestD = Infinity;
    for (const id of ids) {
      if (used.has(id)) continue;
      const tr = tracks.get(id);
      const d = dist(dets[i], tr);
      if (d < bestD) { bestD = d; bestId = id; }
    }
    if (bestId != null && bestD <= MAX_DIST_PX) {
      used.add(bestId);
      assigned.set(i, bestId);
    }
  }

  for (let i = 0; i < dets.length; i++) {
    const p = dets[i];
    if (assigned.has(i)) {
      const id = assigned.get(i);
      const tr = tracks.get(id);
      // smooth
      tr.x = 0.75 * tr.x + 0.25 * p.x;
      tr.y = 0.75 * tr.y + 0.25 * p.y;
      tr.lastSeen = tMs;
    } else {
      tracks.set(nextId, { x: p.x, y: p.y, lastSeen: tMs });
      nextId++;
    }
  }

  for (const [id, tr] of tracks.entries()) {
    if ((tMs - tr.lastSeen) > MAX_AGE_MS) tracks.delete(id);
  }
}

function drawPoseAndIds(result, tMs) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const lms = result?.landmarks || [];
  if (!lms.length) return;

  // A few bones so you see it working
  const edges = [
    [11,12], [11,23], [12,24], [23,24],
    [23,25], [25,27], [24,26], [26,28],
    [11,13], [13,15], [12,14], [14,16]
  ];

  const dets = [];

  for (const lm of lms) {
    // footpoint: ankles midpoint; fallback hips midpoint
    const a1 = lm[27], a2 = lm[28];
    const h1 = lm[23], h2 = lm[24];

    let fx, fy;
    if (a1 && a2) {
      fx = (a1.x + a2.x) * 0.5 * canvas.width;
      fy = (a1.y + a2.y) * 0.5 * canvas.height;
    } else {
      fx = (h1.x + h2.x) * 0.5 * canvas.width;
      fy = (h1.y + h2.y) * 0.5 * canvas.height;
    }
    dets.push({ x: fx, y: fy });

    // draw skeleton
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(255,255,255,0.75)";
    for (const [a, b] of edges) {
      const pa = lm[a], pb = lm[b];
      if (!pa || !pb) continue;
      ctx.beginPath();
      ctx.moveTo(pa.x * canvas.width, pa.y * canvas.height);
      ctx.lineTo(pb.x * canvas.width, pb.y * canvas.height);
      ctx.stroke();
    }

    // draw a few keypoints
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    for (const k of [11,12,23,24,27,28]) {
      const p = lm[k];
      if (!p) continue;
      ctx.beginPath();
      ctx.arc(p.x * canvas.width, p.y * canvas.height, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  updateTracks(dets, tMs);

  // IDs
  ctx.font = "16px system-ui";
  for (const [id, tr] of tracks.entries()) {
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(tr.x + 6, tr.y - 22, 54, 20);
    ctx.fillStyle = "rgba(255,255,0,0.95)";
    ctx.fillText(`#${id}`, tr.x + 10, tr.y - 7);

    ctx.beginPath();
    ctx.arc(tr.x, tr.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,255,0,0.9)";
    ctx.fill();
  }
}

function stopLoop() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
}

function startLoop() {
  if (!landmarker) {
    setStatus("Landmarker noch nicht bereit.");
    return;
  }
  if (running) return;
  running = true;
  lastFrameMs = 0;
  rafId = requestAnimationFrame(loop);
}

function loop(tMs) {
  if (!running) return;

  const minDt = 1000 / fpsLimit;
  if ((tMs - lastFrameMs) < minDt) {
    rafId = requestAnimationFrame(loop);
    return;
  }
  lastFrameMs = tMs;

  // Only process when we actually have decoded frame data
  // readyState >= 2 => HAVE_CURRENT_DATA
  if (video.readyState >= 2 && video.videoWidth > 0) {
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      resizeCanvasToVideo();
    }
    try {
      const res = landmarker.detectForVideo(video, tMs);
      drawPoseAndIds(res, tMs);
    } catch (e) {
      setStatus("Inference-Fehler: " + String(e));
      console.error(e);
      stopLoop();
      return;
    }
  }

  rafId = requestAnimationFrame(loop);
}

// ---------- Init MediaPipe ----------
async function init() {
  setStatus("JS läuft ✅ – lade MediaPipe…");

  try {
    const fileset = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    landmarker = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numPoses: 10
    });

    setStatus("Bereit. Webcam starten oder Video laden.");
  } catch (e) {
    setStatus("MediaPipe konnte nicht geladen werden (Netz/Adblock?): " + String(e));
    console.error(e);
  }
}

// ---------- Events ----------
video.addEventListener("loadedmetadata", () => {
  resizeCanvasToVideo();
});

video.addEventListener("play", () => {
  // If user manually presses play (e.g. autoplay blocked), start tracking
  if (landmarker) startLoop();
});

video.addEventListener("pause", () => {
  // Don’t stop loop hard; just keep it running or stop? We stop to save CPU.
  stopLoop();
});

btnWebcam.addEventListener("click", async () => {
  if (!landmarker) {
    setStatus("MediaPipe noch nicht bereit – bitte kurz warten.");
    return;
  }

  try {
    // stop any file playback
    stopLoop();
    tracks.clear();
    nextId = 1;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // stop existing stream
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    hardResetVideoElement();

    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;

    await video.play();

    btnStop.disabled = false;
    btnWebcam.disabled = true;

    setStatus("Webcam läuft. Tracking aktiv.");
    startLoop();
  } catch (e) {
    setStatus("Webcam Fehler: " + String(e));
    console.error(e);
  }
});

btnStop.addEventListener("click", () => {
  stopLoop();
  tracks.clear();
  nextId = 1;

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }

  hardResetVideoElement();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  btnStop.disabled = true;
  btnWebcam.disabled = false;

  setStatus("Gestoppt.");
});

fileVideo.addEventListener("change", async () => {
  if (!landmarker) {
    setStatus("MediaPipe noch nicht bereit – bitte kurz warten.");
    return;
  }

  const f = fileVideo.files?.[0];
  if (!f) return;

  try {
    stopLoop();
    tracks.clear();
    nextId = 1;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // stop webcam stream
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }

    hardResetVideoElement();

    setStatus(`Datei gewählt: ${f.name} – lade…`);

    const url = URL.createObjectURL(f);
    video.dataset.objectUrl = url;

    video.src = url;
    video.controls = true;
    video.muted = true;
    video.playsInline = true;

    // Wait for metadata or error with timeout
    await new Promise((resolve, reject) => {
      const to = setTimeout(() => reject(new Error("Timeout: loadedmetadata")), 10000);

      video.onloadedmetadata = () => {
        clearTimeout(to);
        resolve();
      };

      video.onerror = () => {
        clearTimeout(to);
        reject(video.error || new Error("Video element error"));
      };
    });

    resizeCanvasToVideo();

    // Try autoplay (may fail). If it fails, we instruct user to press play.
    try {
      await video.play();
      setStatus(`Video läuft (${Math.round(video.duration)}s). Tracking aktiv.`);
      btnStop.disabled = false;
      btnWebcam.disabled = true;
      startLoop();
    } catch (e) {
      // Autoplay blocked: show first frame via seek
      try {
        video.currentTime = Math.min(0.05, Math.max(0, (video.duration || 1) * 0.01));
        await new Promise(r => (video.onseeked = () => r()));
      } catch {}
      btnStop.disabled = false;
      btnWebcam.disabled = true;
      setStatus("Video geladen. Autoplay blockiert – drücke ▶︎ im Player, dann startet Tracking.");
      // Loop starts on 'play' event.
    }
  } catch (e) {
    setStatus("Video-Fehler: " + String(e));
    console.error(e);
  }
});

// Start
await init();
