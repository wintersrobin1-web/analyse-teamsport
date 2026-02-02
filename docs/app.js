import { FilesetResolver, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const statusEl = document.getElementById("status");
const btnWebcam = document.getElementById("btnWebcam");
const btnStop = document.getElementById("btnStop");
const fileVideo = document.getElementById("fileVideo");

let landmarker = null;
let stream = null;
let running = false;
let lastFrameMs = 0;
let fpsLimit = 30;

// sehr einfacher ID-Tracker (nearest-neighbor auf Fußpunkt)
let nextId = 1;
const tracks = new Map(); // id -> {x,y,lastSeen}
const MAX_AGE_MS = 600;
const MAX_DIST_PX = 70;

function setStatus(s) {
  statusEl.textContent = s;
  console.log(s);
}

function resizeCanvas() {
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
  // greedy assignment
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

  // update/create
  for (let i = 0; i < dets.length; i++) {
    const p = dets[i];
    if (assigned.has(i)) {
      const id = assigned.get(i);
      const tr = tracks.get(id);
      // smoothing
      tr.x = 0.7 * tr.x + 0.3 * p.x;
      tr.y = 0.7 * tr.y + 0.3 * p.y;
      tr.lastSeen = tMs;
    } else {
      tracks.set(nextId, { x: p.x, y: p.y, lastSeen: tMs });
      nextId++;
    }
  }

  // prune
  for (const [id, tr] of tracks.entries()) {
    if ((tMs - tr.lastSeen) > MAX_AGE_MS) tracks.delete(id);
  }
}

function drawSkeletons(result) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!result?.landmarks?.length) return;

  // simple connections
  const edges = [
    [11, 12], [11, 23], [12, 24], [23, 24],
    [23, 25], [25, 27], [24, 26], [26, 28],
    [11, 13], [13, 15], [12, 14], [14, 16],
  ];

  // detections for tracking: ankle midpoint
  const dets = [];

  for (const lm of result.landmarks) {
    // footpoint: ankles 27,28; fallback hips 23,24
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

    // draw edges
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

    // draw keypoints (subset)
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    for (const k of [11,12,23,24,25,26,27,28]) {
      const p = lm[k];
      if (!p) continue;
      ctx.beginPath();
      ctx.arc(p.x * canvas.width, p.y * canvas.height, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  updateTracks(dets, performance.now());

  // draw IDs at tracked points
  ctx.font = "16px system-ui";
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  for (const [id, tr] of tracks.entries()) {
    ctx.beginPath();
    ctx.arc(tr.x, tr.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fill();

    ctx.fillStyle = "rgba(255,255,0,0.95)";
    ctx.fillText(`#${id}`, tr.x + 8, tr.y - 8);
  }
}

async function init() {
  setStatus("Lade MediaPipe…");
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
}

async function loop(tMs) {
  if (!running) return;

  const minDt = 1000 / fpsLimit;
  if ((tMs - lastFrameMs) < minDt) {
    requestAnimationFrame(loop);
    return;
  }
  lastFrameMs = tMs;

  if (video.readyState >= 2 && video.videoWidth) {
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      resizeCanvas();
    }
    const res = landmarker.detectForVideo(video, tMs);
    drawSkeletons(res);
  }

  requestAnimationFrame(loop);
}

btnWebcam.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    video.muted = true;
    await video.play();
    resizeCanvas();
    running = true;
    btnStop.disabled = false;
    btnWebcam.disabled = true;
    setStatus("Webcam läuft. Tracking aktiv.");
    requestAnimationFrame(loop);
  } catch (e) {
    setStatus("Webcam Fehler: " + e);
  }
});

btnStop.addEventListener("click", () => {
  running = false;
  btnStop.disabled = true;
  btnWebcam.disabled = false;

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.pause();
  video.srcObject = null;

  tracks.clear();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  setStatus("Gestoppt.");
});

fileVideo.addEventListener("change", async () => {
  const f = fileVideo.files?.[0];
  if (!f) return;

  // stop webcam if running
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;

  const url = URL.createObjectURL(f);
  video.src = url;
  video.muted = true;
  await video.play();
  resizeCanvas();

  tracks.clear();
  running = true;
  btnStop.disabled = false;
  btnWebcam.disabled = true;

  setStatus("Video läuft. Tracking aktiv.");
  requestAnimationFrame(loop);
});

await init();
