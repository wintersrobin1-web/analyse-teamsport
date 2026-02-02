// app.js (ES module)

// MediaPipe Tasks Vision (PoseLandmarker) via CDN
import {
  FilesetResolver,
  PoseLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// --------------------------
// UI refs
// --------------------------
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");

const minimap = document.getElementById("minimap");
const mctx = minimap.getContext("2d");

const statusEl = document.getElementById("status");
const btnWebcam = document.getElementById("btnWebcam");
const btnStop = document.getElementById("btnStop");
const fileVideo = document.getElementById("fileVideo");
const btnCalibrate = document.getElementById("btnCalibrate");

const sportProfile = document.getElementById("sportProfile");
const fpsLimitInput = document.getElementById("fpsLimit");

const showSkeleton = document.getElementById("showSkeleton");
const showBoxes = document.getElementById("showBoxes");
const showIds = document.getElementById("showIds");
const showTeams = document.getElementById("showTeams");

const teamAColor = document.getElementById("teamA");
const teamBColor = document.getElementById("teamB");
const teamTol = document.getElementById("teamTol");

// --------------------------
// Sport profiles
// --------------------------
const PROFILES = {
  handball: { w: 40, h: 20 },
  basketball: { w: 28, h: 15 },
  football: { w: 105, h: 68 },
  volleyball: { w: 18, h: 9 },
};

// --------------------------
// Marker HSV ranges (Hütchen)
// Default: TL pink, TR cyan, BR yellow, BL green
// HSV in OpenCV-like ranges: H 0-179, S 0-255, V 0-255
// We approximate in JS with H 0-360, S/V 0-1; we’ll convert accordingly.
// --------------------------
const MARKERS = {
  corner_tl: { name: "TL", hsvLow: [300, 0.30, 0.30], hsvHigh: [360, 1.00, 1.00] }, // pink-ish
  corner_tr: { name: "TR", hsvLow: [170, 0.30, 0.30], hsvHigh: [205, 1.00, 1.00] }, // cyan
  corner_br: { name: "BR", hsvLow: [40, 0.40, 0.40], hsvHigh: [70, 1.00, 1.00] },   // yellow
  corner_bl: { name: "BL", hsvLow: [85, 0.30, 0.30], hsvHigh: [150, 1.00, 1.00] },  // green
};

// --------------------------
// State
// --------------------------
let landmarker = null;
let stream = null;
let running = false;

let lastFrameTime = 0;
let fpsLimit = 30;

let H = null; // homography (3x3) mapping image -> field coords
let cornersPx = null; // last detected corner centroids

// Offscreen canvas for pixel access
const off = document.createElement("canvas");
const offCtx = off.getContext("2d", { willReadFrequently: true });

// --------------------------
// Simple multi-person tracking by footpoint nearest-neighbor
// --------------------------
let nextTrackId = 1;
const tracks = new Map(); // id -> {id, x,y, vx,vy, lastSeen, team, color}
const MAX_MISSES_MS = 600; // remove track if not seen
const ASSIGN_MAX_DIST_PX = 70;

// --------------------------
// Helpers
// --------------------------
function setStatus(msg) {
  statusEl.textContent = msg;
}

function resizeCanvases() {
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 360;
  overlay.width = w;
  overlay.height = h;
  off.width = w;
  off.height = h;
}

function hexToRgb(hex) {
  const h = hex.replace("#", "").trim();
  const n = parseInt(h, 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

function rgbToHsv(r, g, b) {
  // r,g,b: 0..255 -> hsv: h 0..360, s 0..1, v 0..1
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === r) h = 60 * (((g - b) / d) % 6);
    else if (max === g) h = 60 * (((b - r) / d) + 2);
    else h = 60 * (((r - g) / d) + 4);
  }
  if (h < 0) h += 360;
  const s = max === 0 ? 0 : d / max;
  const v = max;
  return { h, s, v };
}

function hsvDist(hsv, hsvCenter) {
  // weighted distance (h circular)
  let dh = Math.abs(hsv.h - hsvCenter.h);
  dh = Math.min(dh, 360 - dh);
  const ds = Math.abs(hsv.s - hsvCenter.s);
  const dv = Math.abs(hsv.v - hsvCenter.v);
  return 2.0 * dh + 1.0 * (ds * 100) + 0.5 * (dv * 100);
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

// --------------------------
// Homography (DLT) for 4 point pairs
// Returns 3x3 matrix mapping src(x,y,1) -> dst(u,v,1)
// --------------------------
function computeHomography4(srcPts, dstPts) {
  // srcPts/dstPts: array of 4 points: {x,y}
  // Build linear system Ah=0 with h33=1 constraint => solve 8 unknowns
  // We solve A * h = b (8x8), where h = [h11 h12 h13 h21 h22 h23 h31 h32]
  const A = [];
  const b = [];
  for (let i = 0; i < 4; i++) {
    const x = srcPts[i].x, y = srcPts[i].y;
    const u = dstPts[i].x, v = dstPts[i].y;

    // u = (h11 x + h12 y + h13) / (h31 x + h32 y + 1)
    // v = (h21 x + h22 y + h23) / (h31 x + h32 y + 1)

    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }

  const h = solveLinearSystem(A, b); // length 8
  if (!h) return null;

  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1.0],
  ];
}

function solveLinearSystem(A, b) {
  // Gaussian elimination for small systems (8x8)
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    // pivot
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    if (Math.abs(M[pivot][col]) < 1e-9) return null;
    [M[col], M[pivot]] = [M[pivot], M[col]];

    // normalize
    const div = M[col][col];
    for (let c = col; c <= n; c++) M[col][c] /= div;

    // eliminate
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
    }
  }

  return M.map(row => row[n]);
}

function applyHomography(H, x, y) {
  const a = H[0][0] * x + H[0][1] * y + H[0][2];
  const b = H[1][0] * x + H[1][1] * y + H[1][2];
  const c = H[2][0] * x + H[2][1] * y + H[2][2];
  if (Math.abs(c) < 1e-9) return { x: NaN, y: NaN };
  return { x: a / c, y: b / c };
}

// --------------------------
// Marker detection (4 colored cones)
// Strategy: downsample frame, threshold by HSV ranges, find centroid of largest blob per marker
// --------------------------
function findMarkerCentroid(imageData, w, h, hsvLow, hsvHigh) {
  // imageData.data: RGBA
  let sumX = 0, sumY = 0, cnt = 0;

  // downsample step to reduce CPU
  const step = 4; // pixel stride
  for (let y = 0; y < h; y += step) {
    for (let x = 0; x < w; x += step) {
      const i = (y * w + x) * 4;
      const r = imageData.data[i];
      const g = imageData.data[i + 1];
      const b = imageData.data[i + 2];
      const hsv = rgbToHsv(r, g, b);

      const inRange =
        hsv.s >= hsvLow[1] && hsv.s <= hsvHigh[1] &&
        hsv.v >= hsvLow[2] && hsv.v <= hsvHigh[2] &&
        hueInRange(hsv.h, hsvLow[0], hsvHigh[0]);

      if (inRange) {
        sumX += x;
        sumY += y;
        cnt++;
      }
    }
  }
  if (cnt < 60) return null; // threshold; adjust if needed
  return { x: sumX / cnt, y: sumY / cnt, count: cnt };
}

function hueInRange(h, low, high) {
  // handle wrap-around (e.g. 300..360)
  if (low <= high) return h >= low && h <= high;
  return (h >= low && h <= 360) || (h >= 0 && h <= high);
}

async function calibrateField() {
  if (!video.videoWidth) {
    setStatus("Kalibrierung: Video/Webcam noch nicht bereit.");
    return;
  }
  offCtx.drawImage(video, 0, 0, off.width, off.height);
  const img = offCtx.getImageData(0, 0, off.width, off.height);

  const found = {};
  for (const key of Object.keys(MARKERS)) {
    const m = MARKERS[key];
    const c = findMarkerCentroid(img, off.width, off.height, m.hsvLow, m.hsvHigh);
    if (c) found[key] = { x: c.x, y: c.y };
  }

  const req = ["corner_tl", "corner_tr", "corner_br", "corner_bl"];
  if (!req.every(k => found[k])) {
    cornersPx = found;
    setStatus("Kalibrierung: Nicht alle 4 Hütchen gefunden. (Licht/Marker/Farben prüfen.)");
    return;
  }

  const prof = PROFILES[sportProfile.value];
  const src = [found.corner_tl, found.corner_tr, found.corner_br, found.corner_bl];
  const dst = [
    { x: 0, y: 0 },
    { x: prof.w, y: 0 },
    { x: prof.w, y: prof.h },
    { x: 0, y: prof.h },
  ];

  const Hnew = computeHomography4(src, dst);
  if (!Hnew) {
    setStatus("Kalibrierung: Homographie konnte nicht berechnet werden.");
    return;
  }

  H = Hnew;
  cornersPx = found;
  setStatus(`Kalibrierung OK: Homographie gesetzt (${sportProfile.value}).`);
}

// --------------------------
// Team classification (torso ROI average HSV)
// --------------------------
function classifyTeamFromTorsoROI(x1, y1, x2, y2) {
  // ROI: upper-middle torso region
  const w = off.width, h = off.height;
  x1 = clamp(Math.floor(x1), 0, w - 1);
  x2 = clamp(Math.floor(x2), 0, w - 1);
  y1 = clamp(Math.floor(y1), 0, h - 1);
  y2 = clamp(Math.floor(y2), 0, h - 1);
  if (x2 <= x1 || y2 <= y1) return { team: "unknown", score: Infinity };

  const rx1 = x1 + (x2 - x1) * 0.25;
  const rx2 = x1 + (x2 - x1) * 0.75;
  const ry1 = y1 + (y2 - y1) * 0.12;
  const ry2 = y1 + (y2 - y1) * 0.55;

  const W = off.width, Hh = off.height;
  const ix1 = clamp(Math.floor(rx1), 0, W - 1);
  const ix2 = clamp(Math.floor(rx2), 0, W - 1);
  const iy1 = clamp(Math.floor(ry1), 0, Hh - 1);
  const iy2 = clamp(Math.floor(ry2), 0, Hh - 1);

  const roiW = Math.max(1, ix2 - ix1);
  const roiH = Math.max(1, iy2 - iy1);

  const img = offCtx.getImageData(ix1, iy1, roiW, roiH);
  // sample sparsely
  let sum = { r: 0, g: 0, b: 0 }, cnt = 0;
  const step = 6;
  for (let y = 0; y < roiH; y += step) {
    for (let x = 0; x < roiW; x += step) {
      const i = (y * roiW + x) * 4;
      sum.r += img.data[i];
      sum.g += img.data[i + 1];
      sum.b += img.data[i + 2];
      cnt++;
    }
  }
  if (cnt < 10) return { team: "unknown", score: Infinity };
  const r = sum.r / cnt, g = sum.g / cnt, b = sum.b / cnt;
  const hsv = rgbToHsv(r, g, b);

  const a = hexToRgb(teamAColor.value);
  const bcol = hexToRgb(teamBColor.value);
  const hsvA = rgbToHsv(a.r, a.g, a.b);
  const hsvB = rgbToHsv(bcol.r, bcol.g, bcol.b);

  const dA = hsvDist(hsv, hsvA);
  const dB = hsvDist(hsv, hsvB);
  const tol = Number(teamTol.value);

  if (Math.min(dA, dB) > tol) return { team: "unknown", score: Math.min(dA, dB) };
  return dA < dB ? { team: "team_A", score: dA } : { team: "team_B", score: dB };
}

// --------------------------
// Tracking assignment
// --------------------------
function updateTracks(detections, tMs) {
  // detections: [{x,y, bbox:{x1,y1,x2,y2}, team}]
  // Create list of active track ids
  const ids = Array.from(tracks.keys());

  // Mark all tracks unseen initially
  const usedTracks = new Set();
  const assigned = new Map(); // detIndex -> trackId

  // Greedy nearest neighbor matching
  for (let di = 0; di < detections.length; di++) {
    let bestId = null;
    let bestDist = Infinity;
    for (const id of ids) {
      if (usedTracks.has(id)) continue;
      const tr = tracks.get(id);
      const dx = detections[di].x - tr.x;
      const dy = detections[di].y - tr.y;
      const d = Math.hypot(dx, dy);
      if (d < bestDist) {
        bestDist = d;
        bestId = id;
      }
    }
    if (bestId != null && bestDist <= ASSIGN_MAX_DIST_PX) {
      usedTracks.add(bestId);
      assigned.set(di, bestId);
    }
  }

  // Update assigned
  for (let di = 0; di < detections.length; di++) {
    const det = detections[di];
    if (assigned.has(di)) {
      const id = assigned.get(di);
      const tr = tracks.get(id);

      // simple smoothing
      const alpha = 0.65;
      const nx = alpha * tr.x + (1 - alpha) * det.x;
      const ny = alpha * tr.y + (1 - alpha) * det.y;

      tr.vx = (nx - tr.x);
      tr.vy = (ny - tr.y);
      tr.x = nx;
      tr.y = ny;
      tr.bbox = det.bbox;

      // team smoothing: only overwrite if confident
      if (det.team !== "unknown") tr.team = det.team;

      tr.lastSeen = tMs;
    } else {
      // create new track
      const id = nextTrackId++;
      tracks.set(id, {
        id,
        x: det.x,
        y: det.y,
        vx: 0,
        vy: 0,
        lastSeen: tMs,
        team: det.team,
        bbox: det.bbox,
      });
    }
  }

  // Remove stale tracks
  for (const [id, tr] of tracks.entries()) {
    if ((tMs - tr.lastSeen) > MAX_MISSES_MS) {
      tracks.delete(id);
    }
  }
}

// --------------------------
// Drawing
// --------------------------
function drawOverlay(poseResult) {
  octx.clearRect(0, 0, overlay.width, overlay.height);

  // draw corners if available
  if (cornersPx) {
    octx.save();
    octx.font = "14px system-ui";
    octx.lineWidth = 3;
    for (const key of Object.keys(cornersPx)) {
      const p = cornersPx[key];
      octx.beginPath();
      octx.arc(p.x, p.y, 7, 0, Math.PI * 2);
      octx.strokeStyle = "rgba(255,255,0,0.9)";
      octx.stroke();
      octx.fillStyle = "rgba(0,0,0,0.5)";
      octx.fillRect(p.x + 8, p.y - 18, 34, 18);
      octx.fillStyle = "white";
      octx.fillText(key.slice(-2).toUpperCase(), p.x + 12, p.y - 5);
    }
    octx.restore();
  }

  // draw tracks
  for (const tr of tracks.values()) {
    const color = tr.team === "team_A" ? teamAColor.value
                : tr.team === "team_B" ? teamBColor.value
                : "#ffffff";

    const x1 = tr.bbox?.x1 ?? (tr.x - 10);
    const y1 = tr.bbox?.y1 ?? (tr.y - 50);
    const x2 = tr.bbox?.x2 ?? (tr.x + 10);
    const y2 = tr.bbox?.y2 ?? (tr.y + 10);

    if (showBoxes.checked) {
      octx.strokeStyle = color;
      octx.lineWidth = 3;
      octx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }

    if (showIds.checked || showTeams.checked) {
      octx.save();
      octx.font = "16px system-ui";
      octx.fillStyle = "rgba(0,0,0,0.55)";
      octx.fillRect(x1, y1 - 22, 150, 22);
      octx.fillStyle = color;
      const txt = `${showIds.checked ? "#" + tr.id : ""}${showTeams.checked ? " " + tr.team : ""}`.trim();
      octx.fillText(txt, x1 + 6, y1 - 6);
      octx.restore();
    }

    // footpoint
    octx.beginPath();
    octx.arc(tr.x, tr.y, 5, 0, Math.PI * 2);
    octx.fillStyle = color;
    octx.fill();
  }

  // skeleton drawing (optional)
  if (showSkeleton.checked && poseResult?.landmarks?.length) {
    octx.save();
    octx.lineWidth = 2;
    for (let i = 0; i < poseResult.landmarks.length; i++) {
      const lm = poseResult.landmarks[i];
      // draw a few keypoints
      for (const idx of [11, 12, 23, 24, 25, 26, 27, 28]) { // shoulders/hips/knees/ankles
        const p = lm[idx];
        const x = p.x * overlay.width;
        const y = p.y * overlay.height;
        octx.beginPath();
        octx.arc(x, y, 3, 0, Math.PI * 2);
        octx.fillStyle = "rgba(255,255,255,0.85)";
        octx.fill();
      }
    }
    octx.restore();
  }
}

function drawMinimap() {
  mctx.clearRect(0, 0, minimap.width, minimap.height);

  const prof = PROFILES[sportProfile.value];
  // draw field rectangle
  mctx.save();
  mctx.strokeStyle = "rgba(255,255,255,0.35)";
  mctx.lineWidth = 2;
  mctx.strokeRect(30, 30, minimap.width - 60, minimap.height - 60);

  // map field coords -> minimap
  function f2m(fx, fy) {
    const x = 30 + (fx / prof.w) * (minimap.width - 60);
    const y = 30 + (fy / prof.h) * (minimap.height - 60);
    return { x, y };
  }

  // draw tracks in field coords if homography exists
  for (const tr of tracks.values()) {
    if (!H) continue;
    const p = applyHomography(H, tr.x, tr.y);
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;

    const inField = (p.x >= 0 && p.x <= prof.w && p.y >= 0 && p.y <= prof.h);
    const mm = f2m(p.x, p.y);

    const color = tr.team === "team_A" ? teamAColor.value
                : tr.team === "team_B" ? teamBColor.value
                : "#ffffff";

    mctx.beginPath();
    mctx.arc(mm.x, mm.y, 6, 0, Math.PI * 2);
    mctx.fillStyle = inField ? color : "rgba(255,255,255,0.25)";
    mctx.fill();

    mctx.font = "12px system-ui";
    mctx.fillStyle = "rgba(255,255,255,0.85)";
    mctx.fillText(String(tr.id), mm.x + 8, mm.y + 4);
  }

  mctx.restore();
}

// --------------------------
// Main loop
// --------------------------
async function init() {
  setStatus("Lade MediaPipe PoseLandmarker…");

  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  // Model: pose_landmarker_full supports multiple poses
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

function collectDetections(poseResult) {
  // We derive a bbox from landmarks and a footpoint from ankles.
  // We also classify team color using bbox torso ROI (pixel sampling).
  const dets = [];

  if (!poseResult?.landmarks?.length) return dets;

  // Need actual frame pixels for team classification
  offCtx.drawImage(video, 0, 0, off.width, off.height);

  for (let i = 0; i < poseResult.landmarks.length; i++) {
    const lm = poseResult.landmarks[i];

    // bbox from min/max of key landmarks
    let minX = 1, minY = 1, maxX = 0, maxY = 0;
    for (const p of lm) {
      if (p.visibility != null && p.visibility < 0.2) continue;
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    }
    // convert to pixels
    const x1 = minX * off.width;
    const y1 = minY * off.height;
    const x2 = maxX * off.width;
    const y2 = maxY * off.height;

    // footpoint: mean of ankles (27,28) if available, else hips (23,24)
    const a1 = lm[27], a2 = lm[28];
    const h1 = lm[23], h2 = lm[24];

    let fx, fy;
    if (a1 && a2) {
      fx = ((a1.x + a2.x) / 2) * off.width;
      fy = ((a1.y + a2.y) / 2) * off.height;
    } else {
      fx = ((h1.x + h2.x) / 2) * off.width;
      fy = ((h1.y + h2.y) / 2) * off.height;
    }

    const team = classifyTeamFromTorsoROI(x1, y1, x2, y2).team;

    dets.push({
      x: fx,
      y: fy,
      bbox: { x1, y1, x2, y2 },
      team
    });
  }
  return dets;
}

async function tick(tMs) {
  if (!running) return;

  const dt = tMs - lastFrameTime;
  const minDt = 1000 / fpsLimit;
  if (dt < minDt) {
    requestAnimationFrame(tick);
    return;
  }
  lastFrameTime = tMs;

  if (!video.videoWidth) {
    requestAnimationFrame(tick);
    return;
  }

  // pose inference
  const poseResult = landmarker.detectForVideo(video, tMs);

  // update tracks
  const dets = collectDetections(poseResult);
  updateTracks(dets, tMs);

  // draw overlays
  drawOverlay(poseResult);
  drawMinimap();

  requestAnimationFrame(tick);
}

// --------------------------
// UI handlers
// --------------------------
btnWebcam.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    await video.play();
    resizeCanvases();
    running = true;
    btnStop.disabled = false;
    btnWebcam.disabled = true;
    setStatus("Webcam läuft. Tracking aktiv.");
    requestAnimationFrame(tick);
  } catch (e) {
    setStatus("Webcam Fehler: " + String(e));
  }
});

btnStop.addEventListener("click", async () => {
  running = false;
  btnStop.disabled = true;
  btnWebcam.disabled = false;

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.pause();
  video.srcObject = null;
  video.removeAttribute("src");

  tracks.clear();
  H = null;
  cornersPx = null;

  octx.clearRect(0, 0, overlay.width, overlay.height);
  mctx.clearRect(0, 0, minimap.width, minimap.height);

  setStatus("Gestoppt.");
});

fileVideo.addEventListener("change", async () => {
  const f = fileVideo.files?.[0];
  if (!f) return;

  running = false;
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;

  tracks.clear();
  H = null;
  cornersPx = null;

  const url = URL.createObjectURL(f);
  video.src = url;
  video.muted = true;
  await video.play();
  resizeCanvases();

  running = true;
  btnStop.disabled = false;
  btnWebcam.disabled = true;
  setStatus("Video läuft. Tracking aktiv.");
  requestAnimationFrame(tick);
});

btnCalibrate.addEventListener("click", calibrateField);

fpsLimitInput.addEventListener("change", () => {
  fpsLimit = clamp(Number(fpsLimitInput.value) || 30, 5, 60);
});

// --------------------------
// Start
// --------------------------
await init();
