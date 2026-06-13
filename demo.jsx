import { useState, useRef, useEffect, useMemo, useCallback } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// =====================================================================
// WINDMESH — incident command console (proof of concept)
// LEFT MAP: live kriged wind field from the 7-node sensor mesh.
// RIGHT MAP: animated fire spread forecast driven by that wind field.
// NEW: satellite imagery underlay (Esri World Imagery, no API key) and
// a Leaflet location picker — search or click to drop the 1 km incident
// grid anywhere on Earth.
// =====================================================================

// ---------- mesh definition (mirrors a LoRa packet) ----------
const NODES = [
  { id: "N1", x: 150, y: 120, spd: 3.2, dir: 40 },
  { id: "N2", x: 480, y: 200, spd: 4.1, dir: 48 },
  { id: "N3", x: 820, y: 150, spd: 6.8, dir: 60 },
  { id: "N4", x: 300, y: 520, spd: 3.5, dir: 42 },
  { id: "N5", x: 700, y: 560, spd: 5.9, dir: 55 },
  { id: "N6", x: 180, y: 850, spd: 2.1, dir: 25 },
  { id: "N7", x: 600, y: 880, spd: 4.4, dir: 50 },
];
const N = NODES.length;
const DOMAIN = 1000;
const G = 48;
const CELL = DOMAIN / G;
const L_SCALE = 350,
  SIG2 = 1.0,
  NUG = 0.05;
const CANVAS = 640;

// ---------- geo helpers ----------
const M_PER_DEG_LAT = 111320;
function domainBBox(center) {
  const mLat = 1 / M_PER_DEG_LAT;
  const mLon = 1 / (M_PER_DEG_LAT * Math.cos((center.lat * Math.PI) / 180));
  return {
    west: center.lon - 500 * mLon,
    east: center.lon + 500 * mLon,
    south: center.lat - 500 * mLat,
    north: center.lat + 500 * mLat,
  };
}
function satelliteUrl(center) {
  const b = domainBBox(center);
  return (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export" +
    `?bbox=${b.west},${b.south},${b.east},${b.north}` +
    "&bboxSR=4326&size=640,640&format=png32&f=image"
  );
}
const ESRI_TILES =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";

// ---------- fire model (Rothermel surrogate) ----------
const R0 = 0.08;
const windFactor = (U) => 0.6 * Math.pow(U, 1.2);
const lengthToBreadth = (U) => Math.min(6, 1 + 0.25 * U);
function spreadRate(U, cosTheta) {
  const Rhead = R0 * (1 + windFactor(U));
  if (U < 0.3) return R0;
  const lb = lengthToBreadth(U);
  const eps = Math.sqrt(1 - 1 / (lb * lb));
  return (Rhead * (1 - eps)) / (1 - eps * cosTheta);
}

// ---------- kriging (exponential kernel) ----------
const cov = (d) => SIG2 * Math.exp(-d / L_SCALE);

function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++)
      if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

function buildKriging() {
  const A = [];
  for (let i = 0; i <= N; i++) A.push(new Array(N + 1).fill(0));
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const d = Math.hypot(NODES[i].x - NODES[j].x, NODES[i].y - NODES[j].y);
      A[i][j] = cov(d) + (i === j ? NUG : 0);
    }
    A[i][N] = 1;
    A[N][i] = 1;
  }
  const W = new Float32Array(G * G * N);
  const VAR = new Float32Array(G * G);
  for (let r = 0; r < G; r++) {
    for (let c = 0; c < G; c++) {
      const qx = ((c + 0.5) / G) * DOMAIN;
      const qy = ((r + 0.5) / G) * DOMAIN;
      const b = new Array(N + 1).fill(1);
      for (let i = 0; i < N; i++)
        b[i] = cov(Math.hypot(NODES[i].x - qx, NODES[i].y - qy));
      const x = solveLinear(A, b);
      const idx = (r * G + c) * N;
      let v = SIG2 + NUG;
      for (let i = 0; i < N; i++) {
        W[idx + i] = x[i];
        v -= x[i] * b[i];
      }
      v -= x[N];
      VAR[r * G + c] = Math.max(0, Math.min(1, v / (SIG2 + NUG)));
    }
  }
  return { W, VAR };
}

// ---------- fire arrival times: Dijkstra ----------
const NBRS = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
  [1, 1],
  [1, -1],
  [-1, 1],
  [-1, -1],
];

function computeArrival(W, su, sv, ign) {
  const cu = new Float32Array(G * G),
    cvv = new Float32Array(G * G);
  for (let g = 0; g < G * G; g++) {
    let u = 0,
      v = 0;
    const base = g * N;
    for (let i = 0; i < N; i++) {
      u += W[base + i] * su[i];
      v += W[base + i] * sv[i];
    }
    cu[g] = u;
    cvv[g] = v;
  }
  const time = new Float32Array(G * G).fill(Infinity);
  const sc = Math.max(0, Math.min(G - 1, Math.floor((ign.x / DOMAIN) * G)));
  const sr = Math.max(0, Math.min(G - 1, Math.floor((ign.y / DOMAIN) * G)));
  const start = sr * G + sc;
  time[start] = 0;

  const heap = [[0, start]];
  const push = (it) => {
    heap.push(it);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (heap[p][0] <= heap[i][0]) break;
      [heap[p], heap[i]] = [heap[i], heap[p]];
      i = p;
    }
  };
  const pop = () => {
    const top = heap[0],
      last = heap.pop();
    if (heap.length) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1,
          r = l + 1;
        let m = i;
        if (l < heap.length && heap[l][0] < heap[m][0]) m = l;
        if (r < heap.length && heap[r][0] < heap[m][0]) m = r;
        if (m === i) break;
        [heap[m], heap[i]] = [heap[i], heap[m]];
        i = m;
      }
    }
    return top;
  };

  while (heap.length) {
    const [t, idx] = pop();
    if (t > time[idx]) continue;
    const r = (idx / G) | 0,
      c = idx % G;
    const u = cu[idx],
      v = cvv[idx];
    const U = Math.hypot(u, v);
    for (const [dr, dc] of NBRS) {
      const nr = r + dr,
        nc = c + dc;
      if (nr < 0 || nr >= G || nc < 0 || nc >= G) continue;
      const dx = dc * CELL,
        dy = dr * CELL;
      const dist = Math.hypot(dx, dy);
      const cosT = U > 0.3 ? (dx * u + dy * v) / (dist * U) : 0;
      const R = spreadRate(U, cosT);
      const nt = t + dist / R;
      const ni = nr * G + nc;
      if (nt < time[ni]) {
        time[ni] = nt;
        push([nt, ni]);
      }
    }
  }
  for (let g = 0; g < G * G; g++) time[g] /= 60;
  return time;
}

// ---------- colors ----------
const STOPS = [
  [0.0, 255, 255, 204],
  [0.25, 254, 217, 118],
  [0.5, 253, 141, 60],
  [0.75, 227, 26, 28],
  [1.0, 128, 0, 38],
];
function ramp(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < STOPS.length; i++) {
    if (t <= STOPS[i][0]) {
      const [t0, r0, g0, b0] = STOPS[i - 1];
      const [t1, r1, g1, b1] = STOPS[i];
      const f = (t - t0) / (t1 - t0);
      return [r0 + f * (r1 - r0), g0 + f * (g1 - g0), b0 + f * (b1 - b0)];
    }
  }
  return [128, 0, 38];
}

const UPS = 4;
const GH = G * UPS;
const HORIZON = 20;
const LOOP_S = 9,
  HOLD_S = 1.6;
const FUEL_RGB = [30, 36, 32];

function burnColor(age) {
  const stops = [
    [0, 255, 214, 90],
    [1.2, 248, 130, 36],
    [3.5, 200, 62, 24],
    [7, 120, 34, 20],
    [13, 52, 26, 22],
    [20, 38, 22, 20],
  ];
  if (age <= 0) return stops[0].slice(1);
  for (let i = 1; i < stops.length; i++) {
    if (age <= stops[i][0]) {
      const [a0, r0, g0, b0] = stops[i - 1];
      const [a1, r1, g1, b1] = stops[i];
      const f = (age - a0) / (a1 - a0);
      return [r0 + f * (r1 - r0), g0 + f * (g1 - g0), b0 + f * (b1 - b0)];
    }
  }
  return stops[stops.length - 1].slice(1);
}

const toUV = (spd, dir) => {
  const th = (dir * Math.PI) / 180;
  return [spd * Math.sin(th), spd * Math.cos(th)];
};
const px = (x) => (x / DOMAIN) * CANVAS;
const py = (y) => CANVAS * (1 - y / DOMAIN);

export default function WindMesh() {
  const [targets, setTargets] = useState(
    NODES.map((n) => ({ spd: n.spd, dir: n.dir })),
  );
  const [selected, setSelected] = useState(2);
  const [showConf, setShowConf] = useState(false);
  const [ignition, setIgnition] = useState({ x: 250, y: 250 });
  const [burnedHa, setBurnedHa] = useState(0);
  const [satellite, setSatellite] = useState(false);
  const [satReady, setSatReady] = useState(false);
  // default: wildland-urban interface near Paradise, CA
  const [center, setCenter] = useState({ lat: 39.755, lon: -121.605 });
  const [showPicker, setShowPicker] = useState(false);
  const [query, setQuery] = useState("");
  const [searchMsg, setSearchMsg] = useState("");

  const windRef = useRef(null);
  const fireRef = useRef(null);
  const dialRef = useRef(null);
  const pickerDivRef = useRef(null);
  const pickerMapRef = useRef(null);
  const pickerMarkRef = useRef(null);
  const pickerRectRef = useRef(null);
  const pendingRef = useRef(null);
  const displayRef = useRef(NODES.map((n) => ({ spd: n.spd, dir: n.dir })));
  const targetsRef = useRef(targets);
  const confRef = useRef(showConf);
  const selRef = useRef(selected);
  const ignRef = useRef(ignition);
  const arrivalRef = useRef(null);
  const arrHiRef = useRef(null);
  const satImgRef = useRef(null);
  const satOnRef = useRef(false);
  targetsRef.current = targets;
  confRef.current = showConf;
  selRef.current = selected;
  ignRef.current = ignition;
  satOnRef.current = satellite && satReady;

  const { W, VAR } = useMemo(buildKriging, []);
  const offWind = useMemo(() => {
    const c = document.createElement("canvas");
    c.width = G;
    c.height = G;
    return c;
  }, []);
  const offFire = useMemo(() => {
    const c = document.createElement("canvas");
    c.width = GH;
    c.height = GH;
    return c;
  }, []);

  // ---------- satellite image fetch ----------
  useEffect(() => {
    if (!satellite) return;
    setSatReady(false);
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      satImgRef.current = img;
      setSatReady(true);
    };
    img.onerror = () => {
      satImgRef.current = null;
      setSatReady(false);
    };
    img.src = satelliteUrl(center);
  }, [satellite, center]);

  // ---------- fire forecast recompute ----------
  useEffect(() => {
    const su = [],
      sv = [];
    for (let i = 0; i < N; i++) {
      const [u, v] = toUV(targets[i].spd, targets[i].dir);
      su.push(u);
      sv.push(v);
    }
    const arr = computeArrival(W, su, sv, ignition);
    arrivalRef.current = arr;

    const hi = new Float32Array(GH * GH);
    for (let hr = 0; hr < GH; hr++) {
      let gy = (hr + 0.5) / UPS - 0.5;
      gy = Math.max(0, Math.min(G - 1.001, gy));
      const r0 = Math.floor(gy),
        fy = gy - r0;
      for (let hc = 0; hc < GH; hc++) {
        let gx = (hc + 0.5) / UPS - 0.5;
        gx = Math.max(0, Math.min(G - 1.001, gx));
        const c0 = Math.floor(gx),
          fx = gx - c0;
        const a = Math.min(arr[r0 * G + c0], 999);
        const b = Math.min(arr[r0 * G + c0 + 1], 999);
        const cc = Math.min(arr[(r0 + 1) * G + c0], 999);
        const d = Math.min(arr[(r0 + 1) * G + c0 + 1], 999);
        hi[hr * GH + hc] =
          a * (1 - fx) * (1 - fy) +
          b * fx * (1 - fy) +
          cc * (1 - fx) * fy +
          d * fx * fy;
      }
    }
    arrHiRef.current = hi;

    let burned = 0;
    for (let g = 0; g < G * G; g++) if (arr[g] <= 10) burned++;
    setBurnedHa((burned * CELL * CELL) / 10000);
  }, [targets, ignition, W]);

  // ---------- Leaflet location picker ----------
  useEffect(() => {
    if (!showPicker || !pickerDivRef.current) return;
    const map = L.map(pickerDivRef.current).setView(
      [center.lat, center.lon],
      13,
    );
    L.tileLayer(ESRI_TILES, {
      attribution: "Imagery © Esri — World Imagery",
      maxZoom: 18,
    }).addTo(map);

    const boundsFor = (lat, lon) => {
      const b = domainBBox({ lat, lon });
      return [
        [b.south, b.west],
        [b.north, b.east],
      ];
    };
    const mark = L.circleMarker([center.lat, center.lon], {
      radius: 8,
      color: "#fff",
      weight: 2,
      fillColor: "#FF7A3D",
      fillOpacity: 1,
    }).addTo(map);
    const rect = L.rectangle(boundsFor(center.lat, center.lon), {
      color: "#FF7A3D",
      weight: 2,
      fill: false,
      dashArray: "6 5",
    }).addTo(map);
    pendingRef.current = { lat: center.lat, lon: center.lon };

    map.on("click", (e) => {
      const { lat, lng } = e.latlng;
      mark.setLatLng([lat, lng]);
      rect.setBounds(boundsFor(lat, lng));
      pendingRef.current = { lat, lon: lng };
    });

    pickerMapRef.current = map;
    pickerMarkRef.current = mark;
    pickerRectRef.current = rect;
    // Leaflet needs a size kick when created inside a fresh modal
    setTimeout(() => map.invalidateSize(), 60);
    return () => {
      map.remove();
      pickerMapRef.current = null;
    };
  }, [showPicker]); // eslint-disable-line react-hooks/exhaustive-deps

  const doSearch = async () => {
    if (!query.trim()) return;
    setSearchMsg("searching…");
    try {
      const r = await fetch(
        "https://nominatim.openstreetmap.org/search?format=json&limit=1&q=" +
          encodeURIComponent(query),
      );
      const j = await r.json();
      if (j && j[0]) {
        const lat = +j[0].lat,
          lon = +j[0].lon;
        const map = pickerMapRef.current;
        if (map) {
          map.setView([lat, lon], 13);
          pickerMarkRef.current.setLatLng([lat, lon]);
          const b = domainBBox({ lat, lon });
          pickerRectRef.current.setBounds([
            [b.south, b.west],
            [b.north, b.east],
          ]);
          pendingRef.current = { lat, lon };
        }
        setSearchMsg(j[0].display_name.split(",").slice(0, 3).join(","));
      } else setSearchMsg("no results — try a different name");
    } catch {
      setSearchMsg("search failed — check your connection");
    }
  };

  const confirmLocation = () => {
    if (pendingRef.current) {
      setCenter({ ...pendingRef.current });
      setSatellite(true);
    }
    setShowPicker(false);
  };

  // ---------- render loop ----------
  useEffect(() => {
    let raf;
    const imgW = offWind.getContext("2d").createImageData(G, G);
    const imgF = offFire.getContext("2d").createImageData(GH, GH);

    const drawArrows = (ctx, su, sv, vmax, color) => {
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 1.4;
      const step = 4,
        cw = CANVAS / G;
      for (let r = 2; r < G; r += step) {
        for (let c = 2; c < G; c += step) {
          let u = 0,
            v = 0;
          const base = (r * G + c) * N;
          for (let i = 0; i < N; i++) {
            u += W[base + i] * su[i];
            v += W[base + i] * sv[i];
          }
          const s = Math.hypot(u, v);
          if (s < 0.2) continue;
          const cx = (c + 0.5) * cw,
            cy = CANVAS - (r + 0.5) * cw;
          const len = 8 + (s / vmax) * 14;
          const dx = (u / s) * len,
            dy = (-v / s) * len;
          ctx.beginPath();
          ctx.moveTo(cx - dx / 2, cy - dy / 2);
          ctx.lineTo(cx + dx / 2, cy + dy / 2);
          ctx.stroke();
          const ang = Math.atan2(dy, dx);
          ctx.beginPath();
          ctx.moveTo(cx + dx / 2, cy + dy / 2);
          ctx.lineTo(
            cx + dx / 2 - 5 * Math.cos(ang - 0.45),
            cy + dy / 2 - 5 * Math.sin(ang - 0.45),
          );
          ctx.lineTo(
            cx + dx / 2 - 5 * Math.cos(ang + 0.45),
            cy + dy / 2 - 5 * Math.sin(ang + 0.45),
          );
          ctx.fill();
        }
      }
    };

    const drawNodes = (ctx, su, sv, vmax, subtle) => {
      NODES.forEach((n, i) => {
        const X = px(n.x),
          Y = py(n.y);
        const isSel = !subtle && i === selRef.current;
        if (!subtle) {
          const u = su[i],
            v = sv[i];
          const s = Math.hypot(u, v) || 1;
          const len = 14 + (s / vmax) * 18;
          ctx.strokeStyle = isSel ? "#FF7A3D" : "#1d4ed8";
          ctx.lineWidth = 2.4;
          ctx.beginPath();
          ctx.moveTo(X, Y);
          ctx.lineTo(X + (u / s) * len, Y - (v / s) * len);
          ctx.stroke();
        }
        ctx.beginPath();
        ctx.arc(X, Y, subtle ? 5 : isSel ? 11 : 8, 0, Math.PI * 2);
        ctx.fillStyle = subtle
          ? "rgba(125,211,252,0.55)"
          : isSel
            ? "#FF7A3D"
            : "#7dd3fc";
        ctx.fill();
        ctx.lineWidth = isSel ? 3 : 1.6;
        ctx.strokeStyle = subtle
          ? "rgba(15,23,42,0.5)"
          : isSel
            ? "#fff"
            : "#0f172a";
        ctx.stroke();
        if (!subtle) {
          ctx.font = "700 13px ui-monospace, Menlo, monospace";
          ctx.fillStyle = "#0f172a";
          ctx.strokeStyle = "rgba(255,255,255,0.85)";
          ctx.lineWidth = 3;
          ctx.strokeText(n.id, X + 13, Y - 10);
          ctx.fillText(n.id, X + 13, Y - 10);
        }
      });
    };

    const draw = () => {
      const disp = displayRef.current;
      const tg = targetsRef.current;
      for (let i = 0; i < N; i++) {
        disp[i].spd += (tg[i].spd - disp[i].spd) * 0.08;
        let dd = ((tg[i].dir - disp[i].dir + 540) % 360) - 180;
        disp[i].dir = (disp[i].dir + dd * 0.08 + 360) % 360;
      }
      const su = [],
        sv = [];
      for (let i = 0; i < N; i++) {
        const [u, v] = toUV(disp[i].spd, disp[i].dir);
        su.push(u);
        sv.push(v);
      }
      const vmax = Math.max(12, ...tg.map((t) => t.spd));
      const satOn = satOnRef.current && satImgRef.current;

      // ===== wind map =====
      for (let gr = 0; gr < G; gr++) {
        for (let gc = 0; gc < G; gc++) {
          let u = 0,
            v = 0;
          const base = (gr * G + gc) * N;
          for (let i = 0; i < N; i++) {
            u += W[base + i] * su[i];
            v += W[base + i] * sv[i];
          }
          const s = Math.hypot(u, v);
          const [r, gg, b] = ramp((s - 1) / (vmax - 1));
          const p = ((G - 1 - gr) * G + gc) * 4;
          imgW.data[p] = r;
          imgW.data[p + 1] = gg;
          imgW.data[p + 2] = b;
          imgW.data[p + 3] = 255;
        }
      }
      offWind.getContext("2d").putImageData(imgW, 0, 0);
      const wc = windRef.current;
      if (wc) {
        const ctx = wc.getContext("2d");
        ctx.imageSmoothingEnabled = true;
        ctx.clearRect(0, 0, CANVAS, CANVAS);
        if (satOn) {
          ctx.drawImage(satImgRef.current, 0, 0, CANVAS, CANVAS);
          ctx.globalAlpha = 0.52;
          ctx.drawImage(offWind, 0, 0, CANVAS, CANVAS);
          ctx.globalAlpha = 1;
        } else {
          ctx.drawImage(offWind, 0, 0, CANVAS, CANVAS);
        }
        if (confRef.current) {
          for (let r = 0; r < G; r++) {
            for (let c = 0; c < G; c++) {
              const a = VAR[r * G + c];
              if (a > 0.08) {
                ctx.fillStyle = `rgba(80,60,140,${a * 0.55})`;
                const cw = CANVAS / G;
                ctx.fillRect(c * cw, CANVAS - (r + 1) * cw, cw + 1, cw + 1);
              }
            }
          }
        }
        drawArrows(ctx, su, sv, vmax, "rgba(20,18,16,0.66)");
        drawNodes(ctx, su, sv, vmax, false);
      }

      // ===== fire forecast map (animated playback) =====
      const hi = arrHiRef.current;
      const fc = fireRef.current;
      if (hi && fc) {
        const nowMs = performance.now();
        const phase = (nowMs / 1000) % (LOOP_S + HOLD_S);
        const tNow = Math.min(HORIZON, (phase / LOOP_S) * HORIZON);

        for (let hr = 0; hr < GH; hr++) {
          for (let hc = 0; hc < GH; hc++) {
            const t = hi[hr * GH + hc];
            let R, Gc, B, Al;
            if (t <= tNow) {
              [R, Gc, B] = burnColor(tNow - t);
              Al = 255;
            } else if (satOn) {
              // unburned shows the satellite terrain through
              R = 255;
              Gc = 255;
              B = 255;
              Al = 0;
              if (t <= HORIZON) {
                const d5 = Math.abs(t - Math.round(t / 5) * 5);
                if (d5 < 0.1 && t > 0.5) {
                  R = 255;
                  Gc = 255;
                  B = 255;
                  Al = 95;
                }
              }
            } else {
              [R, Gc, B] = FUEL_RGB;
              Al = 255;
              if (t <= HORIZON) {
                const d5 = Math.abs(t - Math.round(t / 5) * 5);
                if (d5 < 0.1 && t > 0.5) {
                  R += 26;
                  Gc += 28;
                  B += 26;
                }
              }
            }
            const dt = t - tNow;
            if (Math.abs(dt) < 1.6 && tNow > 0.05 && tNow < HORIZON) {
              const flick =
                0.78 +
                0.22 * Math.sin(nowMs * 0.012 + ((hr * 97 + hc * 57) % 251));
              const glow = Math.exp(-(dt * dt) / 0.32) * flick;
              R += (255 - R) * glow;
              Gc += (235 - Gc) * glow * 0.9;
              B += (160 - B) * glow * 0.55;
              Al = Math.max(Al, glow * 240);
            }
            const p = ((GH - 1 - hr) * GH + hc) * 4;
            imgF.data[p] = R;
            imgF.data[p + 1] = Gc;
            imgF.data[p + 2] = B;
            imgF.data[p + 3] = Al;
          }
        }
        offFire.getContext("2d").putImageData(imgF, 0, 0);
        const ctx = fc.getContext("2d");
        ctx.imageSmoothingEnabled = true;
        ctx.clearRect(0, 0, CANVAS, CANVAS);
        if (satOn) ctx.drawImage(satImgRef.current, 0, 0, CANVAS, CANVAS);
        else {
          ctx.fillStyle = `rgb(${FUEL_RGB.join(",")})`;
          ctx.fillRect(0, 0, CANVAS, CANVAS);
        }
        ctx.drawImage(offFire, 0, 0, CANVAS, CANVAS);
        drawArrows(
          ctx,
          su,
          sv,
          vmax,
          satOn ? "rgba(255,255,255,0.35)" : "rgba(232,227,216,0.22)",
        );
        drawNodes(ctx, su, sv, vmax, true);
        const ix = px(ignRef.current.x),
          iy = py(ignRef.current.y);
        const pulse = 11 + 3 * Math.sin(nowMs * 0.006);
        ctx.beginPath();
        ctx.arc(ix, iy, pulse, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(255,255,255,0.9)";
        ctx.lineWidth = 2.5;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(ix, iy, 5.5, 0, Math.PI * 2);
        ctx.fillStyle = "#fde047";
        ctx.fill();
        ctx.font = "800 12px ui-monospace, monospace";
        ctx.strokeStyle = "rgba(0,0,0,0.7)";
        ctx.lineWidth = 3;
        ctx.strokeText("IGN", ix + 16, iy + 4);
        ctx.fillStyle = "#fde047";
        ctx.fillText("IGN", ix + 16, iy + 4);
        const mm = Math.floor(tNow),
          ss = Math.floor((tNow - mm) * 60);
        const label = `T+${mm}:${ss.toString().padStart(2, "0")}`;
        ctx.font = "800 19px ui-monospace, monospace";
        ctx.strokeStyle = "rgba(0,0,0,0.75)";
        ctx.lineWidth = 4;
        ctx.strokeText(label, 16, 32);
        ctx.fillStyle = "#fde68a";
        ctx.fillText(label, 16, 32);
        ctx.fillStyle = "rgba(255,255,255,0.18)";
        ctx.fillRect(16, 42, 110, 4);
        ctx.fillStyle = "#fb923c";
        ctx.fillRect(16, 42, (tNow / HORIZON) * 110, 4);
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [W, VAR, offWind, offFire]);

  // ---------- clicks ----------
  const canvasCoords = (e, ref) => {
    const rect = ref.current.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * CANVAS;
    const my = ((e.clientY - rect.top) / rect.height) * CANVAS;
    return { x: (mx / CANVAS) * DOMAIN, y: (1 - my / CANVAS) * DOMAIN, mx, my };
  };
  const onWindClick = (e) => {
    const { mx, my } = canvasCoords(e, windRef);
    let best = 0,
      bd = 1e9;
    NODES.forEach((n, i) => {
      const d = Math.hypot(px(n.x) - mx, py(n.y) - my);
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    if (bd < 60) setSelected(best);
  };
  const onFireClick = (e) => {
    const { x, y } = canvasCoords(e, fireRef);
    setIgnition({
      x: Math.max(0, Math.min(DOMAIN, x)),
      y: Math.max(0, Math.min(DOMAIN, y)),
    });
  };

  // ---------- compass dial ----------
  const setDirFromEvent = useCallback((e) => {
    const rect = dialRef.current.getBoundingClientRect();
    const cx = rect.left + rect.width / 2,
      cy = rect.top + rect.height / 2;
    const pt = e.touches ? e.touches[0] : e;
    const dx = pt.clientX - cx,
      dy = pt.clientY - cy;
    const bearing = ((Math.atan2(dx, -dy) * 180) / Math.PI + 360) % 360;
    setTargets((t) =>
      t.map((v, i) =>
        i === selRef.current ? { ...v, dir: Math.round(bearing) } : v,
      ),
    );
  }, []);
  const dialDrag = (e) => {
    e.preventDefault();
    setDirFromEvent(e);
    const move = (ev) => setDirFromEvent(ev);
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      window.removeEventListener("touchmove", move);
      window.removeEventListener("touchend", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    window.addEventListener("touchmove", move);
    window.addEventListener("touchend", up);
  };

  const sel = targets[selected];
  const compass = (d) => {
    const pts = [
      "N",
      "NNE",
      "NE",
      "ENE",
      "E",
      "ESE",
      "SE",
      "SSE",
      "S",
      "SSW",
      "SW",
      "WSW",
      "W",
      "WNW",
      "NW",
      "NNW",
    ];
    return pts[Math.round(d / 22.5) % 16];
  };
  const reset = () =>
    setTargets(NODES.map((n) => ({ spd: n.spd, dir: n.dir })));
  const gust = () =>
    setTargets((t) =>
      t.map((v, i) =>
        i === selected
          ? { spd: Math.min(20, v.spd * 1.9), dir: (v.dir + 58) % 360 }
          : v,
      ),
    );

  const DIAL = 168;
  const dialAng = ((sel.dir - 90) * Math.PI) / 180;

  const mapBox = {
    flex: "1 1 360px",
    minWidth: 300,
    maxWidth: 700,
    display: "flex",
    flexDirection: "column",
    gap: 8,
  };
  const mapTitle = {
    fontSize: 11.5,
    letterSpacing: "0.12em",
    color: "#9aa6ad",
  };
  const canvasStyle = {
    width: "100%",
    borderRadius: 10,
    cursor: "pointer",
    border: "1px solid #2a3138",
    display: "block",
    boxShadow: "0 8px 30px rgba(0,0,0,0.45)",
  };
  const btn = (primary) => ({
    padding: "9px 14px",
    borderRadius: 9,
    cursor: "pointer",
    fontSize: 13,
    fontWeight: primary ? 800 : 600,
    border: primary ? "1px solid #FF7A3D" : "1px solid #313b44",
    background: primary ? "#FF7A3D" : "transparent",
    color: primary ? "#15191d" : "#cfd6da",
  });

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#15191d",
        color: "#e8e3d8",
        fontFamily: "'Segoe UI', system-ui, sans-serif",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: 16,
          flexWrap: "wrap",
          padding: "14px 22px",
          borderBottom: "1px solid #2a3138",
        }}
      >
        <div style={{ fontSize: 21, fontWeight: 800, letterSpacing: "0.06em" }}>
          WINDMESH<span style={{ color: "#FF7A3D" }}>▲</span>
        </div>
        <div
          style={{
            fontFamily: "ui-monospace, Menlo, monospace",
            fontSize: 12.5,
            color: "#9aa6ad",
          }}
        >
          INCIDENT GRID 1 km² · {N} NODES ONLINE
          {satellite &&
            satReady &&
            ` · ${center.lat.toFixed(4)}, ${center.lon.toFixed(4)}`}
        </div>
        <div
          style={{
            marginLeft: "auto",
            display: "flex",
            gap: 18,
            fontFamily: "ui-monospace, monospace",
            fontSize: 13,
          }}
        >
          <span style={{ color: "#FFC857" }}>
            {NODES[selected].id} → {sel.spd.toFixed(1)} m/s @ {sel.dir}°{" "}
            {compass(sel.dir)}
          </span>
          <span style={{ color: "#f87171" }}>
            BURNED @ +10 MIN: {burnedHa.toFixed(1)} ha
          </span>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 18,
          padding: 18,
          flex: 1,
        }}
      >
        <div style={mapBox}>
          <div style={mapTitle}>WIND FIELD · tap a node to select</div>
          <canvas
            ref={windRef}
            width={CANVAS}
            height={CANVAS}
            onClick={onWindClick}
            style={canvasStyle}
          />
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              fontFamily: "ui-monospace, monospace",
              fontSize: 12,
              color: "#9aa6ad",
            }}
          >
            <span>1 m/s</span>
            <div
              style={{
                flex: 1,
                height: 10,
                borderRadius: 5,
                background:
                  "linear-gradient(90deg,#ffffcc,#fed976,#fd8d3c,#e31a1c,#800026)",
              }}
            />
            <span>peak</span>
          </div>
          {satellite && satReady && (
            <div style={{ fontSize: 10.5, color: "#6c7a83" }}>
              Imagery © Esri — World Imagery
            </div>
          )}
        </div>

        <div style={mapBox}>
          <div style={mapTitle}>
            FIRE SPREAD FORECAST · 20-min playback loop · tap to move ignition
          </div>
          <canvas
            ref={fireRef}
            width={CANVAS}
            height={CANVAS}
            onClick={onFireClick}
            style={canvasStyle}
          />
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              flexWrap: "wrap",
              fontFamily: "ui-monospace, monospace",
              fontSize: 12,
              color: "#9aa6ad",
            }}
          >
            <span>front</span>
            <div
              style={{
                width: 130,
                height: 11,
                borderRadius: 5,
                background:
                  "linear-gradient(90deg,#ffd65a,#f88224,#c83e18,#782214,#34161a)",
              }}
            />
            <span>char</span>
            <span style={{ marginLeft: 8 }}>
              faint rings = 5-min isochrones
            </span>
          </div>
          <div style={{ fontSize: 11, color: "#6c7a83" }}>
            model: wind-driven Rothermel surrogate · uniform fuel · flat terrain
            {satellite &&
              satReady &&
              " · unburned area shows live satellite terrain"}
          </div>
        </div>

        <div
          style={{
            flex: "0 1 320px",
            minWidth: 290,
            display: "flex",
            flexDirection: "column",
            gap: 16,
          }}
        >
          {/* location / imagery */}
          <div
            style={{
              background: "#1b2127",
              borderRadius: 12,
              padding: 14,
              border: "1px solid #2a3138",
              display: "flex",
              flexDirection: "column",
              gap: 10,
            }}
          >
            <div
              style={{
                fontSize: 11.5,
                letterSpacing: "0.12em",
                color: "#9aa6ad",
              }}
            >
              MAP LAYER
            </div>
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: 9,
                fontSize: 13,
                color: "#cfd6da",
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                checked={satellite}
                onChange={(e) => setSatellite(e.target.checked)}
                style={{ accentColor: "#FF7A3D", width: 16, height: 16 }}
              />
              Satellite imagery underlay
            </label>
            <button onClick={() => setShowPicker(true)} style={btn(false)}>
              Choose incident location…
            </button>
            {satellite && !satReady && (
              <div style={{ fontSize: 11.5, color: "#FFC857" }}>
                loading imagery…
              </div>
            )}
          </div>

          <div>
            <div
              style={{
                fontSize: 11.5,
                letterSpacing: "0.12em",
                color: "#9aa6ad",
                marginBottom: 8,
              }}
            >
              SENSOR NODES
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
              {NODES.map((n, i) => (
                <button
                  key={n.id}
                  onClick={() => setSelected(i)}
                  style={{
                    padding: "7px 11px",
                    borderRadius: 8,
                    cursor: "pointer",
                    fontFamily: "ui-monospace, monospace",
                    fontSize: 12.5,
                    background: i === selected ? "#FF7A3D" : "#222a31",
                    color: i === selected ? "#15191d" : "#cfd6da",
                    border:
                      i === selected
                        ? "1px solid #FF7A3D"
                        : "1px solid #313b44",
                    fontWeight: i === selected ? 800 : 500,
                  }}
                >
                  {n.id}{" "}
                  <span style={{ opacity: 0.75 }}>
                    {targets[i].spd.toFixed(1)}
                  </span>
                </button>
              ))}
            </div>
          </div>

          <div
            style={{
              background: "#1b2127",
              borderRadius: 12,
              padding: 16,
              border: "1px solid #2a3138",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline",
              }}
            >
              <span
                style={{
                  fontSize: 11.5,
                  letterSpacing: "0.12em",
                  color: "#9aa6ad",
                }}
              >
                WIND SPEED
              </span>
              <span
                style={{
                  fontFamily: "ui-monospace, monospace",
                  fontSize: 22,
                  fontWeight: 800,
                  color: "#FFC857",
                }}
              >
                {sel.spd.toFixed(1)}
                <span style={{ fontSize: 12, color: "#9aa6ad" }}> m/s</span>
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={20}
              step={0.1}
              value={sel.spd}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                setTargets((t) =>
                  t.map((x, i) => (i === selected ? { ...x, spd: v } : x)),
                );
              }}
              style={{ width: "100%", accentColor: "#FF7A3D", marginTop: 10 }}
            />
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: 11,
                color: "#6c7a83",
                fontFamily: "ui-monospace, monospace",
              }}
            >
              <span>calm</span>
              <span>10</span>
              <span>20 m/s</span>
            </div>
          </div>

          <div
            style={{
              background: "#1b2127",
              borderRadius: 12,
              padding: 16,
              border: "1px solid #2a3138",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline",
                marginBottom: 8,
              }}
            >
              <span
                style={{
                  fontSize: 11.5,
                  letterSpacing: "0.12em",
                  color: "#9aa6ad",
                }}
              >
                WIND DIRECTION
              </span>
              <span
                style={{
                  fontFamily: "ui-monospace, monospace",
                  fontSize: 22,
                  fontWeight: 800,
                  color: "#FFC857",
                }}
              >
                {sel.dir}°
                <span style={{ fontSize: 12, color: "#9aa6ad" }}>
                  {" "}
                  {compass(sel.dir)}
                </span>
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "center" }}>
              <svg
                ref={dialRef}
                width={DIAL}
                height={DIAL}
                viewBox={`0 0 ${DIAL} ${DIAL}`}
                onMouseDown={dialDrag}
                onTouchStart={dialDrag}
                style={{ cursor: "grab", touchAction: "none" }}
              >
                <circle
                  cx={DIAL / 2}
                  cy={DIAL / 2}
                  r={DIAL / 2 - 4}
                  fill="#222a31"
                  stroke="#313b44"
                  strokeWidth="2"
                />
                {[...Array(16)].map((_, i) => {
                  const a = ((i * 22.5 - 90) * Math.PI) / 180;
                  const major = i % 4 === 0;
                  const r1 = DIAL / 2 - (major ? 16 : 10),
                    r2 = DIAL / 2 - 6;
                  return (
                    <line
                      key={i}
                      x1={DIAL / 2 + r1 * Math.cos(a)}
                      y1={DIAL / 2 + r1 * Math.sin(a)}
                      x2={DIAL / 2 + r2 * Math.cos(a)}
                      y2={DIAL / 2 + r2 * Math.sin(a)}
                      stroke={major ? "#9aa6ad" : "#4a565f"}
                      strokeWidth={major ? 2.4 : 1.4}
                    />
                  );
                })}
                {["N", "E", "S", "W"].map((c, i) => {
                  const a = ((i * 90 - 90) * Math.PI) / 180;
                  const r = DIAL / 2 - 28;
                  return (
                    <text
                      key={c}
                      x={DIAL / 2 + r * Math.cos(a)}
                      y={DIAL / 2 + r * Math.sin(a) + 5}
                      textAnchor="middle"
                      fill={c === "N" ? "#FF7A3D" : "#9aa6ad"}
                      fontSize="13"
                      fontWeight="800"
                      fontFamily="ui-monospace, monospace"
                    >
                      {c}
                    </text>
                  );
                })}
                <line
                  x1={DIAL / 2 - (DIAL / 2 - 46) * 0.35 * Math.cos(dialAng)}
                  y1={DIAL / 2 - (DIAL / 2 - 46) * 0.35 * Math.sin(dialAng)}
                  x2={DIAL / 2 + (DIAL / 2 - 40) * Math.cos(dialAng)}
                  y2={DIAL / 2 + (DIAL / 2 - 40) * Math.sin(dialAng)}
                  stroke="#FF7A3D"
                  strokeWidth="4"
                  strokeLinecap="round"
                />
                <polygon
                  points={`0,-7 14,0 0,7`}
                  transform={`translate(${DIAL / 2 + (DIAL / 2 - 36) * Math.cos(dialAng)},${DIAL / 2 + (DIAL / 2 - 36) * Math.sin(dialAng)}) rotate(${sel.dir - 90})`}
                  fill="#FF7A3D"
                />
                <circle
                  cx={DIAL / 2}
                  cy={DIAL / 2}
                  r="7"
                  fill="#FFC857"
                  stroke="#15191d"
                  strokeWidth="2.5"
                />
              </svg>
            </div>
            <div
              style={{
                textAlign: "center",
                fontSize: 11,
                color: "#6c7a83",
                marginTop: 4,
              }}
            >
              drag the needle — arrow shows where the wind pushes
            </div>
          </div>

          <div style={{ display: "flex", gap: 9 }}>
            <button onClick={gust} style={{ ...btn(true), flex: 1 }}>
              Simulate gust at {NODES[selected].id}
            </button>
            <button onClick={reset} style={{ ...btn(false), flex: 1 }}>
              Reset to baseline
            </button>
          </div>

          <label
            style={{
              display: "flex",
              alignItems: "center",
              gap: 9,
              fontSize: 13,
              color: "#cfd6da",
              cursor: "pointer",
            }}
          >
            <input
              type="checkbox"
              checked={showConf}
              onChange={(e) => setShowConf(e.target.checked)}
              style={{ accentColor: "#FF7A3D", width: 16, height: 16 }}
            />
            Show estimation confidence (purple = model is guessing — fewer
            nearby nodes)
          </label>
        </div>
      </div>

      {/* location picker modal */}
      {showPicker && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(8,10,12,0.78)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000,
            padding: 16,
          }}
        >
          <div
            style={{
              background: "#1b2127",
              border: "1px solid #2a3138",
              borderRadius: 14,
              width: "min(860px, 96vw)",
              padding: 16,
              display: "flex",
              flexDirection: "column",
              gap: 12,
            }}
          >
            <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
              <div style={{ fontWeight: 800, fontSize: 16 }}>
                Choose incident location
              </div>
              <div style={{ fontSize: 12, color: "#9aa6ad" }}>
                search or click the map — dashed box = your 1 km sensor grid
              </div>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") doSearch();
                }}
                placeholder="Search a place — e.g. Paradise CA, Amherst MA…"
                style={{
                  flex: 1,
                  padding: "9px 12px",
                  borderRadius: 8,
                  fontSize: 13.5,
                  background: "#15191d",
                  border: "1px solid #313b44",
                  color: "#e8e3d8",
                  outline: "none",
                }}
              />
              <button onClick={doSearch} style={btn(false)}>
                Search
              </button>
            </div>
            {searchMsg && (
              <div style={{ fontSize: 12, color: "#9aa6ad" }}>{searchMsg}</div>
            )}
            <div
              ref={pickerDivRef}
              style={{
                height: "min(440px, 55vh)",
                borderRadius: 10,
                overflow: "hidden",
                border: "1px solid #2a3138",
              }}
            />
            <div
              style={{ display: "flex", gap: 9, justifyContent: "flex-end" }}
            >
              <button onClick={() => setShowPicker(false)} style={btn(false)}>
                Cancel
              </button>
              <button onClick={confirmLocation} style={btn(true)}>
                Use this location
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
