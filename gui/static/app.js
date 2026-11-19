"use strict";
/* CV Monitor GUI – vanilla JS, no external libraries (works offline). */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtT = s => { s = Math.max(0, s || 0); return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`; };
async function api(path, body, method) {
  const opt = body === undefined ? {} : { method: method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  let j = null; try { j = await r.json(); } catch (e) { }
  if (!r.ok) throw new Error((j && j.error) || r.statusText);
  return j;
}
function nest(map) {            // {"a.b": 1} -> {a:{b:1}}
  const out = {};
  for (const [k, v] of Object.entries(map)) { let c = out; const p = k.split("."); p.slice(0, -1).forEach(x => c = (c[x] = c[x] || {})); c[p.at(-1)] = v; }
  return out;
}
function toast(msg, ms = 6000) {
  const d = document.createElement("div"); d.innerHTML = msg; $("#toast").appendChild(d); setTimeout(() => d.remove(), ms);
}
let actx;
function beep() {
  if (!$("#sound").checked) return;
  try {
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    [0, 0.3].forEach(dt => {
      const o = actx.createOscillator(), g = actx.createGain(); o.type = "square"; o.frequency.value = 880;
      g.gain.value = 0.06; o.connect(g); g.connect(actx.destination); o.start(actx.currentTime + dt); o.stop(actx.currentTime + dt + 0.2);
    });
  } catch (e) { }
}

/* ------------------------------------------------------------------ navigation */
const loaders = { setup: () => { loadSources(); loadSetup(); }, settings: loadSettings, history: loadRuns, results: loadResults };
function goto(page) {
  $$("#nav button").forEach(x => x.classList.toggle("on", x.dataset.p === page));
  $$(".page").forEach(p => p.classList.toggle("on", p.id === "p-" + page));
  (loaders[page] || (() => { }))(); window.scrollTo({ top: 0, behavior: "smooth" });
}
$("#nav").addEventListener("click", e => { const b = e.target.closest("button"); if (b) goto(b.dataset.p); });
$("#logo").onclick = () => goto("monitor");
document.addEventListener("click", e => { const l = e.target.closest("[data-go]"); if (l) goto(l.dataset.go); });
const startFromAnywhere = () => { goto("monitor"); if (!$("#bStart").disabled) $("#bStart").click(); };
$("#navStart").onclick = startFromAnywhere; $("#footStart").onclick = startFromAnywhere;
$("#rtabs").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  $$("#rtabs button").forEach(x => x.classList.toggle("on", x === b));
  ["events", "objects", "gallery"].forEach(t => $("#t-" + t).style.display = t === b.dataset.t ? "" : "none");
});
$("#modal").addEventListener("click", e => { if (e.target.id === "modal") $("#modal").classList.remove("on"); });
function modal(html) { $("#modalBox").innerHTML = html; $("#modal").classList.add("on"); }

/* ------------------------------------------------------------------ sources & controls */
let pending = {};                     // settings changed in the GUI but not saved as defaults
async function loadSources() {
  const list = await api("/api/sources");
  for (const sel of [$("#src"), $("#setupSrc")]) {
    const cur = sel.value;
    sel.innerHTML = list.length ? list.map(s => `<option value="${esc(s.value)}">${esc(s.label)}</option>`).join("") :
      `<option value="">(no videos found – upload one or run tools/make_demo_clip.py)</option>`;
    if (cur) sel.value = cur;
  }
}
$("#file").addEventListener("change", async e => {
  const f = e.target.files[0]; if (!f) return;
  toast(`Uploading ${esc(f.name)}…`, 3000);
  const fd = new FormData(); fd.append("file", f);
  const r = await fetch("/api/upload", { method: "POST", body: fd }); const j = await r.json();
  if (!r.ok) return toast("Upload failed: " + esc(j.error));
  await loadSources(); $("#src").value = j.value; $("#setupSrc").value = j.value; toast("Uploaded. Press Start.", 3000);
});
$("#bStart").onclick = async () => {
  const source = $("#customSrc").value.trim() || $("#src").value;
  try {
    evAfter = 0; allEvents = [];
    await api("/api/start", { source, speed: +$("#speed").value, overrides: nest(pending) });
  } catch (e) { toast("Cannot start: " + esc(e.message)); }
};
$("#bPause").onclick = () => api(S.state === "paused" ? "/api/resume" : "/api/pause", {});
$("#bStop").onclick = () => api("/api/stop", {});

/* ------------------------------------------------------------------ live polling */
let S = { state: "idle" }, allEvents = [], evAfter = 0, lastRun = null, streamOn = false;
async function poll() {
  try {
    S = await api("/api/state");
    if (S.run !== lastRun) { lastRun = S.run; evAfter = 0; allEvents = []; renderEvents(); renderGallery(); }
    renderState();
    if (S.n_events > evAfter) {
      const fresh = await api("/api/events?after=" + evAfter);
      evAfter += fresh.length; allEvents.push(...fresh);
      fresh.filter(e => e.level === "alarm").forEach(e => { beep(); toast(`<b>${esc(e.kind.toUpperCase())}</b> at ${fmtT(e.t)}<br><small>${esc(e.info)}</small>`); });
      renderEvents(); renderGallery();
    }
  } catch (e) { /* server busy */ }
  setTimeout(poll, 500);
}
function renderState() {
  const run = ["running", "paused", "starting"].includes(S.state);
  const alarmNow = (S.tracks || []).some(t => t.state === "ALARM");
  const pill = $("#statePill"); pill.textContent = alarmNow ? "ALARM" : S.state;
  pill.className = "pill " + (alarmNow ? "alarm" : S.state);
  $("#videoBox").classList.toggle("alarm", alarmNow);
  if (run && !streamOn) { $("#video").src = "/stream.mjpg?" + Date.now(); $("#video").style.display = ""; $("#ph").style.display = "none"; streamOn = true; }
  if (!run && streamOn) { streamOn = false; }
  $("#bStart").disabled = run; $("#bStop").disabled = !run; $("#bPause").disabled = !(S.state === "running" || S.state === "paused");
  $("#bPause").textContent = S.state === "paused" ? "Resume" : "Pause";
  $("#progress").style.width = S.progress != null ? Math.min(100, S.progress * 100) + "%" : (S.state === "running" ? "100%" : "0");
  $("#kT").textContent = fmtT(S.t); $("#kF").textContent = S.fps ? S.fps.toFixed(0) : "–";
  $("#kO").textContent = (S.tracks || []).length; $("#kA").textContent = S.counts?.alarm || 0; $("#kW").textContent = S.counts?.warn || 0;
  const b = $("#badge"); b.style.display = S.banner ? "" : "none"; b.textContent = S.banner || "";
  const c = S.components || {};
  $("#chips").innerHTML = S.run ? Object.entries(c).map(([k, v]) => `<span class="chip">${esc(k)}: <b>${esc(v)}</b></span>`).join("") : "";
  $("#warnBox").innerHTML = (S.warnings || []).map(w => `<div class="note">${esc(w)}</div>`).join("") + (S.error ? `<div class="note" style="color:#ff9b94;border-color:#6b2226;background:#2a1618">${esc(S.error)}</div>` : "");
  renderObjects(); drawCharts();
}
function renderEvents() {
  const el = $("#t-events");
  if (!allEvents.length) { el.innerHTML = `<div class="empty">No events yet.</div>`; return; }
  el.innerHTML = [...allEvents].reverse().map(e => `
    <div class="ev ${e.level}" data-id="${e.id}"><span class="t">${fmtT(e.t)}</span>
    <div><div class="k">${esc(e.kind)}${e.track_id != null && e.track_id !== "" ? " #" + esc(e.track_id) : ""}<em>${esc(e.level)}</em></div><div class="i" title="${esc(e.info)}">${esc(e.info)}</div></div>
    ${e.snapshot ? `<img src="${e.snapshot}" alt="">` : ""}</div>`).join("");
}
$("#t-events").addEventListener("click", e => {
  const row = e.target.closest(".ev"); if (!row) return;
  const ev = allEvents.find(x => x.id == row.dataset.id); if (ev) showEvent(ev);
});
function showEvent(ev) {
  modal(`<h3 style="margin:0 0 8px">${esc(ev.kind)} <small style="color:var(--muted)">at ${fmtT(ev.t)}</small></h3>
    <p style="color:var(--muted)">${esc(ev.info)}</p>${ev.snapshot ? `<img src="${ev.snapshot}">` : "<p>No snapshot saved.</p>"}`);
}
function renderObjects() {
  const el = $("#t-objects"), tr = S.tracks || [];
  if (!tr.length) { el.innerHTML = `<div class="empty">No objects being tracked.</div>`; return; }
  el.innerHTML = tr.map(t => `<div class="obj ${t.state}"><b>#${t.id}</b><span>${esc(t.state)} ${t.verdict ? "· " + esc(t.verdict) : ""}</span>
    <small>owner</small><span>${t.owner_away ? "out of sight / away" : t.owner_dist != null ? t.owner_dist + " m from object" : "unknown"}</span></div>`).join("");
}
function renderGallery() {
  const al = allEvents.filter(e => e.level === "alarm" && e.snapshot);
  $("#t-gallery").innerHTML = al.length ? [...al].reverse().map(e => `<figure data-id="${e.id}"><img src="${e.snapshot}"><figcaption>${esc(e.kind)} · ${fmtT(e.t)}</figcaption></figure>`).join("") :
    `<div class="empty" style="grid-column:1/-1">No alarm snapshots yet.</div>`;
}
$("#t-gallery").addEventListener("click", e => { const f = e.target.closest("figure"); if (!f) return; const ev = allEvents.find(x => x.id == f.dataset.id); if (ev) showEvent(ev); });

/* ------------------------------------------------------------------ tiny charts */
function drawLine(cv, series, color, max, min = 0) {
  const dpr = devicePixelRatio || 1, w = cv.clientWidth * dpr, h = cv.clientHeight * dpr;
  if (cv.width !== w) { cv.width = w; cv.height = h; }
  const g = cv.getContext("2d"); g.clearRect(0, 0, w, h);
  if (!series.length) return;
  const mx = max ?? Math.max(1, ...series), n = series.length;
  g.strokeStyle = color; g.lineWidth = 1.6 * dpr; g.beginPath();
  series.forEach((v, i) => { const x = i / Math.max(1, n - 1) * (w - 4) + 2, y = h - 3 - (v - min) / (mx - min || 1) * (h - 8); i ? g.lineTo(x, y) : g.moveTo(x, y); });
  g.stroke();
}
function drawCharts() {
  const s = S.series || [];
  drawLine($("#c1"), s.map(p => p.fps), "#4c9aff");
  drawLine($("#c2"), s.map(p => p.lf), "#e3a008", 1);
  drawLine($("#c3"), s.map(p => p.people), "#3fb950"); 
  const c3 = $("#c3"), g = c3.getContext("2d"); const mx = Math.max(1, ...s.map(p => Math.max(p.people, p.tracks)));
  g.strokeStyle = "#4c9aff"; g.lineWidth = 1.6 * (devicePixelRatio || 1); g.beginPath();
  s.forEach((p, i) => { const x = i / Math.max(1, s.length - 1) * (c3.width - 4) + 2, y = c3.height - 3 - p.tracks / mx * (c3.height - 8); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
}

/* ------------------------------------------------------------------ SETUP: zones & calibration editor */
const E = { img: null, tool: "inc", pts: [], zones: { include: [], ignore: [] }, cal: [], meas: [], mouse: null, hom: {}, ppm: 60 };
const cv = $("#cv"), ctx = cv.getContext("2d");
const HINTS = {
  inc: "Click to place corners of a WATCH zone. Close it by clicking the first point, double-click or Enter.",
  ign: "Click to place corners of an IGNORE zone (door, screen, trees…). Close it by clicking the first point, double-click or Enter.",
  cal: "Click 4 floor points that form a rectangle you know the size of (e.g. floor tiles). Then enter their real coordinates.",
  meas: "Click 2 points on the floor whose real distance you know."
};
async function loadSetup() {
  const s = await api("/api/settings");
  E.zones = { include: s.zones.include || [], ignore: s.zones.ignore || [] }; E.hom = s.homography || {}; E.ppm = s.px_per_m;
  renderZones(); renderCalStatus(); setTool(E.tool); if ($("#setupSrc").value && !E.img) loadRef();
}
function loadRef() {
  const src = $("#setupSrc").value; if (!src) return toast("Pick a video first");
  const im = new Image(); im.onload = () => { E.img = im; cv.width = im.naturalWidth; cv.height = im.naturalHeight; draw(); };
  im.onerror = () => toast("Could not read a frame from that source"); im.src = "/api/ref_frame.jpg?source=" + encodeURIComponent(src) + "&_=" + Date.now();
}
$("#loadRef").onclick = loadRef;
function setTool(t) {
  E.tool = t; E.pts = []; $$("#tools [data-tool]").forEach(b => b.classList.toggle("on", b.dataset.tool === t));
  $("#toolHint").textContent = HINTS[t]; $("#calForm").style.display = "none"; $("#measForm").style.display = "none"; draw();
}
$("#tools").addEventListener("click", e => { const b = e.target.closest("[data-tool]"); if (b) setTool(b.dataset.tool); });
$("#undo").onclick = () => { E.pts.pop(); draw(); $("#calForm").style.display = "none"; $("#measForm").style.display = "none"; };
$("#clearTool").onclick = () => setTool(E.tool);
function pos(e) { const r = cv.getBoundingClientRect(); return [Math.round((e.clientX - r.left) * cv.width / r.width), Math.round((e.clientY - r.top) * cv.height / r.height)]; }
cv.addEventListener("mousemove", e => { E.mouse = pos(e); if (E.pts.length) draw(); });
cv.addEventListener("click", e => {
  if (!E.img) return; const p = pos(e);
  if ((E.tool === "inc" || E.tool === "ign") && E.pts.length >= 3 && Math.hypot(p[0] - E.pts[0][0], p[1] - E.pts[0][1]) < 12) return closePoly();
  if (E.tool === "cal" && E.pts.length >= 4) return;
  if (E.tool === "meas" && E.pts.length >= 2) return;
  E.pts.push(p);
  if (E.tool === "cal" && E.pts.length === 4) showCalForm();
  if (E.tool === "meas" && E.pts.length === 2) $("#measForm").style.display = "";
  draw();
});
cv.addEventListener("dblclick", () => { if (E.tool === "inc" || E.tool === "ign") { E.pts.pop(); closePoly(); } });
document.addEventListener("keydown", e => { if (e.key === "Enter" && $("#p-setup").classList.contains("on") && (E.tool === "inc" || E.tool === "ign")) closePoly(); if (e.key === "Escape") setTool(E.tool); });
function closePoly() {
  if (E.pts.length < 3) return;
  E.zones[E.tool === "inc" ? "include" : "ignore"].push(E.pts.map(p => [p[0], p[1]])); E.pts = []; renderZones(); draw();
}
function showCalForm() {
  $("#calForm").style.display = "";
  const dflt = [[0, 0], [4, 0], [4, 2], [0, 2]];
  $("#calInputs").innerHTML = E.pts.map((p, i) => `<div class="row">Point ${i + 1} (${p[0]}, ${p[1]}px) → X <input type="number" step="0.1" id="wx${i}" value="${dflt[i][0]}"> m &nbsp; Y <input type="number" step="0.1" id="wy${i}" value="${dflt[i][1]}"> m</div>`).join("");
}
function renderZones() {
  const rows = [];
  ["include", "ignore"].forEach(k => E.zones[k].forEach((z, i) => rows.push(`<div class="zone"><span><i style="background:${k === "include" ? "#3fb950" : "#f85149"}"></i>${k === "include" ? "watch" : "ignore"} zone ${i + 1} (${z.length} points)</span><button class="btn" data-k="${k}" data-i="${i}">delete</button></div>`)));
  $("#zoneList").innerHTML = rows.join("") || `<div class="empty">No zones: the whole frame is watched.</div>`;
}
$("#zoneList").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; E.zones[b.dataset.k].splice(+b.dataset.i, 1); renderZones(); draw(); });
$("#saveZones").onclick = async () => { await api("/api/settings", { zones: E.zones }); toast("Zones saved. They apply the next time you press Start."); };
function renderCalStatus() {
  const h = E.hom && E.hom.image_points && E.hom.image_points.length >= 4;
  $("#calStatus").innerHTML = h ? `<b style="color:var(--ok)">Homography calibrated</b> – distances are in real metres.` :
    `Not calibrated – distances use ${(+E.ppm).toFixed(1)} pixels per metre (approximate, ignores perspective). Calibrate for accurate owner distances.`;
}
$("#saveCal").onclick = async () => {
  const world = E.pts.map((_, i) => [+$("#wx" + i).value, +$("#wy" + i).value]);
  await api("/api/settings", { homography: { image_points: E.pts, world_points: world } });
  E.hom = { image_points: E.pts, world_points: world }; renderCalStatus(); toast("Homography saved. Applies on next Start.");
};
$("#saveMeas").onclick = async () => {
  const px = Math.hypot(E.pts[0][0] - E.pts[1][0], E.pts[0][1] - E.pts[1][1]), m = +$("#measM").value;
  if (!(m > 0)) return; E.ppm = px / m; await api("/api/settings", { px_per_m: E.ppm }); renderCalStatus(); toast(`Scale saved: ${E.ppm.toFixed(1)} px/m.`);
};
$("#resetCal").onclick = async () => { await api("/api/settings", { homography: {}, px_per_m: null }); E.hom = {}; E.ppm = 60; renderCalStatus(); };
function poly(pts, col, fill, close = true) {
  if (!pts.length) return; ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
  if (close) ctx.closePath(); if (fill) { ctx.fillStyle = fill; ctx.fill(); } ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke();
}
function dots(pts, col, label) { pts.forEach((p, i) => { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(p[0], p[1], 5, 0, 7); ctx.fill(); if (label) { ctx.fillStyle = "#fff"; ctx.font = "bold 14px sans-serif"; ctx.fillText(i + 1, p[0] + 8, p[1] - 8); } }); }
function draw() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!E.img) { ctx.fillStyle = "#111"; ctx.fillRect(0, 0, cv.width, cv.height); ctx.fillStyle = "#789"; ctx.font = "16px sans-serif"; ctx.fillText("Pick a video and press ‘Load frame’", 180, 180); return; }
  ctx.drawImage(E.img, 0, 0);
  E.zones.include.forEach(z => poly(z, "#3fb950", "rgba(63,185,80,.18)"));
  E.zones.ignore.forEach(z => poly(z, "#f85149", "rgba(248,81,73,.22)"));
  if (E.hom && E.hom.image_points) { poly(E.hom.image_points, "#58a6ff", "rgba(88,166,255,.12)"); }
  const col = E.tool === "ign" ? "#f85149" : E.tool === "inc" ? "#3fb950" : "#ffd33d";
  const live = E.mouse && E.pts.length && (E.tool === "inc" || E.tool === "ign") ? [...E.pts, E.mouse] : E.pts;
  poly(live, col, null, false); dots(E.pts, col, E.tool === "cal" || E.tool === "meas");
  if (E.tool === "cal" && E.pts.length === 4) poly(E.pts, "#ffd33d", "rgba(255,211,61,.15)");
}

/* ------------------------------------------------------------------ SETTINGS */
async function loadSettings() {
  const s = await api("/api/settings"), box = $("#settingsBox"); box.innerHTML = "";
  s.schema.forEach((g, gi) => {
    const det = document.createElement("details"); det.className = "acc"; if (gi === 0) det.open = true;
    const sm = document.createElement("summary"); sm.textContent = g.group; det.appendChild(sm);
    g.items.forEach(it => {
      const key = it.key, cur = key in pending ? pending[key] : it.value, row = document.createElement("div"); row.className = "set";
      row.innerHTML = `<label>${esc(it.label)}<span class="tag ${it.live ? "" : "rs"}">${it.live ? "live" : "restart"}</span><small>${esc(key)}</small></label><div class="ctl"></div><div class="val"></div>`;
      const ctl = $(".ctl", row), val = $(".val", row);
      const set = v => { pending[key] = v; if (it.live && ["running", "paused"].includes(S.state)) api("/api/live_config", nest({ [key]: v })); };
      if (it.type === "range") {
        ctl.innerHTML = `<input type="range" min="${it.min}" max="${it.max}" step="${it.step}" value="${cur}">`; val.textContent = cur;
        $("input", ctl).oninput = e => { val.textContent = e.target.value; set(+e.target.value); };
      } else if (it.type === "bool") {
        ctl.innerHTML = `<label class="switch"><input type="checkbox" ${cur ? "checked" : ""}><span></span></label>`; val.textContent = "";
        $("input", ctl).onchange = e => set(e.target.checked);
      } else {
        ctl.innerHTML = `<select>${it.options.map(o => `<option ${o === cur ? "selected" : ""}>${o}</option>`).join("")}</select>`;
        $("select", ctl).onchange = e => set(e.target.value);
      }
      det.appendChild(row);
    });
    box.appendChild(det);
  });
}
$("#saveSet").onclick = async () => { await api("/api/settings", { overrides: nest(pending) }); toast("Saved as defaults."); };
$("#resetSet").onclick = async () => { if (!confirm("Reset all saved settings, zones and calibration to config.yaml?")) return; await api("/api/settings/reset", {}); pending = {}; loadSettings(); toast("Reset."); };

/* ------------------------------------------------------------------ HISTORY */
async function loadRuns() {
  const rows = await api("/api/runs");
  $("#runEmpty").style.display = rows.length ? "none" : "";
  $("#runRows").innerHTML = rows.map(r => `<tr class="click" data-n="${esc(r.name)}"><td>${esc(r.name)}</td><td>${r.when}</td><td>${r.events}</td><td style="color:${r.alarms ? "var(--alarm)" : "inherit"}">${r.alarms}</td><td>${r.clips}</td></tr>`).join("");
}
$("#runRows").addEventListener("click", async e => {
  const tr = e.target.closest("tr"); if (!tr) return; const d = await api("/api/runs/" + encodeURIComponent(tr.dataset.n));
  $("#runTitle").textContent = tr.dataset.n;
  $("#runClips").innerHTML = d.clips.map(c => `<button class="btn" data-url="${c.url}" style="margin:0 6px 8px 0">▶ ${esc(c.name)}</button>`).join("");
  window._runEv = d.events;
  $("#runEvents").innerHTML = d.events.length ? [...d.events].reverse().map(ev => `<div class="ev ${ev.level}" data-id="${ev.id}"><span class="t">${fmtT(ev.t)}</span><div><div class="k">${esc(ev.kind)}</div><div class="i">${esc(ev.info)}</div></div>${ev.snapshot ? `<img src="${ev.snapshot}">` : ""}</div>`).join("") : `<div class="empty">No events.</div>`;
});
$("#runClips").addEventListener("click", e => { const b = e.target.closest("button"); if (b) modal(`<video src="${b.dataset.url}" controls autoplay></video><p><a href="${b.dataset.url}" download>download clip</a> <small style="color:var(--muted)">(if it does not play in the browser, download it)</small></p>`); });
$("#runEvents").addEventListener("click", e => { const r = e.target.closest(".ev"); if (!r) return; const ev = (window._runEv || []).find(x => x.id == r.dataset.id); if (ev) showEvent(ev); });

/* ------------------------------------------------------------------ RESULTS */
function mdTable(md) {
  const lines = md.split("\n").filter(l => l.trim().startsWith("|")); if (lines.length < 2) return `<pre class="log">${esc(md)}</pre>`;
  const cells = l => l.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim());
  const head = cells(lines[0]), body = lines.slice(2).map(cells);
  return `<div style="overflow:auto"><table><thead><tr>${head.map(h => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${body.map(r => `<tr>${r.map(c => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}
async function loadResults() {
  const r = await api("/api/results");
  if (r.eval) {
    const all = r.eval.table.find(x => x.type === "ALL") || r.eval.table[r.eval.table.length - 1] || {};
    const f = v => v == null || v === "" ? "–" : (+v).toFixed(2).replace(/\.?0+$/, "");
    $("#sF1").textContent = f(all.F1); $("#sPR").textContent = `${f(all.precision)} / ${f(all.recall)}`;
    $("#sFA").textContent = f(r.eval.fa_per_hour); $("#sFPS").textContent = f(r.eval.fps);
    const cols = ["type", "TP", "FP", "FN", "precision", "recall", "F1", "mean_delay_s"];
    $("#evalTable").className = ""; $("#evalTable").innerHTML = `<table><thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>${r.eval.table.map(row => `<tr>${cols.map(c => `<td>${row[c] ?? "–"}</td>`).join("")}</tr>`).join("")}</tbody></table>
      <p class="hint">False alarms per hour: <b>${r.eval.fa_per_hour}</b> · pipeline speed: <b>${r.eval.fps} FPS</b></p>`;
  }
  if (r.ablations_md) { $("#ablTable").className = ""; $("#ablTable").innerHTML = mdTable(r.ablations_md); }
  $("#figs").innerHTML = r.figures.length ? r.figures.map(f => `<a href="/figures/${f}" target="_blank"><img src="/figures/${f}?_=${Date.now()}" title="${f}"></a>`).join("") : `<div class="empty">No figures yet (run the ablations / CNN training).</div>`;
  pollJob();
}
async function startJob(name) { try { await api("/api/job", { name }); $("#jobLog").style.display = ""; pollJob(); } catch (e) { toast(esc(e.message)); } }
$("#jEval").onclick = () => startJob("eval"); $("#jAbl").onclick = () => startJob("ablations");
let jobTimer;
async function pollJob() {
  clearTimeout(jobTimer);
  const j = await api("/api/job"); const log = $("#jobLog");
  if (j.name) { log.style.display = ""; log.textContent = `[${j.name}] ${j.running ? "running " + j.elapsed + " s…" : "finished (exit " + j.returncode + ")"}\n` + j.log.join("\n"); log.scrollTop = log.scrollHeight; }
  $("#jEval").disabled = $("#jAbl").disabled = j.running;
  if (j.running) jobTimer = setTimeout(pollJob, 1500); else if (j.name && $("#p-results").classList.contains("on") && !pollJob.done) { pollJob.done = true; loadResults(); setTimeout(() => pollJob.done = false, 3000); }
}

/* ------------------------------------------------------------------ boot */
loadSources(); poll();
