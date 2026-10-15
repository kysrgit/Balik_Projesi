// Ortak "Hata bildir / etiketi duzelt" penceresi.
// Kullanim: openFixEditor({path, src, gt, preds, env, onSaved})
//   path : veri setindeki goruntu yolu (ornek: dataset/test/images/x.jpg)
//   gt   : mevcut etiket kutulari [[x1,y1,x2,y2], ...] (0-1 normalize)
//   preds: (istege bagli) model tahminleri [[x1,y1,x2,y2,conf], ...]
(function () {
  const ISSUES = [
    ["eksik_kutu", "Balık var ama kutusu yok"],
    ["fazla_kutu", "Balık olmayan yerde kutu var"],
    ["kayik_kutu", "Kutu kaymış / balığı tam sarmıyor"],
    ["yanlis_tur", "Puffer değil (yanlış tür)"],
    ["yanlis_ortam", "Ortam (su altı/üstü/dışarı) yanlış"],
    ["kopya", "Başka bir fotoğrafın kopyası"],
    ["kalite", "Fotoğraf bozuk / bulanık / anlaşılmıyor"],
    ["diger", "Diğer (nota yazın)"],
  ];
  const ENVS = [["su_alti", "1 Su altı"], ["su_ustu", "2 Su üstünden"], ["kara", "3 Su dışında"], ["haric", "4 Hariç"]];

  const css = `
  #fx{position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:50;display:none;overflow:auto;padding:16px}
  #fx.open{display:block}
  #fx .wrap{max-width:1250px;margin:0 auto;display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:14px}
  #fx .stage{position:relative;line-height:0;user-select:none;cursor:crosshair;background:#000;border-radius:6px;overflow:hidden;align-self:start}
  #fx .stage img{width:100%;display:block}
  #fx .b{position:absolute;border:2px solid #3fb950;background:rgba(63,185,80,.08);cursor:move}
  #fx .b.sel{border-color:#f0a030;background:rgba(240,160,48,.12)}
  #fx .b span{position:absolute;top:-18px;left:-2px;font:600 11px/16px system-ui;background:#3fb950;color:#04121f;padding:0 4px;border-radius:3px}
  #fx .b.sel span{background:#f0a030}
  #fx .p{position:absolute;border:2px dashed #f85149;cursor:copy}
  #fx .p span{position:absolute;bottom:-18px;left:-2px;font:600 11px/16px system-ui;background:#f85149;color:#fff;padding:0 4px;border-radius:3px;white-space:nowrap}
  #fx .h{position:absolute;width:12px;height:12px;background:#f0a030;border:1px solid #000}
  #fx .h.tl{left:-7px;top:-7px;cursor:nwse-resize}#fx .h.br{right:-7px;bottom:-7px;cursor:nwse-resize}
  #fx .h.tr{right:-7px;top:-7px;cursor:nesw-resize}#fx .h.bl{left:-7px;bottom:-7px;cursor:nesw-resize}
  #fx aside{background:#182029;border:1px solid #2a3542;border-radius:8px;padding:12px;color:#e6edf3;font:13px/1.45 system-ui;align-self:start}
  #fx h3{font-size:14px;margin:0 0 6px}
  #fx .envs{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-bottom:10px}
  #fx button{background:#0f1419;color:#e6edf3;border:1px solid #2a3542;border-radius:6px;padding:6px 8px;font:inherit;cursor:pointer}
  #fx button.on{border-color:#58a6ff;color:#58a6ff}
  #fx button.primary{background:#58a6ff;color:#04121f;border-color:#58a6ff;font-weight:600}
  #fx label{display:flex;flex-direction:row;gap:6px;align-items:flex-start;margin:3px 0;cursor:pointer;color:#e6edf3;font-size:13px}
  #fx input[type=checkbox]{margin-top:3px}
  #fx textarea{width:100%;min-height:60px;background:#0f1419;color:#e6edf3;border:1px solid #2a3542;border-radius:6px;padding:6px;font:inherit}
  #fx .muted{color:#8b98a5;font-size:12px}
  #fx .row{display:flex;gap:6px;margin-top:10px}
  @media(max-width:850px){#fx .wrap{grid-template-columns:1fr}}`;
  const st = document.createElement("style"); st.textContent = css; document.head.appendChild(st);
  const root = document.createElement("div"); root.id = "fx"; document.body.appendChild(root);

  let S = null; // aktif durum

  function el(id) { return root.querySelector("#" + id); }
  function pos(e) {
    const r = el("fximg").getBoundingClientRect();
    return [Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1), Math.min(Math.max((e.clientY - r.top) / r.height, 0), 1)];
  }
  function draw() {
    const stage = el("fxstage");
    stage.querySelectorAll(".b,.p").forEach(n => n.remove());
    if (el("fxshowpred").checked) (S.preds || []).forEach((p, k) => {
      if (p[4] < S.thr) return;
      const d = document.createElement("div"); d.className = "p"; d.dataset.p = k;
      Object.assign(d.style, { left: p[0] * 100 + "%", top: p[1] * 100 + "%", width: (p[2] - p[0]) * 100 + "%", height: (p[3] - p[1]) * 100 + "%" });
      d.title = "Model tahmini — tıklayınca etikete eklenir";
      d.innerHTML = `<span>model ${(p[4] * 100).toFixed(0)}% · + ekle</span>`;
      stage.appendChild(d);
    });
    S.boxes.forEach((b, k) => {
      const d = document.createElement("div"); d.className = "b" + (k === S.sel ? " sel" : ""); d.dataset.k = k;
      Object.assign(d.style, { left: b[0] * 100 + "%", top: b[1] * 100 + "%", width: (b[2] - b[0]) * 100 + "%", height: (b[3] - b[1]) * 100 + "%" });
      d.innerHTML = `<span>etiket ${k + 1}</span>` + (k === S.sel ? ["tl", "tr", "bl", "br"].map(c => `<i class="h ${c}" data-c="${c}"></i>`).join("") : "");
      stage.appendChild(d);
    });
    for (const b of root.querySelectorAll("[data-env]")) b.classList.toggle("on", b.dataset.env === S.env);
    el("fxcount").textContent = `${S.boxes.length} etiket kutusu`;
  }
  function onDown(e) {
    const [x, y] = pos(e);
    if (e.target.dataset.c) S.drag = { mode: "resize", k: S.sel, c: e.target.dataset.c };
    else if (e.target.closest(".b")) { S.sel = +e.target.closest(".b").dataset.k; S.drag = { mode: "move", k: S.sel, x, y, o: S.boxes[S.sel].slice() }; }
    else if (e.target.closest(".p")) { const p = S.preds[+e.target.closest(".p").dataset.p]; S.boxes.push(p.slice(0, 4)); S.sel = S.boxes.length - 1; mark("eksik_kutu"); }
    else { S.boxes.push([x, y, x, y]); S.sel = S.boxes.length - 1; S.drag = { mode: "new", k: S.sel, x, y }; }
    e.preventDefault(); draw();
  }
  function onMove(e) {
    if (!S || !S.drag) return;
    const [x, y] = pos(e), d = S.drag, b = S.boxes[d.k];
    if (d.mode === "new") S.boxes[d.k] = [Math.min(d.x, x), Math.min(d.y, y), Math.max(d.x, x), Math.max(d.y, y)];
    else if (d.mode === "move") {
      const o = d.o, dx = Math.min(Math.max(x - d.x, -o[0]), 1 - o[2]), dy = Math.min(Math.max(y - d.y, -o[1]), 1 - o[3]);
      S.boxes[d.k] = [o[0] + dx, o[1] + dy, o[2] + dx, o[3] + dy];
    } else {
      if (d.c.includes("l")) b[0] = x; if (d.c.includes("r")) b[2] = x;
      if (d.c.includes("t")) b[1] = y; if (d.c.includes("b")) b[3] = y;
    }
    draw();
  }
  function onUp() {
    if (!S || !S.drag) return;
    const b = S.boxes[S.drag.k];
    if (b) {
      const n = [Math.min(b[0], b[2]), Math.min(b[1], b[3]), Math.max(b[0], b[2]), Math.max(b[1], b[3])];
      if (n[2] - n[0] < 0.005 || n[3] - n[1] < 0.005) { S.boxes.splice(S.drag.k, 1); S.sel = -1; } else S.boxes[S.drag.k] = n;
    }
    S.drag = null; draw();
  }
  function mark(issue) { const cb = root.querySelector(`[data-issue="${issue}"]`); if (cb) cb.checked = true; }
  function close() { root.classList.remove("open"); S = null; }
  function changed() {
    const r = a => JSON.stringify(a.map(b => b.slice(0, 4).map(v => Math.round(v * 1000))));
    return r(S.boxes) !== r(S.origGt);
  }
  async function save() {
    const issues = [...root.querySelectorAll("[data-issue]:checked")].map(c => c.dataset.issue);
    const note = el("fxnote").value.trim();
    const boxesChanged = changed();
    const envChanged = S.origEnv !== null && S.env !== S.origEnv;
    if (!issues.length && !note && !boxesChanged && !envChanged) { close(); return; }
    // Ortam yalnizca kullanici sectiyse ya da zaten biliniyorsa kaydedilir (varsayilan deger kaydedilmez)
    const env = (S.envTouched || S.origEnv) ? S.env : (S.savedEnv || null);
    const body = { path: S.path, issues, note, env, boxes: boxesChanged ? S.boxes : null, confirm: S.confirm };
    const r = await fetch("/api/fix", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) { alert("Kaydedilemedi: " + await r.text()); return; }
    const saved = (await r.json()).fix;
    const cb = S.onSaved; close(); if (cb) cb(saved);
  }
  async function clearFix() {
    if (!confirm("Bu fotoğraf için kaydedilen düzeltme silinsin mi? (orijinal etiket geri gelir)")) return;
    const r = await fetch("/api/fix", { method: "DELETE", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: S.path }) });
    if (!r.ok) { alert(await r.text()); return; }
    const cb = S.onSaved; close(); if (cb) cb(null);
  }

  window.openFixEditor = async function (opt) {
    const r = await fetch("/api/fix?path=" + encodeURIComponent(opt.path));
    const d = r.ok ? await r.json() : { fix: null, gt: opt.gt };
    const fix = d.fix;
    S = { path: opt.path, preds: opt.preds || [], thr: opt.thr ?? 0.25, onSaved: opt.onSaved,
          origGt: (d.gt || opt.gt || []).map(b => b.slice(0, 4)),
          boxes: (fix && fix.boxes ? fix.boxes : (d.gt || opt.gt || [])).map(b => b.slice(0, 4)),
          env: (fix && fix.env) || opt.env || null, savedEnv: (fix && fix.env) || null, envTouched: false,
          confirm: (fix && fix.confirm) || [], sel: -1, drag: null };
    S.origEnv = opt.env || null;  // bilinmiyorsa (inceleme sayfasi) ortam degisikligi hata sayilmaz
    root.innerHTML = `<div class="wrap">
      <div class="stage" id="fxstage"><img id="fximg" src="${opt.src}" alt="" draggable="false"></div>
      <aside>
        <h3>✎ Hata bildir / etiketi düzelt</h3>
        <div class="muted">${opt.path.split("/").pop()}${fix ? ` · <b style="color:#f0a030">daha önce düzeltildi (${fix.zaman})</b>` : ""}</div>
        <p class="muted">Yeşil = etiket (doğru kabul edilen konum). Kırmızı kesikli = model tahmini; doğruysa tıklayıp etikete ekleyin.
          Boş yere sürükle: yeni kutu · kutuya tıkla: seç/taşı · köşe: boyutlandır · <b>Delete</b>: seçili kutuyu sil.</p>
        <label class="muted"><input type="checkbox" id="fxshowpred" ${S.preds.length ? "checked" : "disabled"}> Model tahminlerini göster</label>
        <div class="muted" id="fxcount"></div>
        <h3 style="margin-top:10px">Ortam</h3>
        <div class="envs">${ENVS.map(([e, l]) => `<button data-env="${e}">${l}</button>`).join("")}</div>
        <h3>Ne yanlış?</h3>
        <div class="iss">${ISSUES.map(([k, l]) => `<label><input type="checkbox" data-issue="${k}" ${fix && fix.issues.includes(k) ? "checked" : ""}> ${l}</label>`).join("")}</div>
        <h3 style="margin-top:8px">Not</h3>
        <textarea id="fxnote" placeholder="Gördüğünüzü kendi cümlenizle yazın (ör. 'arkadaki ikinci balık etiketlenmemiş')">${fix ? fix.note : ""}</textarea>
        <div class="row"><button class="primary" id="fxsave">Kaydet (Ctrl+Enter)</button><button id="fxcancel">Vazgeç (Esc)</button></div>
        ${fix ? `<div class="row"><button id="fxclear">Düzeltmeyi geri al</button></div>` : ""}
      </aside></div>`;
    root.classList.add("open");
    el("fxstage").onmousedown = onDown;
    el("fxshowpred").onchange = draw;
    el("fxsave").onclick = save; el("fxcancel").onclick = close;
    if (fix) el("fxclear").onclick = clearFix;
    root.querySelectorAll("[data-env]").forEach(b => b.onclick = () => { S.env = b.dataset.env; S.envTouched = true; if (S.origEnv && S.env !== S.origEnv) mark("yanlis_ortam"); draw(); });
    draw();
  };
  window.fixEditorOpen = () => !!S;

  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  window.addEventListener("keydown", e => {
    if (!S) return;
    if (e.target.tagName === "TEXTAREA") { if (e.key === "Enter" && e.ctrlKey) { save(); e.preventDefault(); } return; }
    if (e.key === "Escape") close();
    else if ((e.key === "Delete" || e.key === "Backspace") && S.sel >= 0) { S.boxes.splice(S.sel, 1); S.sel = -1; draw(); }
    else if (e.key === "Enter" && e.ctrlKey) save();
    else if (e.key >= "1" && e.key <= "4") { S.env = ENVS[+e.key - 1][0]; S.envTouched = true; if (S.origEnv && S.env !== S.origEnv) mark("yanlis_ortam"); draw(); }
    else return;
    e.preventDefault(); e.stopImmediatePropagation();
  }, true);
})();
