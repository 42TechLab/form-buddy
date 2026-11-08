// FORM_BUDDY — pagina di test Fase 0.
// Esegue Gemma 3n on-device (MediaPipe LLM Inference, WebGPU) e misura
// estrazione di campi da testo, foto e voce. Nessun dato lascia il dispositivo.

import { FilesetResolver, LlmInference } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@0.10.30/genai_bundle.mjs';
import { CATALOGO, CASI_TESTO, CASI_FOTO } from './casi.js';

const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@0.10.30/wasm';
const OPFS_NAME = 'modello.litertlm';
const $ = (id) => document.getElementById(id);

const state = {
  llm: null,
  stop: false,
  results: { versione: 1, avviato: new Date().toISOString(), ambiente: null, modello: null, testo: [], foto: [], audio: [] },
};

// ---------- normalizzazione e confronto (identica a gen_casi.py) ----------

export function norm(v) {
  if (v === null || v === undefined) return '';
  let s = String(v).toUpperCase().trim().replace(/[\s*"'’]/g, '');
  let m = s.match(/^(\d{1,2})[./-](\d{1,2})[./-](\d{4})$/);
  if (m) return `${m[1].padStart(2, '0')}/${m[2].padStart(2, '0')}/${m[3]}`;
  m = s.match(/^(\d{1,2})[./-](\d{4})$/);
  if (m) return `${m[1].padStart(2, '0')}/${m[2]}`;
  m = s.match(/^(\d{1,2}):(\d{2})(:\d{2})?$/);
  if (m) return `${m[1].padStart(2, '0')}:${m[2]}`;
  if (/^[1-9]\d{0,2}(\.\d{3})+$/.test(s)) s = s.replace(/\./g, '');
  const t = s.replace(',', '.');
  if (/^-?\d+(\.\d+)?$/.test(t)) return String(parseFloat(t));
  return s;
}

async function sha(key, value) {
  const data = new TextEncoder().encode(`fb0:${key}:${norm(value)}`);
  const buf = await crypto.subtle.digest('SHA-256', data);
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

// expected: { chiave: [valori accettati] } in chiaro, oppure { chiave: [hash] } con hashed=true
async function confronta(estratti, expected, hashed, vietati = []) {
  const righe = [];
  let ok = 0;
  for (const [k, accettati] of Object.entries(expected)) {
    const got = estratti[k];
    let match = false;
    if (got !== undefined && got !== null && got !== '') {
      if (hashed) match = accettati.includes(await sha(k, got));
      else match = accettati.some((a) => norm(a) === norm(got));
    }
    if (match) ok++;
    righe.push({ campo: k, atteso: hashed ? '(nascosto)' : accettati.join(' | '), ottenuto: got ?? '—', esito: match ? 'ok' : 'ko' });
  }
  const extra = Object.keys(estratti).filter((k) => !(k in expected));
  const allucinati = extra.filter((k) => vietati.includes(k));
  for (const k of extra) {
    righe.push({ campo: k, atteso: '—', ottenuto: estratti[k], esito: vietati.includes(k) ? 'vietato' : 'extra' });
  }
  return { totale: Object.keys(expected).length, corretti: ok, extra: extra.length, vietati: allucinati.length, righe };
}

// ---------- prompt ----------

function catalogoTesto() {
  return CATALOGO.map(([k, d]) => `${k}: ${d}`).join('\n');
}

const ISTRUZIONI = `Sei l'assistente di un esperto che compila verbali di collaudo ATP (celle isotermiche e frigorifere per trasporto di derrate).
Estrai dall'input SOLO i dati presenti, usando le chiavi del catalogo. Non inventare nulla. Se un dato è illeggibile o incompleto, omettilo.
Formati: date GG/MM/AAAA; mese e anno MM/AAAA; ore HH:MM; numeri senza separatore delle migliaia e con il punto per i decimali; lunghezze in millimetri; masse in kg; testo in MAIUSCOLO; codici e targhe senza spazi né asterischi.
Numeri e lettere possono essere dettati a parole o con lo spelling (es. "zeta bi tre" = ZB3, "doppia vu" = W).
Se per un campo l'utente dice "non ricorre", "niente" o "nessuno", usa il valore "N.R."; per i campi SI/NO usa SI o NO.
Se l'utente si corregge, vale l'ultimo valore detto.
Rispondi SOLO con un oggetto JSON piatto {"chiave": valore}. Se non trovi nulla rispondi {}.

Catalogo dei campi:
${catalogoTesto()}`;

function promptTesto(frase) {
  return `<start_of_turn>user\n${ISTRUZIONI}\n\nInput dettato:\n"${frase}"<end_of_turn>\n<start_of_turn>model\n`;
}

function promptFoto(canvas) {
  return [
    `<start_of_turn>user\n${ISTRUZIONI}\n\nInput: la foto seguente (targhetta, documento o vista del mezzo). Leggi con attenzione ogni carattere; se la foto non contiene dati leggibili rispondi {}.\n`,
    { imageSource: canvas },
    '<end_of_turn>\n<start_of_turn>model\n',
  ];
}

function promptAudio(url) {
  return [
    `<start_of_turn>user\n${ISTRUZIONI}\n\nInput: la registrazione audio seguente, in italiano. Rispondi con un JSON {"trascrizione": "testo esatto ascoltato", "campi": {chiave: valore}}.\n`,
    { audioSource: url },
    '<end_of_turn>\n<start_of_turn>model\n',
  ];
}

function parseJSON(text) {
  const t = text.replace(/```json|```/g, '');
  const a = t.indexOf('{');
  const b = t.lastIndexOf('}');
  if (a < 0 || b <= a) return { ok: false, valore: {} };
  try { return { ok: true, valore: JSON.parse(t.slice(a, b + 1)) }; } catch { return { ok: false, valore: {} }; }
}

// ---------- diagnostica: registro che sopravvive a un crash della scheda ----------

const LOG_KEY = 'fb0_log';
function traccia(passo, extra = {}) {
  try {
    const log = JSON.parse(localStorage.getItem(LOG_KEY) || '[]');
    log.push({ t: new Date().toISOString(), passo, ...extra });
    localStorage.setItem(LOG_KEY, JSON.stringify(log.slice(-40)));
  } catch { /* storage non disponibile */ }
}
function ultimoLog() {
  try { return JSON.parse(localStorage.getItem(LOG_KEY) || '[]'); } catch { return []; }
}

// ---------- esecuzione con misure ----------

async function genera(prompt) {
  traccia('generazione-inizio', { tipo: Array.isArray(prompt) ? 'multimodale' : 'testo' });
  const t0 = performance.now();
  let tFirst = null;
  const out = await state.llm.generateResponse(prompt, (partial) => {
    if (tFirst === null && partial) tFirst = performance.now();
  });
  const t1 = performance.now();
  let tokOut = null;
  try { tokOut = state.llm.sizeInTokens(out) ?? null; } catch { /* non disponibile */ }
  const genSec = (t1 - (tFirst ?? t0)) / 1000;
  traccia('generazione-fine', { ms: Math.round(t1 - t0) });
  return {
    testo: out,
    ms_totali: Math.round(t1 - t0),
    ms_primo_token: tFirst ? Math.round(tFirst - t0) : null,
    token_uscita: tokOut,
    token_al_sec: tokOut && genSec > 0 ? +(tokOut / genSec).toFixed(1) : null,
  };
}

// ---------- UI helpers ----------

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function cardRisultato(titolo, misure, cmp, extraHtml = '', img = null) {
  const div = document.createElement('div');
  div.className = 'res';
  const esito = cmp
    ? `<span class="${cmp.corretti === cmp.totale && !cmp.vietati ? 'ok' : 'ko'}">${cmp.corretti}/${cmp.totale}</span>` +
      (cmp.vietati ? ` <span class="ko">${cmp.vietati} inventati</span>` : '') +
      (cmp.extra - cmp.vietati > 0 ? ` <span class="warn">+${cmp.extra - cmp.vietati} extra</span>` : '')
    : '';
  const righe = cmp
    ? `<table>${cmp.righe.map((r) => `<tr><td>${esc(r.campo)}</td><td>${esc(r.ottenuto)}</td><td>${esc(r.atteso)}</td><td class="${r.esito === 'ok' ? 'ok' : r.esito === 'extra' ? 'warn' : 'ko'}">${r.esito}</td></tr>`).join('')}</table>`
    : '';
  div.innerHTML = `${img ? `<img src="${img}" alt="">` : ''}<h3>${esc(titolo)} ${esito}</h3>
    <div class="meta">${misure.ms_totali} ms totali · primo token ${misure.ms_primo_token ?? '—'} ms · ${misure.token_al_sec ?? '—'} token/s</div>
    ${extraHtml}${righe}`;
  return div;
}

function aggiornaSommario() {
  const r = state.results;
  const somma = (arr) => arr.reduce((acc, x) => {
    if (x.confronto) { acc.c += x.confronto.corretti; acc.t += x.confronto.totale; acc.v += x.confronto.vietati; }
    acc.ms.push(x.misure.ms_totali);
    if (x.misure.token_al_sec) acc.tps.push(x.misure.token_al_sec);
    return acc;
  }, { c: 0, t: 0, v: 0, ms: [], tps: [] });
  const media = (a) => (a.length ? Math.round(a.reduce((x, y) => x + y, 0) / a.length) : '—');
  const riga = (nome, arr) => {
    if (!arr.length) return `${nome}: nessun test`;
    const s = somma(arr);
    return `${nome}: ${arr.length} test · campi corretti ${s.c}/${s.t} (${s.t ? Math.round((100 * s.c) / s.t) : 0}%) · inventati ${s.v} · tempo medio ${media(s.ms)} ms · ${media(s.tps)} token/s`;
  };
  $('summary').textContent = [
    r.modello ? `Modello caricato in ${r.modello.ms_caricamento} ms (immagini: ${r.modello.immagini}, audio: ${r.modello.audio})` : 'Modello non caricato',
    riga('Testo', r.testo), riga('Foto', r.foto), riga('Voce', r.audio),
  ].join('\n');
}

function abilita(on) {
  for (const id of ['btn-text-all', 'btn-text-free', 'btn-rec']) $(id).disabled = !on;
}

// ---------- 1. Ambiente ----------

async function controllaAmbiente() {
  const info = { userAgent: navigator.userAgent, data: new Date().toISOString() };
  try {
    if (!navigator.gpu) throw new Error('WebGPU non disponibile');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('nessun adattatore WebGPU');
    const ai = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
    info.webgpu = { vendor: ai.vendor, architecture: ai.architecture, device: ai.device, description: ai.description,
      maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      shaderF16: adapter.features.has('shader-f16') };
  } catch (e) { info.webgpu = { errore: String(e.message || e) }; }
  try {
    const est = await navigator.storage.estimate();
    info.memoria = { quota_GB: +(est.quota / 1e9).toFixed(1), usata_GB: +(est.usage / 1e9).toFixed(2) };
    info.memoria.persistente = await navigator.storage.persist();
  } catch (e) { info.memoria = { errore: String(e) }; }
  info.ram_GB = navigator.deviceMemory ?? null;
  info.copia_modello = await statoCopia();
  state.results.ambiente = info;
  $('env-out').textContent = JSON.stringify(info, null, 2);
  return info;
}

// ---------- 2. Modello (OPFS) ----------

async function statoCopia() {
  try {
    const root = await navigator.storage.getDirectory();
    const fh = await root.getFileHandle(OPFS_NAME);
    const f = await fh.getFile();
    return { presente: true, MB: Math.round(f.size / 1e6) };
  } catch { return { presente: false }; }
}

async function copiaInOPFS(file) {
  const root = await navigator.storage.getDirectory();
  const fh = await root.getFileHandle(OPFS_NAME, { create: true });
  const w = await fh.createWritable();
  const prog = $('model-progress');
  prog.hidden = false; prog.max = file.size; prog.value = 0;
  let scritti = 0;
  const conta = new TransformStream({
    transform(chunk, ctrl) { scritti += chunk.byteLength; prog.value = scritti; ctrl.enqueue(chunk); },
  });
  await file.stream().pipeThrough(conta).pipeTo(w);
  prog.hidden = true;
}

let wakeLock = null;
async function tieniSveglio(on) {
  try {
    if (on && 'wakeLock' in navigator) wakeLock = await navigator.wakeLock.request('screen');
    if (!on && wakeLock) { await wakeLock.release(); wakeLock = null; }
  } catch { /* opzionale */ }
}

async function caricaModello() {
  const out = $('model-out');
  $('btn-load').disabled = true;
  await tieniSveglio(true);
  try {
    const scelto = $('model-file').files[0];
    if (scelto) {
      out.textContent = `Copio ${scelto.name} (${Math.round(scelto.size / 1e6)} MB) nella memoria della pagina…`;
      traccia('copia-inizio', { MB: Math.round(scelto.size / 1e6) });
      await copiaInOPFS(scelto);
      traccia('copia-fine');
    }
    const root = await navigator.storage.getDirectory();
    const file = await (await root.getFileHandle(OPFS_NAME)).getFile();
    out.textContent = `Avvio del modello (${Math.round(file.size / 1e6)} MB)… può richiedere 1–2 minuti.`;
    if (state.llm) { state.llm.close(); state.llm = null; }
    const immagini = $('opt-image').checked;
    const audio = $('opt-audio').checked;
    const t0 = performance.now();
    traccia('avvio-inizio', { immagini, audio, maxTokens: +$('opt-tokens').value });
    const fileset = await FilesetResolver.forGenAiTasks(WASM_PATH);
    state.llm = await LlmInference.createFromOptions(fileset, {
      baseOptions: { modelAssetBuffer: file.stream().getReader() },
      maxTokens: +$('opt-tokens').value,
      topK: 1,
      temperature: 0,
      randomSeed: 1,
      maxNumImages: immagini ? 1 : 0,
      supportAudio: audio,
    });
    const ms = Math.round(performance.now() - t0);
    traccia('avvio-fine', { ms });
    state.results.modello = { nome: scelto?.name ?? '(copia salvata)', MB: Math.round(file.size / 1e6), ms_caricamento: ms, immagini, audio };
    out.textContent = `Modello pronto in ${(ms / 1000).toFixed(1)} s.`;
    abilita(true);
  } catch (e) {
    traccia('avvio-errore', { errore: String(e.message || e) });
    out.textContent = `Errore: ${e.message || e}`;
    state.results.modello = { errore: String(e.message || e) };
  } finally {
    $('btn-load').disabled = false;
    await tieniSveglio(false);
    aggiornaSommario();
  }
}

// ---------- 3. Testo ----------

async function eseguiTesto(caso) {
  const misure = await genera(promptTesto(caso.frase));
  const parsed = parseJSON(misure.testo);
  const cmp = caso.atteso ? await confronta(parsed.valore, caso.atteso, false) : null;
  const rec = { id: caso.id, frase: caso.frase, json_valido: parsed.ok, estratti: parsed.valore, grezzo: misure.testo, misure: { ...misure, testo: undefined }, confronto: cmp };
  state.results.testo.push(rec);
  $('text-results').prepend(cardRisultato(`${caso.id} · “${caso.frase}”`, rec.misure, cmp,
    parsed.ok ? '' : `<pre class="out">${esc(misure.testo)}</pre>`));
  aggiornaSommario();
}

async function testoTutte() {
  state.stop = false;
  $('btn-text-all').disabled = true; $('btn-text-stop').disabled = false;
  await tieniSveglio(true);
  try {
    for (const caso of CASI_TESTO) {
      if (state.stop) break;
      await eseguiTesto(caso);
    }
  } catch (e) { alert(`Errore: ${e.message || e}`); }
  $('btn-text-all').disabled = false; $('btn-text-stop').disabled = true;
  await tieniSveglio(false);
}

// ---------- 4. Foto ----------

function casoDaNome(nome) {
  const base = nome.toLowerCase();
  return CASI_FOTO.find((c) => base.includes(c.id.toLowerCase())) || null;
}

async function preparaImmagine(file, lato) {
  const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const scala = Math.min(1, lato / Math.max(bmp.width, bmp.height));
  const c = document.createElement('canvas');
  c.width = Math.round(bmp.width * scala); c.height = Math.round(bmp.height * scala);
  c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);
  bmp.close();
  return c;
}

async function eseguiFoto(files) {
  if (!state.llm) { alert('Carica prima il modello.'); return; }
  if (!state.results.modello?.immagini) { alert('Ricarica il modello con "immagini" attivo.'); return; }
  const lato = +$('photo-res').value;
  await tieniSveglio(true);
  for (const file of files) {
    try {
      const canvas = await preparaImmagine(file, lato);
      const thumb = canvas.toDataURL('image/jpeg', 0.6);
      const caso = casoDaNome(file.name);
      const misure = await genera(promptFoto(canvas));
      const parsed = parseJSON(misure.testo);
      const cmp = caso ? await confronta(parsed.valore, caso.atteso, true, caso.vietati || []) : null;
      const rec = { file: file.name, caso: caso?.id ?? null, descrizione: caso?.descrizione ?? null, risoluzione: lato,
        json_valido: parsed.ok, estratti: parsed.valore, grezzo: misure.testo, misure: { ...misure, testo: undefined }, confronto: cmp };
      state.results.foto.push(rec);
      $('photo-results').prepend(cardRisultato(`${file.name}${caso ? ` · ${caso.descrizione}` : ' · (caso non riconosciuto)'} · ${lato}px`,
        rec.misure, cmp, parsed.ok ? '' : `<pre class="out">${esc(misure.testo)}</pre>`, thumb));
      aggiornaSommario();
    } catch (e) {
      $('photo-results').prepend(Object.assign(document.createElement('div'), { className: 'res ko', textContent: `${file.name}: errore ${e.message || e}` }));
    }
  }
  await tieniSveglio(false);
}

// ---------- 5. Voce ----------

let rec = null;
let chunks = [];

async function avviaRegistrazione() {
  if (!state.results.modello?.audio) { alert('Ricarica il modello con "audio" attivo.'); return; }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  chunks = [];
  rec = new MediaRecorder(stream);
  rec.ondataavailable = (e) => chunks.push(e.data);
  rec.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    const blob = new Blob(chunks, { type: rec.mimeType || 'audio/webm' });
    await eseguiAudio(blob);
  };
  rec.start();
  rec._t0 = performance.now();
  $('btn-rec').disabled = true; $('btn-rec-stop').disabled = false;
}

function fermaRegistrazione() {
  if (rec && rec.state === 'recording') { rec._durata = Math.round(performance.now() - rec._t0); rec.stop(); }
  $('btn-rec').disabled = false; $('btn-rec-stop').disabled = true;
}

async function eseguiAudio(blob) {
  const idx = +$('audio-case').value;
  const caso = idx >= 0 ? CASI_TESTO[idx] : null;
  const url = URL.createObjectURL(blob);
  try {
    const misure = await genera(promptAudio(url));
    const parsed = parseJSON(misure.testo);
    const campi = parsed.valore.campi && typeof parsed.valore.campi === 'object' ? parsed.valore.campi : parsed.valore;
    const cmp = caso ? await confronta(campi, caso.atteso, false) : null;
    const r = { caso: caso?.id ?? null, frase: caso?.frase ?? null, durata_audio_ms: rec?._durata ?? null,
      trascrizione: parsed.valore.trascrizione ?? null, json_valido: parsed.ok, estratti: campi, grezzo: misure.testo,
      misure: { ...misure, testo: undefined }, confronto: cmp };
    state.results.audio.push(r);
    $('audio-results').prepend(cardRisultato(`${caso ? caso.id : 'Libera'} · audio ${((r.durata_audio_ms || 0) / 1000).toFixed(1)} s`,
      r.misure, cmp, `<p><b>Trascrizione:</b> ${esc(r.trascrizione ?? '—')}</p><audio controls src="${url}"></audio>`));
    aggiornaSommario();
  } catch (e) {
    $('audio-results').prepend(Object.assign(document.createElement('div'), { className: 'res ko', textContent: `Errore: ${e.message || e}` }));
  }
}

// ---------- 6. Risultati ----------

function scaricaRisultati() {
  const blob = new Blob([JSON.stringify(state.results, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `fase0_risultati_${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

// ---------- avvio ----------

function init() {
  const sel = $('audio-case');
  sel.innerHTML = '<option value="-1">Frase libera (nessun confronto)</option>' +
    CASI_TESTO.map((c, i) => `<option value="${i}">${c.id} · ${esc(c.frase)}</option>`).join('');

  $('btn-env').onclick = controllaAmbiente;
  $('model-file').onchange = () => { $('btn-load').disabled = !$('model-file').files.length; };
  $('btn-load').onclick = caricaModello;
  $('btn-forget').onclick = async () => {
    try { await (await navigator.storage.getDirectory()).removeEntry(OPFS_NAME); } catch { /* assente */ }
    $('model-out').textContent = 'Copia salvata eliminata.';
    $('btn-load').disabled = !$('model-file').files.length;
  };
  $('btn-text-all').onclick = testoTutte;
  $('btn-text-stop').onclick = () => { state.stop = true; };
  $('btn-text-free').onclick = async () => {
    const f = $('text-free').value.trim();
    if (!f) return;
    $('btn-text-free').disabled = true;
    try { await eseguiTesto({ id: 'Libera', frase: f, atteso: null }); } finally { $('btn-text-free').disabled = false; }
  };
  $('photo-file').onchange = (e) => eseguiFoto([...e.target.files]);
  $('photo-cam').onchange = (e) => eseguiFoto([...e.target.files]);
  $('btn-rec').onclick = avviaRegistrazione;
  $('btn-rec-stop').onclick = fermaRegistrazione;
  $('btn-download').onclick = scaricaRisultati;
  $('btn-clear').onclick = () => {
    state.results.testo = []; state.results.foto = []; state.results.audio = [];
    for (const id of ['text-results', 'photo-results', 'audio-results']) $(id).innerHTML = '';
    aggiornaSommario();
  };

  const log = ultimoLog();
  const ultimo = log[log.length - 1];
  if (ultimo && !['avvio-fine', 'generazione-fine', 'avvio-errore'].includes(ultimo.passo)) {
    $('crash-out').textContent = `⚠ La sessione precedente si è interrotta durante: ${ultimo.passo} (${ultimo.t}).\n` +
      log.slice(-8).map((l) => `${l.t.slice(11, 19)} ${l.passo} ${JSON.stringify({ ...l, t: undefined, passo: undefined })}`).join('\n');
  }
  state.results.log_precedente = log;
  $('btn-log-clear').onclick = () => { localStorage.removeItem(LOG_KEY); $('crash-out').textContent = ''; };

  statoCopia().then((s) => {
    if (s.presente) {
      $('btn-load').disabled = false;
      $('model-out').textContent = `Copia del modello già salvata (${s.MB} MB): premi "Carica modello".`;
    }
  });
  aggiornaSommario();
}

init();
