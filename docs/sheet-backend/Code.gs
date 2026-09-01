/**
 * PCFBench Derby: shared record backend.
 *
 * Container-bound Apps Script (Extensions > Apps Script from the spreadsheet). The manifest
 * requests only the `spreadsheets.currentonly` scope, so the deployment can read and write
 * this one spreadsheet and nothing else in the account.
 *
 * POST appends one event row (answer | flag | unflag). GET returns the most recent rows as
 * JSON; the page rebuilds per-player records from them client-side.
 */
const SHEET_NAME = 'events';
const HEADERS = ['ts', 'kind', 'pid', 'name', 'task', 'item_id', 'item', 'answer', 'correct', 'beats', 'faced', 'received_at'];
const MAX_ROWS_RETURNED = 5000;
const MAX_CELL_CHARS = 500;
// Hard ceiling on stored events. Past it, writes are refused so a flood cannot fill the
// sheet or burn the account's daily Apps Script quota indefinitely.
const MAX_TOTAL_ROWS = 50000;
// Per-player write budget (events per minute), enforced with the script cache.
const MAX_EVENTS_PER_MINUTE = 60;
const KINDS = ['answer', 'flag', 'unflag'];
const TASKS = ['triage', 'mapping', 'estimate'];
const PID_RE = /^[a-z0-9]{6,40}$/;
const MAX_NAME_CHARS = 40;
// Optional shared token. If non-empty, POSTs must carry the same value in `token`.
// It is visible in the page source, so it only stops accidental or drive-by writes.
const TOKEN = '';

function eventsSheet_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName(SHEET_NAME);
  if (!sh) {
    sh = ss.insertSheet(SHEET_NAME);
    sh.appendRow(HEADERS);
    sh.setFrozenRows(1);
  }
  return sh;
}

function cell_(v) {
  if (v === null || v === undefined) return '';
  let s = typeof v === 'object' ? JSON.stringify(v) : String(v);
  if (s.length > MAX_CELL_CHARS) s = s.slice(0, MAX_CELL_CHARS);
  // Sheets treats a value beginning with = + - @ or a tab/CR as a formula. A leading apostrophe
  // forces text (it is not displayed), so a player cannot plant IMPORTXML/HYPERLINK in the sheet.
  if (/^\s*[=+\-@]|^[\t\r\n]/.test(s)) s = "'" + s;
  return s;
}

// Coerce untrusted fields to the shapes the page expects; anything odd becomes empty.
function bool_(v) { return v === true || v === 'true' ? 'true' : v === false || v === 'false' ? 'false' : ''; }
function int_(v, max) { const n = Number(v); return Number.isInteger(n) && n >= 0 && n <= max ? n : ''; }
function str_(v, max) { return v === null || v === undefined ? '' : String(v).slice(0, max); }

function overBudget_(pid) {
  const cache = CacheService.getScriptCache();
  const key = 'rate:' + pid + ':' + Math.floor(Date.now() / 60000);
  const n = Number(cache.get(key) || 0) + 1;
  cache.put(key, String(n), 120);
  return n > MAX_EVENTS_PER_MINUTE;
}

function doPost(e) {
  let body;
  try {
    body = JSON.parse(e.postData.contents);
  } catch (err) {
    return json_({ ok: false, error: 'body is not JSON' });
  }
  if (TOKEN && body.token !== TOKEN) return json_({ ok: false, error: 'bad token' });
  if (!KINDS.includes(body.kind)) return json_({ ok: false, error: 'unknown kind' });
  if (!TASKS.includes(body.task)) return json_({ ok: false, error: 'unknown task' });
  if (!PID_RE.test(String(body.pid))) return json_({ ok: false, error: 'bad pid' });
  if (overBudget_(body.pid)) return json_({ ok: false, error: 'slow down' });
  const row = [
    str_(body.ts, 40), body.kind, body.pid, str_(body.name, MAX_NAME_CHARS), body.task, str_(body.id, 80),
    str_(body.item, 200), str_(body.answer, 200), bool_(body.correct), int_(body.beats, 50), int_(body.faced, 50),
    new Date().toISOString(),
  ].map(cell_);
  const lock = LockService.getScriptLock();
  lock.waitLock(5000);
  try {
    const sh = eventsSheet_();
    if (sh.getLastRow() > MAX_TOTAL_ROWS) return json_({ ok: false, error: 'record full' });
    sh.appendRow(row);
  } finally {
    lock.releaseLock();
  }
  return json_({ ok: true });
}

function doGet() {
  const values = eventsSheet_().getDataRange().getValues();
  const head = values.shift() || HEADERS;
  const rows = values.slice(-MAX_ROWS_RETURNED).map(r => {
    const o = {};
    head.forEach((h, i) => { o[h] = r[i] instanceof Date ? r[i].toISOString() : r[i]; });
    return o;
  });
  return json_({ rows: rows });
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
