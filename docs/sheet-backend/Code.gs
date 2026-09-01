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
  const s = typeof v === 'object' ? JSON.stringify(v) : String(v);
  return s.length > MAX_CELL_CHARS ? s.slice(0, MAX_CELL_CHARS) : s;
}

function doPost(e) {
  let body;
  try {
    body = JSON.parse(e.postData.contents);
  } catch (err) {
    return json_({ ok: false, error: 'body is not JSON' });
  }
  if (TOKEN && body.token !== TOKEN) return json_({ ok: false, error: 'bad token' });
  if (!['answer', 'flag', 'unflag'].includes(body.kind)) return json_({ ok: false, error: 'unknown kind' });
  const row = [body.ts, body.kind, body.pid, body.name, body.task, body.id, body.item, body.answer, body.correct, body.beats, body.faced, new Date().toISOString()].map(cell_);
  const lock = LockService.getScriptLock();
  lock.waitLock(5000);
  try {
    eventsSheet_().appendRow(row);
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
