# PCFBench Derby

A single-file web page for playing "beat the model" on PCFBench items and browsing
every per-item model output next to the expert ground truth. Built for the
BrightCon 2026 Demo Derby (station DD6) and served from this folder with GitHub Pages.

## What is in this folder

| File | Purpose |
| ---- | ------- |
| `index.html` | The page. Data, styling and fonts are all inlined, so it works from GitHub Pages, from a local file, or as an attachment, and makes no third-party requests. Only the optional sheet backend is contacted. |
| `demo_data.json` | The per-item results behind the page: eight models across the six scored tasks, plus the expert answers. Reusable on its own. |
| `sheet-backend/` | Optional Google Apps Script that collects players' answers and flags into one Google Sheet. |

## How the page works

**Beat the model** serves a random benchmark item for one of three tasks. Triage:
should this candidate material be mapped to a single ecoinvent activity or decomposed
further? Map: which ecoinvent reference product fits this purchased material? Estimate:
guess the cradle-to-gate kg CO₂e of an EPD product. After each answer the page reveals
what all eight models said, what the expert said, and how many models the player beat.
A player can flag an item where they disagree with the expert.

**Explore results** shows the paper's headline scorecard, recomputed live for any
product category, six automatically selected failure stories, and a browser for every
item with all model outputs.

**Field record** aggregates every player's answers against the models on exactly the
items humans answered, plus a leaderboard and the list of flagged items. It only has
content when the shared record below is configured; without it, answers stay in each
visitor's browser.

## Sharing it as a file

`index.html` is self-contained. Rename it (for example `PCFBench-Derby.html`), send it
through Drive or Slack, and it opens from disk in any modern browser with the quiz and
explorer fully working offline. A player's own record persists in that browser. The
shared record works from a local file too, as long as the machine is online and the
sheet endpoint below has been set before the file was shared. Copies cannot be updated
once sent, so prefer the hosted page when the audience can reach it.

## Turning on the shared record

Answers and flags are posted to a Google Apps Script bound to a single spreadsheet.
The script's manifest requests only the `spreadsheets.currentonly` scope, so the
deployment can touch that one sheet and nothing else in the account.

1. Create a new Google Sheet. Open **Extensions → Apps Script**.
2. In the script editor, open **Project Settings** and tick
   *Show "appsscript.json" manifest file in editor*.
3. Replace the contents of `Code.gs` and `appsscript.json` with the files in
   `sheet-backend/`.
4. **Deploy → New deployment → Web app.** Execute as *Me*; who has access: *Anyone*.
   Authorize when prompted. The consent screen lists only "See, edit, create, and
   delete the spreadsheets this application has been installed in", which is the
   single-sheet scope.
5. Copy the deployment's `/exec` URL into `sheetEndpoint` inside
   `window.PCFBENCH_CONFIG` near the top of `index.html`, and commit.

The script validates every field (known event kinds and tasks, bounded lengths and
integers, a per-player rate limit, a hard cap on total rows) and neutralises any value
that Sheets would otherwise read as a formula, so a player cannot plant `=IMPORTXML`
or similar in your sheet. Each answer or flag becomes one row on an `events` tab. The sheet is the export:
filter or pivot it directly. To change the script later, use
**Deploy → Manage deployments → Edit → New version** so the URL stays the same.

Optional: set `TOKEN` in `Code.gs` and the matching `token` in `PCFBENCH_CONFIG`
to stop accidental posts from other copies of the page. The token is visible in the
page source, so it is a speed bump, not a secret.

## Where the data comes from

`demo_data.json` is generated from the per-item run outputs of the paper's sweep
(the same runs behind the headline table in the preprint). Tasks 2 and 3 use the
agentic harness, direct Task 7 uses the name-plus-description setting, and the
composed Task 7 rows are the canonical stepwise results. Item metadata such as
product category comes from the public dataset on Hugging Face.
