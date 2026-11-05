"use strict";

const RUN_KEY = "research-agent-run";
// How many times the page will start another request for one run before asking the visitor.
const MAX_REQUESTS = 8;
const BUSY_WAIT_MS = 4000;

const $ = (id) => document.getElementById(id);
let runId = null;
let draftedPlan = [];

// Reports hold text from models and web pages, so nothing here goes through innerHTML.
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function safeUrl(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : null;
  } catch {
    return null;
  }
}

function externalLink(text, url) {
  const href = safeUrl(url);
  if (!href) return document.createTextNode(text);
  const link = el("a", "", text);
  link.href = href;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

const INLINE = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[[^\]]+\]\(https?:\/\/[^\s)]+\)|\[\d{1,3}\])/g;

function inlineTokens(text) {
  return text.split(INLINE).filter(Boolean).map((part) => {
    let match;
    if ((match = part.match(/^\[(\d{1,3})\]$/))) return { cite: Number(match[1]) };
    if ((match = part.match(/^\*\*(.+)\*\*$/))) return { node: el("strong", "", match[1]) };
    if ((match = part.match(/^`(.+)`$/))) return { node: el("code", "", match[1]) };
    if ((match = part.match(/^\*(.+)\*$/))) return { node: el("em", "", match[1]) };
    if ((match = part.match(/^\[(.+)\]\((.+)\)$/))) return { node: externalLink(match[1], match[2]) };
    return { text: part };
  });
}

// A claim is everything up to and including its citations, so it can be marked as one piece.
function appendInline(parent, text, report) {
  let claim = el("span", "claim");
  let cited = [];

  const finish = () => {
    if (!claim.hasChildNodes()) return;
    if (cited.length) {
      report.claims.push({ node: claim, cited });
      parent.append(claim);
    } else {
      parent.append(...claim.childNodes);
    }
    claim = el("span", "claim");
    cited = [];
  };

  for (const token of inlineTokens(text)) {
    if (token.cite !== undefined) {
      const cite = el("a", "cite", `[${token.cite}]`);
      cite.href = `#${report.id}-source-${token.cite}`;
      claim.append(cite);
      cited.push(token.cite);
    } else if (token.node) {
      if (cited.length) finish();
      claim.append(token.node);
    } else {
      let rest = token.text;
      if (cited.length) {
        const punctuation = rest.match(/^[.,;:!?)]+/);
        if (punctuation) {
          claim.append(punctuation[0]);
          rest = rest.slice(punctuation[0].length);
        }
        finish();
      }
      if (rest) claim.append(rest);
    }
  }
  finish();
}

function sourceRow(number, title, url, report) {
  const row = el("li", "source");
  row.id = `${report.id}-source-${number}`;
  row.append(el("span", "source-number", String(number)));
  const text = el("span");
  text.append(externalLink(title, url));
  const href = safeUrl(url);
  if (href) text.append(el("span", "source-host", new URL(href).hostname.replace(/^www\./, "")));
  row.append(text);
  report.sources.set(number, row);
  return row;
}

function renderReport(container, markdown, id) {
  const report = { id, claims: [], sources: new Map() };
  const body = el("div", "report-body");
  const sources = el("ol", "sources");
  sources.setAttribute("aria-label", "Sources");
  let heading = null;
  let list = null;
  let inSources = false;

  for (const raw of markdown.split("\n")) {
    const line = raw.trim();
    const bullet = line.match(/^[-*]\s+(.+)$/);
    const numbered = line.match(/^(\d+)\.\s+(.+)$/);
    const title = line.match(/^#\s+(.+)$/);
    const subheading = line.match(/^#{2,4}\s+(.+)$/);

    if (!line) {
      list = null;
    } else if (title) {
      heading = el("h3", "", title[1]);
    } else if (subheading && subheading[1].toLowerCase() === "sources") {
      inSources = true;
    } else if (inSources && numbered) {
      const link = numbered[2].match(/^\[(.+)\]\((.+)\)$/);
      if (link) sources.append(sourceRow(Number(numbered[1]), link[1], link[2], report));
    } else if (subheading) {
      body.append(el("h4", "", subheading[1]));
    } else if (bullet || numbered) {
      const tag = bullet ? "UL" : "OL";
      if (!list || list.tagName !== tag) {
        list = document.createElement(tag.toLowerCase());
        body.append(list);
      }
      const item = el("li");
      appendInline(item, bullet ? bullet[1] : numbered[2], report);
      list.append(item);
    } else {
      list = null;
      const paragraph = el("p");
      appendInline(paragraph, line, report);
      body.append(paragraph);
    }
  }

  container.replaceChildren(...[heading, body].filter(Boolean));
  if (sources.childElementCount) container.append(sources);
  linkClaimsToSources(report);
}

// Point at a source to mark its claims, or at a claim to mark its sources. A tap on a touch
// screen keeps them marked until the next tap.
function linkClaimsToSources(report) {
  const mark = (nodes, on) => nodes.forEach((node) => node?.classList.toggle("marked", on));
  const claimsOf = (number) => report.claims.filter((c) => c.cited.includes(number)).map((c) => c.node);
  let pinned = [];

  const wire = (node, related) => {
    node.addEventListener("pointerenter", (event) => {
      if (event.pointerType === "mouse") mark(related(), true);
    });
    node.addEventListener("pointerleave", (event) => {
      if (event.pointerType === "mouse") mark(related(), false);
    });
    node.addEventListener("focusin", () => mark(related(), true));
    node.addEventListener("focusout", () => mark(related(), false));
    node.addEventListener("pointerup", (event) => {
      if (event.pointerType === "mouse") return;
      mark(pinned, false);
      pinned = related();
      mark(pinned, true);
    });
  };

  for (const [number, source] of report.sources) wire(source, () => [source, ...claimsOf(number)]);
  for (const claim of report.claims) {
    wire(claim.node, () => [claim.node, ...claim.cited.map((n) => report.sources.get(n))]);
  }
}

function remember(id) {
  try {
    if (id) localStorage.setItem(RUN_KEY, id);
    else localStorage.removeItem(RUN_KEY);
  } catch {
    // Private windows can refuse storage. The run still works, it just won't survive a reload.
  }
}

function remembered() {
  try {
    return localStorage.getItem(RUN_KEY);
  } catch {
    return null;
  }
}

function notice(message, kind) {
  $("notice").textContent = message || "";
  $("notice").hidden = !message;
  $("notice").classList.toggle("error", kind === "error");
}

function untilText(seconds) {
  if (seconds < 90) return "in about a minute";
  const minutes = Math.round(seconds / 60);
  if (minutes < 90) return `in about ${minutes} minutes`;
  return `in about ${Math.round(minutes / 60)} hours`;
}

function closeForm(closed) {
  $("draft").disabled = closed;
  $("question").disabled = closed;
}

async function loadLimits() {
  try {
    const response = await fetch("/api/limits");
    if (!response.ok) return;
    const limits = await response.json();
    const { sub_questions: questions, rounds, pages } = limits.caps;
    $("quota").textContent =
      `${limits.runs_left} of ${limits.runs_per_day} runs left today. ` +
      `Each run is capped at ${questions} sub-questions, ${rounds} round and ${pages} pages.`;
    if (limits.runs_left === 0) showDailyCap(limits.resets_in);
  } catch {
    // The page still works without the count.
  }
}

function showDailyCap(seconds) {
  closeForm(true);
  notice(
    `Today's demo runs are used up. New runs open at midnight UTC, ${untilText(seconds)}. ` +
    "A finished report is below.",
  );
}

function showRefusal(detail) {
  if (detail?.reason === "daily_cap") {
    showDailyCap(detail.retry_after);
  } else if (detail?.reason) {
    notice(`${detail.message} Try again ${untilText(detail.retry_after)}.`);
  } else {
    notice("Too many requests. Try again in a minute.");
  }
}

function showPlan(plan) {
  draftedPlan = plan;
  const items = plan.map((text, index) => {
    const row = el("li");
    const input = el("input");
    input.type = "text";
    input.value = text;
    input.maxLength = 300;
    input.setAttribute("aria-label", `Sub-question ${index + 1}`);
    row.append(input);
    return row;
  });
  $("plan-items").replaceChildren(...items);
  $("plan").hidden = false;
  $("approve").disabled = false;
  $("reject").disabled = false;
  $("plan").scrollIntoView({ block: "nearest" });
}

function logLine(message) {
  $("progress").hidden = false;
  $("progress-items").append(el("li", "", message));
}

function working(on) {
  $("progress").classList.toggle("working", on);
}

function showDone(run) {
  working(false);
  $("continue").hidden = true;
  closeForm(false);
  if (!run.report) {
    logLine(run.stop_reason ? `stopped: ${run.stop_reason}` : "stopped");
    return;
  }
  renderReport($("result-report"), run.report, "run");
  const { tavily_credits: credits, model_calls: calls } = run.cost;
  $("result-cost").textContent =
    `This run used ${credits} Tavily ${credits === 1 ? "credit" : "credits"} ` +
    `and ${calls} Gemini ${calls === 1 ? "call" : "calls"}.`;
  $("result").hidden = false;
  $("result").scrollIntoView({ block: "start" });
}

function canContinue(message) {
  working(false);
  notice(message, "error");
  $("continue").hidden = false;
}

function stopped(message) {
  working(false);
  closeForm(false);
  notice(message, "error");
}

// Reads one request's event stream. Returns how it ended: "done", "paused", "failed", or
// "dropped" if the connection closed with no ending, which is what a killed request looks like.
async function readEvents(response) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  let ending = "dropped";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) return ending;
    buffer += value.replaceAll("\r\n", "\n");

    let split;
    while ((split = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      const event = block.match(/^event: (.+)$/m)?.[1];
      const data = block.match(/^data: (.+)$/m)?.[1];
      if (!event || !data) continue;
      const payload = JSON.parse(data);

      if (event === "progress") {
        logLine(payload.message);
      } else if (event === "paused") {
        logLine("time limit for this request reached, continuing in a new one");
        ending = "paused";
      } else if (event === "done") {
        showDone(payload);
        ending = "done";
      } else if (event === "error") {
        canContinue(payload.message);
        ending = "failed";
      }
    }
  }
}

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Keeps a run moving: each request advances it and stops near the platform's time limit, and
// this starts the next one from where the last checkpoint left it.
async function drive(firstUrl, firstBody) {
  notice("");
  $("continue").hidden = true;
  working(true);

  let url = firstUrl;
  let body = firstBody;
  for (let request = 0; request < MAX_REQUESTS; request += 1) {
    let ending;
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: body ? { "content-type": "application/json" } : {},
        body: body ? JSON.stringify(body) : undefined,
      });
      if (response.status === 409) {
        // Another request still holds the run. It either finishes or its lease runs out.
        await wait(BUSY_WAIT_MS);
        ending = "dropped";
      } else if (response.status === 429) {
        showRefusal((await response.json()).detail);
        working(false);
        $("continue").hidden = false;
        return;
      } else if (response.status === 404) {
        stopped("This run isn't available any more. Ask another question to start again.");
        remember(null);
        return;
      } else if (!response.ok) {
        ending = "dropped";
      } else {
        ending = await readEvents(response);
      }
    } catch {
      ending = "dropped";
    }

    if (ending === "done" || ending === "failed") return;
    if (ending === "dropped") {
      const run = await fetchRun();
      if (!run) return;
      if (run.status === "done") {
        showDone(run);
        return;
      }
      if (run.status === "stopped") {
        stopped("This run reached its limit for the demo, so it stopped here.");
        return;
      }
      if (run.status === "awaiting_approval") {
        working(false);
        showPlan(run.plan);
        return;
      }
    }
    url = `/api/runs/${runId}/resume`;
    body = null;
  }
  canContinue("This run is taking longer than expected. You can continue it.");
}

async function fetchRun() {
  try {
    const response = await fetch(`/api/runs/${runId}`);
    if (response.status === 404) {
      stopped("This run isn't available any more. Ask another question to start again.");
      remember(null);
      return null;
    }
    if (!response.ok) throw new Error(String(response.status));
    return await response.json();
  } catch {
    canContinue("Lost the connection. You can continue the run.");
    return null;
  }
}

async function startRun(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  if (question.length < 3) return;

  resetRun();
  closeForm(true);
  $("draft").textContent = "Drafting a plan...";
  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (response.status === 429) {
      closeForm(false);
      showRefusal((await response.json()).detail);
      return;
    }
    if (!response.ok) {
      closeForm(false);
      const detail = (await response.json().catch(() => null))?.detail;
      notice(typeof detail === "string" ? detail : "That didn't work. Check the question and try again.", "error");
      return;
    }
    const run = await response.json();
    runId = run.run_id;
    remember(runId);
    showPlan(run.plan);
    loadLimits();
  } catch {
    closeForm(false);
    notice("Couldn't reach the server. Check your connection and try again.", "error");
  } finally {
    $("draft").textContent = "Draft a plan";
  }
}

function reviewPlan(approved) {
  $("approve").disabled = true;
  $("reject").disabled = true;
  const edited = [...$("plan-items").querySelectorAll("input")]
    .map((input) => input.value.trim())
    .filter(Boolean);
  const changed = edited.join("\n") !== draftedPlan.join("\n");
  $("plan").hidden = true;
  drive(`/api/runs/${runId}/plan`, changed && approved ? { approved, plan: edited } : { approved });
}

function resetRun() {
  runId = null;
  remember(null);
  notice("");
  for (const id of ["plan", "progress", "result", "continue"]) $(id).hidden = true;
  $("progress-items").replaceChildren();
  working(false);
}

// Coming back to the page, or reloading it, picks up the visitor's own run if they have one.
async function restoreRun() {
  runId = remembered();
  if (!runId) return;

  const response = await fetch(`/api/runs/${runId}`).catch(() => null);
  if (!response?.ok) {
    runId = null;
    remember(null);
    return;
  }
  const run = await response.json();
  $("question").value = run.question;
  if (run.status === "awaiting_approval") {
    showPlan(run.plan);
  } else if (run.status === "done") {
    showDone(run);
  } else if (run.status === "stopped") {
    stopped("This run reached its limit for the demo, so it stopped here.");
  } else {
    logLine("picking up where this run left off");
    drive(`/api/runs/${runId}/resume`, null);
  }
}

async function loadExample() {
  try {
    const response = await fetch("/example.md");
    if (response.ok) renderReport($("example-report"), await response.text(), "example");
  } catch {
    // The noscript link stays hidden, so say where to read it instead.
    $("example-report").textContent = "The example didn't load. It's in the examples folder on GitHub.";
  }
}

$("ask").addEventListener("submit", startRun);
$("approve").addEventListener("click", () => reviewPlan(true));
$("reject").addEventListener("click", () => reviewPlan(false));
$("continue-run").addEventListener("click", () => drive(`/api/runs/${runId}/resume`, null));
$("again").addEventListener("click", () => {
  resetRun();
  $("question").value = "";
  $("question").focus();
  loadLimits();
});

loadExample();
loadLimits();
restoreRun();
