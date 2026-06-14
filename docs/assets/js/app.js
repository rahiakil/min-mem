/* Min-Mem GitHub Pages — charts, demo, agent visualization */

const DEMO_RULES = [
  [/in order to/gi, "to"],
  [/in spite of/gi, "though"],
  [/prior to/gi, "before"],
  [/nevertheless/gi, "yet"],
  [/nonetheless/gi, "yet"],
  [/furthermore/gi, "also"],
  [/moreover/gi, "also"],
  [/therefore/gi, "so"],
  [/however/gi, "but"],
  [/consequently/gi, "so"],
  [/subsequently/gi, "then"],
  [/previously/gi, "before"],
  [/approximately/gi, "about"],
  [/frequently/gi, "often"],
  [/particularly/gi, "very"],
  [/extremely/gi, "very"],
  [/numerous/gi, "many"],
  [/additional/gi, "more"],
  [/utilize/gi, "use"],
  [/utilized/gi, "used"],
  [/facilitate/gi, "aid"],
  [/accomplish/gi, "do"],
  [/investigate/gi, "check"],
  [/demonstrate/gi, "show"],
  [/required/gi, "needed"],
  [/require/gi, "need"],
  [/establish/gi, "set"],
  [/communicate/gi, "tell"],
  [/implement/gi, "do"],
  [/attempted/gi, "tried"],
  [/determined/gi, "found"],
  [/advantageous/gi, "good"],
  [/considerable/gi, "much"],
  [/endeavor/gi, "try"],
];

const NOUNS = /\b(python|redis|kubernetes|docker|github|obsidian|typescript|javascript|pandas|numpy|workflow|acme|phoenix)\b/gi;

function clientMinify(text) {
  let out = text;
  for (const [re, rep] of DEMO_RULES) out = out.replace(re, rep);
  return out;
}

function runDemo() {
  const input = document.getElementById("demo-input").value;
  const output = clientMinify(input);
  const saved = input.length - output.length;
  const pct = input.length ? ((saved / input.length) * 100).toFixed(1) : 0;
  document.getElementById("demo-output").innerHTML = `
    <div class="compare-box"><div class="label">Original (${input.length} chars)</div>${highlight(input, false)}</div>
    <div class="compare-arrow">→</div>
    <div class="compare-box"><div class="label">Minified (${output.length} chars, −${pct}%)</div>${highlight(output, true)}</div>`;
}

function highlight(text, minified) {
  return text.replace(/\b(\w+)\b/g, (w) => {
    if (NOUNS.test(w)) { NOUNS.lastIndex = 0; return `<span class="code-noun">${w}</span>`; }
    if (minified) return w;
    const lower = w.toLowerCase();
    if (["utilize","utilized","facilitate","accomplish","however","previously","additional","nevertheless","investigate","required","demonstrate"].includes(lower))
      return `<span class="code-verb">${w}</span>`;
    return w;
  });
}

Chart.defaults.color = "#8b9cb3";
Chart.defaults.borderColor = "#243044";
Chart.defaults.font.family = "Inter, system-ui, sans-serif";

async function loadData() {
  const [bench, agent] = await Promise.all([
    fetch("data/benchmark.json").then((r) => r.json()),
    fetch("data/agent-demo.json").then((r) => r.json()),
  ]);
  return { bench, agent };
}

function renderHero(bench, agent) {
  const s = bench.summary["min-mem (full)"];
  const ctx = agent.context;
  document.getElementById("stat-char").textContent = s.char_savings_pct_mean.toFixed(1) + "%";
  document.getElementById("stat-noun").textContent = s.nouns_preserved_pct_mean.toFixed(1) + "%";
  document.getElementById("stat-agent").textContent = ctx.chars_saved.toLocaleString() + " chars";
  document.getElementById("stat-qa").textContent = (agent.qa_match_rate ?? "—") + "%";
  document.getElementById("llm-model").textContent = agent.model + " (CPU via Ollama)";
}

function renderMethodChart(bench) {
  const s = bench.summary;
  const labels = ["Min-Mem", "Naive", "Phrase", "No infl."];
  const methods = ["min-mem (full)", "naive-dict", "phrase-only", "no-inflection"];
  new Chart(document.getElementById("chart-methods"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Char reduction %", data: methods.map((m) => s[m].char_savings_pct_mean), backgroundColor: "#3b9eff" },
        { label: "Token reduction %", data: methods.map((m) => s[m].token_savings_pct_mean), backgroundColor: "#22d3a6" },
      ],
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom" } }, scales: { y: { beginAtZero: true, title: { display: true, text: "%" } } } },
  });
}

function renderCategoryChart(bench) {
  const cats = bench.by_category_char_savings;
  const labels = Object.keys(cats).map((k) => k.replace(/_/g, " "));
  new Chart(document.getElementById("chart-categories"), {
    type: "bar",
    data: { labels, datasets: [{ label: "Char reduction %", data: Object.values(cats), backgroundColor: "#6366f1" }] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true } } },
  });
}

function renderSafetyChart(bench) {
  const s = bench.summary;
  const methods = ["min-mem (full)", "naive-dict", "phrase-only", "no-inflection"];
  const labels = ["Min-Mem", "Naive", "Phrase", "No infl."];
  new Chart(document.getElementById("chart-safety"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Char reduction %", data: methods.map((m) => s[m].char_savings_pct_mean), backgroundColor: "#3b9eff" },
        { label: "Noun preservation %", data: methods.map((m) => s[m].nouns_preserved_pct_mean), backgroundColor: "#22d3a6" },
        { label: "Synonym retention %", data: methods.map((m) => s[m].synonym_aware_pct_mean), backgroundColor: "#8b5cf6" },
      ],
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom" } }, scales: { y: { beginAtZero: true, max: 100 } } },
  });
}

function renderScaleChart(agent) {
  const p = agent.scale_projections;
  new Chart(document.getElementById("chart-scale"), {
    type: "line",
    data: {
      labels: p.map((x) => x.memory_blocks + " blocks"),
      datasets: [
        { label: "Verbose tokens (% of 8K)", data: p.map((x) => x.pct_of_8k_verbose), borderColor: "#f472b6", backgroundColor: "rgba(244,114,182,.1)", fill: true, tension: .3 },
        { label: "Minified tokens (% of 8K)", data: p.map((x) => x.pct_of_8k_minified), borderColor: "#22d3a6", backgroundColor: "rgba(34,211,166,.1)", fill: true, tension: .3 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "bottom" } },
      scales: { y: { beginAtZero: true, title: { display: true, text: "% of 8K context window" } } },
    },
  });
}

function renderStack(agent) {
  const v = agent.context.verbose;
  const m = agent.context.minified;
  const sys = 200, skills = agent.skills_chars;
  const memV = v.memory_tokens, memM = m.memory_tokens;
  const totalV = v.tokens, totalM = m.tokens;
  const freed = totalV - totalM;

  function bar(label, total, mem, freedTok) {
    const sysPct = (sys / total) * 100;
    const skPct = (skills / total) * 100 * 0.5;
    const memPct = (mem / total) * 100;
    const frPct = freedTok > 0 ? (freedTok / total) * 100 : 0;
    return `<div class="stack-label"><span>${label}</span><span>${total} tokens</span></div>
      <div class="stack-track">
        <div class="stack-seg seg-system" style="width:${sysPct}%">sys</div>
        <div class="stack-seg seg-skills" style="width:${skPct}%">skills</div>
        <div class="stack-seg seg-memory" style="width:${memPct}%">memory</div>
        ${frPct > 0 ? `<div class="stack-seg seg-freed" style="width:${frPct}%">freed</div>` : ""}
      </div>`;
  }

  document.getElementById("stack-verbose").innerHTML = bar("Verbose agent prompt", totalV, memV, 0);
  document.getElementById("stack-minified").innerHTML = bar("Minified agent prompt", totalM, memM, freed);
  document.getElementById("agent-summary").textContent =
    `${agent.memory_count} memory blocks + skills (${skills} chars). Saved ${agent.context.chars_saved} chars (${agent.context.char_reduction_pct}%) and ${agent.context.tokens_saved} tokens. Memory layer alone: −${agent.context.memory_char_reduction_pct}%.`;
}

function renderQA(agent) {
  if (!agent.qa_parity?.length) {
    document.getElementById("qa-table").innerHTML = "<p>Run <code>experiments/agent_context_demo.py</code> with Ollama to generate QA data.</p>";
    return;
  }
  const rows = agent.qa_parity.map((q) => `
    <tr>
      <td>${q.question}</td>
      <td style="font-size:.8rem">${q.answer_verbose.slice(0, 80)}…</td>
      <td style="font-size:.8rem">${q.answer_minified.slice(0, 80)}…</td>
      <td class="${q.match ? "match-yes" : "match-no"}">${q.match ? "✓ retained" : "△ check"}</td>
    </tr>`).join("");
  document.getElementById("qa-table").innerHTML = `
    <table><thead><tr><th>Question</th><th>Verbose memory</th><th>Minified memory</th><th>Facts</th></tr></thead><tbody>${rows}</tbody></table>
    <p style="margin-top:.75rem;font-size:.85rem;color:var(--accent2)">Keyword fact retention: ${agent.qa_match_rate}% on ${agent.model}</p>`;
}

async function init() {
  try {
    const { bench, agent } = await loadData();
    renderHero(bench, agent);
    renderMethodChart(bench);
    renderCategoryChart(bench);
    renderSafetyChart(bench);
    renderScaleChart(agent);
    renderStack(agent);
    renderQA(agent);
    runDemo();
  } catch (e) {
    console.error("Failed to load experiment data:", e);
  }
}

init();
