const ENGINES = {
  mongodb: {
    label: "MongoDB Vector Search",
    color: "#059669",
    score: "Validation complète",
    caption: "Cinq modes de recherche, RAG cité et artefacts offline.",
    bars: [
      ["Index vectoriel", 100],
      ["Recherche hybride", 100],
      ["RAG cité", 100],
      ["Benchmark live Atlas", 40],
    ],
  },
  qdrant: {
    label: "Qdrant",
    color: "#0284c7",
    score: "Smoke intégré",
    caption: "Backend corrigé et testable ; métriques qrels encore à exporter.",
    bars: [
      ["Collection vectorielle", 82],
      ["Filtres payload", 75],
      ["Benchmark dense", 58],
      ["RAG complet", 30],
    ],
  },
  comparison: {
    label: "Comparaison finale",
    color: "#d97706",
    score: "Préliminaire",
    caption: "La comparaison reste honnête tant que les environnements diffèrent.",
    bars: [
      ["Contrat partagé", 100],
      ["Corpus gelé", 100],
      ["Métriques alignées", 66],
      ["Runs live équivalents", 35],
    ],
  },
};

const QUALITY_ROWS = [
  { label: "exact", backend: "MongoDB", value: 1.0, color: "#059669", note: "validé" },
  { label: "ann", backend: "MongoDB", value: 1.0, color: "#059669", note: "offline" },
  { label: "filtered", backend: "MongoDB", value: 1.0, color: "#059669", note: "validé" },
  { label: "text", backend: "MongoDB", value: 1.0, color: "#059669", note: "validé" },
  { label: "hybrid", backend: "MongoDB", value: 1.0, color: "#059669", note: "validé" },
  { label: "qrels", backend: "Qdrant", value: 0.0, color: "#0284c7", note: "à exporter" },
];

const LATENCY_ROWS = [
  { label: "MongoDB offline p50", value: 0.36, color: "#059669" },
  { label: "MongoDB offline p95", value: 0.72, color: "#059669" },
  { label: "Qdrant smoke p50", value: 1.0, color: "#0284c7" },
  { label: "Qdrant smoke p95", value: 2.0, color: "#0284c7" },
];

const QUERIES = [
  {
    id: "q1",
    question: "Comment MongoDB et Qdrant stockent les vecteurs avec métadonnées ?",
    answer:
      "MongoDB conserve les embeddings avec les documents BSON. Qdrant stocke les vecteurs dans une collection et garde les métadonnées dans les payloads.",
    cites: ["mongodb-overview:0", "qdrant-overview:0"],
  },
  {
    id: "q2",
    question: "Quel est le compromis recall / latence dans HNSW ?",
    answer:
      "Augmenter la largeur de recherche améliore généralement le rappel, mais augmente aussi la latence et le coût de calcul.",
    cites: ["hnsw-tradeoffs:0"],
  },
  {
    id: "q3",
    question: "Comment combiner recherche textuelle et recherche vectorielle ?",
    answer:
      "Le projet utilise une fusion de rangs réciproques pour mélanger les signaux lexicaux et sémantiques sans comparer directement les scores bruts.",
    cites: ["hybrid-search:0"],
  },
  {
    id: "q4",
    question: "Que doit faire un RAG quand les preuves sont insuffisantes ?",
    answer:
      "Il doit refuser d’inventer une réponse et garder un résultat valide avec une justification explicite.",
    cites: ["rag-grounding:0"],
  },
];

const CORPUS = [
  {
    chunkId: "mongodb-overview:0",
    title: "MongoDB Vector Search",
    category: "mongodb",
    text:
      "MongoDB Vector Search garde les embeddings avec les documents BSON et les métadonnées, donc la recherche sémantique, les filtres et les mises à jour restent dans une base opérationnelle.",
  },
  {
    chunkId: "qdrant-overview:0",
    title: "Qdrant Collections",
    category: "qdrant",
    text:
      "Qdrant est une base vectorielle spécialisée. Ses collections stockent les vecteurs avec des payloads, et les index de payload accélèrent les filtres.",
  },
  {
    chunkId: "hnsw-tradeoffs:0",
    title: "HNSW Trade-offs",
    category: "indexing",
    text:
      "HNSW est un graphe de voisins approximatifs. Une recherche plus large améliore souvent le rappel, mais augmente la latence et le coût.",
  },
  {
    chunkId: "hybrid-search:0",
    title: "Hybrid Retrieval",
    category: "retrieval",
    text:
      "La recherche hybride combine un matching lexical et une similarité vectorielle. La fusion de rangs réciproques mélange les classements sans comparer les scores bruts.",
  },
  {
    chunkId: "rag-grounding:0",
    title: "Grounded RAG",
    category: "rag",
    text:
      "Un système RAG robuste récupère des preuves avant génération, cite les chunks utilisés et refuse d’inventer une réponse quand le contexte est insuffisant.",
  },
  {
    chunkId: "metadata-filtering:0",
    title: "Metadata Filtering",
    category: "retrieval",
    text:
      "Les filtres de métadonnées réduisent la recherche vectorielle par langue, catégorie, source ou date. Les deux bases doivent exécuter les mêmes cas de filtre.",
  },
];

const SYNONYMS = {
  combiner: ["hybride", "fusion", "lexical", "vectorielle"],
  textuelle: ["lexical", "matching", "hybride"],
  vectorielle: ["embedding", "sémantique", "vecteurs", "vector"],
  metadata: ["métadonnées", "payloads", "filtres"],
  metadonnees: ["métadonnées", "payloads", "filtres"],
  preuve: ["citations", "chunks", "rag"],
  preuves: ["citations", "chunks", "rag"],
  inventer: ["refuse", "insuffisant", "rag"],
  latence: ["hnsw", "rappel", "coût"],
};

const state = {
  selectedEngine: "mongodb",
  selectedQuery: QUERIES[0],
  lastSearchResults: [],
  pointers: {
    global: { x: -9999, y: -9999 },
    hero: { x: -9999, y: -9999 },
  },
  spaceNodes: [],
};

function setCssMousePosition(event) {
  document.documentElement.style.setProperty("--mouse-x", `${event.clientX}px`);
  document.documentElement.style.setProperty("--mouse-y", `${event.clientY}px`);
  state.pointers.global = { x: event.clientX, y: event.clientY };
}

function svgElement(tag, attrs = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function drawText(svg, text, x, y, attrs = {}) {
  const element = svgElement("text", { x, y, ...attrs });
  element.textContent = text;
  svg.appendChild(element);
  return element;
}

function clearSvg(svg, width = 940, height = 430) {
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  return { width, height };
}

function drawChartFrame(svg, bounds, ticks, yScale) {
  ticks.forEach((tick) => {
    const y = yScale(tick);
    svg.appendChild(svgElement("line", { class: "grid-line", x1: bounds.left, x2: bounds.right, y1: y, y2: y }));
    drawText(svg, `${Math.round(tick * 100)}%`, bounds.left - 16, y + 4, {
      "font-size": 12,
      "text-anchor": "end",
      class: "chart-label",
    });
  });
  svg.appendChild(svgElement("line", { class: "axis-line", x1: bounds.left, x2: bounds.right, y1: bounds.bottom, y2: bounds.bottom }));
  svg.appendChild(svgElement("line", { class: "axis-line", x1: bounds.left, x2: bounds.left, y1: bounds.top, y2: bounds.bottom }));
}

function drawQualityChart() {
  const svg = document.getElementById("qualityChart");
  const { width, height } = clearSvg(svg, 960, 430);
  const bounds = { left: 72, right: width - 34, top: 38, bottom: height - 74 };
  const yScale = (value) => bounds.bottom - value * (bounds.bottom - bounds.top);
  const gap = (bounds.right - bounds.left) / QUALITY_ROWS.length;

  drawChartFrame(svg, bounds, [0.25, 0.5, 0.75, 1], yScale);

  QUALITY_ROWS.forEach((row, index) => {
    const x = bounds.left + index * gap + gap * 0.16;
    const barWidth = gap * 0.62;
    const value = row.note === "à exporter" ? 0.18 : row.value;
    const y = yScale(value);
    svg.appendChild(svgElement("rect", {
      x,
      y,
      width: barWidth,
      height: bounds.bottom - y,
      rx: 6,
      fill: row.color,
      "fill-opacity": row.note ? 0.26 : 0.88,
    }));
    drawText(svg, row.label, x + barWidth / 2, bounds.bottom + 28, {
      "font-size": 12,
      "text-anchor": "middle",
      class: "chart-label",
    });
    drawText(svg, row.note || "validé", x + barWidth / 2, y - 10, {
      "font-size": 12,
      "font-weight": 900,
      "text-anchor": "middle",
      fill: row.color,
    });
  });
}

function drawLatencyChart() {
  const svg = document.getElementById("latencyChart");
  const { width, height } = clearSvg(svg, 960, 450);
  const bounds = { left: 190, right: width - 44, top: 44, bottom: height - 44 };
  const maxValue = Math.max(...LATENCY_ROWS.map((row) => row.value));
  const rowGap = (bounds.bottom - bounds.top) / LATENCY_ROWS.length;

  LATENCY_ROWS.forEach((row, index) => {
    const y = bounds.top + index * rowGap + 16;
    const barWidth = ((bounds.right - bounds.left) * row.value) / maxValue;
    drawText(svg, row.label, bounds.left - 18, y + 18, {
      "font-size": 14,
      "font-weight": 900,
      "text-anchor": "end",
      class: "chart-label",
    });
    svg.appendChild(svgElement("rect", {
      x: bounds.left,
      y,
      width: bounds.right - bounds.left,
      height: 30,
      rx: 15,
      fill: "#edf2f4",
    }));
    svg.appendChild(svgElement("rect", {
      x: bounds.left,
      y,
      width: barWidth,
      height: 30,
      rx: 15,
      fill: row.color,
      "fill-opacity": 0.86,
    }));
    drawText(svg, `${row.value.toFixed(2)} ms`, bounds.left + barWidth + 12, y + 21, {
      "font-size": 13,
      "font-weight": 900,
      fill: row.color,
    });
  });
}

function renderProfile() {
  const engine = ENGINES[state.selectedEngine];
  document.querySelector('[data-profile="name"]').textContent = engine.label;
  document.querySelector('[data-profile="score"]').textContent = engine.score;
  document.querySelector('[data-profile="caption"]').textContent = engine.caption;
  document.querySelector('[data-profile="dot"]').style.background = engine.color;

  const bars = document.getElementById("profileBars");
  bars.replaceChildren();
  engine.bars.forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "profile-bar";
    row.innerHTML = `
      <div class="bar-head"><span>${label}</span><strong>${value}%</strong></div>
      <div class="bar-track"><div class="bar-fill" style="width:${value}%; background:${engine.color}"></div></div>
    `;
    bars.appendChild(row);
  });

  document.querySelectorAll(".engine-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.engine === state.selectedEngine);
  });
}

function renderLeaderboard() {
  const rows = [
    ["1", "MongoDB : track Ossama complet", "5 modes + RAG"],
    ["2", "Qdrant : track Hamza intégré", "smoke OK"],
    ["3", "À finir : run Qdrant qrels", "qualité stricte"],
    ["4", "À finir : run Atlas live", "latence réelle"],
  ];
  const leaderboard = document.getElementById("leaderboard");
  leaderboard.replaceChildren();
  rows.forEach(([rank, label, value]) => {
    const row = document.createElement("div");
    row.className = "leaderboard-row";
    row.innerHTML = `<span>${rank}. ${label}</span><strong>${value}</strong>`;
    leaderboard.appendChild(row);
  });
}

function normalizeText(text) {
  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .replace(/[^\p{L}\p{N}\s-]/gu, " ");
}

function tokenize(text) {
  const baseTokens = normalizeText(text)
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 2);
  const expanded = [...baseTokens];
  baseTokens.forEach((token) => {
    (SYNONYMS[token] || []).forEach((synonym) => expanded.push(normalizeText(synonym)));
  });
  return expanded;
}

function semanticSearch(query, { category = "all", topK = 3, backend = "mongodb" } = {}) {
  const queryTokens = tokenize(query);
  const filtered = CORPUS.filter((chunk) => category === "all" || chunk.category === category);
  return filtered
    .map((chunk) => {
      const haystack = tokenize(`${chunk.title} ${chunk.category} ${chunk.text}`);
      const overlap = queryTokens.filter((token) => haystack.includes(token)).length;
      const categoryBoost = queryTokens.includes(chunk.category) ? 0.14 : 0;
      const backendBoost =
        backend === "mongodb" && chunk.category === "mongodb"
          ? 0.04
          : backend === "qdrant" && chunk.category === "qdrant"
            ? 0.04
            : 0;
      const score = Math.min(0.99, overlap / Math.max(4, queryTokens.length) + categoryBoost + backendBoost);
      return { ...chunk, score };
    })
    .filter((chunk) => chunk.score > 0)
    .sort((a, b) => b.score - a.score || a.title.localeCompare(b.title))
    .slice(0, topK);
}

function searchCorpus() {
  const query = document.getElementById("semanticQueryInput").value.trim();
  const category = document.getElementById("categoryFilter").value;
  const topK = Number(document.getElementById("topKFilter").value);
  const mongodb = semanticSearch(query, { category, topK, backend: "mongodb" });
  const qdrant = semanticSearch(query, { category, topK, backend: "qdrant" });
  state.lastSearchResults = mongodb.length >= qdrant.length ? mongodb : qdrant;

  renderSearchResults("mongodbResults", mongodb, "vector + text + metadata");
  renderSearchResults("qdrantResults", qdrant, "dense vector + payload");
  generateGroundedAnswer(query, state.lastSearchResults);

  const status = document.getElementById("semanticStatus");
  status.textContent = `${mongodb.length + qdrant.length} résultats affichés · filtre = ${category} · top-k = ${topK}`;
}

function renderSearchResults(containerId, rows, modeLabel) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (rows.length === 0) {
    const empty = document.createElement("div");
    empty.className = "result-card";
    empty.innerHTML = "<strong>Aucun chunk pertinent</strong><p>Le système refuserait de répondre sans preuve suffisante.</p>";
    container.appendChild(empty);
    return;
  }
  rows.forEach((row, index) => {
    const card = document.createElement("article");
    card.className = "result-card";
    card.innerHTML = `
      <strong>#${index + 1} · ${row.title}</strong>
      <p>${row.text}</p>
      <div class="result-meta">
        <span>${row.chunkId}</span>
        <span>${row.category}</span>
        <span>score ${(row.score * 100).toFixed(0)}%</span>
        <span>${modeLabel}</span>
      </div>
    `;
    container.appendChild(card);
  });
}

function generateGroundedAnswer(query, rows) {
  const target = document.getElementById("ragGeneratedAnswer");
  if (!rows.length) {
    target.innerHTML = `
      <strong>Refus contrôlé</strong>
      <p>Preuves insuffisantes dans le contexte récupéré. Le RAG ne doit pas inventer une réponse.</p>
    `;
    return;
  }
  const citations = rows.slice(0, 2).map((row) => row.chunkId);
  const answer = rows
    .slice(0, 2)
    .map((row) => row.text)
    .join(" ");
  target.innerHTML = `
    <strong>Réponse proposée</strong>
    <p>${answer}</p>
    <p><b>Question :</b> ${query}</p>
    <div>${citations.map((citation) => `<cite>${citation}</cite>`).join(" ")}</div>
  `;
}

function updateDecisionComparator() {
  const documentNeed = Number(document.getElementById("documentNeedRange").value);
  const vectorNeed = Number(document.getElementById("vectorNeedRange").value);
  const mongoScore = 6 + documentNeed * 1.4 + Math.max(0, 5 - vectorNeed) * 0.4;
  const qdrantScore = 6 + vectorNeed * 1.35 + Math.max(0, 5 - documentNeed) * 0.25;
  const winner = mongoScore >= qdrantScore ? "MongoDB Vector Search" : "Qdrant";
  const reason =
    winner === "MongoDB Vector Search"
      ? "choix plus naturel quand le modèle document, les filtres et la recherche textuelle font partie du même système."
      : "choix plus naturel quand la spécialisation vectorielle et les collections dédiées deviennent prioritaires.";
  document.getElementById("decisionOutput").innerHTML = `
    <strong>${winner}</strong>
    <p>Score MongoDB : ${mongoScore.toFixed(1)} · Score Qdrant : ${qdrantScore.toFixed(1)}</p>
    <p>${reason}</p>
  `;
}

function renderQueries() {
  const list = document.getElementById("queryList");
  list.replaceChildren();
  QUERIES.forEach((query) => {
    const button = document.createElement("button");
    button.className = "query-button";
    button.type = "button";
    button.textContent = query.question;
    button.classList.toggle("active", query.id === state.selectedQuery.id);
    button.addEventListener("click", () => selectQuery(query.id));
    list.appendChild(button);
  });
}

function selectQuery(id) {
  const query = QUERIES.find((item) => item.id === id);
  if (!query) return;
  state.selectedQuery = query;
  document.querySelector('[data-rag="qid"]').textContent = query.id;
  document.getElementById("answerTitle").textContent = query.question;
  document.getElementById("answerText").textContent = query.answer;
  document.getElementById("citationBox").textContent = `Cite : ${query.cites.join(" · ")}`;
  renderQueries();
  drawRagCanvas();
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function startGlobalSpaceField() {
  const canvas = document.getElementById("spaceFieldCanvas");
  state.spaceNodes = Array.from({ length: 58 }, (_, index) => ({
    x: (index * 97) % window.innerWidth,
    y: (index * 53) % window.innerHeight,
    vx: ((index % 5) - 2) * 0.05,
    vy: (((index + 2) % 5) - 2) * 0.05,
  }));

  function drawSpaceField() {
    const { ctx, width, height } = resizeCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    state.spaceNodes.forEach((node, index) => {
      node.x = (node.x + node.vx + width) % width;
      node.y = (node.y + node.vy + height) % height;
      for (let j = index + 1; j < state.spaceNodes.length; j += 1) {
        const other = state.spaceNodes[j];
        const dx = node.x - other.x;
        const dy = node.y - other.y;
        const distance = Math.hypot(dx, dy);
        if (distance < 135) {
          ctx.strokeStyle = `rgba(15, 124, 128, ${0.12 * (1 - distance / 135)})`;
          ctx.beginPath();
          ctx.moveTo(node.x, node.y);
          ctx.lineTo(other.x, other.y);
          ctx.stroke();
        }
      }
      ctx.fillStyle = index % 3 === 0 ? "rgba(143, 29, 44, 0.28)" : "rgba(15, 124, 128, 0.24)";
      ctx.beginPath();
      ctx.arc(node.x, node.y, 2.1, 0, Math.PI * 2);
      ctx.fill();
    });
    requestAnimationFrame(drawSpaceField);
  }

  drawSpaceField();
}

function drawHeroNetwork() {
  const canvas = document.getElementById("heroCanvas");
  const { ctx, width, height } = resizeCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const nodes = [
    { x: width * 0.58, y: height * 0.25, label: "Corpus", color: "#059669" },
    { x: width * 0.78, y: height * 0.34, label: "Embeddings", color: "#0284c7" },
    { x: width * 0.64, y: height * 0.57, label: "MongoDB", color: "#059669" },
    { x: width * 0.84, y: height * 0.64, label: "Qdrant", color: "#0284c7" },
    { x: width * 0.72, y: height * 0.82, label: "RAG cité", color: "#d97706" },
  ];
  ctx.lineWidth = 2;
  nodes.forEach((node, i) => {
    nodes.slice(i + 1).forEach((other) => {
      ctx.strokeStyle = "rgba(203, 213, 225, 0.16)";
      ctx.beginPath();
      ctx.moveTo(node.x, node.y);
      ctx.quadraticCurveTo((node.x + other.x) / 2, (node.y + other.y) / 2 - 30, other.x, other.y);
      ctx.stroke();
    });
  });
  nodes.forEach((node) => {
    ctx.fillStyle = node.color;
    ctx.beginPath();
    ctx.arc(node.x, node.y, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#0f172a";
    ctx.font = "800 13px Inter, system-ui, sans-serif";
    ctx.fillText(node.label, node.x + 15, node.y + 5);
  });
}

function drawRagCanvas() {
  const canvas = document.getElementById("ragCanvas");
  const { ctx, width, height } = resizeCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const active = state.selectedQuery;
  const steps = [
    ["Question", active.id, "#059669"],
    ["Vector search", "top-k", "#0284c7"],
    ["Chunks", active.cites.length, "#2563eb"],
    ["Answer", "cité", "#d97706"],
  ];
  const y = height * 0.5;
  const gap = width / (steps.length + 1);
  steps.forEach(([label, value, color], index) => {
    const x = gap * (index + 1);
    if (index > 0) {
      const prevX = gap * index;
      ctx.strokeStyle = "rgba(20, 33, 61, 0.25)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(prevX + 42, y);
      ctx.quadraticCurveTo((prevX + x) / 2, y - 48, x - 42, y);
      ctx.stroke();
    }
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 42, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.font = "900 18px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(String(value), x, y + 6);
    ctx.fillStyle = "#14213d";
    ctx.font = "900 14px Inter, system-ui, sans-serif";
    ctx.fillText(label, x, y + 70);
  });
}

function bindEvents() {
  window.addEventListener("pointermove", setCssMousePosition);
  window.addEventListener("resize", () => {
    drawHeroNetwork();
    drawRagCanvas();
  });
  document.querySelectorAll(".engine-button").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedEngine = button.dataset.engine;
      renderProfile();
    });
  });
  document.getElementById("runSemanticSearch").addEventListener("click", searchCorpus);
  document.getElementById("semanticQueryInput").addEventListener("input", searchCorpus);
  document.getElementById("categoryFilter").addEventListener("change", searchCorpus);
  document.getElementById("topKFilter").addEventListener("change", searchCorpus);
  document.getElementById("documentNeedRange").addEventListener("input", updateDecisionComparator);
  document.getElementById("vectorNeedRange").addEventListener("input", updateDecisionComparator);
}

function init() {
  bindEvents();
  renderProfile();
  renderLeaderboard();
  renderQueries();
  selectQuery("q1");
  drawQualityChart();
  drawLatencyChart();
  searchCorpus();
  updateDecisionComparator();
  drawHeroNetwork();
  drawRagCanvas();
  startGlobalSpaceField();
}

init();
