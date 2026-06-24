const ENGINES = {
  mongodb: {
    label: "MongoDB Vector Search",
    color: "#8f1d2c",
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
    color: "#0f7c80",
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
    color: "#b7791f",
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
  { label: "exact", backend: "MongoDB", value: 1.0, color: "#8f1d2c" },
  { label: "ann", backend: "MongoDB", value: 1.0, color: "#8f1d2c" },
  { label: "filtered", backend: "MongoDB", value: 1.0, color: "#8f1d2c" },
  { label: "text", backend: "MongoDB", value: 1.0, color: "#8f1d2c" },
  { label: "hybrid", backend: "MongoDB", value: 1.0, color: "#8f1d2c" },
  { label: "dense", backend: "Qdrant", value: 0.0, color: "#0f7c80", note: "à exporter" },
];

const LATENCY_ROWS = [
  { label: "MongoDB p50", value: 0.36, color: "#8f1d2c" },
  { label: "MongoDB p95", value: 0.72, color: "#8f1d2c" },
  { label: "Qdrant p50", value: 1.0, color: "#0f7c80" },
  { label: "Qdrant p95", value: 2.0, color: "#0f7c80" },
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

const state = {
  selectedEngine: "mongodb",
  selectedQuery: QUERIES[0],
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
    const value = row.note ? 0.18 : row.value;
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
    drawText(svg, row.note || row.value.toFixed(2), x + barWidth / 2, y - 10, {
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
    { x: width * 0.58, y: height * 0.25, label: "Corpus", color: "#8f1d2c" },
    { x: width * 0.78, y: height * 0.34, label: "Embeddings", color: "#0f7c80" },
    { x: width * 0.64, y: height * 0.57, label: "MongoDB", color: "#8f1d2c" },
    { x: width * 0.84, y: height * 0.64, label: "Qdrant", color: "#0f7c80" },
    { x: width * 0.72, y: height * 0.82, label: "RAG cité", color: "#b7791f" },
  ];
  ctx.lineWidth = 2;
  nodes.forEach((node, i) => {
    nodes.slice(i + 1).forEach((other) => {
      ctx.strokeStyle = "rgba(20, 33, 61, 0.14)";
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
    ctx.fillStyle = "#14213d";
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
    ["Question", active.id, "#8f1d2c"],
    ["Vector search", "top-k", "#0f7c80"],
    ["Chunks", active.cites.length, "#4b5c96"],
    ["Answer", "cité", "#b7791f"],
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
}

function init() {
  bindEvents();
  renderProfile();
  renderLeaderboard();
  renderQueries();
  selectQuery("q1");
  drawQualityChart();
  drawLatencyChart();
  drawHeroNetwork();
  drawRagCanvas();
  startGlobalSpaceField();
}

init();
