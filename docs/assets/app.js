const queries = [
  {
    id: "q1",
    text: "How do MongoDB and Qdrant store vectors with metadata?",
    answer:
      "MongoDB keeps embeddings beside BSON documents; Qdrant stores vectors with payload metadata. Both designs can support semantic search and filters.",
    cites: ["mongodb-overview:0", "qdrant-overview:0"],
  },
  {
    id: "q2",
    text: "What is the recall and latency trade-off in HNSW search?",
    answer:
      "Increasing search breadth generally improves recall, but it also raises query latency and compute cost.",
    cites: ["hnsw-tradeoffs:0"],
  },
  {
    id: "q3",
    text: "How can keyword and semantic rankings be combined?",
    answer:
      "Hybrid retrieval can merge lexical and vector rankings with reciprocal rank fusion, without requiring raw scores to mean the same thing.",
    cites: ["hybrid-search:0"],
  },
  {
    id: "q4",
    text: "What should grounded RAG do when evidence is weak?",
    answer:
      "A grounded RAG system should cite the chunks it used and refuse to invent an answer when the retrieved context is insufficient.",
    cites: ["rag-grounding:0"],
  },
  {
    id: "q5",
    text: "Which metadata fields can narrow semantic search?",
    answer:
      "Language, category, source, and date-style metadata can narrow semantic retrieval when both engines use the same filter cases.",
    cites: ["metadata-filtering:0"],
  },
];

const queryList = document.querySelector("#queryList");
const answerLabel = document.querySelector("#answerLabel");
const answerTitle = document.querySelector("#answerTitle");
const answerText = document.querySelector("#answerText");
const citationBox = document.querySelector("#citationBox");

function renderQueryButtons() {
  queries.forEach((query, index) => {
    const button = document.createElement("button");
    button.className = "query-button";
    button.type = "button";
    button.textContent = query.text;
    button.setAttribute("aria-pressed", String(index === 0));
    button.addEventListener("click", () => selectQuery(query.id));
    queryList.appendChild(button);
  });
}

function selectQuery(id) {
  const query = queries.find((item) => item.id === id);
  if (!query) return;

  document.querySelectorAll(".query-button").forEach((button, index) => {
    button.setAttribute("aria-pressed", String(queries[index].id === id));
  });

  answerLabel.textContent = `Selected query · ${query.id}`;
  answerTitle.textContent = query.text;
  answerText.textContent = query.answer;
  citationBox.textContent = `Cites: ${query.cites.join(" · ")}`;
}

renderQueryButtons();
