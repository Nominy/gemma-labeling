const statusNode = document.querySelector("#status");
const taxonomyGrid = document.querySelector("#taxonomy-grid");
const taxonomySummary = document.querySelector("#taxonomy-summary");
const form = document.querySelector("#label-form");

const baselineRaw = document.querySelector("#baseline-raw");
const baselineTags = document.querySelector("#baseline-tags");
const constrainedRaw = document.querySelector("#constrained-raw");
const constrainedTags = document.querySelector("#constrained-tags");
const traceList = document.querySelector("#trace-list");
const finalState = document.querySelector("#final-state");
const tokenStats = document.querySelector("#token-stats");

const defaultPrompts = {
  system_prompt:
    "You are a booru-style image tagger.\nReturn only canonical comma-separated tags with no prose, numbering, or explanations.\nPrefer concrete visual attributes that are visible in the image.",
  user_prompt:
    "Tag the image using the provided canonical vocabulary.\nPrioritize subject count, composition, setting, visible appearance details, attire, and pose.",
};

for (const [name, value] of Object.entries(defaultPrompts)) {
  const field = form.elements.namedItem(name);
  if (field) {
    field.value = value;
  }
}

await loadTaxonomy();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = new FormData(form);
  setStatus("Running inference...");

  try {
    const response = await fetch("/label", {
      method: "POST",
      body: payload,
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    const data = await response.json();
    renderResult(data);
    setStatus(`Completed with ${data.model_id}`);
  } catch (error) {
    setStatus(`Request failed: ${error.message}`);
  }
});

async function loadTaxonomy() {
  try {
    const response = await fetch("/taxonomy");
    const data = await response.json();
    const starterPreview = data.starter_tags.slice(0, 16).join(", ");
    const starterSuffix =
      data.starter_tags.length > 16
        ? `, ... (+${data.starter_tags.length - 16} more)`
        : "";
    taxonomySummary.textContent = `${data.tag_count} tags across ${Object.keys(data.categories).length} categories. Starter tags: ${starterPreview}${starterSuffix}`;
    taxonomyGrid.innerHTML = data.rules
      .map(
        (rule) => `
          <article class="taxonomy-card">
            <h3>${rule.canonical}</h3>
            <p><strong>Category:</strong> ${rule.category}</p>
            <p><strong>Prereqs:</strong> ${rule.prerequisites.join(", ") || "none"}</p>
            <p><strong>Exclusions:</strong> ${rule.exclusions.join(", ") || "none"}</p>
            <p><strong>Implications:</strong> ${rule.implications.join(", ") || "none"}</p>
          </article>
        `,
      )
      .join("");
    setStatus("Ready.");
  } catch (error) {
    setStatus(`Failed to load taxonomy: ${error.message}`);
  }
}

function renderResult(data) {
  baselineRaw.textContent = data.baseline.raw_text || "(no baseline output)";
  constrainedRaw.textContent = data.constrained.raw_text || "(no constrained output)";
  renderTags(baselineTags, data.baseline.normalized_tags);
  renderTags(constrainedTags, data.constrained.normalized_tags);

  traceList.innerHTML = data.trace
    .map(
      (step) => `
        <li>
          <strong>${step.tag}</strong>
          implied ${step.implied_tags.join(", ") || "nothing"}
          ; unlocked ${step.unlocked_tags.join(", ") || "nothing new"}
        </li>
      `,
    )
    .join("");
  finalState.textContent = JSON.stringify(data.final_state, null, 2);
  tokenStats.textContent = JSON.stringify(data.invalid_token_stats, null, 2);
}

function renderTags(node, tags) {
  node.innerHTML = tags.length
    ? tags.map((tag) => `<span class="chip">${tag}</span>`).join("")
    : '<span class="chip">No normalized tags</span>';
}

function setStatus(message) {
  statusNode.textContent = message;
}
