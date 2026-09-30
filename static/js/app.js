document.addEventListener("DOMContentLoaded", () => {
  // Global State
  let currentSampleDatasets = {};
  let currentAnalysisData = null;
  let activeSelectedObjectIdx = 0;
  let currentUploadedBase64 = null;
  let webcamStream = null;

  // DOM Elements
  const navTabs = document.querySelectorAll(".nav-tab");
  const tabPages = document.querySelectorAll(".tab-page");

  // Tab Switching
  navTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const target = tab.getAttribute("data-tab");
      navTabs.forEach(t => t.classList.remove("active"));
      tabPages.forEach(p => p.classList.add("hidden"));

      tab.classList.add("active");
      document.getElementById(`tab-${target}`).classList.remove("hidden");

      if (target === "knowledge") loadKnowledgeBase();
    });
  });

  // -------------------------------------------------------------
  // TAB 1: VISUAL PERCEPTION PIPELINE
  // -------------------------------------------------------------
  const presetDatasetSelect = document.getElementById("preset-dataset-select");
  const presetImageSelect = document.getElementById("preset-image-select");
  const inputPreviewImg = document.getElementById("input-preview-img");
  const btnRunAnalysis = document.getElementById("btn-run-analysis");
  const fileInput = document.getElementById("file-input");
  const dropZone = document.getElementById("drop-zone");

  const modeBtns = document.querySelectorAll(".mode-btn");
  const presetControls = document.getElementById("preset-controls");
  const uploadControls = document.getElementById("upload-controls");

  let inputMode = "preset"; // preset or upload

  modeBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      modeBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      inputMode = btn.getAttribute("data-mode");

      if (inputMode === "preset") {
        presetControls.classList.remove("hidden");
        uploadControls.classList.add("hidden");
        updatePresetPreview();
      } else {
        presetControls.classList.add("hidden");
        uploadControls.classList.remove("hidden");
        if (currentUploadedBase64) inputPreviewImg.src = currentUploadedBase64;
      }
    });
  });

  // Fetch Preset Datasets
  async function fetchSampleDatasets() {
    try {
      const res = await fetch("/api/sample_datasets");
      const json = await res.json();
      if (json.success) {
        currentSampleDatasets = json.datasets;
        populatePresetImageSelect();
      }
    } catch (err) {
      console.error("Error fetching datasets:", err);
    }
  }

  function populatePresetImageSelect() {
    const dsKey = presetDatasetSelect.value;
    const files = currentSampleDatasets[dsKey] || [];

    presetImageSelect.innerHTML = "";
    if (files.length === 0) {
      presetImageSelect.innerHTML = `<option value="">No samples found</option>`;
      return;
    }

    files.forEach(item => {
      const opt = document.createElement("option");
      opt.value = item.rel_path;
      opt.textContent = item.filename;
      presetImageSelect.appendChild(opt);
    });

    updatePresetPreview();
  }

  presetDatasetSelect.addEventListener("change", populatePresetImageSelect);
  presetImageSelect.addEventListener("change", updatePresetPreview);

  function updatePresetPreview() {
    const val = presetImageSelect.value;
    if (val) {
      inputPreviewImg.src = `/${val}`;
    }
  }

  // Handle File Upload & Drag-and-Drop
  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", e => handleFileUpload(e.target.files[0]));

  dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  function handleFileUpload(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
      currentUploadedBase64 = e.target.result;
      inputPreviewImg.src = currentUploadedBase64;
    };
    reader.readAsDataURL(file);
  }

  // Run Perception Pipeline
  btnRunAnalysis.addEventListener("click", runSingleAnalysis);

  async function runSingleAnalysis() {
    let payload = {};
    if (inputMode === "preset") {
      const samplePath = presetImageSelect.value;
      if (!samplePath) {
        alert("Please select a sample image.");
        return;
      }
      payload = { sample_path: samplePath };
    } else {
      if (!currentUploadedBase64) {
        alert("Please upload an image first.");
        return;
      }
      payload = { image_base64: currentUploadedBase64 };
    }

    btnRunAnalysis.disabled = true;
    btnRunAnalysis.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing...`;

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const json = await res.json();
      if (json.success) {
        currentAnalysisData = json.data;
        renderAnalysisResults();
      } else {
        alert("Analysis error: " + json.error);
      }
    } catch (err) {
      console.error(err);
      alert("Failed to analyze image.");
    } finally {
      btnRunAnalysis.disabled = false;
      btnRunAnalysis.innerHTML = `<i class="fa-solid fa-play"></i> Run Classical AI Perception`;
    }
  }

  // Render Pipeline & Reasoning Results
  const stageDisplayImg = document.getElementById("stage-display-img");
  const stageCaptionText = document.getElementById("stage-caption-text");
  const stageTabs = document.querySelectorAll(".stage-tab");
  const otsuValBadge = document.getElementById("otsu-val-badge");

  const stageCaptions = {
    stage_1_original: "Stage 1: Raw Input Perception - Image read as BGR/RGB pixel matrix.",
    stage_2_grayscale: "Stage 2: Grayscale Conversion - Intensity reduction (Luminance channel).",
    stage_3_blurred: "Stage 3: Noise Reduction - Gaussian Blur (5x5 kernel) to smooth noise.",
    stage_4_otsu_threshold: "Stage 4: Segmentation - Otsu's Thresholding (Optimal variance split).",
    stage_5_morphological: "Stage 5: Morphological Filtering - Opening (noise removal) + Closing (hole fill).",
    stage_6_detected_contours: "Stage 6: Multi-Object Contours - Polygon approximation & bounding box detection."
  };

  stageTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      stageTabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      const stageKey = tab.getAttribute("data-stage");
      displayStageImage(stageKey);
    });
  });

  function displayStageImage(stageKey) {
    if (!currentAnalysisData) return;
    const b64 = currentAnalysisData.pipeline_stages[stageKey];
    if (b64) {
      stageDisplayImg.src = b64;
      stageCaptionText.textContent = stageCaptions[stageKey] || stageKey;
    }
  }

  function renderAnalysisResults() {
    if (!currentAnalysisData) return;

    // Set Otsu badge
    const otsuVal = currentAnalysisData.pipeline_stages.otsu_threshold_value;
    otsuValBadge.textContent = `Otsu Threshold: ${otsuVal.toFixed(1)}`;

    // Default to stage 6 (contours)
    stageTabs.forEach(t => t.classList.remove("active"));
    const contourTab = document.querySelector('.stage-tab[data-stage="stage_6_detected_contours"]');
    if (contourTab) contourTab.classList.add("active");
    displayStageImage("stage_6_detected_contours");

    // Hide empty state & show analysis content
    document.getElementById("no-analysis-msg").classList.add("hidden");
    document.getElementById("analysis-content").classList.remove("hidden");
    document.getElementById("matrix-section").classList.remove("hidden");

    // Render Object Selector Pills
    const objectPicker = document.getElementById("detected-objects-picker");
    const objectPills = document.getElementById("object-pills");
    objectPills.innerHTML = "";

    const objects = currentAnalysisData.objects;
    if (objects.length > 0) {
      objectPicker.classList.remove("hidden");
      objects.forEach((obj, idx) => {
        const pill = document.createElement("div");
        pill.className = `obj-pill ${idx === 0 ? "active" : ""}`;
        pill.textContent = `#${obj.id}: ${obj.reasoning.recognized_name}`;
        pill.addEventListener("click", () => {
          document.querySelectorAll(".obj-pill").forEach(p => p.classList.remove("active"));
          pill.classList.add("active");
          activeSelectedObjectIdx = idx;
          renderObjectDetails(objects[idx]);
        });
        objectPills.appendChild(pill);
      });

      activeSelectedObjectIdx = 0;
      renderObjectDetails(objects[0]);
    } else {
      objectPicker.classList.add("hidden");
      alert("No distinct objects detected in the image.");
    }
  }

  function renderObjectDetails(obj) {
    const facts = obj.symbolic_facts;
    const feats = obj.features;
    const reasoning = obj.reasoning;

    // Facts
    document.getElementById("fact-color").textContent = facts.color;
    document.getElementById("fact-shape").textContent = facts.shape;
    document.getElementById("fact-size").textContent = facts.size;

    // Low Level CV Metrics
    document.getElementById("metric-vertices").textContent = feats.num_vertices;
    document.getElementById("metric-aspect").textContent = feats.aspect_ratio;
    document.getElementById("metric-circularity").textContent = feats.circularity;
    document.getElementById("metric-hsv").textContent = `H:${feats.mean_hsv[0]}, S:${feats.mean_hsv[1]}, V:${feats.mean_hsv[2]}`;
    document.getElementById("metric-bbox").textContent = `(${obj.bounding_box.x}, ${obj.bounding_box.y}, ${obj.bounding_box.w}x${obj.bounding_box.h})`;

    // Decision Banner
    const banner = document.getElementById("decision-banner");
    const statusEl = document.getElementById("decision-status");
    const nameEl = document.getElementById("decision-name");
    const confFill = document.getElementById("confidence-fill");
    const confText = document.getElementById("confidence-text");
    const expText = document.getElementById("explanation-text");

    banner.className = "decision-banner";
    if (reasoning.status === "RECOGNIZED") {
      banner.classList.add("banner-recognized");
      statusEl.textContent = "EXACT PRODUCTION RULE MATCH";
    } else if (reasoning.status === "RECOGNIZED_FALLBACK") {
      banner.classList.add("banner-fallback");
      statusEl.textContent = "WEIGHTED FALLBACK CANDIDATE";
    } else {
      banner.classList.add("banner-unknown");
      statusEl.textContent = "UNKNOWN OBJECT";
    }

    nameEl.textContent = reasoning.recognized_name;
    const pct = Math.round(reasoning.confidence * 100);
    confFill.style.width = `${pct}%`;
    confText.textContent = `${pct}%`;
    expText.textContent = reasoning.explanation;

    // Similarity Matrix Table
    const tbody = document.getElementById("matrix-tbody");
    tbody.innerHTML = "";

    reasoning.matrix.forEach(item => {
      const tr = document.createElement("tr");
      if (item.exact_match) tr.style.background = "rgba(16, 185, 129, 0.1)";

      tr.innerHTML = `
        <td><strong>${item.object}</strong></td>
        <td>${item.color_score === 1.0 ? '<span class="badge badge-status">Match (1.0)</span>' : item.color_score > 0 ? `<span class="badge badge-info">Fuzzy (${item.color_score})</span>` : '<span class="badge badge-danger">0.0</span>'}</td>
        <td>${item.shape_score === 1.0 ? '<span class="badge badge-status">Match (1.0)</span>' : item.shape_score > 0 ? `<span class="badge badge-info">Fuzzy (${item.shape_score})</span>` : '<span class="badge badge-danger">0.0</span>'}</td>
        <td>${item.size_score === 1.0 ? '<span class="badge badge-status">Match (1.0)</span>' : item.size_score > 0 ? `<span class="badge badge-info">Fuzzy (${item.size_score})</span>` : '<span class="badge badge-danger">0.0</span>'}</td>
        <td><strong>${item.weighted_score.toFixed(2)}</strong></td>
        <td>${item.exact_match ? '<span class="badge badge-status"><i class="fa-solid fa-check"></i> IF-THEN Match</span>' : '<span class="text-muted">-</span>'}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // -------------------------------------------------------------
  // TAB 2: BATCH PROCESSING
  // -------------------------------------------------------------
  const btnRunBatch = document.getElementById("btn-run-batch");
  const batchKpis = document.getElementById("batch-kpis");
  const batchTbody = document.getElementById("batch-tbody");
  const btnExportCsv = document.getElementById("btn-export-csv");

  let lastBatchResults = [];

  const radioCards = document.querySelectorAll(".radio-card");
  radioCards.forEach(card => {
    card.addEventListener("click", () => {
      radioCards.forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      const radio = card.querySelector('input[type="radio"]');
      if (radio) radio.checked = true;
    });
  });

  btnRunBatch.addEventListener("click", async () => {
    const dsRadio = document.querySelector('input[name="batch-ds"]:checked');
    const datasetKey = dsRadio ? dsRadio.value : "synthetic";

    btnRunBatch.disabled = true;
    btnRunBatch.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running Batch Inference...`;

    try {
      const res = await fetch("/api/batch_analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_key: datasetKey })
      });
      const json = await res.json();
      if (json.success) {
        lastBatchResults = json.results;
        renderBatchResults(json.summary, json.results);
      } else {
        alert("Batch error: " + json.error);
      }
    } catch (err) {
      console.error(err);
      alert("Failed to execute batch inference.");
    } finally {
      btnRunBatch.disabled = false;
      btnRunBatch.innerHTML = `<i class="fa-solid fa-play"></i> Execute Batch Inference`;
    }
  });

  function renderBatchResults(summary, results) {
    batchKpis.classList.remove("hidden");
    document.getElementById("kpi-total-img").textContent = summary.total_images;
    document.getElementById("kpi-total-obj").textContent = summary.total_objects;
    document.getElementById("kpi-recognized").textContent = summary.recognized_count;
    document.getElementById("kpi-unknown").textContent = summary.unknown_count;
    document.getElementById("kpi-accuracy").textContent = `${summary.recognition_rate}%`;

    batchTbody.innerHTML = "";
    btnExportCsv.classList.remove("hidden");

    results.forEach((row, idx) => {
      const tr = document.createElement("tr");
      const isRec = row.status.startsWith("RECOGNIZED");
      const statusBadge = isRec
        ? `<span class="badge badge-status">${row.status}</span>`
        : `<span class="badge badge-danger">UNKNOWN</span>`;

      tr.innerHTML = `
        <td>${idx + 1}</td>
        <td><code>${row.filename}</code></td>
        <td><span class="fact-tag">${row.color}</span></td>
        <td><span class="fact-tag">${row.shape}</span></td>
        <td><span class="fact-tag">${row.size}</span></td>
        <td><strong>${row.recognized_name}</strong></td>
        <td>${statusBadge}</td>
        <td><small>${row.matched_rule}</small></td>
        <td>${Math.round(row.confidence * 100)}%</td>
      `;
      batchTbody.appendChild(tr);
    });
  }

  // Export CSV
  btnExportCsv.addEventListener("click", async () => {
    if (!lastBatchResults || lastBatchResults.length === 0) return;

    try {
      const res = await fetch("/api/export_csv", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ results: lastBatchResults })
      });
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "classical_ai_recognition_report.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      console.error(err);
      alert("Failed to download CSV.");
    }
  });

  // -------------------------------------------------------------
  // TAB 3: KNOWLEDGE BASE MANAGER
  // -------------------------------------------------------------
  const kbTbody = document.getElementById("kb-tbody");
  const kbCountBadge = document.getElementById("kb-count-badge");
  const btnResetKb = document.getElementById("btn-reset-kb");
  const btnOpenAddModal = document.getElementById("btn-open-add-modal");
  const ruleModal = document.getElementById("rule-modal");
  const btnCloseModal = document.getElementById("btn-close-modal");
  const btnCancelRule = document.getElementById("btn-cancel-rule");
  const btnSaveRule = document.getElementById("btn-save-rule");

  async function loadKnowledgeBase() {
    try {
      const res = await fetch("/api/knowledge_base");
      const json = await res.json();
      if (json.success) {
        kbCountBadge.innerHTML = `<i class="fa-solid fa-book-bookmark"></i> KB: ${json.total_rules} Rules`;
        renderKnowledgeBaseTable(json.knowledge_base);
      }
    } catch (err) {
      console.error(err);
    }
  }

  function renderKnowledgeBaseTable(kbList) {
    kbTbody.innerHTML = "";
    kbList.forEach(entry => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${entry.name}</strong></td>
        <td><span class="badge badge-info">${entry.color}</span></td>
        <td><span class="badge badge-info">${entry.shape}</span></td>
        <td><span class="badge badge-info">${entry.size}</span></td>
        <td><code>${entry.name} Rule</code></td>
        <td>
          <button class="btn btn-danger btn-sm btn-del-rule" data-name="${entry.name}">
            <i class="fa-solid fa-trash"></i> Delete
          </button>
        </td>
      `;
      kbTbody.appendChild(tr);
    });

    document.querySelectorAll(".btn-del-rule").forEach(btn => {
      btn.addEventListener("click", async () => {
        const name = btn.getAttribute("data-name");
        if (confirm(`Delete rule '${name}' from Knowledge Base?`)) {
          await deleteRule(name);
        }
      });
    });
  }

  async function deleteRule(name) {
    try {
      const res = await fetch("/api/delete_rule", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name })
      });
      const json = await res.json();
      if (json.success) {
        loadKnowledgeBase();
      }
    } catch (err) {
      console.error(err);
    }
  }

  btnResetKb.addEventListener("click", async () => {
    if (confirm("Reset Knowledge Base to default 13 entries?")) {
      try {
        const res = await fetch("/api/reset_kb", { method: "POST" });
        const json = await res.json();
        if (json.success) loadKnowledgeBase();
      } catch (err) {
        console.error(err);
      }
    }
  });

  // Modal Actions
  btnOpenAddModal.addEventListener("click", () => {
    document.getElementById("modal-rule-name").value = "";
    ruleModal.classList.remove("hidden");
  });

  const closeModal = () => ruleModal.classList.add("hidden");
  btnCloseModal.addEventListener("click", closeModal);
  btnCancelRule.addEventListener("click", closeModal);

  btnSaveRule.addEventListener("click", async () => {
    const name = document.getElementById("modal-rule-name").value.trim();
    const color = document.getElementById("modal-rule-color").value;
    const shape = document.getElementById("modal-rule-shape").value;
    const size = document.getElementById("modal-rule-size").value;

    if (!name) {
      alert("Please enter object name.");
      return;
    }

    try {
      const res = await fetch("/api/add_rule", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, color, shape, size })
      });
      const json = await res.json();
      if (json.success) {
        closeModal();
        loadKnowledgeBase();
      } else {
        alert("Error: " + json.error);
      }
    } catch (err) {
      console.error(err);
      alert("Failed to save rule.");
    }
  });

  // Acquire Knowledge button from Tab 1
  document.getElementById("btn-acquire-knowledge").addEventListener("click", () => {
    if (!currentAnalysisData || currentAnalysisData.objects.length === 0) return;
    const activeObj = currentAnalysisData.objects[activeSelectedObjectIdx];
    const facts = activeObj.symbolic_facts;

    document.getElementById("modal-rule-name").value = `${capitalize(facts.color)} ${capitalize(facts.shape)}`;
    document.getElementById("modal-rule-color").value = facts.color;
    document.getElementById("modal-rule-shape").value = facts.shape;
    document.getElementById("modal-rule-size").value = facts.size;

    ruleModal.classList.remove("hidden");
  });

  function capitalize(str) {
    return str ? str.charAt(0).toUpperCase() + str.slice(1) : "";
  }

  // -------------------------------------------------------------
  // TAB 4: SYNTHETIC STUDIO
  // -------------------------------------------------------------
  const btnGenerateSynth = document.getElementById("btn-generate-synth");
  const synthResultBox = document.getElementById("synth-result-box");

  btnGenerateSynth.addEventListener("click", async () => {
    const shape = document.getElementById("synth-shape").value;
    const color = document.getElementById("synth-color").value;
    const size = document.getElementById("synth-size").value;

    btnGenerateSynth.disabled = true;
    btnGenerateSynth.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Generating...`;

    try {
      const res = await fetch("/api/generate_synthetic", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ shape, color, size })
      });
      const json = await res.json();
      if (json.success) {
        renderSyntheticResult(json.data);
      }
    } catch (err) {
      console.error(err);
      alert("Failed to generate shape.");
    } finally {
      btnGenerateSynth.disabled = false;
      btnGenerateSynth.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Generate & Analyze Shape`;
    }
  });

  function renderSyntheticResult(data) {
    if (!data.objects || data.objects.length === 0) return;
    const obj = data.objects[0];
    const reasoning = obj.reasoning;

    synthResultBox.innerHTML = `
      <div class="analysis-grid">
        <div class="text-center">
          <img src="${data.pipeline_stages.stage_6_detected_contours}" style="max-width:100%; border-radius:12px;" alt="Synthetic">
        </div>
        <div>
          <h4>Generated Synthetic Object</h4>
          <div class="fact-pills mt-2">
            <div class="fact-pill"><span class="lbl">Color</span><strong class="val">${obj.symbolic_facts.color}</strong></div>
            <div class="fact-pill"><span class="lbl">Shape</span><strong class="val">${obj.symbolic_facts.shape}</strong></div>
            <div class="fact-pill"><span class="lbl">Size</span><strong class="val">${obj.symbolic_facts.size}</strong></div>
          </div>
          <div class="decision-banner ${reasoning.status === 'RECOGNIZED' ? 'banner-recognized' : 'banner-unknown'} mt-3">
            <div class="decision-status">${reasoning.status}</div>
            <div class="decision-name">${reasoning.recognized_name}</div>
          </div>
          <p class="explanation-p mt-3">${reasoning.explanation}</p>
        </div>
      </div>
    `;
  }

  // -------------------------------------------------------------
  // TAB 5: LIVE WEBCAM
  // -------------------------------------------------------------
  const btnToggleCam = document.getElementById("btn-toggle-cam");
  const btnSnapCam = document.getElementById("btn-snap-cam");
  const webcamVideo = document.getElementById("webcam-video");
  const webcamCanvas = document.getElementById("webcam-canvas");
  const camResults = document.getElementById("cam-results");

  btnToggleCam.addEventListener("click", async () => {
    if (webcamStream) {
      // Stop webcam
      webcamStream.getTracks().forEach(t => t.stop());
      webcamStream = null;
      webcamVideo.srcObject = null;
      btnToggleCam.innerHTML = `<i class="fa-solid fa-power-off"></i> Start Camera`;
      btnSnapCam.disabled = true;
    } else {
      // Start webcam
      try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcamVideo.srcObject = webcamStream;
        btnToggleCam.innerHTML = `<i class="fa-solid fa-stop"></i> Stop Camera`;
        btnSnapCam.disabled = false;
      } catch (err) {
        alert("Camera access denied or unavailable: " + err.message);
      }
    }
  });

  btnSnapCam.addEventListener("click", async () => {
    if (!webcamStream) return;
    const ctx = webcamCanvas.getContext("2d");
    webcamCanvas.width = webcamVideo.videoWidth;
    webcamCanvas.height = webcamVideo.videoHeight;
    ctx.drawImage(webcamVideo, 0, 0);

    const dataUrl = webcamCanvas.toDataURL("image/png");

    btnSnapCam.disabled = true;
    btnSnapCam.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Frame...`;

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: dataUrl })
      });
      const json = await res.json();
      if (json.success) {
        renderWebcamResults(json.data);
      }
    } catch (err) {
      console.error(err);
    } finally {
      btnSnapCam.disabled = false;
      btnSnapCam.innerHTML = `<i class="fa-solid fa-camera-retro"></i> Capture & Analyze Frame`;
    }
  });

  function renderWebcamResults(data) {
    if (!data.objects || data.objects.length === 0) {
      camResults.innerHTML = `<h4>No distinct object detected in frame.</h4>`;
      return;
    }

    const obj = data.objects[0];
    camResults.innerHTML = `
      <h4>Webcam Frame Result</h4>
      <img src="${data.pipeline_stages.stage_6_detected_contours}" style="max-width:100%; border-radius:12px; margin-top:10px;">
      <div class="decision-banner ${obj.reasoning.status === 'RECOGNIZED' ? 'banner-recognized' : 'banner-unknown'} mt-3">
        <div class="decision-status">${obj.reasoning.status}</div>
        <div class="decision-name">${obj.reasoning.recognized_name}</div>
      </div>
      <p class="explanation-p mt-2">${obj.reasoning.explanation}</p>
    `;
  }

  // Initial Setup
  fetchSampleDatasets();
  loadKnowledgeBase();
});
