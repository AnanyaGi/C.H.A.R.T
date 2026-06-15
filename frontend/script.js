// ================================================================
//  VisuAlly — script.js
//  Stateful continuous-feedback loop with per-user log files
// ================================================================

const backendUrl = "http://127.0.0.1:8000";

// ── Global state ─────────────────────────────────────────────────
let userId         = null;
let imageId        = null;
let iteration      = 0;
let currentSummary = "";
let selectedRating = 0;
let currentFontSize = 16;
let availableVoices = [];

// ================================================================
//  INIT
// ================================================================
document.addEventListener("DOMContentLoaded", () => {
  console.log("VisuAlly loaded.");

  populateVoices();
  if (speechSynthesis.onvoiceschanged !== undefined) {
    speechSynthesis.onvoiceschanged = populateVoices;
  }

  // Accessibility bar
  document.getElementById("increaseText").addEventListener("click", () => adjustFontSize(2));
  document.getElementById("decreaseText").addEventListener("click", () => adjustFontSize(-2));
  document.getElementById("resetText").addEventListener("click", () => {
    currentFontSize = 16;
    document.documentElement.style.setProperty("--base-font-size", "16px");
  });
  document.getElementById("contrastBtn").addEventListener("click",
    () => document.body.classList.toggle("high-contrast"));
  document.getElementById("dyslexiaBtn").addEventListener("click",
    () => document.body.classList.toggle("dyslexia-mode"));
  document.getElementById("cursorBtn").addEventListener("click",
    () => document.body.classList.toggle("big-cursor"));

  // Login tabs
  document.getElementById("tabNew").addEventListener("click",       () => switchTab("new"));
  document.getElementById("tabReturning").addEventListener("click", () => switchTab("returning"));

  // Login actions
  document.getElementById("registerBtn").addEventListener("click",  doRegister);
  document.getElementById("returningBtn").addEventListener("click", doReturningLogin);
  document.getElementById("continueBtn").addEventListener("click",  enterApp);
  document.getElementById("copyIdBtn").addEventListener("click",    copyUserId);

  // Allow Enter key on returning user field
  document.getElementById("existingIdInput").addEventListener("keypress", e => {
    if (e.key === "Enter") doReturningLogin();
  });

  // Upload
  document.getElementById("uploadBtn").addEventListener("click", uploadImage);

  // Star rating
  document.querySelectorAll(".star-btn").forEach(btn =>
    btn.addEventListener("click", () => selectRating(parseInt(btn.dataset.value)))
  );
  document.getElementById("submitRatingBtn").addEventListener("click", submitRating);

  // Regenerate
  document.getElementById("regenBtn").addEventListener("click", regenerateDescription);

  // Chat
  document.getElementById("question").addEventListener("keypress", e => {
    if (e.key === "Enter") askQuestion();
  });
  document.getElementById("askBtn").addEventListener("click", askQuestion);
});

// ================================================================
//  CAI DISPLAY HELPERS
// ================================================================
function renderCAIResults(score, iterations, critiqueReport) {
  const bar     = document.getElementById("caiScoreBar");
  const badge   = document.getElementById("caiScoreBadge");
  const iterLbl = document.getElementById("caiIterLabel");

  if (score === undefined || score === null) { bar.classList.add("hidden"); return; }

  const pct       = Math.round(score * 100);
  const satisfied = Math.round(score * 10);
  badge.textContent = `${pct}% (${satisfied}/10)`;
  badge.className   = "cai-badge " +
    (score >= 0.9 ? "high" : score >= 0.7 ? "medium" : "low");
  iterLbl.textContent = `${iterations} CAI iteration${iterations !== 1 ? "s" : ""}`;
  bar.classList.remove("hidden");

  // Build critique table
  const box = document.getElementById("critiqueBox");
  if (!critiqueReport || Object.keys(critiqueReport).length === 0) {
    box.classList.add("hidden"); return;
  }

  const principles = {
    P1:"Visual Faithfulness", P2:"Numeric Grounding",   P3:"Structural Order",
    P4:"Readability",         P5:"No Over-Interpretation", P6:"Accessibility Referencing",
    P7:"Completeness",        P8:"Consistency",          P9:"Uncertainty Handling",
    P10:"Neutral Tone"
  };

  let rows = "";
  for (const [pid, data] of Object.entries(critiqueReport)) {
    const cls    = data.status === "satisfied" ? "status-satisfied" : "status-violated";
    const symbol = data.status === "satisfied" ? "✔" : "✘";
    rows += `<tr>
      <td><strong>${pid}</strong></td>
      <td>${principles[pid] || ""}</td>
      <td class="${cls}">${symbol} ${data.status}</td>
      <td>${data.reason || ""}</td>
    </tr>`;
  }

  box.innerHTML = `
    <h4>Constitutional Critique Report</h4>
    <table class="critique-table">
      <thead><tr><th>ID</th><th>Principle</th><th>Status</th><th>Reason</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  box.classList.remove("hidden");
}

function resetCAIDisplay() {
  document.getElementById("caiScoreBar").classList.add("hidden");
  document.getElementById("critiqueBox").classList.add("hidden");
  document.getElementById("critiqueBox").innerHTML = "";
}

// ================================================================
//  ACCESSIBILITY HELPERS
// ================================================================
function populateVoices() {
  availableVoices = window.speechSynthesis.getVoices();
}

function speakText(text) {
  window.speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  const voice =
    availableVoices.find(v => v.name.includes("Google US English")) ||
    availableVoices.find(v => v.name.includes("Samantha")) ||
    availableVoices.find(v => v.name.includes("Zira")) ||
    availableVoices.find(v => v.lang === "en-US");
  if (voice) utt.voice = voice;
  utt.pitch = 1; utt.rate = 1;
  window.speechSynthesis.speak(utt);
}

function adjustFontSize(delta) {
  currentFontSize = Math.min(32, Math.max(12, currentFontSize + delta));
  document.documentElement.style.setProperty("--base-font-size", `${currentFontSize}px`);
}

// ================================================================
//  LOGIN TABS
// ================================================================
function switchTab(tab) {
  const isNew = tab === "new";
  document.getElementById("tabNew").classList.toggle("active", isNew);
  document.getElementById("tabReturning").classList.toggle("active", !isNew);
  document.getElementById("tabNew").setAttribute("aria-selected", isNew);
  document.getElementById("tabReturning").setAttribute("aria-selected", !isNew);
  document.getElementById("newUserPanel").classList.toggle("hidden", !isNew);
  document.getElementById("returningPanel").classList.toggle("hidden", isNew);
  clearLoginFeedback();
}

function clearLoginFeedback() {
  document.getElementById("loginError").classList.add("hidden");
  document.getElementById("loginError").textContent = "";
  document.getElementById("userIdDisplay").classList.add("hidden");
}

function showLoginError(msg) {
  const el = document.getElementById("loginError");
  el.textContent = msg;
  el.classList.remove("hidden");
  speakText(msg);
}

// ================================================================
//  LOGIN — NEW USER
// ================================================================
async function doRegister() {
  const language    = document.getElementById("langSelect").value;
  const vision_type = document.getElementById("visionSelect").value;
  const familiarity = document.getElementById("familiaritySelect").value;
  const primary_use = document.getElementById("useSelect").value;

  setLoginLoading(true);
  clearLoginFeedback();

  const fd = new FormData();
  fd.append("language",    language);
  fd.append("vision_type", vision_type);
  fd.append("familiarity", familiarity);
  fd.append("primary_use", primary_use);

  try {
    const resp = await fetch(`${backendUrl}/login`, { method: "POST", body: fd });
    const data = await resp.json();

    if (data.error) throw new Error(data.error);

    userId = data.user_id;
    showUserIdDisplay(
      "🎉 Profile created! Your unique User ID is:",
      data
    );
    speakText(`Welcome to VisuAlly! Your user ID has been created. Please save it for future sessions.`);

  } catch (err) {
    showLoginError(err.message || "Registration failed. Please try again.");
  } finally {
    setLoginLoading(false);
  }
}

// ================================================================
//  LOGIN — RETURNING USER
// ================================================================
async function doReturningLogin() {
  const id = document.getElementById("existingIdInput").value.trim();
  if (!id) {
    showLoginError("Please enter your User ID.");
    return;
  }

  setLoginLoading(true);
  clearLoginFeedback();

  const fd = new FormData();
  fd.append("user_id", id);

  try {
    const resp = await fetch(`${backendUrl}/login`, { method: "POST", body: fd });
    const data = await resp.json();

    if (!resp.ok || data.error) throw new Error(data.error || "Sign-in failed.");

    userId = data.user_id;
    showUserIdDisplay(
      ` Welcome back!`,
      data
    );
    speakText(`Welcome back to VisuAlly!`);

  } catch (err) {
    showLoginError(err.message || "Sign-in failed. Please check your User ID.");
  } finally {
    setLoginLoading(false);
  }
}

function showUserIdDisplay(message, profileData) {
  document.getElementById("userIdMessage").textContent = message;
  document.getElementById("userIdText").textContent    = profileData.user_id;

  const profileSummary =
    `${profileData.vision_type} · ${profileData.language} · ${profileData.familiarity}`;
  document.getElementById("userIdDisplay").dataset.profile = profileSummary;
  document.getElementById("userIdDisplay").classList.remove("hidden");
}

function copyUserId() {
  const id = document.getElementById("userIdText").textContent;
  navigator.clipboard.writeText(id).then(() => {
    const btn = document.getElementById("copyIdBtn");
    btn.textContent = " Copied!";
    setTimeout(() => { btn.textContent = " Copy"; }, 2000);
  });
}

function setLoginLoading(on) {
  document.getElementById("loginLoader").classList.toggle("hidden", !on);
  document.getElementById("registerBtn").disabled  = on;
  document.getElementById("returningBtn").disabled = on;
}

// ================================================================
//  ENTER APP  (after ID shown)
// ================================================================
function enterApp() {
  // Show user badge in upload section
  document.getElementById("badgeUserId").textContent = userId;
  const profile = document.getElementById("userIdDisplay").dataset.profile || "";
  document.getElementById("badgeProfile").textContent = profile ? `(${profile})` : "";

  document.getElementById("loginSection").classList.add("hidden");
  document.getElementById("uploadSection").classList.remove("hidden");
  speakText("Please upload an image to begin.");
}

// ================================================================
//  UPLOAD IMAGE
// ================================================================
async function uploadImage() {
  const fileInput = document.getElementById("imageInput");
  const file      = fileInput.files[0];
  const loader    = document.getElementById("loader");

  if (!file) {
    speakText("Please select an image first.");
    alert("Please select an image first!");
    return;
  }

  // Reset UI
  document.getElementById("summarySection").classList.add("hidden");
  document.getElementById("chatSection").classList.add("hidden");
  document.getElementById("description").textContent = "";
  document.getElementById("vlmPromptBox").textContent = "";
  document.getElementById("chatBox").innerHTML = "";
  resetCAIDisplay();
  resetRatingPanel();
  lockRegenControls();
  iteration      = 0;
  currentSummary = "";
  imageId        = null;

  // Preview
  document.getElementById("previewImg").src = URL.createObjectURL(file);
  document.getElementById("imagePreview").classList.remove("hidden");
  loader.classList.remove("hidden");
  speakText("Analysing image. Please wait.");

  const fd = new FormData();
  fd.append("file",    file);
  fd.append("user_id", userId);

  try {
    const resp = await fetch(`${backendUrl}/upload/`, { method: "POST", body: fd });
    if (!resp.ok) throw new Error("Backend error");
    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    currentSummary = data.description;
    imageId        = data.image_id;
    iteration      = 1;

    document.getElementById("description").textContent  = data.description;
    document.getElementById("vlmPromptBox").textContent = data.vlm_prompt || "(not available)";
    renderCAIResults(data.compliance_score, data.cai_iterations, data.critique_report);

    document.getElementById("summarySection").classList.remove("hidden");
    document.getElementById("chatSection").classList.remove("hidden");
    loader.classList.add("hidden");

    speakText("Description generated. Constitutional AI compliance score is " +
      Math.round((data.compliance_score || 0) * 10) + " out of 10. Please rate this description.");
    document.getElementById("description").focus();

  } catch (err) {
    console.error(err);
    loader.classList.add("hidden");
    speakText("Upload failed. Please try again.");
    alert("Upload failed: " + err.message);
  }
}

// ================================================================
//  RATING UI
// ================================================================
function selectRating(value) {
  selectedRating = value;
  document.querySelectorAll(".star-btn").forEach(btn => {
    btn.classList.toggle("active", parseInt(btn.dataset.value) <= value);
  });
  document.getElementById("ratingValue").textContent =
    `${value} star${value > 1 ? "s" : ""} selected`;
  document.getElementById("submitRatingBtn").disabled = false;
}

function resetRatingPanel() {
  selectedRating = 0;
  document.querySelectorAll(".star-btn").forEach(btn => btn.classList.remove("active"));
  document.getElementById("ratingValue").textContent      = "Select a rating above";
  document.getElementById("commentBox").value             = "";
  document.getElementById("submitRatingBtn").disabled     = true;
  document.getElementById("ratingStatus").textContent     = "";
}

// ================================================================
//  SUBMIT RATING  →  unlocks regen controls
// ================================================================
async function submitRating() {
  if (!selectedRating) {
    speakText("Please select a star rating first.");
    return;
  }

  const comment      = document.getElementById("commentBox").value.trim();
  const tone         = document.getElementById("toneSelect").value;
  const summaryType  = document.getElementById("lengthSelect").value;
  const submitBtn    = document.getElementById("submitRatingBtn");
  const ratingStatus = document.getElementById("ratingStatus");

  submitBtn.disabled       = true;
  ratingStatus.textContent = "Saving rating…";

  const fd = new FormData();
  fd.append("user_id",           userId);
  fd.append("image_id",          imageId);
  fd.append("iteration",         iteration);
  fd.append("generated_summary", currentSummary);
  fd.append("rating",            selectedRating);
  fd.append("comment",           comment);
  fd.append("tone",              tone);
  fd.append("summary_type",      summaryType);

  try {
    const resp = await fetch(`${backendUrl}/rate`, { method: "POST", body: fd });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    ratingStatus.textContent =
      ` Saved! Readability — FRE: ${data.fre.toFixed(1)} | FK Grade: ${data.fkgl.toFixed(1)}`;

    document.getElementById("commentBox").value = "";
    unlockRegenControls();
    speakText("Rating saved. You can now refine the description or ask questions.");

  } catch (err) {
    console.error("Rating error:", err);
    ratingStatus.textContent = "Error saving rating. Please try again.";
    submitBtn.disabled = false;
  }
}

// ================================================================
//  REGENERATE  →  re-locks controls until next rating
// ================================================================
async function regenerateDescription() {
  const tone        = document.getElementById("toneSelect").value;
  const summaryType = document.getElementById("lengthSelect").value;
  const regenLoader = document.getElementById("regenLoader");

  lockRegenControls();
  regenLoader.classList.remove("hidden");
  speakText("Regenerating description. Please wait.");

  const fd = new FormData();
  fd.append("user_id",          userId);
  fd.append("image_id",         imageId);
  fd.append("iteration",        iteration + 1);
  fd.append("tone",             tone);
  fd.append("summary_type",     summaryType);
  fd.append("previous_summary", currentSummary);

  try {
    const resp = await fetch(`${backendUrl}/regenerate`, { method: "POST", body: fd });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    currentSummary = data.description;
    iteration     += 1;

    document.getElementById("description").textContent  = data.description;
    document.getElementById("chatBox").innerHTML = ""; 
    document.getElementById("vlmPromptBox").textContent =
      `[Iteration ${iteration} — Rewrite Prompt]\n\nTone: ${tone} | Length: ${summaryType}`;
    renderCAIResults(data.compliance_score, data.cai_iterations, data.critique_report);

    resetRatingPanel();
    regenLoader.classList.add("hidden");
    speakText("New description generated. Constitutional AI compliance score is " +
      Math.round((data.compliance_score || 0) * 10) + " out of 10. Please rate this version.");
    document.getElementById("description").focus();

  } catch (err) {
    console.error("Regenerate error:", err);
    regenLoader.classList.add("hidden");
    speakText("Regeneration failed. Please try again.");
    unlockRegenControls();
  }
}

// ================================================================
//  REGEN LOCK / UNLOCK
// ================================================================
function lockRegenControls() {
  document.getElementById("toneSelect").disabled   = true;
  document.getElementById("lengthSelect").disabled = true;
  document.getElementById("regenBtn").disabled     = true;
}
function unlockRegenControls() {
  document.getElementById("toneSelect").disabled   = false;
  document.getElementById("lengthSelect").disabled = false;
  document.getElementById("regenBtn").disabled     = false;
}

// ================================================================
//  CHAT Q&A
// ================================================================
async function askQuestion() {
  const qInput  = document.getElementById("question");
  const question = qInput.value.trim();
  if (!question) return;

  const chatBox = document.getElementById("chatBox");
  const userMsg = document.createElement("div");
  userMsg.className   = "message user-msg";
  userMsg.textContent = question;
  chatBox.appendChild(userMsg);
  qInput.value = "";

  const fd = new FormData();
  fd.append("user_id", userId);       
  fd.append("image_id", imageId); 
  fd.append("question", question);
  fd.append("context",  currentSummary);

  try {
    const resp = await fetch(`${backendUrl}/ask/`, { method: "POST", body: fd });
    const data = await resp.json();
    const botMsg = document.createElement("div");
    botMsg.className   = "message bot-msg";
    botMsg.textContent = data.answer;
    chatBox.appendChild(botMsg);
    chatBox.scrollTop  = chatBox.scrollHeight;
    speakText(data.answer);
  } catch {
    const errMsg = document.createElement("div");
    errMsg.className   = "message bot-msg";
    errMsg.textContent = "Error connecting to AI.";
    chatBox.appendChild(errMsg);
    speakText("Error connecting to AI.");
  }
}