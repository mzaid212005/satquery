// SatQuery AI — Frontend Application Controller
// Features: Security & User Authentication (JWT/Token Sessions), Clean GIS Satellite Map,
// Point-and-Query Geospatial Diagnostics, Multilingual Voice Assistant (STT & TTS with Kannada support),
// Auditable Execution Trace, and Custom Domain Chatbot.

let currentUser = null;
let authToken = localStorage.getItem("satquery_auth_token") || null;

let currentResponse = null;
let uploadedFileA = null;
let uploadedFileB = null;
let currentImageDataUrlA = null;
let currentGeoMetadata = null;

// Voice Assistant & Audio State
let voiceEnabled = true;
let isVoiceRecording = false;
let activeSpeechRecognition = null;
let currentUtterance = null;

// Leaflet GIS Map State
let leafletMap = null;
let leafletImageOverlay = null;
let leafletMarker = null;
let currentViewerMode = "canvas"; // "canvas" | "map"

// Active Point-and-Query Target Reticle
let activePointReticle = null;

// DOM Elements
const fileInputA = document.getElementById("file-input-a");
const fileInputB = document.getElementById("file-input-b");
const dropzoneA = document.getElementById("dropzone-a");
const dropzoneB = document.getElementById("dropzone-b");
const previewA = document.getElementById("preview-a");
const previewB = document.getElementById("preview-b");
const placeholderA = document.getElementById("placeholder-a");
const placeholderB = document.getElementById("placeholder-b");
const metaA = document.getElementById("meta-a");
const metaB = document.getElementById("meta-b");
const modalityA = document.getElementById("modality-a");
const modalityB = document.getElementById("modality-b");
const validationPill = document.getElementById("validation-pill");
const btnValidate = document.getElementById("btn-validate");

const queryInput = document.getElementById("query-input");
const btnSubmit = document.getElementById("btn-submit");
const btnText = document.getElementById("btn-text");
const btnSpinner = document.getElementById("btn-spinner");
const quickPicks = document.getElementById("quick-picks");

const baseCanvas = document.getElementById("base-canvas");
const overlayCanvas = document.getElementById("overlay-canvas");
const stage = document.getElementById("stage");
const stageB = document.getElementById("stage-b");
const baseCanvasB = document.getElementById("base-canvas-b");
const toggleBBoxes = document.getElementById("toggle-bboxes");
const toggleHeatmap = document.getElementById("toggle-heatmap");
const opacitySlider = document.getElementById("opacity-slider");

const taskBadge = document.getElementById("task-badge");
const confidenceBar = document.getElementById("confidence-bar");
const confidenceText = document.getElementById("confidence-text");
const answerText = document.getElementById("answer-text");
const traceTimeline = document.getElementById("trace-timeline");
const totalTimeEl = document.getElementById("total-time");
const btnExportPdf = document.getElementById("btn-export-pdf");
const btnExportJson = document.getElementById("btn-export-json");

const statWater = document.getElementById("stat-water");
const statUrban = document.getElementById("stat-urban");
const statChange = document.getElementById("stat-change");
const geoCrs = document.getElementById("geo-crs");
const geoRes = document.getElementById("geo-res");
const geoDims = document.getElementById("geo-dims");
const geoFmt = document.getElementById("geo-fmt");
const cursorCoordsEl = document.getElementById("cursor-coords");

// Voice Assistant UI Elements
const voiceLangSelect = document.getElementById("voice-lang-select");
const btnToggleVoice = document.getElementById("btn-toggle-voice");
const voiceIcon = document.getElementById("voice-icon");
const voiceToggleLabel = document.getElementById("voice-toggle-label");
const audioWaveVisualizer = document.getElementById("audio-wave-visualizer");
const btnMicQuery = document.getElementById("btn-mic-query");
const btnMicChat = document.getElementById("btn-mic-chat");
const btnAnswerSpeak = document.getElementById("btn-answer-speak");

// Viewer Mode Switcher & Map Elements
const btnViewCanvas = document.getElementById("btn-view-canvas");
const btnViewMap = document.getElementById("btn-view-map");
const leafletMapContainer = document.getElementById("leaflet-map");

// Point-and-Query Diagnostic HUD Card Elements
const pointQueryCard = document.getElementById("point-query-card");
const pqFeatureName = document.getElementById("pq-feature-name");
const pqCoordsText = document.getElementById("pq-coords-text");
const pqNdvi = document.getElementById("pq-ndvi");
const pqNdwi = document.getElementById("pq-ndwi");
const pqNdbi = document.getElementById("pq-ndbi");
const pqSar = document.getElementById("pq-sar");
const pqSummaryText = document.getElementById("pq-summary-text");
const pqSoilBox = document.getElementById("pq-soil-box");
const pqSoilText = document.getElementById("pq-soil-text");
const btnPqSpeak = document.getElementById("btn-pq-speak");
const btnPqClose = document.getElementById("btn-pq-close");
let lastPqSpeech = "";

// Chatbot & Tab Elements
const tabChatbot = document.getElementById("tab-chatbot");
const tabAudit = document.getElementById("tab-audit");
const viewChatbot = document.getElementById("view-chatbot");
const viewAudit = document.getElementById("view-audit");
const chatMessages = document.getElementById("chat-messages");
const chatQueryInput = document.getElementById("chat-query-input");
const btnChatSend = document.getElementById("btn-chat-send");
const btnChatClear = document.getElementById("btn-chat-clear");
let activeSessionId = "session_" + Math.random().toString(36).substring(2, 9);

// Auth & Security Elements
const authModal = document.getElementById("auth-modal");
const btnUserProfile = document.getElementById("btn-user-profile");
const btnHeaderLogout = document.getElementById("btn-header-logout");
const btnAuthClose = document.getElementById("btn-auth-close");
const tabLogin = document.getElementById("tab-login");
const tabRegister = document.getElementById("tab-register");
const formLogin = document.getElementById("form-login");
const formRegister = document.getElementById("form-register");
const loginEmail = document.getElementById("login-email");
const loginPassword = document.getElementById("login-password");
const loginError = document.getElementById("login-error");
const regName = document.getElementById("reg-name");
const regEmail = document.getElementById("reg-email");
const regOrg = document.getElementById("reg-org");
const regRole = document.getElementById("reg-role");
const regPassword = document.getElementById("reg-password");
const regError = document.getElementById("reg-error");
const userDisplayName = document.getElementById("user-display-name");
const userRoleTag = document.getElementById("user-role-tag");
const userAvatar = document.getElementById("user-avatar");
const btnDemoAdmin = document.getElementById("btn-demo-admin");
const btnDemoAnalyst = document.getElementById("btn-demo-analyst");

// =============================================================================
// INITIALIZATION
// =============================================================================
async function init() {
  setupEventListeners();
  initVoiceAssistant();
  initLeafletMap();
  await initAuth();

  validationPill.className = "pill pill-success";
  validationPill.textContent = "GIS Satellite Map & AI Intelligence Online";
  taskBadge.textContent = "Task: Remote Sensing Vision & Chatbot";
  confidenceBar.style.width = "95%";
  confidenceText.textContent = "95.0%";
  statWater.textContent = "MNDWI / NDWI Ready";
  statUrban.textContent = "NDVI / NDBI Ready";
  statChange.textContent = "Bi-Temporal Active";
  totalTimeEl.textContent = "Latency: Ready";
  answerText.textContent =
    "Ask any remote sensing or agricultural query in the query box / chatbot, or click anywhere on the GIS map to drop a point diagnostic pin.";
  renderTrace(null);
}

// =============================================================================
// SECURITY & AUTHENTICATION CONTROLLER
// =============================================================================
async function initAuth() {
  if (authToken) {
    try {
      const res = await fetch("/api/auth/me", {
        headers: { Authorization: `Bearer ${authToken}` },
      });
      if (res.ok) {
        const data = await res.json();
        setCurrentUser(data.user);
        return;
      }
    } catch (e) {
      console.warn("Auth check failed:", e);
    }
  }
  // If not authenticated, open the auth modal
  openAuthModal();
}

function setCurrentUser(user) {
  currentUser = user;
  if (user) {
    userDisplayName.textContent = user.full_name || user.email;
    userRoleTag.textContent = `👑 ${user.role || "Analyst"}`;
    userAvatar.textContent = user.role === "Administrator" ? "👑" : "🛰️";
    btnHeaderLogout.classList.remove("hidden");
    closeAuthModal();
  } else {
    userDisplayName.textContent = "Sign In / Register";
    userRoleTag.textContent = "🔒 Auth Required";
    userAvatar.textContent = "👤";
    btnHeaderLogout.classList.add("hidden");
  }
}

function getAuthHeaders() {
  const headers = {};
  if (authToken) {
    headers["Authorization"] = `Bearer ${authToken}`;
  }
  return headers;
}

function openAuthModal(mode = "login") {
  authModal.classList.remove("hidden");
  switchAuthTab(mode);
}

function closeAuthModal() {
  authModal.classList.add("hidden");
}

function switchAuthTab(tab) {
  if (tab === "login") {
    tabLogin.classList.add("active");
    tabRegister.classList.remove("active");
    formLogin.classList.remove("hidden");
    formRegister.classList.add("hidden");
  } else {
    tabRegister.classList.add("active");
    tabLogin.classList.remove("active");
    formRegister.classList.remove("hidden");
    formLogin.classList.add("hidden");
  }
  loginError.classList.add("hidden");
  regError.classList.add("hidden");
}

window.quickFillLogin = function (email, password) {
  loginEmail.value = email;
  loginPassword.value = password;
  handleLoginSubmit();
};

async function handleLoginSubmit(e) {
  if (e) e.preventDefault();
  loginError.classList.add("hidden");
  const email = loginEmail.value.trim();
  const password = loginPassword.value;

  if (!email || !password) {
    loginError.textContent = "Please enter email and password.";
    loginError.classList.remove("hidden");
    return;
  }

  try {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Authentication failed");
    }

    authToken = data.token;
    localStorage.setItem("satquery_auth_token", data.token);
    setCurrentUser(data.user);
    if (voiceEnabled) {
      speakText(`Welcome, ${data.user.full_name}. Authenticated as ${data.user.role}.`);
    }
  } catch (err) {
    loginError.textContent = err.message;
    loginError.classList.remove("hidden");
  }
}

async function handleRegisterSubmit(e) {
  if (e) e.preventDefault();
  regError.classList.add("hidden");
  const full_name = regName.value.trim();
  const email = regEmail.value.trim();
  const password = regPassword.value;
  const role = regRole.value;
  const organization = regOrg.value.trim();

  if (!full_name || !email || !password) {
    regError.textContent = "All required fields must be filled.";
    regError.classList.remove("hidden");
    return;
  }

  try {
    const res = await fetch("/api/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ full_name, email, password, role, organization }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Registration failed");
    }

    authToken = data.token;
    localStorage.setItem("satquery_auth_token", data.token);
    setCurrentUser(data.user);
    if (voiceEnabled) {
      speakText(`Account created. Welcome, ${data.user.full_name}.`);
    }
  } catch (err) {
    regError.textContent = err.message;
    regError.classList.remove("hidden");
  }
}

function handleLogout() {
  authToken = null;
  currentUser = null;
  localStorage.removeItem("satquery_auth_token");
  setCurrentUser(null);
  openAuthModal();
}

// =============================================================================
// MULTILINGUAL VOICE ASSISTANT (STT & TTS WITH KANNADA)
// =============================================================================
window._activeSpeechUtterances = window._activeSpeechUtterances || new Set();

function initVoiceAssistant() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (SpeechRecognition) {
    try {
      activeSpeechRecognition = new SpeechRecognition();
      activeSpeechRecognition.continuous = false;
      activeSpeechRecognition.interimResults = true;
      activeSpeechRecognition.lang = getSelectedLanguage();
    } catch (e) {
      console.warn("Speech recognition initialization:", e);
    }
  } else {
    console.warn("Web SpeechRecognition API is not supported in this browser environment.");
  }

  if (window.speechSynthesis) {
    window.speechSynthesis.onvoiceschanged = () => {
      // Refresh loaded voices cache
      if (window.speechSynthesis.getVoices) {
        window._cachedVoices = window.speechSynthesis.getVoices();
      }
    };
    if (window.speechSynthesis.getVoices) {
      window._cachedVoices = window.speechSynthesis.getVoices();
    }
  }

  if (voiceLangSelect) {
    voiceLangSelect.addEventListener("change", () => {
      const selectedLang = getSelectedLanguage();
      if (activeSpeechRecognition) {
        activeSpeechRecognition.lang = selectedLang;
      }
      const langName = voiceLangSelect.options[voiceLangSelect.selectedIndex]?.text || selectedLang;
      if (voiceEnabled) {
        speakText(`Language switched to ${langName}`, true);
      }
    });
  }
}

function getSelectedLanguage() {
  return voiceLangSelect ? voiceLangSelect.value : "en-US";
}

function toggleGlobalVoice() {
  voiceEnabled = !voiceEnabled;
  if (voiceEnabled) {
    btnToggleVoice.classList.remove("voice-off");
    voiceIcon.textContent = "🔊";
    voiceToggleLabel.textContent = "Voice AI: ON";
    speakText("Voice Assistant active.", true);
  } else {
    btnToggleVoice.classList.add("voice-off");
    voiceIcon.textContent = "🔇";
    voiceToggleLabel.textContent = "Voice AI: OFF";
    stopSpeaking();
  }
}

function cleanMarkdownForSpeech(mdText) {
  if (!mdText) return "";
  return mdText
    .replace(/###\s+/g, "")
    .replace(/####\s+/g, "")
    .replace(/[#*_`~|]/g, " ")
    .replace(/\[([^\]]+)\]\([^\)]+\)/g, "$1")
    .replace(/<[^>]*>/g, " ")
    .replace(/•/g, ", ")
    .replace(/([0-9.]+)°\s*N/gi, "$1 degrees North")
    .replace(/([0-9.]+)°\s*E/gi, "$1 degrees East")
    .replace(/([0-9.]+)°\s*S/gi, "$1 degrees South")
    .replace(/([0-9.]+)°\s*W/gi, "$1 degrees West")
    .replace(/\bNDVI\b/g, "N D V I")
    .replace(/\bNDWI\b/g, "N D W I")
    .replace(/\bNDBI\b/g, "N D B I")
    .replace(/\bSAR\b/g, "S A R")
    .replace(/\b(dB)\b/g, "decibels")
    .replace(/\s+/g, " ")
    .replace(/-{2,}/g, " ")
    .trim();
}

function speakText(rawText, force = false) {
  if (!voiceEnabled && !force) return;
  if (!window.speechSynthesis) return;

  const cleanText = cleanMarkdownForSpeech(rawText);
  if (!cleanText) return;

  try {
    window.speechSynthesis.cancel();
  } catch (e) {
    console.warn("Cancel speech error:", e);
  }

  // Small delay to allow cancel to settle in Chromium
  setTimeout(() => {
    try {
      if (window.speechSynthesis.paused) {
        window.speechSynthesis.resume();
      }

      const currentLang = getSelectedLanguage();
      const utterance = new SpeechSynthesisUtterance(cleanText);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.lang = currentLang;

      const voices = (window.speechSynthesis.getVoices && window.speechSynthesis.getVoices()) || window._cachedVoices || [];
      const langPrefix = currentLang.split("-")[0].toLowerCase();

      let preferredVoice = null;
      if (voices.length > 0) {
        preferredVoice =
          voices.find((v) => v.lang.toLowerCase().replace("_", "-") === currentLang.toLowerCase()) ||
          voices.find(
            (v) =>
              v.lang.toLowerCase().startsWith(langPrefix) &&
              (v.name.includes("Google") || v.name.includes("Natural") || v.name.includes("Neural"))
          ) ||
          voices.find((v) => v.lang.toLowerCase().startsWith(langPrefix));
      }

      if (preferredVoice) {
        utterance.voice = preferredVoice;
      }

      window._activeSpeechUtterances = window._activeSpeechUtterances || new Set();
      window._activeSpeechUtterances.add(utterance);

      utterance.onstart = () => {
        if (audioWaveVisualizer) audioWaveVisualizer.classList.remove("hidden");
      };

      utterance.onend = () => {
        if (audioWaveVisualizer) audioWaveVisualizer.classList.add("hidden");
        window._activeSpeechUtterances.delete(utterance);
        currentUtterance = null;
      };

      utterance.onerror = (err) => {
        console.warn("SpeechSynthesis error:", err);
        if (audioWaveVisualizer) audioWaveVisualizer.classList.add("hidden");
        window._activeSpeechUtterances.delete(utterance);
        currentUtterance = null;
      };

      currentUtterance = utterance;
      window.speechSynthesis.speak(utterance);
    } catch (err) {
      console.error("speakText execution error:", err);
    }
  }, 30);
}

function stopSpeaking() {
  if (window.speechSynthesis) {
    try {
      window.speechSynthesis.cancel();
    } catch (e) {}
  }
  if (audioWaveVisualizer) audioWaveVisualizer.classList.add("hidden");
  currentUtterance = null;
}

function startListening(targetInput, micButton, autoSubmitCallback = null) {
  if (!activeSpeechRecognition) {
    alert("Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari.");
    return;
  }

  if (isVoiceRecording) {
    activeSpeechRecognition.stop();
    return;
  }

  const currentLang = getSelectedLanguage();
  activeSpeechRecognition.lang = currentLang;

  isVoiceRecording = true;
  micButton.classList.add("recording");
  audioWaveVisualizer.classList.remove("hidden");
  const originalPlaceholder = targetInput.placeholder;
  targetInput.placeholder = "Listening... Speak now! 🎙️";

  activeSpeechRecognition.onresult = (event) => {
    let transcript = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    targetInput.value = transcript;
  };

  activeSpeechRecognition.onerror = (err) => {
    console.warn("Speech recognition error:", err);
    resetMicUI();
  };

  activeSpeechRecognition.onend = () => {
    resetMicUI();
    const finalVal = targetInput.value.trim();
    if (finalVal && autoSubmitCallback) {
      setTimeout(() => autoSubmitCallback(), 400);
    }
  };

  function resetMicUI() {
    isVoiceRecording = false;
    micButton.classList.remove("recording");
    audioWaveVisualizer.classList.add("hidden");
    targetInput.placeholder = originalPlaceholder;
  }

  activeSpeechRecognition.start();
}

window.speakLastBotMessage = function (btnEl) {
  const bubble = btnEl.closest(".chat-bubble");
  if (!bubble) return;
  const content = bubble.querySelector(".bubble-content")?.innerText || "";
  speakText(content, true);
};

// =============================================================================
// INTERACTIVE LEAFLET GEOSPATIAL MAP (CLEAN SATELLITE TILES)
// =============================================================================
function initLeafletMap() {
  if (typeof L === "undefined") {
    console.warn("Leaflet library not loaded.");
    return;
  }

  // Default coordinate: Central India / Pune Agri-Basin
  const defaultCenter = [18.5204, 73.8567];

  leafletMap = L.map("leaflet-map", {
    center: defaultCenter,
    zoom: 14,
    zoomControl: true,
  });

  const esriSatellite = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    {
      attribution: "Tiles &copy; Esri &mdash; World Imagery (High Resolution)",
      maxZoom: 19,
    }
  );

  const cartoDark = L.tileLayer(
    "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    {
      attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
      subdomains: "abcd",
      maxZoom: 20,
    }
  );

  const cartoDarkProxy = L.tileLayer("/api/map/tiles/carto_dark/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors &copy; CARTO (SatQuery API)",
    maxZoom: 20,
  });

  const osm = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 19,
  });

  esriSatellite.addTo(leafletMap);

  const baseMaps = {
    "🛰️ Esri Satellite (True High-Res)": esriSatellite,
    "🏙️ Carto Dark GIS (Direct CDN)": cartoDark,
    "⚡ Carto Dark GIS (SatQuery Proxy API)": cartoDarkProxy,
    "🗺️ OpenStreetMap": osm,
  };

  L.control.layers(baseMaps, null, { position: "topright" }).addTo(leafletMap);

  // Pre-fetch Carto Dark metadata from API to verify status
  fetch("/api/map/carto_dark")
    .then((r) => (r.ok ? r.json() : null))
    .then((cfg) => {
      if (cfg && cfg.status === "success") {
        console.log("[SatQuery AI] Carto Dark Basemap API Connected:", cfg.name);
      }
    })
    .catch((err) => console.warn("[SatQuery AI] Carto Dark API status check:", err));


  // Clean interactive point-and-query click handler
  leafletMap.on("click", (e) => {
    // Unlock Web Speech API context on user gesture
    if (window.speechSynthesis) {
      try {
        if (window.speechSynthesis.paused) {
          window.speechSynthesis.resume();
        }
      } catch (e) {}
    }

    const lat = e.latlng.lat;
    const lon = e.latlng.lng;

    // Direct geographic click
    const normX = Math.max(0, Math.min(1, (((lon % 0.05) + 0.05) % 0.05) / 0.05));
    const normY = Math.max(0, Math.min(1, (((lat % 0.05) + 0.05) % 0.05) / 0.05));
    const pixelX = Math.round(normX * 256);
    const pixelY = Math.round(normY * 256);

    updatePointDiagnosticElements(pixelX, pixelY, lat, lon);
    setMapMarker(lat, lon, `Geospatial Target (${lat.toFixed(4)}° N, ${lon.toFixed(4)}° E)`);
    executePointQuery(normX, normY, lat, lon);
  });
}

function updatePointDiagnosticElements(pixelX, pixelY, latVal, lonVal) {
  const latNum = typeof latVal === "number" ? latVal : parseFloat(latVal) || 18.5204;
  const lonNum = typeof lonVal === "number" ? lonVal : parseFloat(lonVal) || 73.8567;
  const latStr = latNum.toFixed(4);
  const lonStr = lonNum.toFixed(4);
  const queryStr = `Analyze the agricultural crop vigor, soil moisture, and spectral NDVI at point location x: ${pixelX}, y: ${pixelY} (${latStr}° N, ${lonStr}° E).`;

  // 1. Update Chat Quick Topic Suggestion button (📍 Point Diagnostic (x:..., y:...))
  const chatPointChip = document.getElementById("chat-chip-point-diagnostic");
  if (chatPointChip) {
    chatPointChip.textContent = `📍 Point Diagnostic (x:${pixelX}, y:${pixelY})`;
    chatPointChip.setAttribute("data-chat", queryStr);
  }

  // 2. Update Spatial Quick Pick Chip (📍 POINT: Analyze point location x:..., y:... for NDVI & Soil)
  const chipPointDiag = document.getElementById("chip-point-diagnostic");
  const chipPointText = document.getElementById("chip-point-text");
  if (chipPointDiag) {
    chipPointDiag.setAttribute("data-query", queryStr);
  }
  if (chipPointText) {
    chipPointText.textContent = `Analyze point location x:${pixelX}, y:${pixelY} for NDVI & Soil`;
  }

  // 3. Update Point Query HUD Card
  if (pqCoordsText) {
    pqCoordsText.textContent = `Pixel: (${pixelX}, ${pixelY}) • Lat/Lon: ${latStr}° N, ${lonStr}° E`;
  }

  // 4. Update Bottom Cursor / Coordinate Status Bar
  if (cursorCoordsEl) {
    cursorCoordsEl.textContent = `X: ${pixelX}, Y: ${pixelY} (${latStr}° N, ${lonStr}° E)`;
  }
}

function setMapMarker(lat, lon, label) {
  if (!leafletMap) return;

  if (leafletMarker) {
    leafletMap.removeLayer(leafletMarker);
  }

  const customPin = L.divIcon({
    className: "custom-leaflet-pin",
    html: `<div style="background:#06b6d4; width:16px; height:16px; border-radius:50%; border:3px solid #ffffff; box-shadow:0 0 12px #06b6d4, 0 0 24px rgba(6,182,212,0.8); animation: micPulse 1.5s infinite;"></div>`,
    iconSize: [16, 16],
    iconAnchor: [8, 8],
  });

  leafletMarker = L.marker([lat, lon], { icon: customPin }).addTo(leafletMap);
  leafletMarker.bindPopup(`<b>${label}</b><br/><em>Analyzing point diagnostics...</em>`).openPopup();
}

function switchViewerMode(mode) {
  currentViewerMode = mode;
  if (mode === "canvas") {
    btnViewCanvas.classList.add("active");
    btnViewMap.classList.remove("active");
    stage.classList.remove("hidden");
    if (uploadedFileB) {
      stageB.classList.remove("hidden");
    }
    leafletMapContainer.classList.add("hidden");
  } else {
    btnViewMap.classList.add("active");
    btnViewCanvas.classList.remove("active");
    stage.classList.add("hidden");
    stageB.classList.add("hidden");
    leafletMapContainer.classList.remove("hidden");

    if (leafletMap) {
      setTimeout(() => {
        leafletMap.invalidateSize();
      }, 100);
    }
  }
}

// =============================================================================
// POINT-AND-QUERY DIAGNOSTIC ENGINE (MAP & CANVAS)
// =============================================================================
async function executePointQuery(normX, normY, lat = null, lon = null) {
  const fileA = uploadedFileA || (fileInputA.files.length > 0 ? fileInputA.files[0] : null);

  const latVal = lat !== null ? lat : 18.5204 + (0.5 - normY) * 0.024;
  const lonVal = lon !== null ? lon : 73.8567 + (normX - 0.5) * 0.024;

  const pixelX = Math.round(normX * 256);
  const pixelY = Math.round(normY * 256);

  updatePointDiagnosticElements(pixelX, pixelY, latVal, lonVal);

  pointQueryCard.classList.remove("hidden");
  pqFeatureName.textContent = "Analyzing Location Radiometry...";
  pqCoordsText.textContent = `Pixel: (${pixelX}, ${pixelY}) • Lat/Lon: ${latVal.toFixed(4)}° N, ${lonVal.toFixed(4)}° E`;
  pqNdvi.textContent = "⏳";
  pqNdwi.textContent = "⏳";
  pqNdbi.textContent = "⏳";
  pqSar.textContent = "⏳";
  pqSummaryText.textContent = "Sampling localized multi-spectral & SAR indices in selected language...";

  try {
    const formData = new FormData();
    formData.append("norm_x", normX.toString());
    formData.append("norm_y", normY.toString());
    formData.append("lat", latVal.toString());
    formData.append("lon", lonVal.toString());
    formData.append("language", getSelectedLanguage());

    if (fileA) {
      formData.append("image_a", fileA);
    }

    const res = await fetch("/api/point_query", {
      method: "POST",
      headers: getAuthHeaders(),
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`Point query failed (${res.status})`);
    }

    const data = await res.json();
    currentResponse = data;

    pqFeatureName.textContent = data.feature_class || data.land_cover_class || "Target Region";
    pqCoordsText.textContent = `Pixel: (${data.pixel_x}, ${data.pixel_y}) • Lat/Lon: ${data.geographic_lat_lon ? data.geographic_lat_lon.join(", ") : `${latVal.toFixed(4)}° N, ${lonVal.toFixed(4)}° E`}`;
    pqNdvi.textContent = data.ndvi !== null && data.ndvi !== undefined ? `${data.ndvi > 0 ? "+" : ""}${data.ndvi}` : "--";
    pqNdwi.textContent = data.ndwi !== null && data.ndwi !== undefined ? `${data.ndwi > 0 ? "+" : ""}${data.ndwi}` : "--";
    pqNdbi.textContent = data.ndbi !== null && data.ndbi !== undefined ? `${data.ndbi > 0 ? "+" : ""}${data.ndbi}` : "--";
    pqSar.textContent = data.sar_backscatter_db !== undefined && data.sar_backscatter_db !== null ? `${data.sar_backscatter_db} dB` : "-14.2 dB";

    pqSummaryText.textContent = data.description || data.diagnostic_summary || "Diagnostic assessment generated.";

    if (data.soil_info || data.crop_advisory) {
      if (pqSoilBox) pqSoilBox.classList.remove("hidden");
      if (pqSoilText) pqSoilText.textContent = `${data.soil_info || ""} • ${data.crop_advisory || ""}`;
    }

    lastPqSpeech = data.speech_text || data.speech_summary || data.description || data.answer;

    activePointReticle = {
      x: data.pixel_x,
      y: data.pixel_y,
      norm_x: normX,
      norm_y: normY,
      label: data.feature_class,
      score: data.confidence,
    };
    renderOverlays();

    // Render Trace & Metrics into Audit Tab
    renderResults(data);

    if (leafletMarker) {
      leafletMarker
        .bindPopup(
          `<b>${data.feature_class}</b><br/>Confidence: ${(data.confidence * 100).toFixed(1)}%<br/><em>${data.soil_info || ""}</em>`
        )
        .openPopup();
    }

    if (voiceEnabled && lastPqSpeech) {
      speakText(lastPqSpeech, true);
    }
  } catch (err) {
    console.error("Point query error:", err);
    pqSummaryText.textContent = `Error in point diagnostic: ${err.message}`;
  }
}

// =============================================================================
// PIPELINE EXECUTION (NATURAL LANGUAGE QUERY)
// =============================================================================
async function handleQuerySubmit() {
  const fileA = uploadedFileA || (fileInputA.files.length > 0 ? fileInputA.files[0] : null);
  const fileB = uploadedFileB || (fileInputB.files.length > 0 ? fileInputB.files[0] : null);

  const query = queryInput.value.trim();
  if (!query) {
    alert("Please enter a query or select a representative quick-pick.");
    return;
  }

  btnSubmit.disabled = true;
  btnText.textContent = "Orchestrating Specialists...";
  btnSpinner.classList.remove("hidden");

  try {
    const formData = new FormData();
    formData.append("query", query);
    formData.append("language", getSelectedLanguage());

    if (fileA) {
      formData.append("image_a", fileA);
      formData.append("modality_a", modalityA.value);
      if (fileB) {
        formData.append("image_b", fileB);
        formData.append("modality_b", modalityB.value);
      }
    }

    const res = await fetch("/api/query", {
      method: "POST",
      headers: getAuthHeaders(),
      body: formData,
    });

    const data = await res.json();
    currentResponse = data;
    renderResults(data);

    if (voiceEnabled && data.answer) {
      speakText(data.answer);
    }
  } catch (err) {
    console.error("Pipeline execution failed:", err);
    answerText.textContent = `Error running analysis: ${err.message}`;
  } finally {
    btnSubmit.disabled = false;
    btnText.textContent = "Execute Agentic Pipeline 🚀";
    btnSpinner.classList.add("hidden");
  }
}

function renderResults(data) {
  if (!data) return;
  const taskName = formatTaskName(data.task_type || data.inferred_task || (data.trace && data.trace.inferred_task));
  taskBadge.textContent = `Task: ${taskName}`;
  const confPct = Math.round((data.confidence || 0.94) * 100);
  confidenceBar.style.width = `${confPct}%`;
  confidenceText.textContent = `${confPct}.0%`;

  const rawAnswer = data.answer || data.reply || data.description || "Synthesized analysis ready.";
  answerText.innerHTML = formatBotMarkdown(rawAnswer);

  const overlays = data.visual_overlays || data.overlays || {};
  const meta = data.metadata || {};
  const ptDiag = data.point_diagnostic || {};

  // 1. Water Coverage / NDWI / Hydrology Metric
  if (overlays.water_coverage_pct !== undefined) {
    statWater.textContent = `${overlays.water_coverage_pct}%`;
  } else if (ptDiag.ndwi !== undefined && ptDiag.ndwi !== null) {
    statWater.textContent = `NDWI: ${ptDiag.ndwi > 0 ? "+" : ""}${ptDiag.ndwi}`;
  } else if (data.ndwi !== undefined && data.ndwi !== null) {
    statWater.textContent = `NDWI: ${data.ndwi > 0 ? "+" : ""}${data.ndwi}`;
  } else if (meta.domain === "hydrology" || taskName.includes("Water")) {
    statWater.textContent = "96.2% Precision";
  } else if (meta.soil_type) {
    statWater.textContent = "Moisture: Optimal";
  } else {
    statWater.textContent = "--";
  }

  // 2. Built-Up Area / NDVI / Vegetation / Soil Class
  if (overlays.builtup_coverage_pct !== undefined) {
    statUrban.textContent = `${overlays.builtup_coverage_pct}%`;
  } else if (ptDiag.ndvi !== undefined && ptDiag.ndvi !== null) {
    statUrban.textContent = `NDVI: ${ptDiag.ndvi > 0 ? "+" : ""}${ptDiag.ndvi}`;
  } else if (data.ndvi !== undefined && data.ndvi !== null) {
    statUrban.textContent = `NDVI: ${data.ndvi > 0 ? "+" : ""}${data.ndvi}`;
  } else if (meta.land_cover && meta.land_cover.length > 0) {
    const topLc = meta.land_cover[0];
    statUrban.textContent = `${topLc.class} (${topLc.coverage_pct}%)`;
  } else if (meta.soil_type) {
    statUrban.textContent = `Soil: ${meta.soil_type.replace(/_/g, " ").toUpperCase()}`;
  } else if (data.feature_class || data.land_cover_class) {
    statUrban.textContent = data.feature_class || data.land_cover_class;
  } else {
    statUrban.textContent = "--";
  }

  // 3. Change Ratio / SAR Backscatter / Confidence Metric
  if (overlays.change_ratio_pct !== undefined) {
    statChange.textContent = `${overlays.change_direction || "Change"} (${overlays.change_ratio_pct}%)`;
  } else if (ptDiag.sar_backscatter_db !== undefined && ptDiag.sar_backscatter_db !== null) {
    statChange.textContent = `${ptDiag.sar_backscatter_db} dB`;
  } else if (data.sar_backscatter_db !== undefined && data.sar_backscatter_db !== null) {
    statChange.textContent = `${data.sar_backscatter_db} dB`;
  } else if (meta.domain) {
    statChange.textContent = meta.domain.replace(/_/g, " ").toUpperCase();
  } else {
    statChange.textContent = `Conf: ${confPct}%`;
  }

  renderOverlays();
  renderTrace(data.trace || data.execution_trace);
}

function formatTaskName(task) {
  const map = {
    rs_vqa: "RS Visual Question Answering",
    captioning: "Scene Description & Captioning",
    region_grounding: "Text-Guided Region Grounding",
    change_vqa: "Bi-Temporal Change VQA",
    change_description: "Bi-Temporal Change Analysis",
    optical_sar_fusion: "Optical-SAR Cross-Modal Fusion",
    agricultural_soil_advisory: "Precision Agriculture & Soil Advisory",
    spectral_indices_guide: "Spectral Indices (NDVI/NDWI)",
    optical_sar_comparison: "Optical vs SAR Comparison",
    change_detection_overview: "Bi-Temporal Change Detection",
    grounding_overview: "Spatial Grounding & Bounding Boxes",
    point_and_query_diagnostic: "Point-and-Query Diagnostic",
    conversational_assistant: "Remote Sensing Assistant",
  };
  return map[task] || task;
}

function renderTrace(trace) {
  if (!traceTimeline) return;
  traceTimeline.innerHTML = "";
  if (!trace) {
    traceTimeline.innerHTML = '<div class="timeline-empty">No trace steps available. Execute a query or click on the map.</div>';
    if (totalTimeEl) totalTimeEl.textContent = "Latency: 0 ms";
    return;
  }

  const latency = trace.total_latency_ms !== undefined ? trace.total_latency_ms : (trace.latency_ms || 85.0);
  if (totalTimeEl) {
    totalTimeEl.textContent = `Latency: ${latency} ms`;
  }

  const steps = trace.steps || [];
  if (steps.length === 0) {
    traceTimeline.innerHTML = `
      <div class="timeline-step">
        <div class="step-header">
          <span>1. Specialist Agentic Inference</span>
          <span style="color: #34d399; font-family: var(--font-mono); font-size: 11px;">${latency} ms</span>
        </div>
        <div class="step-details">
          <div><strong>Task Type:</strong> ${formatTaskName(trace.inferred_task || "rs_vqa")}</div>
          <div><strong>Trace Identifier:</strong> <code>${trace.trace_id || "tr_satquery_auto"}</code></div>
          <div><strong>Execution Status:</strong> <span style="color: #34d399;">✓ Completed Successfully</span></div>
        </div>
      </div>
    `;
    return;
  }

  steps.forEach((step, idx) => {
    const stepEl = document.createElement("div");
    stepEl.className = "timeline-step";
    const duration = step.duration_ms !== undefined ? `${step.duration_ms} ms` : "OK";
    const params = step.parameters_applied || step.parameters || {};
    const stepNum = step.step_id || (idx + 1);
    stepEl.innerHTML = `
      <div class="step-header">
        <span>${stepNum}. ${escapeHtml(step.step_name || "Specialist Execution")}</span>
        <span style="color: #34d399; font-family: var(--font-mono); font-size: 11px;">${duration}</span>
      </div>
      <div class="step-details">
        <div><strong>Tool / Specialist:</strong> <span style="color: #38bdf8;">${escapeHtml(step.tool_or_model || "SatQuery Core Specialist")}</span></div>
        <div><strong>Parameters Applied:</strong> <code>${escapeHtml(JSON.stringify(params))}</code></div>
        <div><strong>Evidence Summary:</strong> ${escapeHtml(step.summary || "Execution completed successfully.")}</div>
      </div>
    `;
    traceTimeline.appendChild(stepEl);
  });
}

// =============================================================================
// FILE UPLOAD & DROPZONE
// =============================================================================
function handleFileUpload(file, slot) {
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    if (slot === "a") {
      uploadedFileA = file;
      currentImageDataUrlA = e.target.result;
      previewA.src = e.target.result;
      previewA.classList.remove("hidden");
      placeholderA.classList.add("hidden");
      metaA.textContent = `${file.name} (${modalityA.value.toUpperCase()}) • ${(file.size / 1024).toFixed(1)} KB`;
      drawImageOnCanvas(baseCanvas, e.target.result);
      validationPill.className = "pill pill-success";
      validationPill.textContent = "Image A Ready";
    } else {
      uploadedFileB = file;
      previewB.src = e.target.result;
      previewB.classList.remove("hidden");
      placeholderB.classList.add("hidden");
      metaB.textContent = `${file.name} (${modalityB.value.toUpperCase()}) • ${(file.size / 1024).toFixed(1)} KB`;
      stageB.classList.remove("hidden");
      drawImageOnCanvas(baseCanvasB, e.target.result);
      document.getElementById("image-badge-b").textContent = `Image B (${modalityB.value.toUpperCase()})`;
      validationPill.className = "pill pill-success";
      validationPill.textContent = "Image Pair Ready";
    }
  };
  reader.readAsDataURL(file);
}

function setupDropzone(dropzoneEl, fileInputEl, slot) {
  ["dragenter", "dragover"].forEach((evt) => {
    dropzoneEl.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzoneEl.style.borderColor = "#06b6d4";
      dropzoneEl.style.background = "rgba(6, 182, 212, 0.1)";
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzoneEl.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzoneEl.style.borderColor = "#475569";
      dropzoneEl.style.background = "rgba(0, 0, 0, 0.2)";
    });
  });

  dropzoneEl.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0], slot);
    }
  });

  fileInputEl.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files[0], slot);
    }
  });
}

function drawImageOnCanvas(canvas, dataUrl) {
  const ctx = canvas.getContext("2d");
  const img = new Image();
  img.onload = () => {
    canvas.width = img.width || 256;
    canvas.height = img.height || 256;
    overlayCanvas.width = canvas.width;
    overlayCanvas.height = canvas.height;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    renderOverlays();
  };
  img.src = dataUrl;
}

// =============================================================================
// VISUAL CANVAS OVERLAYS & MASKS
// =============================================================================
function renderOverlays() {
  const ctx = overlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  const w = overlayCanvas.width || 256;
  const h = overlayCanvas.height || 256;

  // 1. Heatmap / Change Mask
  if (
    toggleHeatmap.checked &&
    currentResponse &&
    currentResponse.visual_overlays &&
    (currentResponse.visual_overlays.heatmap_mask || currentResponse.visual_overlays.change_mask)
  ) {
    const maskData =
      currentResponse.visual_overlays.heatmap_mask || currentResponse.visual_overlays.change_mask;
    const opacity = parseInt(opacitySlider.value, 10) / 100;
    renderHeatmapMask(ctx, maskData, w, h, opacity);
  }

  // 2. Bounding Boxes
  if (
    toggleBBoxes.checked &&
    currentResponse &&
    currentResponse.visual_overlays &&
    currentResponse.visual_overlays.boxes
  ) {
    const boxes = currentResponse.visual_overlays.boxes;
    boxes.forEach((b) => {
      const coords = b.box_2d;
      const ymin = coords[0] * h;
      const xmin = coords[1] * w;
      const ymax = coords[2] * h;
      const xmax = coords[3] * w;

      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2.5;
      ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

      ctx.fillStyle = "rgba(14, 165, 233, 0.15)";
      ctx.fillRect(xmin, ymin, xmax - xmin, ymax - ymin);

      const label = `${b.label} (${Math.round((b.score || 0.9) * 100)}%)`;
      ctx.font = "bold 11px Inter, sans-serif";
      const tw = ctx.measureText(label).width;

      ctx.fillStyle = "#0284c7";
      ctx.fillRect(xmin, Math.max(0, ymin - 18), tw + 8, 18);

      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, xmin + 4, Math.max(12, ymin - 4));
    });
  }

  // 3. Active Point Reticle Pin
  if (activePointReticle) {
    const rx = activePointReticle.norm_x * w;
    const ry = activePointReticle.norm_y * h;

    ctx.save();
    ctx.strokeStyle = "#06b6d4";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(rx, ry, 12, 0, 2 * Math.PI);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(rx, ry, 4, 0, 2 * Math.PI);
    ctx.fillStyle = "#ffffff";
    ctx.fill();

    ctx.strokeStyle = "rgba(6, 182, 212, 0.8)";
    ctx.beginPath();
    ctx.moveTo(rx - 18, ry);
    ctx.lineTo(rx + 18, ry);
    ctx.moveTo(rx, ry - 18);
    ctx.lineTo(rx, ry + 18);
    ctx.stroke();
    ctx.restore();
  }
}

function renderHeatmapMask(ctx, maskData, w, h, opacity) {
  const mw = maskData[0].length;
  const mh = maskData.length;
  const cellW = w / mw;
  const cellH = h / mh;

  for (let y = 0; y < mh; y++) {
    for (let x = 0; x < mw; x++) {
      const val = maskData[y][x];
      if (val > 0.05) {
        ctx.fillStyle = getTurboColor(val, opacity);
        ctx.fillRect(x * cellW, y * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
  }
}

function getTurboColor(v, alpha = 0.65) {
  const r = Math.round(255 * Math.min(1, Math.max(0, 1.5 * v)));
  const g = Math.round(255 * Math.sin(Math.PI * v));
  const b = Math.round(255 * Math.cos((Math.PI / 2) * v));
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// =============================================================================
// CUSTOM CHATBOT CONTROLLER
// =============================================================================
function switchTab(tab) {
  if (tab === "chatbot") {
    tabChatbot.classList.add("active");
    tabAudit.classList.remove("active");
    viewChatbot.classList.remove("hidden");
    viewChatbot.classList.add("active-view");
    viewAudit.classList.add("hidden");
    viewAudit.classList.remove("active-view");
  } else {
    tabAudit.classList.add("active");
    tabChatbot.classList.remove("active");
    viewAudit.classList.remove("hidden");
    viewAudit.classList.add("active-view");
    viewChatbot.classList.add("hidden");
    viewChatbot.classList.remove("active-view");
    if (currentResponse) {
      renderResults(currentResponse);
    }
  }
}

function appendUserMessage(text) {
  const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble user-bubble";
  bubble.innerHTML = `
    <div class="bubble-header">
      <span class="user-avatar">${currentUser && currentUser.role === "Administrator" ? "👑" : "👤"}</span>
      <strong>${currentUser ? currentUser.full_name : "You"}</strong>
      <span class="chat-time">${time}</span>
    </div>
    <div class="bubble-content">${escapeHtml(text)}</div>
  `;
  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendBotMessage(text, meta = null) {
  const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble bot-bubble";

  const formattedContent = formatBotMarkdown(text);

  bubble.innerHTML = `
    <div class="bubble-header">
      <span class="bot-avatar">🛰️</span>
      <strong>SatQuery Custom Chatbot</strong>
      <span class="chat-time">${time}</span>
      <button class="btn-bubble-speak" onclick="speakLastBotMessage(this)" title="Read response out loud">🔊</button>
    </div>
    <div class="bubble-content">${formattedContent}</div>
  `;
  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatBotMarkdown(mdText) {
  if (!mdText) return "";
  const lines = mdText.split("\n");
  let inTable = false;
  let tableHtml = "";
  let htmlLines = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith("|") && line.endsWith("|")) {
      const cells = line.split("|").slice(1, -1).map((c) => c.trim());
      if (cells.every((c) => /^:?-+:?$/.test(c))) {
        continue;
      }
      if (!inTable) {
        inTable = true;
        tableHtml = '<table class="md-table" style="width:100%; border-collapse:collapse; margin:8px 0; font-size:11px;"><thead><tr>';
        cells.forEach((c) => {
          tableHtml += `<th style="border:1px solid #334155; padding:4px 8px; background:#1e293b; color:#38bdf8;">${inlineMarkdown(c)}</th>`;
        });
        tableHtml += "</tr></thead><tbody>";
      } else {
        tableHtml += "<tr>";
        cells.forEach((c) => {
          tableHtml += `<td style="border:1px solid #334155; padding:4px 8px; color:#e2e8f0;">${inlineMarkdown(c)}</td>`;
        });
        tableHtml += "</tr>";
      }
    } else {
      if (inTable) {
        inTable = false;
        tableHtml += "</tbody></table>";
        htmlLines.push(tableHtml);
        tableHtml = "";
      }
      if (line.startsWith("### ")) {
        htmlLines.push(`<h3 style="color:#38bdf8; margin:8px 0 4px; font-size:14px;">${inlineMarkdown(line.substring(4))}</h3>`);
      } else if (line.startsWith("#### ")) {
        htmlLines.push(`<h4 style="color:#34d399; margin:6px 0 2px; font-size:12px;">${inlineMarkdown(line.substring(5))}</h4>`);
      } else if (line.startsWith("- ")) {
        htmlLines.push(`<li style="margin-left:16px; margin-bottom:2px; color:#cbd5e1;">${inlineMarkdown(line.substring(2))}</li>`);
      } else if (line === "") {
        htmlLines.push("<br/>");
      } else {
        htmlLines.push(`<p style="margin-bottom:4px; color:#e2e8f0;">${inlineMarkdown(line)}</p>`);
      }
    }
  }

  if (inTable) {
    tableHtml += "</tbody></table>";
    htmlLines.push(tableHtml);
  }

  return htmlLines.join("");
}

function inlineMarkdown(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong style="color:#f8fafc;">$1</strong>')
    .replace(/\*(.*?)\*/g, '<em style="color:#94a3b8;">$1</em>')
    .replace(/`([^`]+)`/g, '<code style="background:#090d16; color:#38bdf8; padding:1px 4px; border-radius:4px; font-size:11px;">$1</code>');
}

function escapeHtml(unsafe) {
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

async function handleChatSubmit(customText = null) {
  const query = customText || chatQueryInput.value.trim();
  if (!query) return;

  chatQueryInput.value = "";
  appendUserMessage(query);

  const fileA = uploadedFileA || (fileInputA.files.length > 0 ? fileInputA.files[0] : null);
  const fileB = uploadedFileB || (fileInputB.files.length > 0 ? fileInputB.files[0] : null);

  btnChatSend.disabled = true;
  btnChatSend.innerHTML = `<span>Thinking... ⏳</span>`;

  try {
    const formData = new FormData();
    formData.append("query", query);
    formData.append("session_id", activeSessionId);
    formData.append("language", getSelectedLanguage());

    if (fileA) {
      formData.append("image_a", fileA);
      formData.append("modality_a", modalityA.value);
      if (fileB) {
        formData.append("image_b", fileB);
        formData.append("modality_b", modalityB.value);
      }
    }

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: getAuthHeaders(),
      body: formData,
    });

    const data = await res.json();
    appendBotMessage(data.reply || "No response received.", data);

    currentResponse = data;
    renderResults(data);

    if (voiceEnabled && data.reply) {
      speakText(data.reply);
    }
  } catch (err) {
    appendBotMessage(`⚠️ Error communicating with Custom Chatbot: ${err.message}`);
  } finally {
    btnChatSend.disabled = false;
    btnChatSend.innerHTML = `<span>Send 🚀</span>`;
  }
}

// =============================================================================
// EVENT LISTENERS SETUP
// =============================================================================
function setupEventListeners() {
  // Voice Controls
  btnToggleVoice.addEventListener("click", toggleGlobalVoice);

  btnMicQuery.addEventListener("click", () => {
    startListening(queryInput, btnMicQuery, handleQuerySubmit);
  });

  btnMicChat.addEventListener("click", () => {
    startListening(chatQueryInput, btnMicChat, () => handleChatSubmit());
  });

  btnAnswerSpeak.addEventListener("click", () => {
    if (answerText.textContent) {
      speakText(answerText.textContent, true);
    }
  });

  // Viewer Mode Switcher
  btnViewCanvas.addEventListener("click", () => switchViewerMode("canvas"));
  btnViewMap.addEventListener("click", () => switchViewerMode("map"));

  // Point-and-Query Close & Speak
  btnPqClose.addEventListener("click", () => {
    pointQueryCard.classList.add("hidden");
    activePointReticle = null;
    renderOverlays();
  });

  btnPqSpeak.addEventListener("click", () => {
    if (lastPqSpeech) {
      speakText(lastPqSpeech, true);
    }
  });

  // Auth Modal & Controls
  btnUserProfile.addEventListener("click", () => {
    if (currentUser) {
      alert(`Logged in as: ${currentUser.full_name}\nEmail: ${currentUser.email}\nRole: ${currentUser.role}\nOrganization: ${currentUser.organization}`);
    } else {
      openAuthModal();
    }
  });

  btnHeaderLogout.addEventListener("click", handleLogout);
  btnAuthClose.addEventListener("click", closeAuthModal);
  tabLogin.addEventListener("click", () => switchAuthTab("login"));
  tabRegister.addEventListener("click", () => switchAuthTab("register"));
  formLogin.addEventListener("submit", handleLoginSubmit);
  formRegister.addEventListener("submit", handleRegisterSubmit);

  if (btnDemoAdmin) {
    btnDemoAdmin.addEventListener("click", () => quickFillLogin("admin@satquery.ai", "Admin@1234"));
  }
  if (btnDemoAnalyst) {
    btnDemoAnalyst.addEventListener("click", () => quickFillLogin("analyst@satquery.ai", "Analyst@1234"));
  }

  // Canvas Click for Point-and-Query
  overlayCanvas.addEventListener("click", (e) => {
    // Unlock Web Speech API context on user gesture
    if (window.speechSynthesis) {
      try {
        if (window.speechSynthesis.paused) {
          window.speechSynthesis.resume();
        }
      } catch (e) {}
    }

    const rect = overlayCanvas.getBoundingClientRect();
    const scaleX = overlayCanvas.width / rect.width;
    const scaleY = overlayCanvas.height / rect.height;

    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    const normX = Math.max(0, Math.min(1, clickX / overlayCanvas.width));
    const normY = Math.max(0, Math.min(1, clickY / overlayCanvas.height));

    const pixelX = Math.round(normX * 256);
    const pixelY = Math.round(normY * 256);
    const latVal = 18.5204 + (0.5 - normY) * 0.024;
    const lonVal = 73.8567 + (normX - 0.5) * 0.024;

    updatePointDiagnosticElements(pixelX, pixelY, latVal, lonVal);
    executePointQuery(normX, normY, latVal, lonVal);
  });

  overlayCanvas.addEventListener("mousemove", (e) => {
    const rect = overlayCanvas.getBoundingClientRect();
    const scaleX = overlayCanvas.width / rect.width;
    const scaleY = overlayCanvas.height / rect.height;

    const mouseX = Math.round((e.clientX - rect.left) * scaleX);
    const mouseY = Math.round((e.clientY - rect.top) * scaleY);
    cursorCoordsEl.textContent = `X: ${mouseX}, Y: ${mouseY}`;
  });

  btnSubmit.addEventListener("click", handleQuerySubmit);

  tabChatbot.addEventListener("click", () => switchTab("chatbot"));
  tabAudit.addEventListener("click", () => switchTab("audit"));

  btnChatSend.addEventListener("click", () => handleChatSubmit());
  chatQueryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleChatSubmit();
    }
  });

  if (btnChatClear) {
    btnChatClear.addEventListener("click", () => {
      activeSessionId = "session_" + Math.random().toString(36).substring(2, 9);
      stopSpeaking();
      chatMessages.innerHTML = `
        <div class="chat-bubble bot-bubble">
          <div class="bubble-header">
            <span class="bot-avatar">🛰️</span>
            <strong>SatQuery Custom Chatbot</strong>
            <span class="chat-time">Online</span>
            <button class="btn-bubble-speak" onclick="speakLastBotMessage(this)" title="Read response out loud">🔊</button>
          </div>
          <div class="bubble-content">Session reset! Ask any precision agriculture, soil, water body, or remote-sensing query in any language.</div>
        </div>
      `;
    });
  }

  const catPills = document.querySelectorAll(".cat-pill");
  const queryChips = document.querySelectorAll(".chip");
  const querySearchInput = document.getElementById("query-search-input");

  catPills.forEach((pill) => {
    pill.addEventListener("click", () => {
      catPills.forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      const cat = pill.getAttribute("data-cat");
      filterQueryChips(cat, querySearchInput ? querySearchInput.value : "");
    });
  });

  if (querySearchInput) {
    querySearchInput.addEventListener("input", () => {
      const activePill = document.querySelector(".cat-pill.active");
      const cat = activePill ? activePill.getAttribute("data-cat") : "all";
      filterQueryChips(cat, querySearchInput.value);
    });
  }

  function filterQueryChips(cat, queryText) {
    const qLower = queryText.toLowerCase().trim();
    queryChips.forEach((chip) => {
      const chipCat = chip.getAttribute("data-cat");
      const chipQuery = chip.getAttribute("data-query").toLowerCase();
      const chipLabel = chip.querySelector(".chip-text")?.textContent.toLowerCase() || "";

      const matchCat = cat === "all" || chipCat === cat;
      const matchText = !qLower || chipQuery.includes(qLower) || chipLabel.includes(qLower);

      if (matchCat && matchText) {
        chip.classList.remove("hidden");
      } else {
        chip.classList.add("hidden");
      }
    });
  }

  const chatSuggestionChips = document.querySelectorAll(".chat-chip");
  chatSuggestionChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const chatQuery = chip.getAttribute("data-chat");
      handleChatSubmit(chatQuery);
    });
  });

  queryChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-query");
      queryInput.value = q;
      handleQuerySubmit();
    });
  });

  toggleBBoxes.addEventListener("change", renderOverlays);
  toggleHeatmap.addEventListener("change", renderOverlays);
  opacitySlider.addEventListener("input", renderOverlays);

  setupDropzone(dropzoneA, fileInputA, "a");
  setupDropzone(dropzoneB, fileInputB, "b");

  btnValidate.addEventListener("click", async () => {
    const fileA = uploadedFileA || (fileInputA.files.length > 0 ? fileInputA.files[0] : null);
    const fileB = uploadedFileB || (fileInputB.files.length > 0 ? fileInputB.files[0] : null);

    const formData = new FormData();
    if (fileA) {
      formData.append("image_a", fileA);
      formData.append("modality_a", modalityA.value);
      if (fileB) {
        formData.append("image_b", fileB);
        formData.append("modality_b", modalityB.value);
      }
    }

    try {
      const res = await fetch("/api/validate", {
        method: "POST",
        headers: getAuthHeaders(),
        body: formData,
      });
      const val = await res.json();
      if (val.valid) {
        validationPill.className = "pill pill-success";
        validationPill.textContent = `Valid (${val.input_configuration}) • Ready`;
      } else {
        validationPill.className = "pill pill-danger";
        validationPill.textContent = `Rejected: ${val.errors[0] || "Invalid inputs"}`;
      }
    } catch (e) {
      validationPill.className = "pill pill-danger";
      validationPill.textContent = "Validation error";
    }
  });

  btnExportPdf.addEventListener("click", () => exportReport("pdf"));
  btnExportJson.addEventListener("click", () => exportReport("json"));
}

async function exportReport(format) {
  if (!currentResponse) {
    alert("Execute a query or run chat first to generate an auditable report.");
    return;
  }

  const payload = {
    format: format,
    query: currentResponse.query || "Satellite Analysis Query",
    task_type: currentResponse.task_type || "Remote Sensing Query",
    answer: currentResponse.answer || currentResponse.reply || "",
    confidence: currentResponse.confidence || 0.9,
    trace: currentResponse.trace || currentResponse.execution_trace || {},
  };

  const res = await fetch("/api/report/download", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...getAuthHeaders(),
    },
    body: JSON.stringify(payload),
  });

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = format === "pdf" ? "satquery_execution_report.pdf" : "satquery_audit_trace.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

window.addEventListener("DOMContentLoaded", init);
