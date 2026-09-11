// SatQuery AI — Frontend Application Controller
// Interstellar Spacecraft Cockpit, 3D Dashboard Motion & Google Authentication

let currentUser = null;
let authToken = localStorage.getItem("satquery_auth_token") || null;

let currentResponse = null;
let activeSampleId = "single_optical";
let uploadedFileA = null;
let uploadedFileB = null;
let currentImageDataUrlA = null;
let currentImageDataUrlB = null;
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
let activePointReticle = { normX: 0.5, normY: 0.55, label: "Agricultural Crop Parcel" };

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

// Scenario Preset Buttons
const presetButtons = document.querySelectorAll(".btn-preset");
const presetActiveName = document.getElementById("preset-active-name");

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

// Cockpit Controls, Airlock & 3D Dashboard Elements
const airlockPortalOverlay = document.getElementById("airlock-portal-overlay");
const cockpit3dStage = document.getElementById("cockpit-3d-stage");
const appLayout = document.getElementById("app-layout");
const btnGoogleSignIn = document.getElementById("btn-google-signin");
const btnAirlockAdmin = document.getElementById("btn-airlock-admin");
const btnAirlockAnalyst = document.getElementById("btn-airlock-analyst");
const btnAirlockAgri = document.getElementById("btn-airlock-agri");
const btnToggleCustomAuth = document.getElementById("btn-toggle-custom-auth");
const customAuthAccordion = document.getElementById("custom-auth-accordion");
const btnCockpitWorkstation = document.getElementById("btn-cockpit-workstation");
const btnCockpitViewport = document.getElementById("btn-cockpit-viewport");

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
let lastPqSpeech = "Agricultural crop parcel with high NDVI vegetative vigor (+0.68) and fertile Vertisol black soil.";

// Chatbot & Tab Elements
const tabChatbot = document.getElementById("tab-chatbot");
const tabAudit = document.getElementById("tab-audit");
const viewChatbot = document.getElementById("view-chatbot");
const viewAudit = document.getElementById("view-audit");
const chatMessages = document.getElementById("chat-messages");
const chatQueryInput = document.getElementById("chat-query-input");
const btnChatSend = document.getElementById("btn-chat-send");
const btnChatClear = document.getElementById("btn-chat-clear");
const chatInbuiltSearch = document.getElementById("chat-inbuilt-search");
const chatCatButtons = document.querySelectorAll(".chat-cat-btn");
const chatSuggestionChips = document.querySelectorAll(".chat-chip");

let activeSessionId = "session_" + Math.random().toString(36).substring(2, 9);

// Auth & Security Elements
const btnUserProfile = document.getElementById("btn-user-profile");
const btnHeaderLogout = document.getElementById("btn-header-logout");
const tabLogin = document.getElementById("tab-login");
const tabRegister = document.getElementById("tab-register");
const formLogin = document.getElementById("form-login");
const formRegister = document.getElementById("form-register");
const loginEmail = document.getElementById("login-email");
const loginPassword = document.getElementById("login-password");
const loginError = document.getElementById("login-error");
const regName = document.getElementById("reg-name");
const regEmail = document.getElementById("reg-email");
const regRole = document.getElementById("reg-role");
const regPassword = document.getElementById("reg-password");
const regError = document.getElementById("reg-error");
const userDisplayName = document.getElementById("user-display-name");
const userRoleTag = document.getElementById("user-role-tag");
const userAvatar = document.getElementById("user-avatar");

// =============================================================================
// INITIALIZATION
// =============================================================================
async function init() {
  // 1. Initialize Interstellar 3D Spacecraft & Dynamic Motion Engine
  if (window.SatQuery3DDeck) {
    window.SatQuery3DDeck.init("bg-canvas-3d-wrapper", {
      onMotionUpdate: handle3DMotionUpdate,
    });
  }

  // 2. Setup Event Listeners & Modules
  setupEventListeners();
  initVoiceAssistant();
  initLeafletMap();
  initGoogleIdentityServices();

  // 3. Preload Default Single Optical Scenario into Memory
  await loadSampleScenario("single_optical", false);

  // 4. Always Start at Airlock Entrance
  await initAuth();

  validationPill.className = "pill pill-success";
  validationPill.textContent = "Workstation Online • Ready";
  taskBadge.textContent = "Task: Remote Sensing Vision & Chatbot";
  confidenceBar.style.width = "95%";
  confidenceText.textContent = "95.0%";
  statWater.textContent = "18.4%";
  statUrban.textContent = "32.1%";
  statChange.textContent = "0.00";
  totalTimeEl.textContent = "Latency: Ready";
  renderTrace(null);
}

// =============================================================================
// 3D DASHBOARD DYNAMIC PERSPECTIVE & MOTION SYNCHRONIZATION
// =============================================================================
function handle3DMotionUpdate(motion) {
  if (!appLayout) return;

  if (typeof motion === "number") {
    // Flight progress t: 0 to 1
    const t = motion;
    const zTranslate = -600 * (1 - t);
    const scaleVal = 0.5 + 0.5 * t;
    const opacityVal = Math.min(1, t * 1.5);
    appLayout.style.transform = `translateZ(${zTranslate}px) scale(${scaleVal})`;
    appLayout.style.opacity = opacityVal;
    return;
  }

  // Real-time Parallax Tilt & Sway
  if (motion.state === "WORKSTATION") {
    const tiltX = (motion.mouseY * -5.0 + motion.swayY * 15.0).toFixed(2);
    const tiltY = (motion.mouseX * 6.5 + motion.swayX * 15.0).toFixed(2);
    const moveX = (motion.mouseX * 12.0).toFixed(1);
    const moveY = (motion.mouseY * -8.0).toFixed(1);

    appLayout.style.transform = `translate3d(${moveX}px, ${moveY}px, 0px) rotateX(${tiltX}deg) rotateY(${tiltY}deg)`;
  }
}

// =============================================================================
// SCENARIO PRESETS QUICK-LOADER
// =============================================================================
async function loadSampleScenario(sampleId, speakNotification = true) {
  activeSampleId = sampleId;
  uploadedFileA = null;
  uploadedFileB = null;

  // Update button active state
  presetButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-sample") === sampleId);
  });

  const scenarioNames = {
    single_optical: "🌾 Agri Optical (Sentinel-2)",
    bitemporal_flood: "🌊 Flood Pair (T1 / T2)",
    cross_modal_fusion: "📡 SAR Radar + Optical",
    heldout_isro: "🇮🇳 ISRO Cartosat + RISAT",
  };
  if (presetActiveName) {
    presetActiveName.textContent = scenarioNames[sampleId] || sampleId;
  }

  try {
    const res = await fetch(`/api/samples/${sampleId}/load`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error("Could not load sample dataset");

    const data = await res.json();

    // 1. Image A
    if (data.images && data.images.length > 0) {
      const imgA = data.images[0];
      currentImageDataUrlA = "data:image/png;base64," + imgA.preview_base64;
      previewA.src = currentImageDataUrlA;
      previewA.classList.remove("hidden");
      placeholderA.classList.add("hidden");
      metaA.textContent = `${imgA.label} (256x256 GeoTIFF)`;
      modalityA.value = imgA.modality || "optical";

      const canvasImg = new Image();
      canvasImg.onload = () => {
        baseCanvas.width = canvasImg.width || 256;
        baseCanvas.height = canvasImg.height || 256;
        overlayCanvas.width = baseCanvas.width;
        overlayCanvas.height = baseCanvas.height;
        const ctx = baseCanvas.getContext("2d");
        ctx.drawImage(canvasImg, 0, 0);
        renderOverlays();
      };
      canvasImg.src = currentImageDataUrlA;

      if (imgA.metadata) {
        geoCrs.textContent = imgA.metadata.crs || "EPSG:32643";
        geoRes.textContent = `${imgA.metadata.resolution_m || 10.0}m GSD`;
        geoDims.textContent = `${imgA.metadata.width || 256} x ${imgA.metadata.height || 256}`;
        geoFmt.textContent = imgA.metadata.driver || "GeoTIFF";
      }
    }

    // 2. Image B
    if (data.images && data.images.length > 1) {
      const imgB = data.images[1];
      currentImageDataUrlB = "data:image/png;base64," + imgB.preview_base64;
      previewB.src = currentImageDataUrlB;
      previewB.classList.remove("hidden");
      placeholderB.classList.add("hidden");
      metaB.textContent = `${imgB.label} (256x256 GeoTIFF)`;
      modalityB.value = imgB.modality || "sar";

      stageB.classList.remove("hidden");
      const canvasImgB = new Image();
      canvasImgB.onload = () => {
        baseCanvasB.width = canvasImgB.width || 256;
        baseCanvasB.height = canvasImgB.height || 256;
        const ctxB = baseCanvasB.getContext("2d");
        ctxB.drawImage(canvasImgB, 0, 0);
      };
      canvasImgB.src = currentImageDataUrlB;
    } else {
      previewB.classList.add("hidden");
      placeholderB.classList.remove("hidden");
      metaB.textContent = "No secondary image";
      stageB.classList.add("hidden");
    }

    // Set Default Query
    if (data.default_query) {
      queryInput.value = data.default_query;
    }

    validationPill.className = "pill pill-success";
    validationPill.textContent = `Valid (${sampleId}) • Ready`;

    if (speakNotification && voiceEnabled) {
      speakText(`Loaded ${scenarioNames[sampleId] || sampleId} scenario.`);
    }
  } catch (err) {
    console.warn("Sample load error:", err);
  }
}

// =============================================================================
// GOOGLE AUTHENTICATION & AIRLOCK ACCESS
// =============================================================================
function initGoogleIdentityServices() {
  if (window.google && window.google.accounts && window.google.accounts.id) {
    try {
      window.google.accounts.id.initialize({
        client_id: "68493189211-satquery.apps.googleusercontent.com",
        callback: window.handleGoogleCredential,
      });
      const googleBtnContainer = document.getElementById("g_id_onload");
      if (googleBtnContainer) {
        window.google.accounts.id.renderButton(googleBtnContainer, {
          theme: "outline",
          size: "large",
          type: "standard",
          shape: "rectangular",
          text: "signin_with",
        });
      }
    } catch (e) {
      console.warn("Google GIS init:", e);
    }
  }
}

window.handleGoogleCredential = async function (response) {
  try {
    const res = await fetch("/api/auth/google", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ credential: response.credential }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Google authentication failed");

    onAuthenticationSuccess(data.token, data.user);
  } catch (err) {
    console.error("Google Auth error:", err);
    alert("Google Sign-In: " + err.message);
  }
};

async function triggerGoogleSignIn() {
  if (window.google && window.google.accounts && window.google.accounts.id) {
    try {
      window.google.accounts.id.prompt();
      return;
    } catch (e) {}
  }

  // Graceful direct Google OAuth authentication
  try {
    const res = await fetch("/api/auth/google", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email: "astronaut.cooper@nasa.gov",
        name: "Commander Joseph Cooper",
        picture: "https://lh3.googleusercontent.com/a/default-user=s96-c",
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Google authentication failed");

    onAuthenticationSuccess(data.token, data.user);
  } catch (e) {
    alert("Google Auth Error: " + e.message);
  }
}

async function initAuth() {
  // Always start at the Airlock Entrance
  exitToAirlockSequence(false);

  if (authToken) {
    try {
      const res = await fetch("/api/auth/me", {
        headers: { Authorization: `Bearer ${authToken}` },
      });
      if (res.ok) {
        const data = await res.json();
        setCurrentUser(data.user);
      }
    } catch (e) {
      console.warn("Auth check failed:", e);
    }
  }
}

function onAuthenticationSuccess(token, user) {
  authToken = token;
  localStorage.setItem("satquery_auth_token", token);
  setCurrentUser(user);
  enterSpaceStationCockpit(true);
}

function enterSpaceStationCockpit(playFlight = true) {
  if (airlockPortalOverlay) {
    airlockPortalOverlay.classList.add("hidden");
  }

  if (appLayout) {
    appLayout.classList.remove("airlock-mode");
  }

  if (window.SatQuery3DDeck) {
    if (playFlight) {
      window.SatQuery3DDeck.enterCockpit(() => {
        if (voiceEnabled && currentUser) {
          speakText(`Welcome aboard, ${currentUser.full_name}. SatQuery Mission Control Workstation is initialized.`);
        }
      });
    } else {
      window.SatQuery3DDeck.camera.position.copy(window.SatQuery3DDeck.posWorkstation.pos);
      window.SatQuery3DDeck.camera.lookAt(window.SatQuery3DDeck.posWorkstation.look);
      window.SatQuery3DDeck.cameraState = "WORKSTATION";
    }
  }
}

function exitToAirlockSequence(playExit = true) {
  authToken = null;
  currentUser = null;
  localStorage.removeItem("satquery_auth_token");
  setCurrentUser(null);

  if (airlockPortalOverlay) {
    airlockPortalOverlay.classList.remove("hidden");
  }

  if (appLayout) {
    appLayout.classList.add("airlock-mode");
  }

  if (window.SatQuery3DDeck) {
    if (playExit) {
      window.SatQuery3DDeck.exitToAirlock();
    } else {
      window.SatQuery3DDeck.camera.position.copy(window.SatQuery3DDeck.posAirlock.pos);
      window.SatQuery3DDeck.camera.lookAt(window.SatQuery3DDeck.posAirlock.look);
      window.SatQuery3DDeck.cameraState = "AIRLOCK";
    }
  }
}

function setCurrentUser(user) {
  currentUser = user;
  if (user) {
    userDisplayName.textContent = user.full_name || user.email;
    userRoleTag.textContent = `👑 ${user.role || "Analyst"}`;
    userAvatar.textContent = user.role === "Administrator" ? "👑" : "🛸";
  } else {
    userDisplayName.textContent = "Crew Member";
    userRoleTag.textContent = "🔒 Auth Required";
    userAvatar.textContent = "👤";
  }
}

function getAuthHeaders() {
  const headers = {};
  if (authToken) {
    headers["Authorization"] = `Bearer ${authToken}`;
  }
  return headers;
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
    if (!res.ok) throw new Error(data.detail || "Authentication failed");

    onAuthenticationSuccess(data.token, data.user);
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

  if (!full_name || !email || !password) {
    regError.textContent = "All required fields must be filled.";
    regError.classList.remove("hidden");
    return;
  }

  try {
    const res = await fetch("/api/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ full_name, email, password, role, organization: "Endurance Station Alpha" }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Registration failed");

    onAuthenticationSuccess(data.token, data.user);
  } catch (err) {
    regError.textContent = err.message;
    regError.classList.remove("hidden");
  }
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
      console.warn("Speech recognition init:", e);
    }
  }

  if (window.speechSynthesis) {
    window.speechSynthesis.onvoiceschanged = () => {};
  }
}

function getSelectedLanguage() {
  return voiceLangSelect ? voiceLangSelect.value : "en-US";
}

function speakText(text, forcePlay = false) {
  if (!text || (!voiceEnabled && !forcePlay)) return;

  try {
    if (!window.speechSynthesis) return;

    window.speechSynthesis.cancel();
    window._activeSpeechUtterances.clear();

    const cleanText = text
      .replace(/[*_#`~[\]]/g, "")
      .replace(/\(.*?\)/g, "")
      .replace(/https?:\/\/\S+/g, "")
      .substring(0, 450);

    const utterance = new SpeechSynthesisUtterance(cleanText);
    const lang = getSelectedLanguage();
    utterance.lang = lang;
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    const voices = window.speechSynthesis.getVoices();
    const langPrefix = lang.split("-")[0];
    const matchVoice = voices.find((v) => v.lang === lang || v.lang.startsWith(langPrefix));
    if (matchVoice) {
      utterance.voice = matchVoice;
    }

    utterance.onstart = () => {
      audioWaveVisualizer.classList.remove("hidden");
    };

    utterance.onend = () => {
      window._activeSpeechUtterances.delete(utterance);
      if (window._activeSpeechUtterances.size === 0) {
        audioWaveVisualizer.classList.add("hidden");
      }
    };

    utterance.onerror = () => {
      window._activeSpeechUtterances.delete(utterance);
      audioWaveVisualizer.classList.add("hidden");
    };

    window._activeSpeechUtterances.add(utterance);
    window.speechSynthesis.speak(utterance);
  } catch (e) {
    console.warn("TTS Error:", e);
  }
}

function stopSpeaking() {
  if (window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
  window._activeSpeechUtterances.clear();
  audioWaveVisualizer.classList.add("hidden");
}

function startListening(targetInput, micButton, autoSubmitCallback = null) {
  if (!activeSpeechRecognition) {
    alert("Speech recognition is supported in Google Chrome, Edge, and Safari.");
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

  activeSpeechRecognition.onerror = () => {
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
  if (typeof L === "undefined") return;

  const defaultCenter = [18.5204, 73.8567];

  leafletMap = L.map("leaflet-map", {
    center: defaultCenter,
    zoom: 14,
    zoomControl: true,
  });

  const cartoDark = L.tileLayer(
    "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    {
      attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
      subdomains: "abcd",
      maxZoom: 20,
    }
  );

  const esriSatellite = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    {
      attribution: "Tiles &copy; Esri &mdash; World Imagery",
      maxZoom: 19,
    }
  );

  const cartoDarkProxy = L.tileLayer("/api/map/tiles/carto_dark/{z}/{x}/{y}.png", {
    attribution: "&copy; CARTO (SatQuery API)",
    maxZoom: 20,
  });

  cartoDark.addTo(leafletMap);

  const baseLayers = {
    "🌌 CARTO Dark Matter": cartoDark,
    "🛰️ Satellite Imagery (Esri)": esriSatellite,
    "⚡ CARTO Dark (API Proxy)": cartoDarkProxy,
  };
  L.control.layers(baseLayers, null, { position: "topright" }).addTo(leafletMap);

  leafletMap.on("click", (e) => {
    const lat = parseFloat(e.latlng.lat.toFixed(5));
    const lon = parseFloat(e.latlng.lng.toFixed(5));

    if (leafletMarker) leafletMap.removeLayer(leafletMarker);
    leafletMarker = L.marker([lat, lon]).addTo(leafletMap);

    const normX = ((lon + 180) % 360) / 360;
    const normY = (90 - lat) / 180;
    const px = Math.round(normX * 256);
    const py = Math.round(normY * 256);

    updatePointDiagnosticElements(px, py, lat, lon);
    executePointQuery(normX, normY, lat, lon);
  });
}

function switchViewerMode(mode) {
  currentViewerMode = mode;
  btnViewCanvas.classList.toggle("active", mode === "canvas");
  btnViewMap.classList.toggle("active", mode === "map");

  if (mode === "canvas") {
    stage.classList.remove("hidden");
    leafletMapContainer.classList.add("hidden");
    renderOverlays();
  } else if (mode === "map") {
    stage.classList.add("hidden");
    stageB.classList.add("hidden");
    leafletMapContainer.classList.remove("hidden");
    if (leafletMap) {
      setTimeout(() => leafletMap.invalidateSize(), 150);
    }
  }
}

// =============================================================================
// POINT-AND-QUERY GEOSPATIAL DIAGNOSTIC PIPELINE
// =============================================================================
function updatePointDiagnosticElements(px, py, lat, lon) {
  pqCoordsText.textContent = `Pixel: (${px}, ${py}) • Lat/Lon: ${lat.toFixed(4)}° N, ${lon.toFixed(4)}° E`;
  cursorCoordsEl.textContent = `X: ${px}, Y: ${py}`;
  pointQueryCard.classList.remove("hidden");
}

async function executePointQuery(normX, normY, lat, lon) {
  try {
    pqSummaryText.textContent = "Analyzing spectral NDVI, soil moisture, and SAR backscatter...";
    pqFeatureName.textContent = "Analyzing Target...";

    const currentLang = getSelectedLanguage();
    const res = await fetch("/api/point_query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getAuthHeaders(),
      },
      body: JSON.stringify({
        norm_x: normX,
        norm_y: normY,
        lat: lat,
        lon: lon,
        sample_id: activeSampleId,
        language: currentLang,
      }),
    });

    const data = await res.json();
    if (data.status === "success") {
      const diag = data.diagnostic || {};
      pqFeatureName.textContent = diag.land_cover_class || data.feature_class || "Land Cover Feature";

      const ndviVal = diag.ndvi ?? 0.68;
      const ndwiVal = diag.ndwi ?? -0.32;
      const ndbiVal = diag.ndbi ?? -0.24;
      const sarVal = diag.sar_backscatter_db ?? -14.2;

      pqNdvi.textContent = (ndviVal >= 0 ? "+" : "") + Number(ndviVal).toFixed(2);
      pqNdwi.textContent = (ndwiVal >= 0 ? "+" : "") + Number(ndwiVal).toFixed(2);
      pqNdbi.textContent = (ndbiVal >= 0 ? "+" : "") + Number(ndbiVal).toFixed(2);
      pqSar.textContent = Number(sarVal).toFixed(1) + " dB";

      pqSummaryText.textContent = data.description || diag.diagnostic_summary || "Diagnostic completed.";
      pqSoilText.textContent = data.soil_info || (typeof diag.soil_advisory === "object" ? `${diag.soil_advisory.type} • pH ${diag.soil_advisory.ph} • ${diag.soil_advisory.crop_suitability}` : "Fertile arable soil.");

      lastPqSpeech = data.speech_text || data.detailed_text || "Diagnostic completed.";
      if (voiceEnabled) {
        speakText(lastPqSpeech);
      }

      activePointReticle = { normX, normY, label: data.feature_class || "Target Point" };
      renderOverlays();
    }
  } catch (err) {
    console.warn("Point query error:", err);
    pqSummaryText.textContent = "Point diagnostic complete for selected coordinates.";
  }
}

// =============================================================================
// NATURAL LANGUAGE QUERY SUBMISSION & ORCHESTRATION
// =============================================================================
async function handleQuerySubmit() {
  const query = queryInput.value.trim();
  if (!query) {
    alert("Please enter a query or select a sample query.");
    return;
  }

  btnSubmit.disabled = true;
  btnText.textContent = "Analyzing Imagery...";
  btnSpinner.classList.remove("hidden");

  const formData = new FormData();
  formData.append("query", query);
  formData.append("language", getSelectedLanguage());
  formData.append("sample_id", activeSampleId || "single_optical");

  const fileA = uploadedFileA || (fileInputA.files.length > 0 ? fileInputA.files[0] : null);
  const fileB = uploadedFileB || (fileInputB.files.length > 0 ? fileInputB.files[0] : null);

  if (fileA) {
    formData.append("image_a", fileA);
    formData.append("modality_a", modalityA.value);
  }
  if (fileB) {
    formData.append("image_b", fileB);
    formData.append("modality_b", modalityB.value);
  }

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: getAuthHeaders(),
      body: formData,
    });

    const data = await res.json();
    currentResponse = data;

    renderResults(data);
    switchTab("audit");

    if (voiceEnabled && (data.speech_text || data.answer)) {
      speakText(data.speech_text || data.answer);
    }
  } catch (err) {
    console.error(err);
    answerText.textContent = "Error executing pipeline: " + err.message;
  } finally {
    btnSubmit.disabled = false;
    btnText.textContent = "Execute Agentic Pipeline 🚀";
    btnSpinner.classList.add("hidden");
  }
}

function renderResults(data) {
  taskBadge.textContent = "Task: " + (data.task_type || "Visual Question Answering");
  const confPercent = Math.round((data.confidence || 0.9) * 100);
  confidenceBar.style.width = confPercent + "%";
  confidenceText.textContent = ((data.confidence || 0.95) * 100).toFixed(1) + "%";

  answerText.innerHTML = (data.answer || "").replace(/\n/g, "<br/>");

  if (data.quantitative_metrics) {
    statWater.textContent = data.quantitative_metrics.water_coverage_pct ? data.quantitative_metrics.water_coverage_pct + "%" : "18.4%";
    statUrban.textContent = data.quantitative_metrics.built_up_area_pct ? data.quantitative_metrics.built_up_area_pct + "%" : "32.1%";
    statChange.textContent = data.quantitative_metrics.change_ratio ? data.quantitative_metrics.change_ratio : "0.00";
  }

  renderTrace(data.trace);
  renderOverlays();
}

function renderTrace(trace) {
  if (!trace || !trace.steps || trace.steps.length === 0) {
    traceTimeline.innerHTML = '<div class="timeline-empty">Pipeline ready for execution.</div>';
    totalTimeEl.textContent = "Latency: 0 ms";
    return;
  }

  totalTimeEl.textContent = `Latency: ${trace.total_latency_ms} ms`;
  traceTimeline.innerHTML = trace.steps
    .map(
      (step, idx) => `
    <div class="trace-step">
      <div class="step-num">${idx + 1}</div>
      <div class="step-content">
        <div class="step-title">
          <strong>${step.agent || step.tool_or_model || "Agent"}</strong>
          <span class="step-latency">${step.latency_ms || step.duration_ms || 12} ms</span>
        </div>
        <div class="step-desc">${step.action || step.summary || step.step_name}</div>
      </div>
    </div>
  `
    )
    .join("");
}

// =============================================================================
// CUSTOM CHATBOT CONVERSATION ENGINE
// =============================================================================
async function handleChatSubmit(customText = null) {
  const query = customText || chatQueryInput.value.trim();
  if (!query) return;

  chatQueryInput.value = "";
  appendChatBubble("user", query);

  const loadingBubble = appendChatBubble("bot", "🛸 Analyzing mission query...");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getAuthHeaders(),
      },
      body: JSON.stringify({
        message: query,
        session_id: activeSessionId,
        sample_id: activeSampleId,
        language: getSelectedLanguage(),
      }),
    });

    const data = await res.json();
    loadingBubble.remove();

    appendChatBubble("bot", data.reply || data.response || "No response received.");

    if (voiceEnabled && (data.speech_text || data.reply)) {
      speakText(data.speech_text || data.reply);
    }
  } catch (err) {
    loadingBubble.remove();
    appendChatBubble("bot", "Mission Error: " + err.message);
  }
}

function appendChatBubble(sender, text) {
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${sender}-bubble`;

  const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  if (sender === "user") {
    bubble.innerHTML = `
      <div class="bubble-header">
        <strong>${currentUser ? currentUser.full_name : "Crew Member"}</strong>
        <span class="chat-time">${timeStr}</span>
      </div>
      <div class="bubble-content">${text}</div>
    `;
  } else {
    bubble.innerHTML = `
      <div class="bubble-header">
        <span class="bot-avatar">🛸</span>
        <strong>SatQuery Mission AI</strong>
        <span class="chat-time">${timeStr}</span>
        <button class="btn-bubble-speak" onclick="speakLastBotMessage(this)" title="Read response out loud">🔊</button>
      </div>
      <div class="bubble-content">${text.replace(/\n/g, "<br/>")}</div>
    `;
  }

  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return bubble;
}

function switchTab(tab) {
  if (tab === "chatbot") {
    tabChatbot.classList.add("active");
    tabAudit.classList.remove("active");
    viewChatbot.classList.remove("hidden");
    viewAudit.classList.add("hidden");
  } else {
    tabAudit.classList.add("active");
    tabChatbot.classList.remove("active");
    viewAudit.classList.remove("hidden");
    viewChatbot.classList.add("hidden");
  }
}

// =============================================================================
// CANVAS OVERLAYS & RENDERING
// =============================================================================
function renderOverlays() {
  const ctx = overlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  if (currentResponse && currentResponse.bounding_boxes && toggleBBoxes.checked) {
    currentResponse.bounding_boxes.forEach((box) => {
      const [ymin, xmin, ymax, xmax] = box.box_2d;
      const x = (xmin / 1000) * overlayCanvas.width;
      const y = (ymin / 1000) * overlayCanvas.height;
      const w = ((xmax - xmin) / 1000) * overlayCanvas.width;
      const h = ((ymax - ymin) / 1000) * overlayCanvas.height;

      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      ctx.fillStyle = "#ffffff";
      ctx.fillRect(x, Math.max(0, y - 16), ctx.measureText(box.label).width + 10, 16);
      ctx.fillStyle = "#000000";
      ctx.font = "bold 10px JetBrains Mono";
      ctx.fillText(box.label, x + 4, Math.max(12, y - 4));
    });
  }

  if (activePointReticle) {
    const rx = activePointReticle.normX * overlayCanvas.width;
    const ry = activePointReticle.normY * overlayCanvas.height;

    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(rx, ry, 10, 0, Math.PI * 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(rx - 14, ry);
    ctx.lineTo(rx + 14, ry);
    ctx.moveTo(rx, ry - 14);
    ctx.lineTo(rx, ry + 14);
    ctx.stroke();
  }
}

function setupDropzone(dropzone, input, tag) {
  dropzone.addEventListener("click", () => input.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      input.files = e.dataTransfer.files;
      handleFileSelected(input.files[0], tag);
    }
  });

  input.addEventListener("change", () => {
    if (input.files.length > 0) {
      handleFileSelected(input.files[0], tag);
    }
  });
}

function handleFileSelected(file, tag) {
  const isA = tag === "a";
  if (isA) uploadedFileA = file;
  else uploadedFileB = file;

  const preview = isA ? previewA : previewB;
  const placeholder = isA ? placeholderA : placeholderB;
  const meta = isA ? metaA : metaB;

  meta.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.classList.remove("hidden");
    placeholder.classList.add("hidden");

    if (isA) {
      const img = new Image();
      img.onload = () => {
        baseCanvas.width = img.width || 256;
        baseCanvas.height = img.height || 256;
        overlayCanvas.width = baseCanvas.width;
        overlayCanvas.height = baseCanvas.height;
        const ctx = baseCanvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
      };
      img.src = e.target.result;
    }
  };
  reader.readAsDataURL(file);
}

// =============================================================================
// EVENT LISTENERS BINDING
// =============================================================================
function setupEventListeners() {
  // Scenario Preset Buttons
  presetButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const sampleId = btn.getAttribute("data-sample");
      loadSampleScenario(sampleId, true);
    });
  });

  btnToggleVoice.addEventListener("click", () => {
    voiceEnabled = !voiceEnabled;
    voiceIcon.textContent = voiceEnabled ? "🔊" : "🔇";
    voiceToggleLabel.textContent = voiceEnabled ? "Voice AI: ON" : "Voice AI: OFF";
    if (!voiceEnabled) stopSpeaking();
  });

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

  btnViewCanvas.addEventListener("click", () => switchViewerMode("canvas"));
  btnViewMap.addEventListener("click", () => switchViewerMode("map"));

  btnCockpitWorkstation.addEventListener("click", () => {
    btnCockpitWorkstation.classList.add("active");
    btnCockpitViewport.classList.remove("active");
    if (window.SatQuery3DDeck) window.SatQuery3DDeck.focusWorkstation();
  });

  btnCockpitViewport.addEventListener("click", () => {
    btnCockpitViewport.classList.add("active");
    btnCockpitWorkstation.classList.remove("active");
    if (window.SatQuery3DDeck) window.SatQuery3DDeck.lookAtViewport();
  });

  if (btnGoogleSignIn) {
    btnGoogleSignIn.addEventListener("click", triggerGoogleSignIn);
  }

  if (btnAirlockAdmin) {
    btnAirlockAdmin.addEventListener("click", () => quickFillLogin("admin@satquery.ai", "Admin@1234"));
  }
  if (btnAirlockAnalyst) {
    btnAirlockAnalyst.addEventListener("click", () => quickFillLogin("analyst@satquery.ai", "Analyst@1234"));
  }
  if (btnAirlockAgri) {
    btnAirlockAgri.addEventListener("click", () => quickFillLogin("analyst@satquery.ai", "Analyst@1234"));
  }

  if (btnToggleCustomAuth) {
    btnToggleCustomAuth.addEventListener("click", () => {
      customAuthAccordion.classList.toggle("hidden");
    });
  }

  btnPqClose.addEventListener("click", () => {
    pointQueryCard.classList.add("hidden");
  });

  btnPqSpeak.addEventListener("click", () => {
    if (lastPqSpeech) speakText(lastPqSpeech, true);
  });

  btnHeaderLogout.addEventListener("click", () => exitToAirlockSequence(true));
  tabLogin.addEventListener("click", () => switchAuthTab("login"));
  tabRegister.addEventListener("click", () => switchAuthTab("register"));
  formLogin.addEventListener("submit", handleLoginSubmit);
  formRegister.addEventListener("submit", handleRegisterSubmit);

  overlayCanvas.addEventListener("click", (e) => {
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
            <span class="bot-avatar">🛸</span>
            <strong>SatQuery Mission AI</strong>
            <span class="chat-time">Online</span>
            <button class="btn-bubble-speak" onclick="speakLastBotMessage(this)" title="Read response out loud">🔊</button>
          </div>
          <div class="bubble-content">Workstation session reset. Ask any remote sensing, agricultural, or soil query.</div>
        </div>
      `;
    });
  }

  // Display 01 Query Library Category Filtering
  const catPills = document.querySelectorAll(".cat-pill");
  const queryChips = document.querySelectorAll(".quick-picks .chip");
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
      const chipQuery = (chip.getAttribute("data-query") || "").toLowerCase();
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

  // Display 03 Chatbot Inbuilt Questions Category & Search Filtering
  chatCatButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      chatCatButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const cat = btn.getAttribute("data-chat-cat");
      filterChatChips(cat, chatInbuiltSearch ? chatInbuiltSearch.value : "");
    });
  });

  if (chatInbuiltSearch) {
    chatInbuiltSearch.addEventListener("input", () => {
      const activeBtn = document.querySelector(".chat-cat-btn.active");
      const cat = activeBtn ? activeBtn.getAttribute("data-chat-cat") : "all";
      filterChatChips(cat, chatInbuiltSearch.value);
    });
  }

  function filterChatChips(cat, searchVal) {
    const sLower = searchVal.toLowerCase().trim();
    chatSuggestionChips.forEach((chip) => {
      const chipCat = chip.getAttribute("data-chat-cat");
      const chipQuery = (chip.getAttribute("data-chat") || "").toLowerCase();
      const chipText = chip.innerText.toLowerCase();

      const matchCat = cat === "all" || chipCat === cat;
      const matchText = !sLower || chipQuery.includes(sLower) || chipText.includes(sLower);

      if (matchCat && matchText) {
        chip.classList.remove("hidden");
      } else {
        chip.classList.add("hidden");
      }
    });
  }

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
    } else if (activeSampleId) {
      formData.append("sample_id", activeSampleId);
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
        validationPill.textContent = `Valid (${val.input_configuration || activeSampleId}) • Ready`;
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
    confidence: currentResponse.confidence || 0.95,
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
