/**
 * SatQuery AI — 2D.js Interactive Canvas Background Engine
 * Features:
 * - Orbital Satellite Constellations & Telemetry Data Links
 * - Dynamic Multi-Spectral Sensor Nodes (Optical, SAR, Thermal, SWIR)
 * - Mouse Proximity Physics (Gravitational Attraction, Holographic Tether Rays)
 * - Interactive Sonar Radar Pulse Waves on Click / Tap
 * - Rotating Deep-Space Radar Sweep with Phosphor Afterglow
 * - Adaptive High-DPI Rendering & 60 FPS Performance Optimization
 * - Real-Time Telemetry Data HUD Callouts (LEO Satellites, Sensor Bands, Lat/Lon)
 */

(function () {
  "use strict";

  // Configuration & State
  const CONFIG = {
    nodeCount: 65,
    maxDistance: 160,
    mouseRadius: 180,
    radarSpeed: 0.008,
    pulseSpeed: 3.5,
    pulseMaxRadius: 350,
    starCount: 80,
    mode: "orbital", // "orbital" | "radar" | "matrix" | "aurora"
    interactive: true,
  };

  let canvas, ctx;
  let width = 0;
  let height = 0;
  let dpr = 1;
  let animationFrameId = null;

  // Mouse & Touch State
  const mouse = {
    x: -1000,
    y: -1000,
    active: false,
    radius: CONFIG.mouseRadius,
  };

  // Node Collections
  let nodes = [];
  let stars = [];
  let pulses = [];
  let radarAngle = 0;

  const SATELLITE_NAMES = [
    { name: "SENTINEL-2A", band: "B8-NIR", alt: "786km" },
    { name: "LANDSAT-9", band: "OLI-2", alt: "705km" },
    { name: "RISAT-1A", band: "C-SAR", alt: "536km" },
    { name: "CARTOSAT-3", band: "PAN-0.28m", alt: "505km" },
    { name: "TERRA-MODIS", band: "SWIR-2.1µm", alt: "705km" },
    { name: "SENTINEL-1B", band: "C-Band SAR", alt: "693km" },
    { name: "WORLDVIEW-3", band: "VNIR-8B", alt: "617km" },
    { name: "PLANETSCOPE", band: "SuperDove", alt: "475km" },
  ];

  const COLOR_PALETTES = {
    orbital: {
      primary: "rgba(6, 182, 212, ",      // Cyan
      secondary: "rgba(16, 185, 129, ",   // Emerald
      accent: "rgba(56, 189, 248, ",      // Sky
      special: "rgba(168, 85, 247, ",     // Purple
      radar: "rgba(6, 182, 212, 0.12)",
      lineMaxAlpha: 0.35,
    },
    radar: {
      primary: "rgba(16, 185, 129, ",     // Emerald Green
      secondary: "rgba(52, 211, 153, ",
      accent: "rgba(6, 182, 212, ",
      special: "rgba(245, 158, 11, ",     // Amber
      radar: "rgba(16, 185, 129, 0.18)",
      lineMaxAlpha: 0.45,
    },
    matrix: {
      primary: "rgba(56, 189, 248, ",
      secondary: "rgba(168, 85, 247, ",
      accent: "rgba(244, 63, 94, ",
      special: "rgba(234, 179, 8, ",
      radar: "rgba(56, 189, 248, 0.1)",
      lineMaxAlpha: 0.3,
    },
    aurora: {
      primary: "rgba(168, 85, 247, ",     // Purple
      secondary: "rgba(6, 182, 212, ",
      accent: "rgba(244, 63, 94, ",       // Rose
      special: "rgba(16, 185, 129, ",
      radar: "rgba(168, 85, 247, 0.15)",
      lineMaxAlpha: 0.4,
    },
  };

  // Node Class
  class SatelliteNode {
    constructor() {
      this.reset(true);
    }

    reset(initial = false) {
      this.x = Math.random() * width;
      this.y = Math.random() * height;
      this.baseX = this.x;
      this.baseY = this.y;

      const speed = 0.25 + Math.random() * 0.45;
      const angle = Math.random() * Math.PI * 2;
      this.vx = Math.cos(angle) * speed;
      this.vy = Math.sin(angle) * speed;

      this.radius = 1.8 + Math.random() * 2.6;
      this.isSatellite = Math.random() < 0.28;
      this.pulsePhase = Math.random() * Math.PI * 2;
      this.pulseSpeed = 0.03 + Math.random() * 0.04;

      if (this.isSatellite) {
        this.satMeta = SATELLITE_NAMES[Math.floor(Math.random() * SATELLITE_NAMES.length)];
        this.radius = 3.2 + Math.random() * 1.5;
        this.orbitRadius = 20 + Math.random() * 40;
        this.orbitSpeed = 0.015 + Math.random() * 0.02;
        this.orbitAngle = Math.random() * Math.PI * 2;
      }

      this.type = Math.random() < 0.35 ? "sar" : Math.random() < 0.7 ? "optical" : "thermal";
      this.colorType = Math.random() < 0.5 ? "primary" : Math.random() < 0.8 ? "secondary" : "special";
    }

    update() {
      this.x += this.vx;
      this.y += this.vy;

      // Wrap around edges seamlessly
      if (this.x < -30) this.x = width + 30;
      if (this.x > width + 30) this.x = -30;
      if (this.y < -30) this.y = height + 30;
      if (this.y > height + 30) this.y = -30;

      this.pulsePhase += this.pulseSpeed;

      if (this.isSatellite) {
        this.orbitAngle += this.orbitSpeed;
      }

      // Mouse Proximity Gravitational Interaction
      if (mouse.active && CONFIG.interactive) {
        const dx = mouse.x - this.x;
        const dy = mouse.y - this.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < mouse.radius && dist > 0.1) {
          const force = (1 - dist / mouse.radius) * 1.2;
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;

          this.x += fx;
          this.y += fy;
        }
      }
    }

    draw(palette) {
      const pulseGlow = Math.sin(this.pulsePhase) * 0.35 + 0.65;
      const baseColor = palette[this.colorType] || palette.primary;

      ctx.save();

      if (this.isSatellite) {
        // Draw Stylized Satellite Body
        ctx.fillStyle = baseColor + (0.85 * pulseGlow) + ")";
        ctx.shadowColor = baseColor + "0.8)";
        ctx.shadowBlur = 10;

        // Core satellite diamond
        ctx.beginPath();
        ctx.moveTo(this.x, this.y - this.radius * 1.4);
        ctx.lineTo(this.x + this.radius * 1.4, this.y);
        ctx.lineTo(this.x, this.y + this.radius * 1.4);
        ctx.lineTo(this.x - this.radius * 1.4, this.y);
        ctx.closePath();
        ctx.fill();

        // Solar Array Wings
        ctx.strokeStyle = palette.accent + (0.7 * pulseGlow) + ")";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(this.x - this.radius * 3.2, this.y);
        ctx.lineTo(this.x + this.radius * 3.2, this.y);
        ctx.stroke();

        // Mini Solar Panels
        ctx.fillStyle = palette.secondary + "0.7)";
        ctx.fillRect(this.x - this.radius * 3.2, this.y - 2.5, 3.5, 5);
        ctx.fillRect(this.x + this.radius * 3.2 - 3.5, this.y - 2.5, 3.5, 5);

        // Orbital Ring
        ctx.strokeStyle = palette.primary + "0.15)";
        ctx.lineWidth = 0.75;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.orbitRadius, 0, Math.PI * 2);
        ctx.stroke();

        // Telemetry Sub-probe
        const probeX = this.x + Math.cos(this.orbitAngle) * this.orbitRadius;
        const probeY = this.y + Math.sin(this.orbitAngle) * this.orbitRadius;
        ctx.fillStyle = palette.special + "0.8)";
        ctx.beginPath();
        ctx.arc(probeX, probeY, 1.8, 0, Math.PI * 2);
        ctx.fill();

        // Satellite Name Callout (subtle)
        if (CONFIG.mode === "orbital" || CONFIG.mode === "radar") {
          ctx.font = "8px 'JetBrains Mono', monospace";
          ctx.fillStyle = palette.accent + (0.45 * pulseGlow) + ")";
          ctx.fillText(this.satMeta.name, this.x + 8, this.y - 6);
        }
      } else {
        // Normal Sensor Node
        ctx.fillStyle = baseColor + (0.75 * pulseGlow) + ")";
        ctx.shadowColor = baseColor + "0.5)";
        ctx.shadowBlur = 6;

        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fill();

        // Outer signal wave
        if (pulseGlow > 0.8) {
          ctx.strokeStyle = baseColor + ((1 - pulseGlow) * 0.5) + ")";
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.arc(this.x, this.y, this.radius * (1.8 + pulseGlow), 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      ctx.restore();
    }
  }

  // Multi-Spectral Star / Photon Particle
  class Star {
    constructor() {
      this.reset();
    }
    reset() {
      this.x = Math.random() * width;
      this.y = Math.random() * height;
      this.size = Math.random() * 1.3 + 0.3;
      this.alpha = Math.random() * 0.5 + 0.15;
      this.twinkleSpeed = Math.random() * 0.02 + 0.005;
      this.phase = Math.random() * Math.PI * 2;
    }
    update() {
      this.phase += this.twinkleSpeed;
    }
    draw(palette) {
      const a = (Math.sin(this.phase) * 0.5 + 0.5) * this.alpha;
      ctx.fillStyle = `rgba(226, 232, 240, ${a})`;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Interactive Click Sonar Wave
  class SonarPulse {
    constructor(x, y) {
      this.x = x;
      this.y = y;
      this.radius = 5;
      this.maxRadius = CONFIG.pulseMaxRadius;
      this.speed = CONFIG.pulseSpeed;
      this.alpha = 0.9;
      this.dead = false;
    }
    update() {
      this.radius += this.speed;
      this.alpha = Math.max(0, 0.9 * (1 - this.radius / this.maxRadius));
      if (this.radius >= this.maxRadius || this.alpha <= 0.01) {
        this.dead = true;
      }
    }
    draw(palette) {
      if (this.dead) return;
      ctx.save();
      ctx.lineWidth = 1.8;
      ctx.strokeStyle = palette.primary + this.alpha + ")";
      ctx.shadowColor = palette.primary + "0.6)";
      ctx.shadowBlur = 12;

      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.stroke();

      // Secondary echo ripple
      if (this.radius > 30) {
        ctx.lineWidth = 1.0;
        ctx.strokeStyle = palette.secondary + (this.alpha * 0.6) + ")";
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius * 0.65, 0, Math.PI * 2);
        ctx.stroke();
      }

      ctx.restore();
    }
  }

  // Initialize Canvas & Engine
  function init2DBackground() {
    canvas = document.getElementById("bg-canvas-2d");
    if (!canvas) {
      canvas = document.createElement("canvas");
      canvas.id = "bg-canvas-2d";
      canvas.className = "bg-canvas-2d";
      document.body.prepend(canvas);
    }

    ctx = canvas.getContext("2d");
    resize();

    window.addEventListener("resize", debounce(resize, 150));
    setupInteractionListeners();
    createNodes();
    createStars();
    injectFloatingHUDControl();

    if (!animationFrameId) {
      loop();
    }
  }

  function resize() {
    width = window.innerWidth;
    height = window.innerHeight;
    dpr = Math.min(window.devicePixelRatio || 1, 2);

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
  }

  function createNodes() {
    nodes = [];
    const count = Math.min(CONFIG.nodeCount, Math.floor((width * height) / 18000));
    for (let i = 0; i < count; i++) {
      nodes.push(new SatelliteNode());
    }
  }

  function createStars() {
    stars = [];
    for (let i = 0; i < CONFIG.starCount; i++) {
      stars.push(new Star());
    }
  }

  function setupInteractionListeners() {
    window.addEventListener("mousemove", (e) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
      mouse.active = true;
    });

    window.addEventListener("mouseenter", (e) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
      mouse.active = true;
    });

    window.addEventListener("mouseleave", () => {
      mouse.active = false;
      mouse.x = -1000;
      mouse.y = -1000;
    });

    // Interactive Sonar Shockwave on Click
    window.addEventListener("click", (e) => {
      // Avoid creating pulse if user clicked inside interactive input or select
      if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
      pulses.push(new SonarPulse(e.clientX, e.clientY));
      if (pulses.length > 8) pulses.shift();
    });

    // Touch Support for Mobile / Tablets
    window.addEventListener("touchstart", (e) => {
      if (e.touches && e.touches[0]) {
        mouse.x = e.touches[0].clientX;
        mouse.y = e.touches[0].clientY;
        mouse.active = true;
        pulses.push(new SonarPulse(mouse.x, mouse.y));
      }
    }, { passive: true });

    window.addEventListener("touchmove", (e) => {
      if (e.touches && e.touches[0]) {
        mouse.x = e.touches[0].clientX;
        mouse.y = e.touches[0].clientY;
        mouse.active = true;
      }
    }, { passive: true });

    window.addEventListener("touchend", () => {
      mouse.active = false;
    });
  }

  // Draw Connecting Telemetry Laser Beams
  function drawTelemetryConnections(palette) {
    const maxDist = CONFIG.maxDistance;
    const len = nodes.length;

    for (let i = 0; i < len; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < len; j++) {
        const b = nodes[j];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < maxDist) {
          const alpha = (1 - dist / maxDist) * palette.lineMaxAlpha;
          const isSpecialLink = a.isSatellite || b.isSatellite;

          ctx.beginPath();
          ctx.strokeStyle = isSpecialLink
            ? palette.primary + alpha + ")"
            : palette.accent + (alpha * 0.7) + ")";
          ctx.lineWidth = isSpecialLink ? 1.1 : 0.6;

          if (isSpecialLink && dist < maxDist * 0.6) {
            ctx.setLineDash([4, 4]);
          } else {
            ctx.setLineDash([]);
          }

          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }

      // Draw Laser Tether to Mouse Cursor
      if (mouse.active && CONFIG.interactive) {
        const dx = mouse.x - a.x;
        const dy = mouse.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < mouse.radius) {
          const mouseAlpha = (1 - dist / mouse.radius) * 0.65;
          ctx.beginPath();
          ctx.strokeStyle = palette.secondary + mouseAlpha + ")";
          ctx.lineWidth = 1.2;
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(mouse.x, mouse.y);
          ctx.stroke();

          // Reticle Target Ring on Close Satellites
          if (dist < 80) {
            ctx.strokeStyle = palette.primary + (mouseAlpha * 0.8) + ")";
            ctx.beginPath();
            ctx.arc(a.x, a.y, a.radius + 6, 0, Math.PI * 2);
            ctx.stroke();
          }
        }
      }
    }
  }

  // Draw Rotating Deep-Space Radar Sweep
  function drawRadarSweep(palette) {
    if (CONFIG.mode !== "radar" && CONFIG.mode !== "orbital") return;

    radarAngle += CONFIG.radarSpeed;
    if (radarAngle > Math.PI * 2) radarAngle -= Math.PI * 2;

    const centerX = width * 0.5;
    const centerY = height * 0.5;
    const sweepRadius = Math.max(width, height) * 0.85;

    ctx.save();
    ctx.translate(centerX, centerY);

    // Subtle sweeping cone
    const grad = ctx.createConicGradient(radarAngle, 0, 0);
    grad.addColorStop(0, "rgba(6, 182, 212, 0)");
    grad.addColorStop(0.85, "rgba(6, 182, 212, 0.005)");
    grad.addColorStop(0.98, palette.radar);
    grad.addColorStop(1, "rgba(6, 182, 212, 0)");

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(0, 0, sweepRadius, 0, Math.PI * 2);
    ctx.fill();

    // Leading high-tech sweep ray
    const rayX = Math.cos(radarAngle) * sweepRadius;
    const rayY = Math.sin(radarAngle) * sweepRadius;

    ctx.strokeStyle = palette.primary + "0.25)";
    ctx.lineWidth = 1.0;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(rayX, rayY);
    ctx.stroke();

    ctx.restore();
  }

  // Draw Interactive Mouse Cursor Target Reticle
  function drawMouseReticle(palette) {
    if (!mouse.active || !CONFIG.interactive) return;

    ctx.save();
    ctx.strokeStyle = palette.primary + "0.55)";
    ctx.lineWidth = 1.2;

    // Crosshairs
    ctx.beginPath();
    ctx.arc(mouse.x, mouse.y, 14, 0, Math.PI * 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(mouse.x - 20, mouse.y);
    ctx.lineTo(mouse.x - 16, mouse.y);
    ctx.moveTo(mouse.x + 16, mouse.y);
    ctx.lineTo(mouse.x + 20, mouse.y);
    ctx.moveTo(mouse.x, mouse.y - 20);
    ctx.lineTo(mouse.x, mouse.y - 16);
    ctx.moveTo(mouse.x, mouse.y + 16);
    ctx.lineTo(mouse.x, mouse.y + 20);
    ctx.stroke();

    // Coordinates Tag
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillStyle = palette.accent + "0.75)";
    ctx.fillText(`GEO: (${Math.round(mouse.x)}, ${Math.round(mouse.y)})`, mouse.x + 18, mouse.y - 8);
    ctx.restore();
  }

  // Floating Background HUD Control Badge
  function injectFloatingHUDControl() {
    if (document.getElementById("bg-2d-hud-control")) return;

    const hud = document.createElement("div");
    hud.id = "bg-2d-hud-control";
    hud.className = "bg-2d-hud-control";
    hud.innerHTML = `
      <div class="bg-hud-header" id="bg-hud-toggle-btn" title="Toggle 2D Interactive Background Settings">
        <span class="hud-pulse-dot"></span>
        <span class="hud-title">2D.JS ORBITAL</span>
        <span class="hud-mode-tag" id="bg-mode-label">ORBITAL</span>
      </div>
      <div class="bg-hud-panel hidden" id="bg-hud-panel">
        <div class="bg-hud-row">
          <span>Constellation Mode</span>
          <div class="bg-mode-btn-group">
            <button class="bg-btn active" data-mode="orbital">🛰️ Orbital</button>
            <button class="bg-btn" data-mode="radar">📡 Radar</button>
            <button class="bg-btn" data-mode="matrix">✨ Photons</button>
            <button class="bg-btn" data-mode="aurora">🌌 Aurora</button>
          </div>
        </div>
        <div class="bg-hud-row">
          <span>Node Density</span>
          <input type="range" id="bg-density-slider" min="30" max="100" value="65" />
        </div>
        <div class="bg-hud-row">
          <span>Radar Sonar Pulse</span>
          <button class="btn-hud-action" id="btn-trigger-pulse">Trigger Sonar Ping 💥</button>
        </div>
      </div>
    `;

    document.body.appendChild(hud);

    // Event handlers for HUD
    const toggleBtn = hud.querySelector("#bg-hud-toggle-btn");
    const panel = hud.querySelector("#bg-hud-panel");
    const modeLabel = hud.querySelector("#bg-mode-label");
    const modeBtns = hud.querySelectorAll(".bg-btn");
    const densitySlider = hud.querySelector("#bg-density-slider");
    const triggerPulseBtn = hud.querySelector("#btn-trigger-pulse");

    toggleBtn.addEventListener("click", () => {
      panel.classList.toggle("hidden");
    });

    modeBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        modeBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const mode = btn.getAttribute("data-mode");
        CONFIG.mode = mode;
        modeLabel.textContent = mode.toUpperCase();
      });
    });

    if (densitySlider) {
      densitySlider.addEventListener("input", (e) => {
        CONFIG.nodeCount = parseInt(e.target.value, 10);
        createNodes();
      });
    }

    if (triggerPulseBtn) {
      triggerPulseBtn.addEventListener("click", () => {
        pulses.push(new SonarPulse(width * 0.5, height * 0.5));
      });
    }
  }

  // Animation Loop (60 FPS)
  function loop() {
    ctx.clearRect(0, 0, width, height);

    const palette = COLOR_PALETTES[CONFIG.mode] || COLOR_PALETTES.orbital;

    // 1. Draw Starfield
    for (let i = 0; i < stars.length; i++) {
      stars[i].update();
      stars[i].draw(palette);
    }

    // 2. Draw Deep-Space Radar Sweep
    drawRadarSweep(palette);

    // 3. Draw Connecting Telemetry Beams
    drawTelemetryConnections(palette);

    // 4. Update and Draw Satellite Nodes
    for (let i = 0; i < nodes.length; i++) {
      nodes[i].update();
      nodes[i].draw(palette);
    }

    // 5. Update and Draw Interactive Sonar Shockwaves
    for (let i = pulses.length - 1; i >= 0; i--) {
      pulses[i].update();
      pulses[i].draw(palette);
      if (pulses[i].dead) {
        pulses.splice(i, 1);
      }
    }

    // 6. Draw Mouse Cursor Reticle
    drawMouseReticle(palette);

    animationFrameId = requestAnimationFrame(loop);
  }

  // Utility: Debounce
  function debounce(func, wait) {
    let timeout;
    return function (...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  }

  // Public API Export
  window.SatQuery2D = {
    setMode: (mode) => {
      if (COLOR_PALETTES[mode]) {
        CONFIG.mode = mode;
        const lbl = document.getElementById("bg-mode-label");
        if (lbl) lbl.textContent = mode.toUpperCase();
      }
    },
    triggerPulse: (x, y) => {
      pulses.push(new SonarPulse(x || width * 0.5, y || height * 0.5));
    },
    setDensity: (count) => {
      CONFIG.nodeCount = count;
      createNodes();
    },
  };

  // Start on DOM Ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init2DBackground);
  } else {
    init2DBackground();
  }
})();
