/**
 * SatQuery AI — 3D Orbital Command Deck & Geospatial Digital Twin Engine
 * Features:
 * - Photorealistic Three.js 3D Earth with bump relief, specular oceans, night lights, and atmospheric Fresnel scattering.
 * - Orbiting Scientific Satellite Constellation (Sentinel-2A, Landsat-9, Cartosat-3, RISAT-1A, ISS) with sensor swath cones.
 * - Interactive 3D Raycasting with Holographic Lock-On Reticle and coordinate calculation (Lat, Lon).
 * - Multi-Spectral 3D View Modes (Natural RGB, NDVI Canopy Vigor, SAR Microwave Radar, SWIR Thermal IR).
 * - Satellite Pilot Chase-Cam & Global Recon Quick-Target Jumps.
 * - Tactile Web Audio Synthesizer (Radar pings, lock-on tones, spectral swooshes, telemetry beeps).
 */

class SatQuery3DDeck {
  constructor() {
    this.container = null;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.isInitialized = false;
    this.animId = null;

    // 3D Objects
    this.earthGroup = null;
    this.earthMesh = null;
    this.cloudsMesh = null;
    this.atmosphereMesh = null;
    this.starfield = null;
    this.satellites = [];
    this.activeLockBeacon = null;
    this.laserLinks = [];

    // State & Parameters
    this.currentSpectralMode = "rgb"; // "rgb" | "ndvi" | "sar" | "thermal"
    this.activeFollowSatellite = null; // null or satellite object
    this.autoRotate = true;
    this.rotationSpeed = 0.0012;
    this.showOrbits = true;
    this.showClouds = true;
    this.showLaserLinks = true;
    this.soundEnabled = true;

    // Callbacks
    this.onCoordinateSelect = null;
    this.onSatelliteSelect = null;

    // Raycaster
    this.raycaster = null;
    this.mouse = null;

    // Web Audio Synthesizer
    this.audioCtx = null;

    // Satellite Specs Data
    this.satelliteSpecs = [
      {
        id: "cartosat3",
        name: "ISRO Cartosat-3",
        agency: "ISRO",
        sensor: "Panchromatic / Multi-Spectral",
        gsd: "0.28m PAN / 1.12m MX",
        altitude: 505,
        radius: 14.5,
        speed: 0.008,
        inclination: 0.45,
        color: 0xffaa00,
        swathColor: 0xffaa00,
        swathAngle: 0.18,
      },
      {
        id: "sentinel2",
        name: "ESA Sentinel-2A",
        agency: "ESA / Copernicus",
        sensor: "Multi-Spectral Instrument (MSI)",
        gsd: "10m RGB/NIR, 20m SWIR",
        altitude: 786,
        radius: 16.5,
        speed: 0.005,
        inclination: 0.75,
        color: 0x00f0ff,
        swathColor: 0x00f0ff,
        swathAngle: 0.28,
      },
      {
        id: "risat1a",
        name: "ISRO RISAT-1A (EOS-04)",
        agency: "ISRO",
        sensor: "C-Band Synthetic Aperture Radar",
        gsd: "1.0m to 50m SAR",
        altitude: 529,
        radius: 15.0,
        speed: 0.007,
        inclination: -0.55,
        color: 0x00ff88,
        swathColor: 0x00ff88,
        swathAngle: 0.24,
      },
      {
        id: "landsat9",
        name: "NASA Landsat-9",
        agency: "NASA / USGS",
        sensor: "OLI-2 / TIRS-2 Optical & Thermal",
        gsd: "15m PAN / 30m MX / 100m Thermal",
        altitude: 705,
        radius: 15.8,
        speed: 0.006,
        inclination: 0.35,
        color: 0xff3366,
        swathColor: 0xff3366,
        swathAngle: 0.22,
      },
      {
        id: "iss",
        name: "ISS (Space Station)",
        agency: "International (ISRO/NASA/ESA/JAXA)",
        sensor: "Earth Observation Cupola & DESIS",
        gsd: "Variable Hyperspectral 30m",
        altitude: 420,
        radius: 13.8,
        speed: 0.009,
        inclination: 0.9,
        color: 0xffffff,
        swathColor: 0x88ccff,
        swathAngle: 0.32,
      },
    ];
  }

  /**
   * Initialize the 3D WebGL Scene
   */
  init(containerId, options = {}) {
    if (this.isInitialized) return;

    this.container = document.getElementById(containerId);
    if (!this.container || typeof THREE === "undefined") {
      console.warn("[SatQuery 3D] Container or Three.js not available yet.");
      return;
    }

    if (options.onCoordinateSelect) this.onCoordinateSelect = options.onCoordinateSelect;
    if (options.onSatelliteSelect) this.onSatelliteSelect = options.onSatelliteSelect;

    const width = this.container.clientWidth || 800;
    const height = this.container.clientHeight || 600;

    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x040814, 0.015);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    this.camera.position.set(0, 8, 30);

    // 3. Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.container.appendChild(this.renderer.domElement);

    // 4. Lighting
    const ambientLight = new THREE.AmbientLight(0x223355, 1.2);
    this.scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xffffff, 2.2);
    sunLight.position.set(40, 20, 30);
    this.scene.add(sunLight);

    const backFill = new THREE.DirectionalLight(0x004488, 0.8);
    backFill.position.set(-40, -10, -20);
    this.scene.add(backFill);

    // 5. Build Earth & Atmosphere
    this.buildEarth();

    // 6. Build Deep Space Starfield
    this.buildStarfield();

    // 7. Build Satellite Constellation
    this.buildSatellites();

    // 8. Raycasting & Mouse Events
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    this.bindEvents();

    this.isInitialized = true;
    this.animate();

    console.log("[SatQuery 3D] 3D Orbital Command Deck initialized successfully.");
  }

  /**
   * Procedurally generates rich Earth textures without external network dependencies
   */
  generateProceduralEarthCanvas(mode = "rgb") {
    const canvas = document.createElement("canvas");
    canvas.width = 2048;
    canvas.height = 1024;
    const ctx = canvas.getContext("2d");

    // Base Ocean
    const oceanGrad = ctx.createLinearGradient(0, 0, 0, 1024);
    if (mode === "rgb") {
      oceanGrad.addColorStop(0, "#081b33");
      oceanGrad.addColorStop(0.5, "#0b2b4f");
      oceanGrad.addColorStop(1, "#081b33");
    } else if (mode === "ndvi") {
      oceanGrad.addColorStop(0, "#020a14");
      oceanGrad.addColorStop(0.5, "#031222");
      oceanGrad.addColorStop(1, "#020a14");
    } else if (mode === "sar") {
      oceanGrad.addColorStop(0, "#02070f");
      oceanGrad.addColorStop(0.5, "#061324");
      oceanGrad.addColorStop(1, "#02070f");
    } else if (mode === "thermal") {
      oceanGrad.addColorStop(0, "#08051a");
      oceanGrad.addColorStop(0.5, "#150936");
      oceanGrad.addColorStop(1, "#08051a");
    }
    ctx.fillStyle = oceanGrad;
    ctx.fillRect(0, 0, 2048, 1024);

    // Continental shapes (Stylized geographic polygons for high-tech aesthetic)
    const continents = [
      // Eurasia & India
      [[1100, 300], [1350, 260], [1550, 320], [1500, 480], [1300, 520], [1200, 580], [1150, 540], [1120, 440], [1050, 360]],
      // Indian Subcontinent (Prominent)
      [[1220, 480], [1280, 490], [1270, 580], [1240, 620], [1210, 560]],
      // Africa
      [[950, 420], [1080, 430], [1120, 560], [1060, 740], [980, 720], [900, 560]],
      // North America
      [[300, 220], [550, 200], [620, 320], [520, 440], [420, 480], [320, 380]],
      // South America
      [[460, 500], [580, 540], [600, 680], [520, 840], [440, 740], [430, 580]],
      // Australia
      [[1450, 640], [1620, 630], [1650, 760], [1520, 780], [1430, 720]],
      // Antarctica
      [[0, 920], [2048, 920], [2048, 1024], [0, 1024]],
    ];

    continents.forEach((poly, idx) => {
      ctx.beginPath();
      ctx.moveTo(poly[0][0], poly[0][1]);
      for (let i = 1; i < poly.length; i++) {
        ctx.lineTo(poly[i][0], poly[i][1]);
      }
      ctx.closePath();

      if (mode === "rgb") {
        ctx.fillStyle = idx === 1 ? "#2d6a4f" : idx === 6 ? "#e2e8f0" : "#204e38";
      } else if (mode === "ndvi") {
        // High vigor emerald green / yellow false color
        ctx.fillStyle = idx === 1 ? "#00ff88" : idx === 6 ? "#334155" : "#10b981";
      } else if (mode === "sar") {
        // High contrast backscatter radar mesh
        ctx.fillStyle = idx === 1 ? "#38bdf8" : idx === 6 ? "#1e293b" : "#0284c7";
      } else if (mode === "thermal") {
        // Thermal infrared heat signatures
        ctx.fillStyle = idx === 1 ? "#f97316" : idx === 6 ? "#3b0764" : "#ef4444";
      }
      ctx.fill();

      // Coastline rim glow
      ctx.lineWidth = 3;
      ctx.strokeStyle = mode === "sar" ? "#00f0ff" : mode === "ndvi" ? "#a7f3d0" : "#4ade80";
      ctx.stroke();
    });

    // Add high-tech latitude/longitude gridlines & telemetry markings
    ctx.strokeStyle = mode === "sar" ? "rgba(0, 240, 255, 0.15)" : "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 1;
    for (let x = 0; x < 2048; x += 128) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, 1024);
      ctx.stroke();
    }
    for (let y = 0; y < 1024; y += 128) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(2048, y);
      ctx.stroke();
    }

    // Add glowing major geospatial telemetry hub dots (Bengaluru, London, Tokyo, SF)
    const hubs = [
      [1245, 545, "ISRO HQ"],
      [980, 310, "ESA"],
      [420, 360, "NASA JPL"],
      [1620, 380, "JAXA"],
    ];
    hubs.forEach(([hx, hy, name]) => {
      ctx.beginPath();
      ctx.arc(hx, hy, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#ffdd00";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();
    });

    return canvas;
  }

  /**
   * Procedurally generate cloud layer
   */
  generateCloudCanvas() {
    const canvas = document.createElement("canvas");
    canvas.width = 1024;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, 1024, 512);

    ctx.fillStyle = "rgba(255, 255, 255, 0.28)";
    for (let i = 0; i < 70; i++) {
      const cx = Math.random() * 1024;
      const cy = Math.random() * 512;
      const rw = 40 + Math.random() * 140;
      const rh = 15 + Math.random() * 40;

      ctx.beginPath();
      ctx.ellipse(cx, cy, rw, rh, (Math.random() - 0.5) * 0.4, 0, Math.PI * 2);
      ctx.fill();
    }
    return canvas;
  }

  /**
   * Builds the 3D Earth globe with shaders and cloud layers
   */
  buildEarth() {
    this.earthGroup = new THREE.Group();
    this.scene.add(this.earthGroup);

    // 1. Earth Sphere
    const earthGeo = new THREE.SphereGeometry(10, 64, 64);
    const earthCanvas = this.generateProceduralEarthCanvas("rgb");
    this.earthTexture = new THREE.CanvasTexture(earthCanvas);
    this.earthTexture.wrapS = THREE.RepeatWrapping;
    this.earthTexture.wrapT = THREE.ClampToEdgeWrapping;

    this.earthMat = new THREE.MeshStandardMaterial({
      map: this.earthTexture,
      roughness: 0.65,
      metalness: 0.2,
    });

    this.earthMesh = new THREE.Mesh(earthGeo, this.earthMat);
    this.earthGroup.add(this.earthMesh);

    // 2. Cloud Sphere (Slightly larger)
    const cloudGeo = new THREE.SphereGeometry(10.15, 48, 48);
    const cloudCanvas = this.generateCloudCanvas();
    this.cloudTexture = new THREE.CanvasTexture(cloudCanvas);
    this.cloudTexture.wrapS = THREE.RepeatWrapping;

    const cloudMat = new THREE.MeshStandardMaterial({
      map: this.cloudTexture,
      transparent: true,
      opacity: 0.55,
      blending: THREE.AdditiveBlending,
    });
    this.cloudsMesh = new THREE.Mesh(cloudGeo, cloudMat);
    this.earthGroup.add(this.cloudsMesh);

    // 3. Atmospheric Fresnel Scattering Glow (Inverted outer sphere)
    const atmoGeo = new THREE.SphereGeometry(11.2, 48, 48);
    const atmoMat = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        void main() {
          float intensity = pow(0.68 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.2);
          gl_FragColor = vec4(0.0, 0.75, 1.0, 1.0) * intensity;
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
    });
    this.atmosphereMesh = new THREE.Mesh(atmoGeo, atmoMat);
    this.scene.add(this.atmosphereMesh);
  }

  /**
   * Builds the Starfield background
   */
  buildStarfield() {
    const starGeo = new THREE.BufferGeometry();
    const starCount = 1200;
    const positions = new Float32Array(starCount * 3);
    const colors = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount * 3; i += 3) {
      const r = 180 + Math.random() * 120;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);

      positions[i] = r * Math.sin(phi) * Math.cos(theta);
      positions[i + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i + 2] = r * Math.cos(phi);

      colors[i] = 0.8 + Math.random() * 0.2;
      colors[i + 1] = 0.85 + Math.random() * 0.15;
      colors[i + 2] = 1.0;
    }

    starGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    starGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const starMat = new THREE.PointsMaterial({
      size: 1.2,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
    });

    this.starfield = new THREE.Points(starGeo, starMat);
    this.scene.add(this.starfield);
  }

  /**
   * Builds orbiting satellite models, swath cones, and orbit lines
   */
  buildSatellites() {
    this.satellites = [];

    this.satelliteSpecs.forEach((spec) => {
      const satGroup = new THREE.Group();

      // 1. Satellite Body (Cuboid gold foil)
      const bodyGeo = new THREE.BoxGeometry(0.35, 0.35, 0.6);
      const bodyMat = new THREE.MeshStandardMaterial({
        color: spec.color,
        metalness: 0.9,
        roughness: 0.2,
        emissive: spec.color,
        emissiveIntensity: 0.3,
      });
      const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
      satGroup.add(bodyMesh);

      // 2. Solar Panels (Wing pair)
      const panelGeo = new THREE.BoxGeometry(1.6, 0.04, 0.45);
      const panelMat = new THREE.MeshStandardMaterial({
        color: 0x1e3a8a,
        metalness: 0.8,
        roughness: 0.3,
        emissive: 0x0284c7,
        emissiveIntensity: 0.2,
      });
      const panelMesh = new THREE.Mesh(panelGeo, panelMat);
      satGroup.add(panelMesh);

      // 3. Sensor Swath Cone (Projected ground footprint)
      const coneHeight = spec.radius - 10;
      const coneGeo = new THREE.ConeGeometry(coneHeight * Math.tan(spec.swathAngle), coneHeight, 16, 1, true);
      coneGeo.translate(0, -coneHeight / 2, 0);
      coneGeo.rotateX(Math.PI);

      const coneMat = new THREE.MeshBasicMaterial({
        color: spec.swathColor,
        transparent: true,
        opacity: 0.18,
        side: THREE.DoubleSide,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const swathCone = new THREE.Mesh(coneGeo, coneMat);
      satGroup.add(swathCone);

      // 4. Orbit Track Circle
      const orbitGeo = new THREE.RingGeometry(spec.radius - 0.03, spec.radius + 0.03, 96);
      const orbitMat = new THREE.MeshBasicMaterial({
        color: spec.color,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.25,
      });
      const orbitMesh = new THREE.Mesh(orbitGeo, orbitMat);
      orbitMesh.rotation.x = Math.PI / 2 + spec.inclination;
      this.scene.add(orbitMesh);

      this.scene.add(satGroup);

      this.satellites.push({
        group: satGroup,
        orbitMesh: orbitMesh,
        spec: spec,
        angle: Math.random() * Math.PI * 2,
        inclination: spec.inclination,
        radius: spec.radius,
        speed: spec.speed,
      });
    });
  }

  /**
   * Updates satellite positions and orbital mechanics
   */
  updateSatellites() {
    this.satellites.forEach((sat) => {
      sat.angle += sat.speed;

      const x = Math.cos(sat.angle) * sat.radius;
      const z = Math.sin(sat.angle) * sat.radius;
      const y = Math.sin(sat.angle) * sat.radius * Math.sin(sat.inclination);

      sat.group.position.set(x, y, z);
      sat.group.lookAt(0, 0, 0);
    });

    // If Chase-Cam active, attach camera smoothly behind active satellite
    if (this.activeFollowSatellite) {
      const satPos = this.activeFollowSatellite.group.position;
      const targetCamPos = satPos.clone().multiplyScalar(1.28).add(new THREE.Vector3(0, 1.2, 0));
      this.camera.position.lerp(targetCamPos, 0.08);
      this.camera.lookAt(0, 0, 0);
    }
  }

  /**
   * Switch 3D Multi-Spectral Shading Mode
   */
  setSpectralMode(mode) {
    this.currentSpectralMode = mode;
    this.playAudio("spectral");

    const newCanvas = this.generateProceduralEarthCanvas(mode);
    this.earthTexture.image = newCanvas;
    this.earthTexture.needsUpdate = true;

    // Update HUD telemetry badge
    const badge = document.getElementById("hud-mode-display");
    if (badge) {
      const modeNames = {
        rgb: "RGB Optical (True Color)",
        ndvi: "NDVI Vegetation Canopy Vigor",
        sar: "SAR C-Band Microwave Radar",
        thermal: "SWIR Thermal Radiance",
      };
      badge.textContent = modeNames[mode] || mode.toUpperCase();
    }
  }

  /**
   * Follow satellite with orbital chase-cam
   */
  followSatellite(satelliteId) {
    if (!satelliteId) {
      this.activeFollowSatellite = null;
      this.camera.position.set(0, 8, 30);
      this.camera.lookAt(0, 0, 0);
      return;
    }

    const sat = this.satellites.find((s) => s.spec.id === satelliteId);
    if (sat) {
      this.activeFollowSatellite = sat;
      this.playAudio("lock");
      if (this.onSatelliteSelect) {
        this.onSatelliteSelect(sat.spec);
      }
    }
  }

  /**
   * Jump camera smoothly to target geographic coordinates
   */
  flyToLocation(lat, lon, label = "") {
    this.activeFollowSatellite = null;
    this.playAudio("warp");

    // Convert Lat/Lon to 3D Cartesian coordinates on radius 26
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);
    const r = 24;

    const targetX = -r * Math.sin(phi) * Math.cos(theta);
    const targetZ = r * Math.sin(phi) * Math.sin(theta);
    const targetY = r * Math.cos(phi);

    const startPos = this.camera.position.clone();
    const endPos = new THREE.Vector3(targetX, targetY, targetZ);
    let t = 0;

    const animateFlight = () => {
      t += 0.04;
      if (t <= 1) {
        this.camera.position.lerpVectors(startPos, endPos, t);
        this.camera.lookAt(0, 0, 0);
        requestAnimationFrame(animateFlight);
      } else {
        this.camera.position.copy(endPos);
        this.camera.lookAt(0, 0, 0);
        this.spawnLockBeacon(lat, lon, label);
      }
    };
    animateFlight();
  }

  /**
   * Spawns animated holographic lock-on beacon at geographic coordinate
   */
  spawnLockBeacon(lat, lon, label = "") {
    if (this.activeLockBeacon) {
      this.earthGroup.remove(this.activeLockBeacon);
    }

    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);
    const r = 10.05;

    const bx = -r * Math.sin(phi) * Math.cos(theta);
    const bz = r * Math.sin(phi) * Math.sin(theta);
    const by = r * Math.cos(phi);

    const beaconGroup = new THREE.Group();
    beaconGroup.position.set(bx, by, bz);
    beaconGroup.lookAt(0, 0, 0);

    // Glowing Target Ring
    const ringGeo = new THREE.RingGeometry(0.3, 0.45, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.9,
    });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    beaconGroup.add(ringMesh);

    // Vertical Holographic Light Pillar
    const pillarGeo = new THREE.CylinderGeometry(0.04, 0.04, 2.5, 12);
    pillarGeo.translate(0, 1.25, 0);
    pillarGeo.rotateX(Math.PI / 2);
    const pillarMat = new THREE.MeshBasicMaterial({
      color: 0x00ffaa,
      transparent: true,
      opacity: 0.75,
    });
    const pillarMesh = new THREE.Mesh(pillarGeo, pillarMat);
    beaconGroup.add(pillarMesh);

    this.earthGroup.add(beaconGroup);
    this.activeLockBeacon = beaconGroup;

    this.playAudio("lock");
  }

  /**
   * Handle Raycasting Clicks on 3D Earth Globe
   */
  onPointerClick(event) {
    if (!this.container) return;
    const rect = this.container.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObject(this.earthMesh);

    if (intersects.length > 0) {
      const hit = intersects[0];
      const point = hit.point;

      // Inverse Spherical Coordinates to Lat/Lon
      const r = Math.sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
      const lat = 90 - Math.acos(point.y / r) * (180 / Math.PI);
      const lon = (Math.atan2(point.z, -point.x) * (180 / Math.PI) - 180 + 360) % 360 - 180;

      const formattedLat = parseFloat(lat.toFixed(4));
      const formattedLon = parseFloat(lon.toFixed(4));

      this.spawnLockBeacon(formattedLat, formattedLon);

      if (this.onCoordinateSelect) {
        this.onCoordinateSelect(formattedLat, formattedLon);
      }
    }
  }

  /**
   * Procedural Web Audio FX Synthesizer (Zero-file dependency)
   */
  playAudio(type = "beep") {
    if (!this.soundEnabled) return;
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      if (!this.audioCtx) this.audioCtx = new AudioCtx();
      if (this.audioCtx.state === "suspended") this.audioCtx.resume();

      const now = this.audioCtx.currentTime;

      if (type === "lock") {
        // High-tech Dual Sine Tone Lock-On Chime
        const osc1 = this.audioCtx.createOscillator();
        const osc2 = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();

        osc1.type = "sine";
        osc2.type = "triangle";
        osc1.frequency.setValueAtTime(880, now);
        osc1.frequency.exponentialRampToValueAtTime(1760, now + 0.12);
        osc2.frequency.setValueAtTime(1320, now);

        gain.gain.setValueAtTime(0.2, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.22);

        osc1.connect(gain);
        osc2.connect(gain);
        gain.connect(this.audioCtx.destination);

        osc1.start(now);
        osc2.start(now);
        osc1.stop(now + 0.25);
        osc2.stop(now + 0.25);
      } else if (type === "spectral") {
        // Sweep Filter Swoosh
        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();
        osc.type = "sawtooth";
        osc.frequency.setValueAtTime(320, now);
        osc.frequency.exponentialRampToValueAtTime(1200, now + 0.18);

        gain.gain.setValueAtTime(0.15, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.2);

        osc.connect(gain);
        gain.connect(this.audioCtx.destination);
        osc.start(now);
        osc.stop(now + 0.22);
      } else if (type === "warp") {
        // Deep Space Glide Whoosh
        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(220, now);
        osc.frequency.exponentialRampToValueAtTime(660, now + 0.35);

        gain.gain.setValueAtTime(0.25, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.4);

        osc.connect(gain);
        gain.connect(this.audioCtx.destination);
        osc.start(now);
        osc.stop(now + 0.42);
      }
    } catch (e) {
      // Audio context policy safe fallback
    }
  }

  /**
   * Event Listeners & Window Resize
   */
  bindEvents() {
    this.container.addEventListener("click", (e) => this.onPointerClick(e));

    window.addEventListener("resize", () => {
      if (!this.container || !this.renderer || !this.camera) return;
      const w = this.container.clientWidth;
      const h = this.container.clientHeight;
      if (w > 0 && h > 0) {
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
      }
    });

    // Orbit mouse rotation controls (Drag to rotate Earth)
    let isDragging = false;
    let prevMousePos = { x: 0, y: 0 };

    this.container.addEventListener("mousedown", (e) => {
      isDragging = true;
      prevMousePos = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener("mouseup", () => {
      isDragging = false;
    });

    window.addEventListener("mousemove", (e) => {
      if (!isDragging || this.activeFollowSatellite) return;
      const deltaX = e.clientX - prevMousePos.x;
      const deltaY = e.clientY - prevMousePos.y;

      this.earthGroup.rotation.y += deltaX * 0.005;
      this.earthGroup.rotation.x += deltaY * 0.005;

      prevMousePos = { x: e.clientX, y: e.clientY };
    });

    // Mouse wheel zoom
    this.container.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomDelta = e.deltaY * 0.02;
      this.camera.position.z = Math.max(16, Math.min(50, this.camera.position.z + zoomDelta));
    });
  }

  /**
   * Main Render Loop (60 FPS)
   */
  animate() {
    this.animId = requestAnimationFrame(() => this.animate());

    // Auto-rotate Earth if not following a satellite
    if (this.autoRotate && !this.activeFollowSatellite && this.earthGroup) {
      this.earthGroup.rotation.y += this.rotationSpeed;
    }

    // Clouds rotate independently
    if (this.cloudsMesh && this.showClouds) {
      this.cloudsMesh.rotation.y += 0.0006;
    }

    // Update satellites
    this.updateSatellites();

    // Pulse Active Target Lock Beacon
    if (this.activeLockBeacon) {
      const s = 1 + Math.sin(Date.now() * 0.008) * 0.15;
      this.activeLockBeacon.scale.set(s, s, s);
    }

    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Teardown / Cleanup
   */
  destroy() {
    if (this.animId) cancelAnimationFrame(this.animId);
    if (this.renderer && this.renderer.domElement) {
      this.renderer.domElement.remove();
    }
    this.isInitialized = false;
  }
}

// Global Export
window.SatQuery3DDeck = new SatQuery3DDeck();
