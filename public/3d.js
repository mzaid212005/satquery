/**
 * SatQuery AI — Photorealistic Space Station & Cockpit Digital Twin Engine
 * Cinematic 3D space station mission control module inspired by Interstellar.
 * 
 * Features:
 * - Realistic 3D Spacecraft Interior Hull: Textured metallic bulkheads, matte titanium panels, structural ribs, cockpit console table, and interior warm/cool accent lights.
 * - Realistic Earth Observation Viewport: Curved cockpit window showing photorealistic 3D Earth (bump relief, specular oceans, dynamic atmospheric Rayleigh limb glow, independent cloud layer, sun flare, and starfield).
 * - Multi-Monitor Physical Workstation Mounts with realistic bezels, hardware status LEDs, and console controls.
 * - Cinematic Camera Flight Choreography:
 *    * Airlock Entry Position (Pre-Authentication)
 *    * Warp Glide through airlock into the pilot's command seat (Post-Authentication)
 *    * Realistic cockpit breathing parallax and observation modes
 * - Procedural Web Audio Synthesizer (Airlock hiss, computer boot chime, tactile telemetry chirps).
 */

class SatQuery3DDeck {
  constructor() {
    this.container = null;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.isInitialized = false;
    this.animId = null;

    // 3D Environment Groups
    this.spaceGroup = null;
    this.earthGroup = null;
    this.earthMesh = null;
    this.cloudsMesh = null;
    this.atmosphereMesh = null;
    this.starfield = null;
    this.cockpitGroup = null;
    this.cockpitLights = [];
    this.satellites = [];
    this.activeLockBeacon = null;

    // Camera Kinematics
    this.cameraState = "AIRLOCK"; // "AIRLOCK" | "TRANSITION" | "WORKSTATION" | "VIEWPORT"
    this.cameraTargetPos = new THREE.Vector3(0, 1.2, 14);
    this.cameraTargetLook = new THREE.Vector3(0, 0, 0);
    this.camTransitionProgress = 1;
    this.camStartPos = new THREE.Vector3();
    this.camEndPos = new THREE.Vector3();
    this.camStartLook = new THREE.Vector3();
    this.camEndLook = new THREE.Vector3();

    // Workstation & Cockpit Positions
    this.posAirlock = { pos: new THREE.Vector3(0, 1.4, 13.5), look: new THREE.Vector3(0, 0, -5) };
    this.posWorkstation = { pos: new THREE.Vector3(0, 0.45, 2.6), look: new THREE.Vector3(0, 0.25, -2) };
    this.posViewport = { pos: new THREE.Vector3(0, 1.8, -4.5), look: new THREE.Vector3(0, 0, -25) };

    // Mouse Parallax
    this.mouse = { x: 0, y: 0, targetX: 0, targetY: 0 };

    // Audio Context
    this.audioCtx = null;
    this.soundEnabled = true;

    // Callbacks
    this.onCoordinateSelect = null;
    this.onSatelliteSelect = null;

    // Satellite Specs
    this.satelliteSpecs = [
      { id: "cartosat3", name: "ISRO Cartosat-3", agency: "ISRO", altitude: 505, radius: 24, speed: 0.005, inclination: 0.45, color: 0xffaa00 },
      { id: "sentinel2", name: "ESA Sentinel-2A", agency: "ESA", altitude: 786, radius: 26, speed: 0.004, inclination: 0.75, color: 0x00f0ff },
      { id: "risat1a", name: "ISRO RISAT-1A (SAR)", agency: "ISRO", altitude: 529, radius: 25, speed: 0.0045, inclination: -0.55, color: 0x00ff88 },
      { id: "landsat9", name: "NASA Landsat-9", agency: "NASA", altitude: 705, radius: 27, speed: 0.0035, inclination: 0.35, color: 0xff3366 },
      { id: "iss", name: "ISS (Space Station Alpha)", agency: "International", altitude: 420, radius: 22, speed: 0.006, inclination: 0.85, color: 0xffffff },
    ];
  }

  /**
   * Initialize the 3D Canvas & WebGL Scene
   */
  init(containerId = "bg-canvas-3d-wrapper", options = {}) {
    if (this.isInitialized) return;

    this.container = document.getElementById(containerId);
    if (!this.container) {
      this.container = document.createElement("div");
      this.container.id = containerId;
      this.container.className = "global-3d-viewport-canvas";
      document.body.prepend(this.container);
    }

    if (options.onCoordinateSelect) this.onCoordinateSelect = options.onCoordinateSelect;
    if (options.onSatelliteSelect) this.onSatelliteSelect = options.onSatelliteSelect;

    const width = window.innerWidth || 1280;
    const height = window.innerHeight || 800;

    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x02050e, 0.012);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 2000);
    this.camera.position.copy(this.posAirlock.pos);
    this.camera.lookAt(this.posAirlock.look);

    // 3. WebGL Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.renderer.outputEncoding = THREE.sRGBEncoding;
    this.container.appendChild(this.renderer.domElement);

    // 4. Photorealistic Space Environment & Earth
    this.buildDeepSpace();

    // 5. Realistic Physical Spacecraft Cockpit Architecture
    this.buildCockpitInterior();

    // 6. Satellites
    this.buildSatellites();

    // 7. Event Handlers
    this.bindEvents();

    this.isInitialized = true;
    this.animate();

    console.log("[SatQuery 3D] Photorealistic Space Station & Cockpit initialized.");
  }

  /**
   * Build Deep Space Starfield, Lighting & 3D Earth Globe
   */
  buildDeepSpace() {
    this.spaceGroup = new THREE.Group();
    this.spaceGroup.position.set(0, 0, -35); // Position Earth outside the cockpit viewport
    this.scene.add(this.spaceGroup);

    // 1. Deep Starfield
    const starGeo = new THREE.BufferGeometry();
    const starCount = 2000;
    const positions = new Float32Array(starCount * 3);
    const colors = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount * 3; i += 3) {
      const r = 300 + Math.random() * 200;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);

      positions[i] = r * Math.sin(phi) * Math.cos(theta);
      positions[i + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i + 2] = r * Math.cos(phi);

      const brightness = 0.7 + Math.random() * 0.3;
      colors[i] = brightness;
      colors[i + 1] = brightness * (0.9 + Math.random() * 0.1);
      colors[i + 2] = brightness * (1.0);
    }

    starGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    starGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const starMat = new THREE.PointsMaterial({
      size: 1.5,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
    });

    this.starfield = new THREE.Points(starGeo, starMat);
    this.scene.add(this.starfield);

    // 2. Solar Directional Light
    const sunLight = new THREE.DirectionalLight(0xffffff, 2.5);
    sunLight.position.set(60, 40, 50);
    this.scene.add(sunLight);

    const spaceAmbient = new THREE.AmbientLight(0x0a1428, 1.0);
    this.scene.add(spaceAmbient);

    // 3. Earth Group
    this.earthGroup = new THREE.Group();
    this.spaceGroup.add(this.earthGroup);

    // Procedural High-Res Earth Texture
    const earthCanvas = this.generatePhotorealisticEarthCanvas();
    const earthTexture = new THREE.CanvasTexture(earthCanvas);
    earthTexture.wrapS = THREE.RepeatWrapping;

    const earthGeo = new THREE.SphereGeometry(14, 64, 64);
    const earthMat = new THREE.MeshStandardMaterial({
      map: earthTexture,
      roughness: 0.6,
      metalness: 0.15,
    });

    this.earthMesh = new THREE.Mesh(earthGeo, earthMat);
    this.earthGroup.add(this.earthMesh);

    // Cloud Layer
    const cloudCanvas = this.generatePhotorealisticCloudCanvas();
    const cloudTexture = new THREE.CanvasTexture(cloudCanvas);
    cloudTexture.wrapS = THREE.RepeatWrapping;

    const cloudGeo = new THREE.SphereGeometry(14.2, 48, 48);
    const cloudMat = new THREE.MeshStandardMaterial({
      map: cloudTexture,
      transparent: true,
      opacity: 0.45,
      blending: THREE.AdditiveBlending,
    });
    this.cloudsMesh = new THREE.Mesh(cloudGeo, cloudMat);
    this.earthGroup.add(this.cloudsMesh);

    // Atmospheric Rayleigh Scattering Glow
    const atmoGeo = new THREE.SphereGeometry(15.4, 48, 48);
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
          float intensity = pow(0.62 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.8);
          gl_FragColor = vec4(0.2, 0.65, 1.0, 1.0) * intensity;
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
    });
    this.atmosphereMesh = new THREE.Mesh(atmoGeo, atmoMat);
    this.earthGroup.add(this.atmosphereMesh);
  }

  /**
   * Procedural Photorealistic Earth Canvas
   */
  generatePhotorealisticEarthCanvas() {
    const canvas = document.createElement("canvas");
    canvas.width = 2048;
    canvas.height = 1024;
    const ctx = canvas.getContext("2d");

    // Deep Specular Oceans
    const oceanGrad = ctx.createLinearGradient(0, 0, 0, 1024);
    oceanGrad.addColorStop(0, "#061324");
    oceanGrad.addColorStop(0.2, "#0a2240");
    oceanGrad.addColorStop(0.5, "#0d3158");
    oceanGrad.addColorStop(0.8, "#0a2240");
    oceanGrad.addColorStop(1, "#061324");
    ctx.fillStyle = oceanGrad;
    ctx.fillRect(0, 0, 2048, 1024);

    // Continents & Landmasses
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

      // Realistic vegetation / land-cover tones
      if (idx === 1) {
        // India (Lush agricultural green & Deccan plateau)
        ctx.fillStyle = "#2d5a3f";
      } else if (idx === 6) {
        // Antarctica (Ice cap)
        ctx.fillStyle = "#e2e8f0";
      } else if (idx === 2) {
        // Africa (Savanna / Sahara beige to central forest)
        ctx.fillStyle = "#4a4e32";
      } else {
        ctx.fillStyle = "#274834";
      }
      ctx.fill();

      // Subtle coastline elevation rim
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(74, 222, 128, 0.4)";
      ctx.stroke();
    });

    // Night-side city lights dots
    ctx.fillStyle = "rgba(255, 230, 140, 0.75)";
    const cityLights = [
      [1245, 545], [1235, 520], [1250, 500], // India hubs
      [980, 320], [1010, 310], [960, 340],  // Europe hubs
      [420, 360], [450, 380], [360, 340],  // US hubs
      [1620, 380], [1580, 420],             // East Asia hubs
    ];
    cityLights.forEach(([cx, cy]) => {
      ctx.beginPath();
      ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
      ctx.fill();
    });

    return canvas;
  }

  /**
   * Procedural Photorealistic Cloud Texture
   */
  generatePhotorealisticCloudCanvas() {
    const canvas = document.createElement("canvas");
    canvas.width = 1024;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, 1024, 512);

    ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
    for (let i = 0; i < 90; i++) {
      const cx = Math.random() * 1024;
      const cy = Math.random() * 512;
      const rw = 50 + Math.random() * 160;
      const rh = 18 + Math.random() * 50;

      ctx.beginPath();
      ctx.ellipse(cx, cy, rw, rh, (Math.random() - 0.5) * 0.3, 0, Math.PI * 2);
      ctx.fill();
    }
    return canvas;
  }

  /**
   * Builds the Realistic Physical Spacecraft Cockpit & Command Deck
   */
  buildCockpitInterior() {
    this.cockpitGroup = new THREE.Group();
    this.scene.add(this.cockpitGroup);

    // 1. Cockpit Hull Materials (Matte Titanium, Carbon Fiber & Dark Aluminum)
    const hullMat = new THREE.MeshStandardMaterial({
      color: 0x111622,
      roughness: 0.75,
      metalness: 0.65,
    });

    const frameMat = new THREE.MeshStandardMaterial({
      color: 0x1c2436,
      roughness: 0.6,
      metalness: 0.8,
    });

    const metalTrimMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      roughness: 0.4,
      metalness: 0.9,
    });

    // 2. Cockpit Structural Arch & Cylindrical Pressure Hull
    const hullGeo = new THREE.CylinderGeometry(8, 8, 28, 32, 1, true, -Math.PI / 2, Math.PI);
    hullGeo.rotateZ(Math.PI / 2);
    hullGeo.translate(0, 2, 0);
    const hullMesh = new THREE.Mesh(hullGeo, hullMat);
    this.cockpitGroup.add(hullMesh);

    // 3. Structural Bulkhead Ribs
    for (let z = -8; z <= 12; z += 4) {
      const ribGeo = new THREE.TorusGeometry(7.8, 0.18, 12, 32, Math.PI);
      ribGeo.rotateX(Math.PI / 2);
      ribGeo.translate(0, 2, z);
      const ribMesh = new THREE.Mesh(ribGeo, frameMat);
      this.cockpitGroup.add(ribMesh);
    }

    // 4. Realistic Forward Observation Cupola / Viewport Window Frame
    const cupolaRingGeo = new THREE.RingGeometry(4.8, 5.6, 24);
    cupolaRingGeo.translate(0, 1.8, -10.5);
    const cupolaRing = new THREE.Mesh(cupolaRingGeo, frameMat);
    this.cockpitGroup.add(cupolaRing);

    // Viewport Glass with Subtle Reflection
    const glassGeo = new THREE.CircleGeometry(4.8, 24);
    glassGeo.translate(0, 1.8, -10.45);
    const glassMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      roughness: 0.1,
      metalness: 0.95,
      transparent: true,
      opacity: 0.12,
    });
    const glassMesh = new THREE.Mesh(glassGeo, glassMat);
    this.cockpitGroup.add(glassMesh);

    // 5. Cockpit Command Desk / Workstation Console Table
    const consoleGeo = new THREE.BoxGeometry(6.4, 0.8, 2.2);
    consoleGeo.translate(0, -0.4, 0.8);
    const consoleTable = new THREE.Mesh(consoleGeo, frameMat);
    this.cockpitGroup.add(consoleTable);

    // Metallic Upper Console Dashboard
    const dashGeo = new THREE.BoxGeometry(6.2, 0.25, 0.8);
    dashGeo.rotateX(-Math.PI / 8);
    dashGeo.translate(0, 0.05, 0.2);
    const dashMesh = new THREE.Mesh(dashGeo, metalTrimMat);
    this.cockpitGroup.add(dashMesh);

    // Physical Hardware Status LED Indicators on Console
    const ledColors = [0x10b981, 0x06b6d4, 0xf59e0b, 0x3b82f6];
    for (let i = 0; i < 6; i++) {
      const ledGeo = new THREE.SphereGeometry(0.04, 8, 8);
      const c = ledColors[i % ledColors.length];
      const ledMat = new THREE.MeshBasicMaterial({ color: c });
      const led = new THREE.Mesh(ledGeo, ledMat);
      led.position.set(-2.2 + i * 0.9, 0.15, 0.4);
      this.cockpitGroup.add(led);
    }

    // 6. Realistic Multi-Monitor Physical Mounting Bezels
    const createMonitorBezel = (x, y, z, rotY) => {
      const monitorGroup = new THREE.Group();
      // Outer Chamfered Bezel
      const bezelGeo = new THREE.BoxGeometry(2.1, 1.35, 0.08);
      const bezelMesh = new THREE.Mesh(bezelGeo, frameMat);
      monitorGroup.add(bezelMesh);

      // Inner LCD Screen Plane (Matte Glass)
      const screenGeo = new THREE.PlaneGeometry(1.98, 1.22);
      screenGeo.translate(0, 0, 0.045);
      const screenMat = new THREE.MeshStandardMaterial({
        color: 0x070d18,
        roughness: 0.2,
        metalness: 0.8,
        emissive: 0x0284c7,
        emissiveIntensity: 0.15,
      });
      const screenMesh = new THREE.Mesh(screenGeo, screenMat);
      monitorGroup.add(screenMesh);

      // Power LED
      const pwrGeo = new THREE.CircleGeometry(0.015, 8);
      pwrGeo.translate(0.9, -0.62, 0.045);
      const pwrMat = new THREE.MeshBasicMaterial({ color: 0x00ff88 });
      const pwrLed = new THREE.Mesh(pwrGeo, pwrMat);
      monitorGroup.add(pwrLed);

      monitorGroup.position.set(x, y, z);
      monitorGroup.rotation.y = rotY;
      return monitorGroup;
    };

    // Center Main Mission Display Frame
    const centerDisplay = createMonitorBezel(0, 0.85, -0.2, 0);
    this.cockpitGroup.add(centerDisplay);

    // Left Display Frame (Angled 18 deg)
    const leftDisplay = createMonitorBezel(-2.2, 0.85, 0.15, Math.PI / 10);
    this.cockpitGroup.add(leftDisplay);

    // Right Display Frame (Angled -18 deg)
    const rightDisplay = createMonitorBezel(2.2, 0.85, 0.15, -Math.PI / 10);
    this.cockpitGroup.add(rightDisplay);

    // 7. Interior Cockpit Warm / Cool Lighting
    const cockpitWarmLight = new THREE.PointLight(0xffeedd, 1.2, 10);
    cockpitWarmLight.position.set(0, 3.2, 2.0);
    this.scene.add(cockpitWarmLight);

    const cockpitConsoleGlow = new THREE.PointLight(0x38bdf8, 1.8, 6);
    cockpitConsoleGlow.position.set(0, 1.0, 0.5);
    this.scene.add(cockpitConsoleGlow);

    const airlockLight = new THREE.PointLight(0xf59e0b, 1.0, 8);
    airlockLight.position.set(0, 2.5, 12.0);
    this.scene.add(airlockLight);
  }

  /**
   * Build Orbiting Satellites
   */
  buildSatellites() {
    this.satellites = [];
    this.satelliteSpecs.forEach((spec) => {
      const satGroup = new THREE.Group();

      const bodyGeo = new THREE.BoxGeometry(0.3, 0.3, 0.5);
      const bodyMat = new THREE.MeshStandardMaterial({
        color: spec.color,
        metalness: 0.9,
        roughness: 0.2,
      });
      const body = new THREE.Mesh(bodyGeo, bodyMat);
      satGroup.add(body);

      const panelGeo = new THREE.BoxGeometry(1.4, 0.03, 0.4);
      const panelMat = new THREE.MeshStandardMaterial({
        color: 0x1e3a8a,
        metalness: 0.8,
        roughness: 0.3,
      });
      const panels = new THREE.Mesh(panelGeo, panelMat);
      satGroup.add(panels);

      this.earthGroup.add(satGroup);

      this.satellites.push({
        group: satGroup,
        spec: spec,
        angle: Math.random() * Math.PI * 2,
        radius: spec.radius,
        speed: spec.speed,
        inclination: spec.inclination,
      });
    });
  }

  /**
   * Updates Satellite Positions
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
  }

  /**
   * Cinematic Camera Transition: Enter Cockpit from Airlock
   */
  enterCockpit(callback) {
    this.playAudio("airlock");
    this.cameraState = "TRANSITION";
    this.camStartPos.copy(this.camera.position);
    this.camEndPos.copy(this.posWorkstation.pos);

    this.camStartLook.copy(this.posAirlock.look);
    this.camEndLook.copy(this.posWorkstation.look);

    this.camTransitionProgress = 0;

    const startT = performance.now();
    const duration = 2400; // 2.4s smooth cinematic glide

    const animateTransition = (now) => {
      const elapsed = now - startT;
      const t = Math.min(1, elapsed / duration);
      // Ease In Out Cubic
      const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

      this.camera.position.lerpVectors(this.camStartPos, this.camEndPos, ease);

      const curLook = new THREE.Vector3().lerpVectors(this.camStartLook, this.camEndLook, ease);
      this.camera.lookAt(curLook);

      if (t < 1) {
        requestAnimationFrame(animateTransition);
      } else {
        this.cameraState = "WORKSTATION";
        this.playAudio("boot");
        if (callback) callback();
      }
    };

    requestAnimationFrame(animateTransition);
  }

  /**
   * Cinematic Camera Transition: Exit back to Airlock
   */
  exitToAirlock(callback) {
    this.playAudio("airlock");
    this.cameraState = "TRANSITION";
    this.camStartPos.copy(this.camera.position);
    this.camEndPos.copy(this.posAirlock.pos);

    this.camStartLook.copy(this.posWorkstation.look);
    this.camEndLook.copy(this.posAirlock.look);

    const startT = performance.now();
    const duration = 1800;

    const animateExit = (now) => {
      const elapsed = now - startT;
      const t = Math.min(1, elapsed / duration);
      const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

      this.camera.position.lerpVectors(this.camStartPos, this.camEndPos, ease);
      const curLook = new THREE.Vector3().lerpVectors(this.camStartLook, this.camEndLook, ease);
      this.camera.lookAt(curLook);

      if (t < 1) {
        requestAnimationFrame(animateExit);
      } else {
        this.cameraState = "AIRLOCK";
        if (callback) callback();
      }
    };

    requestAnimationFrame(animateExit);
  }

  /**
   * Look directly out the orbital observation viewport
   */
  lookAtViewport() {
    this.playAudio("chirp");
    this.cameraState = "TRANSITION";
    this.camStartPos.copy(this.camera.position);
    this.camEndPos.copy(this.posViewport.pos);

    this.camStartLook.copy(this.posWorkstation.look);
    this.camEndLook.copy(this.posViewport.look);

    const startT = performance.now();
    const duration = 1600;

    const anim = (now) => {
      const elapsed = now - startT;
      const t = Math.min(1, elapsed / duration);
      const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;

      this.camera.position.lerpVectors(this.camStartPos, this.camEndPos, ease);
      const curLook = new THREE.Vector3().lerpVectors(this.camStartLook, this.camEndLook, ease);
      this.camera.lookAt(curLook);

      if (t < 1) {
        requestAnimationFrame(anim);
      } else {
        this.cameraState = "VIEWPORT";
      }
    };
    requestAnimationFrame(anim);
  }

  /**
   * Return back to Workstation Focus
   */
  focusWorkstation() {
    this.playAudio("chirp");
    this.cameraState = "TRANSITION";
    this.camStartPos.copy(this.camera.position);
    this.camEndPos.copy(this.posWorkstation.pos);

    this.camStartLook.copy(this.cameraState === "VIEWPORT" ? this.posViewport.look : this.posAirlock.look);
    this.camEndLook.copy(this.posWorkstation.look);

    const startT = performance.now();
    const duration = 1500;

    const anim = (now) => {
      const elapsed = now - startT;
      const t = Math.min(1, elapsed / duration);
      const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;

      this.camera.position.lerpVectors(this.camStartPos, this.camEndPos, ease);
      const curLook = new THREE.Vector3().lerpVectors(this.camStartLook, this.camEndLook, ease);
      this.camera.lookAt(curLook);

      if (t < 1) {
        requestAnimationFrame(anim);
      } else {
        this.cameraState = "WORKSTATION";
      }
    };
    requestAnimationFrame(anim);
  }

  /**
   * Procedural Web Audio FX Synthesizer
   */
  playAudio(type = "chirp") {
    if (!this.soundEnabled) return;
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      if (!this.audioCtx) this.audioCtx = new AudioCtx();
      if (this.audioCtx.state === "suspended") this.audioCtx.resume();

      const now = this.audioCtx.currentTime;

      if (type === "airlock") {
        // Atmospheric Decompression / Door Slide Hiss
        const bufferSize = this.audioCtx.sampleRate * 1.5;
        const noiseBuffer = this.audioCtx.createBuffer(1, bufferSize, this.audioCtx.sampleRate);
        const output = noiseBuffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) {
          output[i] = Math.random() * 2 - 1;
        }

        const whiteNoise = this.audioCtx.createBufferSource();
        whiteNoise.buffer = noiseBuffer;

        const filter = this.audioCtx.createBiquadFilter();
        filter.type = "lowpass";
        filter.frequency.setValueAtTime(800, now);
        filter.frequency.exponentialRampToValueAtTime(150, now + 1.4);

        const gain = this.audioCtx.createGain();
        gain.gain.setValueAtTime(0.3, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 1.4);

        whiteNoise.connect(filter);
        filter.connect(gain);
        gain.connect(this.audioCtx.destination);

        whiteNoise.start(now);
        whiteNoise.stop(now + 1.5);
      } else if (type === "boot") {
        // High-Tech Workstation Boot Chime (C-Major 7th Harmonic Sweep)
        const freqs = [523.25, 659.25, 783.99, 1046.50];
        freqs.forEach((freq, idx) => {
          const osc = this.audioCtx.createOscillator();
          const gain = this.audioCtx.createGain();
          osc.type = "sine";
          osc.frequency.setValueAtTime(freq, now + idx * 0.08);

          gain.gain.setValueAtTime(0.15, now + idx * 0.08);
          gain.gain.exponentialRampToValueAtTime(0.001, now + idx * 0.08 + 0.6);

          osc.connect(gain);
          gain.connect(this.audioCtx.destination);
          osc.start(now + idx * 0.08);
          osc.stop(now + idx * 0.08 + 0.65);
        });
      } else if (type === "chirp") {
        // Tactile Mechanical / Electronic Beep
        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(1200, now);
        osc.frequency.exponentialRampToValueAtTime(800, now + 0.06);

        gain.gain.setValueAtTime(0.15, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.08);

        osc.connect(gain);
        gain.connect(this.audioCtx.destination);
        osc.start(now);
        osc.stop(now + 0.09);
      }
    } catch (e) {
      // Audio safety fallback
    }
  }

  /**
   * Bind Mouse Parallax & Resize Listeners
   */
  bindEvents() {
    window.addEventListener("resize", () => {
      if (!this.container || !this.renderer || !this.camera) return;
      const w = window.innerWidth;
      const h = window.innerHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    });

    window.addEventListener("mousemove", (e) => {
      this.mouse.targetX = (e.clientX / window.innerWidth - 0.5) * 2;
      this.mouse.targetY = (e.clientY / window.innerHeight - 0.5) * 2;
    });
  }

  /**
   * 60FPS Main Render Loop
   */
  animate() {
    this.animId = requestAnimationFrame(() => this.animate());

    // Earth Orbital Rotation
    if (this.earthGroup) {
      this.earthGroup.rotation.y += 0.0008;
    }
    if (this.cloudsMesh) {
      this.cloudsMesh.rotation.y += 0.0011;
    }

    // Satellites
    this.updateSatellites();

    // Subtle Cockpit Mouse Parallax
    this.mouse.x += (this.mouse.targetX - this.mouse.x) * 0.05;
    this.mouse.y += (this.mouse.targetY - this.mouse.y) * 0.05;

    if (this.cameraState === "WORKSTATION") {
      this.camera.position.x = this.posWorkstation.pos.x + this.mouse.x * 0.12;
      this.camera.position.y = this.posWorkstation.pos.y - this.mouse.y * 0.08;
    } else if (this.cameraState === "AIRLOCK") {
      this.camera.position.x = this.posAirlock.pos.x + this.mouse.x * 0.25;
      this.camera.position.y = this.posAirlock.pos.y - this.mouse.y * 0.15;
    }

    this.renderer.render(this.scene, this.camera);
  }

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
