/**
 * SatQuery AI — Interstellar Spacecraft 3D Engine & Cockpit Kinematics
 * Photorealistic 3D Endurance Command Module & Ranger Flight Deck
 * 
 * Features:
 * - High-Contrast Black & White Interstellar Spacecraft Aesthetics:
 *   * Matte Arctic White ceramic heat tiles + Obsidian Black structural bulkheads & ribs.
 *   * Octagonal Endurance docking airlock with mechanical locking latches.
 *   * Dual pilot console flight sticks, throttle levers, conduits, and monitor mounting arms.
 *   * Rotating 12-module Endurance ring visible in deep orbit outside the viewport.
 * - Photorealistic Earth: Specular ocean shine, Rayleigh atmospheric limb scattering, cloud layer.
 * - Dynamic 3D Dashboard Motion Synchronization:
 *   * Real-time camera & parallax coordinate broadcasting to 3D CSS dashboard.
 *   * Smooth zero-g microgravity floating sway.
 *   * Cinematic airlock-to-cockpit camera flight and workstation docking.
 * - Procedural Web Audio: Airlock decompression hiss, workstation power chime, mechanical clicks.
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
    this.enduranceRing = null;

    // Cockpit & Interstellar Structures
    this.cockpitGroup = null;
    this.joysticks = [];
    this.satellites = [];

    // Camera Kinematics & States
    this.cameraState = "AIRLOCK"; // "AIRLOCK" | "TRANSITION" | "WORKSTATION" | "VIEWPORT"
    this.posAirlock = { pos: new THREE.Vector3(0, 1.4, 14.5), look: new THREE.Vector3(0, 0, -5) };
    this.posWorkstation = { pos: new THREE.Vector3(0, 0.45, 2.5), look: new THREE.Vector3(0, 0.25, -2) };
    this.posViewport = { pos: new THREE.Vector3(0, 1.8, -4.8), look: new THREE.Vector3(0, 0, -30) };

    this.camStartPos = new THREE.Vector3();
    this.camEndPos = new THREE.Vector3();
    this.camStartLook = new THREE.Vector3();
    this.camEndLook = new THREE.Vector3();

    // Mouse & Parallax Motion
    this.mouse = { x: 0, y: 0, targetX: 0, targetY: 0 };
    this.clock = new THREE.Clock();

    // Motion Synchronization Callback for 3D Dashboard
    this.onMotionUpdate = null;

    // Audio Context
    this.audioCtx = null;
    this.soundEnabled = true;

    // Orbiting Satellites Specs
    this.satelliteSpecs = [
      { id: "cartosat3", name: "ISRO Cartosat-3", radius: 24, speed: 0.004, inclination: 0.45, color: 0xffffff },
      { id: "sentinel2", name: "ESA Sentinel-2A", radius: 26, speed: 0.0035, inclination: 0.75, color: 0x38bdf8 },
      { id: "risat1a", name: "ISRO RISAT-1A (SAR)", radius: 25, speed: 0.0038, inclination: -0.55, color: 0x10b981 },
      { id: "landsat9", name: "NASA Landsat-9", radius: 27, speed: 0.003, inclination: 0.35, color: 0xf59e0b },
      { id: "endurance", name: "Endurance Station Alpha", radius: 22, speed: 0.0045, inclination: 0.85, color: 0xffffff },
    ];
  }

  /**
   * Initialize 3D Engine & Scene
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

    if (options.onMotionUpdate) this.onMotionUpdate = options.onMotionUpdate;

    const width = window.innerWidth || 1280;
    const height = window.innerHeight || 800;

    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x04060a, 0.012);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(48, width / height, 0.1, 2500);
    this.camera.position.copy(this.posAirlock.pos);
    this.camera.lookAt(this.posAirlock.look);

    // 3. WebGL Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.renderer.outputEncoding = THREE.sRGBEncoding;
    this.container.appendChild(this.renderer.domElement);

    // 4. Build Deep Space & Earth
    this.buildDeepSpace();

    // 5. Build Interstellar Spacecraft Structures (White & Black Theme)
    this.buildInterstellarCockpit();

    // 6. Build Rotating Endurance Spacecraft Ring & Satellites
    this.buildSatellites();

    // 7. Event Listeners
    this.bindEvents();

    this.isInitialized = true;
    this.animate();

    console.log("[SatQuery 3D] Interstellar Cockpit & 3D Motion Engine initialized.");
  }

  /**
   * Build Deep Space, Starfield, Solar Illumination & 3D Earth Globe
   */
  buildDeepSpace() {
    this.spaceGroup = new THREE.Group();
    this.spaceGroup.position.set(0, 0, -40);
    this.scene.add(this.spaceGroup);

    // 1. Starfield
    const starGeo = new THREE.BufferGeometry();
    const starCount = 2200;
    const positions = new Float32Array(starCount * 3);
    const colors = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount * 3; i += 3) {
      const r = 350 + Math.random() * 250;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);

      positions[i] = r * Math.sin(phi) * Math.cos(theta);
      positions[i + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i + 2] = r * Math.cos(phi);

      const b = 0.75 + Math.random() * 0.25;
      colors[i] = b;
      colors[i + 1] = b * (0.95 + Math.random() * 0.05);
      colors[i + 2] = 1.0;
    }

    starGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    starGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const starMat = new THREE.PointsMaterial({
      size: 1.4,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
    });
    this.starfield = new THREE.Points(starGeo, starMat);
    this.scene.add(this.starfield);

    // 2. Solar Lighting
    const sunLight = new THREE.DirectionalLight(0xffffff, 2.8);
    sunLight.position.set(70, 45, 60);
    this.scene.add(sunLight);

    const spaceAmbient = new THREE.AmbientLight(0x101828, 1.2);
    this.scene.add(spaceAmbient);

    // 3. Earth Globe
    this.earthGroup = new THREE.Group();
    this.spaceGroup.add(this.earthGroup);

    const earthCanvas = this.generateEarthCanvas();
    const earthTexture = new THREE.CanvasTexture(earthCanvas);
    earthTexture.wrapS = THREE.RepeatWrapping;

    const earthGeo = new THREE.SphereGeometry(15, 64, 64);
    const earthMat = new THREE.MeshStandardMaterial({
      map: earthTexture,
      roughness: 0.55,
      metalness: 0.2,
    });
    this.earthMesh = new THREE.Mesh(earthGeo, earthMat);
    this.earthGroup.add(this.earthMesh);

    // Clouds
    const cloudCanvas = this.generateCloudCanvas();
    const cloudTexture = new THREE.CanvasTexture(cloudCanvas);
    cloudTexture.wrapS = THREE.RepeatWrapping;

    const cloudGeo = new THREE.SphereGeometry(15.22, 48, 48);
    const cloudMat = new THREE.MeshStandardMaterial({
      map: cloudTexture,
      transparent: true,
      opacity: 0.48,
      blending: THREE.AdditiveBlending,
    });
    this.cloudsMesh = new THREE.Mesh(cloudGeo, cloudMat);
    this.earthGroup.add(this.cloudsMesh);

    // Rayleigh Scattering Atmosphere
    const atmoGeo = new THREE.SphereGeometry(16.5, 48, 48);
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
          float intensity = pow(0.64 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.6);
          gl_FragColor = vec4(0.3, 0.75, 1.0, 1.0) * intensity;
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
   * Procedural Earth Canvas
   */
  generateEarthCanvas() {
    const canvas = document.createElement("canvas");
    canvas.width = 2048;
    canvas.height = 1024;
    const ctx = canvas.getContext("2d");

    const oceanGrad = ctx.createLinearGradient(0, 0, 0, 1024);
    oceanGrad.addColorStop(0, "#051120");
    oceanGrad.addColorStop(0.2, "#081e38");
    oceanGrad.addColorStop(0.5, "#0c2c50");
    oceanGrad.addColorStop(0.8, "#081e38");
    oceanGrad.addColorStop(1, "#051120");
    ctx.fillStyle = oceanGrad;
    ctx.fillRect(0, 0, 2048, 1024);

    const continents = [
      [[1100, 300], [1350, 260], [1550, 320], [1500, 480], [1300, 520], [1200, 580], [1150, 540], [1120, 440], [1050, 360]],
      [[1220, 480], [1280, 490], [1270, 580], [1240, 620], [1210, 560]],
      [[950, 420], [1080, 430], [1120, 560], [1060, 740], [980, 720], [900, 560]],
      [[300, 220], [550, 200], [620, 320], [520, 440], [420, 480], [320, 380]],
      [[460, 500], [580, 540], [600, 680], [520, 840], [440, 740], [430, 580]],
      [[1450, 640], [1620, 630], [1650, 760], [1520, 780], [1430, 720]],
      [[0, 920], [2048, 920], [2048, 1024], [0, 1024]],
    ];

    continents.forEach((poly, idx) => {
      ctx.beginPath();
      ctx.moveTo(poly[0][0], poly[0][1]);
      for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
      ctx.closePath();
      ctx.fillStyle = idx === 1 ? "#26573c" : idx === 6 ? "#e2e8f0" : idx === 2 ? "#42472d" : "#224430";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(74, 222, 128, 0.4)";
      ctx.stroke();
    });

    // Night city light sparkles
    ctx.fillStyle = "rgba(255, 235, 160, 0.85)";
    [[1245, 545], [1235, 520], [1250, 500], [980, 320], [1010, 310], [420, 360], [450, 380], [1620, 380]].forEach(([cx, cy]) => {
      ctx.beginPath();
      ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
      ctx.fill();
    });

    return canvas;
  }

  /**
   * Procedural Cloud Texture
   */
  generateCloudCanvas() {
    const canvas = document.createElement("canvas");
    canvas.width = 1024;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, 1024, 512);

    ctx.fillStyle = "rgba(255, 255, 255, 0.38)";
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
   * Build Interstellar Spacecraft Cockpit (Endurance / Ranger White & Black Aesthetic)
   */
  buildInterstellarCockpit() {
    this.cockpitGroup = new THREE.Group();
    this.scene.add(this.cockpitGroup);

    // 1. PBR Materials (Interstellar White Ceramic & Obsidian Titanium)
    const whiteCeramicMat = new THREE.MeshStandardMaterial({
      color: 0xf3f6fa,
      roughness: 0.35,
      metalness: 0.4,
    });

    const obsidianRibMat = new THREE.MeshStandardMaterial({
      color: 0x090d14,
      roughness: 0.7,
      metalness: 0.85,
    });

    const carbonDeskMat = new THREE.MeshStandardMaterial({
      color: 0x111622,
      roughness: 0.5,
      metalness: 0.75,
    });

    const metalBezelMat = new THREE.MeshStandardMaterial({
      color: 0x1a2334,
      roughness: 0.4,
      metalness: 0.9,
    });

    // 2. Interstellar Endurance Hull (White Ceramic Shell with Black Compression Ribs)
    const hullGeo = new THREE.CylinderGeometry(8.2, 8.2, 32, 32, 1, true, -Math.PI / 2, Math.PI);
    hullGeo.rotateZ(Math.PI / 2);
    hullGeo.translate(0, 2.2, 0);
    const hullMesh = new THREE.Mesh(hullGeo, whiteCeramicMat);
    this.cockpitGroup.add(hullMesh);

    // Black Structural Arch Ribs
    for (let z = -10; z <= 14; z += 3.5) {
      const ribGeo = new THREE.TorusGeometry(8.05, 0.22, 12, 32, Math.PI);
      ribGeo.rotateX(Math.PI / 2);
      ribGeo.translate(0, 2.2, z);
      const ribMesh = new THREE.Mesh(ribGeo, obsidianRibMat);
      this.cockpitGroup.add(ribMesh);
    }

    // 3. Octagonal Endurance Airlock Hatch (Rear Entrance at z = 14)
    const airlockFrameGeo = new THREE.RingGeometry(3.6, 4.5, 8);
    airlockFrameGeo.translate(0, 1.8, 14.2);
    const airlockFrame = new THREE.Mesh(airlockFrameGeo, obsidianRibMat);
    this.cockpitGroup.add(airlockFrame);

    // 4. Observation Cupola / Forward Viewport Window
    const cupolaRingGeo = new THREE.RingGeometry(4.8, 5.8, 12);
    cupolaRingGeo.translate(0, 1.8, -11.2);
    const cupolaRing = new THREE.Mesh(cupolaRingGeo, obsidianRibMat);
    this.cockpitGroup.add(cupolaRing);

    // Viewport Glass
    const glassGeo = new THREE.CircleGeometry(4.8, 16);
    glassGeo.translate(0, 1.8, -11.15);
    const glassMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      roughness: 0.05,
      metalness: 0.95,
      transparent: true,
      opacity: 0.12,
    });
    const glass = new THREE.Mesh(glassGeo, glassMat);
    this.cockpitGroup.add(glass);

    // 5. Ranger Flight Console Desk (Carbon Surface & White PBR Trim)
    const deskGeo = new THREE.BoxGeometry(6.8, 0.85, 2.4);
    deskGeo.translate(0, -0.42, 0.8);
    const deskMesh = new THREE.Mesh(deskGeo, carbonDeskMat);
    this.cockpitGroup.add(deskMesh);

    const deskTrimGeo = new THREE.BoxGeometry(7.0, 0.15, 2.5);
    deskTrimGeo.translate(0, -0.85, 0.8);
    const deskTrim = new THREE.Mesh(deskTrimGeo, whiteCeramicMat);
    this.cockpitGroup.add(deskTrim);

    // 6. Dual Pilot Flight Sticks / Joysticks (Ranger Style)
    const buildJoystick = (xPos) => {
      const stickGroup = new THREE.Group();
      const baseGeo = new THREE.CylinderGeometry(0.18, 0.22, 0.1, 16);
      const baseMesh = new THREE.Mesh(baseGeo, obsidianRibMat);
      stickGroup.add(baseMesh);

      const poleGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.45, 12);
      poleGeo.translate(0, 0.22, 0);
      const poleMesh = new THREE.Mesh(poleGeo, metalBezelMat);
      stickGroup.add(poleMesh);

      const gripGeo = new THREE.BoxGeometry(0.12, 0.24, 0.14);
      gripGeo.translate(0, 0.45, 0);
      const gripMesh = new THREE.Mesh(gripGeo, obsidianRibMat);
      stickGroup.add(gripMesh);

      stickGroup.position.set(xPos, 0.05, 1.2);
      this.cockpitGroup.add(stickGroup);
      this.joysticks.push(stickGroup);
    };

    buildJoystick(-1.6); // Commander station joystick
    buildJoystick(1.6);  // Pilot station joystick

    // 7. Mechanical Articulating Monitor Support Arms
    const createMonitorArm = (x, y, z, rotY) => {
      const armGroup = new THREE.Group();
      const mountGeo = new THREE.BoxGeometry(0.15, 0.15, 0.8);
      const mountMesh = new THREE.Mesh(mountGeo, metalBezelMat);
      armGroup.add(mountMesh);

      armGroup.position.set(x, y, z);
      armGroup.rotation.y = rotY;
      this.cockpitGroup.add(armGroup);
    };

    createMonitorArm(-2.2, 0.85, -0.2, Math.PI / 10);
    createMonitorArm(0, 0.85, -0.4, 0);
    createMonitorArm(2.2, 0.85, -0.2, -Math.PI / 10);

    // 8. Interstellar Tactical Cockpit Lighting (Cool White Key Light + Cyan/Amber Telemetry Glare)
    const keyLight = new THREE.PointLight(0xffffff, 1.4, 12);
    keyLight.position.set(0, 3.4, 1.5);
    this.scene.add(keyLight);

    const consoleGlare = new THREE.PointLight(0x38bdf8, 2.0, 5);
    consoleGlare.position.set(0, 0.9, 0.4);
    this.scene.add(consoleGlare);

    const airlockAmberLight = new THREE.PointLight(0xf59e0b, 1.2, 8);
    airlockAmberLight.position.set(0, 2.6, 13.0);
    this.scene.add(airlockAmberLight);
  }

  /**
   * Build Orbiting Satellites and the Rotating Endurance Spacecraft Ring
   */
  buildSatellites() {
    this.satellites = [];

    // 1. Rotating Endurance 12-Module Ring Structure
    const ringGroup = new THREE.Group();
    const moduleCount = 12;
    const ringRadius = 5.2;

    const modGeo = new THREE.BoxGeometry(1.2, 0.8, 1.4);
    const modWhiteMat = new THREE.MeshStandardMaterial({ color: 0xf1f5f9, metalness: 0.5, roughness: 0.3 });
    const modBlackMat = new THREE.MeshStandardMaterial({ color: 0x090e18, metalness: 0.8, roughness: 0.6 });

    for (let i = 0; i < moduleCount; i++) {
      const angle = (i / moduleCount) * Math.PI * 2;
      const x = Math.cos(angle) * ringRadius;
      const y = Math.sin(angle) * ringRadius;

      const mod = new THREE.Mesh(modGeo, i % 2 === 0 ? modWhiteMat : modBlackMat);
      mod.position.set(x, y, 0);
      mod.rotation.z = angle + Math.PI / 2;
      ringGroup.add(mod);
    }

    ringGroup.position.set(18, 12, -45);
    ringGroup.rotation.x = Math.PI / 4;
    this.scene.add(ringGroup);
    this.enduranceRing = ringGroup;

    // 2. Scientific Orbiting Satellites
    this.satelliteSpecs.forEach((spec) => {
      const satGroup = new THREE.Group();
      const bodyGeo = new THREE.BoxGeometry(0.3, 0.3, 0.5);
      const bodyMat = new THREE.MeshStandardMaterial({ color: spec.color, metalness: 0.9, roughness: 0.2 });
      const body = new THREE.Mesh(bodyGeo, bodyMat);
      satGroup.add(body);

      const panelGeo = new THREE.BoxGeometry(1.5, 0.03, 0.4);
      const panelMat = new THREE.MeshStandardMaterial({ color: 0x090e18, metalness: 0.8, roughness: 0.3 });
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
   * Updates Satellites and Rotating Ring
   */
  updateSatellites() {
    if (this.enduranceRing) {
      this.enduranceRing.rotation.z += 0.003; // Interstellar artificial gravity spin
    }

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
   * Cinematic Camera Flight: Enter Cockpit from Airlock
   */
  enterCockpit(callback) {
    this.playAudio("airlock");
    this.cameraState = "TRANSITION";
    this.camStartPos.copy(this.camera.position);
    this.camEndPos.copy(this.posWorkstation.pos);

    this.camStartLook.copy(this.posAirlock.look);
    this.camEndLook.copy(this.posWorkstation.look);

    const startT = performance.now();
    const duration = 2400;

    const animateTransition = (now) => {
      const elapsed = now - startT;
      const t = Math.min(1, elapsed / duration);
      const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

      this.camera.position.lerpVectors(this.camStartPos, this.camEndPos, ease);
      const curLook = new THREE.Vector3().lerpVectors(this.camStartLook, this.camEndLook, ease);
      this.camera.lookAt(curLook);

      // Broadcast flight progress to 3D CSS Dashboard for matching zoom/docking
      if (this.onMotionUpdate) {
        this.onMotionUpdate(t, "TRANSITION");
      }

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
   * Cinematic Camera Flight: Exit back to Airlock
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

      if (this.onMotionUpdate) {
        this.onMotionUpdate(1 - t, "EXIT");
      }

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
   * Look directly out Viewport
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
    this.camStartLook.copy(this.posViewport.look);
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
   * Synthesized Web Audio FX
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
        const bufferSize = this.audioCtx.sampleRate * 1.6;
        const noiseBuffer = this.audioCtx.createBuffer(1, bufferSize, this.audioCtx.sampleRate);
        const output = noiseBuffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) output[i] = Math.random() * 2 - 1;

        const whiteNoise = this.audioCtx.createBufferSource();
        whiteNoise.buffer = noiseBuffer;

        const filter = this.audioCtx.createBiquadFilter();
        filter.type = "lowpass";
        filter.frequency.setValueAtTime(900, now);
        filter.frequency.exponentialRampToValueAtTime(140, now + 1.5);

        const gain = this.audioCtx.createGain();
        gain.gain.setValueAtTime(0.35, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 1.5);

        whiteNoise.connect(filter);
        filter.connect(gain);
        gain.connect(this.audioCtx.destination);
        whiteNoise.start(now);
        whiteNoise.stop(now + 1.6);
      } else if (type === "boot") {
        [523.25, 659.25, 783.99, 1046.50].forEach((freq, idx) => {
          const osc = this.audioCtx.createOscillator();
          const gain = this.audioCtx.createGain();
          osc.type = "sine";
          osc.frequency.setValueAtTime(freq, now + idx * 0.08);

          gain.gain.setValueAtTime(0.18, now + idx * 0.08);
          gain.gain.exponentialRampToValueAtTime(0.001, now + idx * 0.08 + 0.6);

          osc.connect(gain);
          gain.connect(this.audioCtx.destination);
          osc.start(now + idx * 0.08);
          osc.stop(now + idx * 0.08 + 0.65);
        });
      } else if (type === "chirp") {
        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(1100, now);
        osc.frequency.exponentialRampToValueAtTime(750, now + 0.06);

        gain.gain.setValueAtTime(0.15, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.08);

        osc.connect(gain);
        gain.connect(this.audioCtx.destination);
        osc.start(now);
        osc.stop(now + 0.09);
      }
    } catch (e) {}
  }

  /**
   * Bind Mouse Parallax & Resize
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
   * Main Render Loop (60 FPS) with 3D Motion Broadcasting
   */
  animate() {
    this.animId = requestAnimationFrame(() => this.animate());
    const time = this.clock.getElapsedTime();

    // 1. Earth & Clouds Orbital Drift
    if (this.earthGroup) this.earthGroup.rotation.y += 0.0006;
    if (this.cloudsMesh) this.cloudsMesh.rotation.y += 0.0009;

    // 2. Satellites & Endurance Ring
    this.updateSatellites();

    // 3. Smooth Mouse Parallax Interpolation
    this.mouse.x += (this.mouse.targetX - this.mouse.x) * 0.06;
    this.mouse.y += (this.mouse.targetY - this.mouse.y) * 0.06;

    // 4. Subtle Microgravity Zero-G Floating Sway
    const swayX = Math.sin(time * 0.8) * 0.03;
    const swayY = Math.cos(time * 0.6) * 0.025;

    // Move Joysticks slightly with sway
    this.joysticks.forEach((j, i) => {
      j.rotation.x = (i === 0 ? this.mouse.y : -this.mouse.y) * 0.15 + swayY;
      j.rotation.z = this.mouse.x * 0.15 + swayX;
    });

    // 5. Position Camera based on State
    if (this.cameraState === "WORKSTATION") {
      this.camera.position.x = this.posWorkstation.pos.x + this.mouse.x * 0.15 + swayX;
      this.camera.position.y = this.posWorkstation.pos.y - this.mouse.y * 0.1 + swayY;
    } else if (this.cameraState === "AIRLOCK") {
      this.camera.position.x = this.posAirlock.pos.x + this.mouse.x * 0.3 + swayX;
      this.camera.position.y = this.posAirlock.pos.y - this.mouse.y * 0.2 + swayY;
    }

    // 6. Broadcast 3D Parallax & Motion to Front Dashboard
    if (this.onMotionUpdate && (this.cameraState === "WORKSTATION" || this.cameraState === "AIRLOCK")) {
      this.onMotionUpdate({
        mouseX: this.mouse.x,
        mouseY: this.mouse.y,
        swayX: swayX,
        swayY: swayY,
        state: this.cameraState
      });
    }

    this.renderer.render(this.scene, this.camera);
  }

  destroy() {
    if (this.animId) cancelAnimationFrame(this.animId);
    if (this.renderer && this.renderer.domElement) this.renderer.domElement.remove();
    this.isInitialized = false;
  }
}

// Global Export
window.SatQuery3DDeck = new SatQuery3DDeck();
