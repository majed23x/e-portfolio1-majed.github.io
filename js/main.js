// Three.js and GSAP Animation Script

import * as THREE from 'https://cdn.skypack.dev/three@0.136.0';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.136.0/examples/jsm/controls/OrbitControls.js';
import gsap from 'https://cdn.skypack.dev/gsap@3.10.4';

// Initialize Three.js scene
let scene, camera, renderer, controls;
let geometry, material, particles;
let mouseX = 0, mouseY = 0;
const windowHalfX = window.innerWidth / 2;
const windowHalfY = window.innerHeight / 2;

// Initialize Three.js scene only on homepage
function initThreeJS() {
  const canvasContainer = document.querySelector('.canvas-container');
  
  // If no canvas container is found, return early (not on homepage)
  if (!canvasContainer) return;
  
  // Scene setup
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.z = 20;
  
  // Renderer setup
  renderer = new THREE.WebGLRenderer({ 
    alpha: true, 
    antialias: true 
  });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  canvasContainer.appendChild(renderer.domElement);
  
  // Add cartoonish particles
  const particleCount = 500;
  geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(particleCount * 3);
  const colors = new Float32Array(particleCount * 3);
  const sizes = new Float32Array(particleCount);
  
  const color = new THREE.Color();
  
  for (let i = 0; i < particleCount; i++) {
    // Position
    positions[i * 3] = (Math.random() - 0.5) * 50; // x
    positions[i * 3 + 1] = (Math.random() - 0.5) * 50; // y
    positions[i * 3 + 2] = (Math.random() - 0.5) * 50; // z
    
    // Colors - use brand colors
    const colorChoice = Math.random();
    if (colorChoice < 0.33) {
      color.set('#6a51cf'); // primary
    } else if (colorChoice < 0.66) {
      color.set('#ff6b6b'); // secondary
    } else {
      color.set('#ffcf5c'); // accent
    }
    
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
    
    // Sizes
    sizes[i] = Math.random() * 2 + 0.5;
  }
  
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
  
  // Shader material for particles
  material = new THREE.ShaderMaterial({
    uniforms: {
      pointTexture: { value: new THREE.TextureLoader().load('https://assets.codepen.io/3685267/circle.png') }
    },
    vertexShader: `
      attribute float size;
      varying vec3 vColor;
      void main() {
        vColor = color;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * (300.0 / -mvPosition.z);
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      uniform sampler2D pointTexture;
      varying vec3 vColor;
      void main() {
        gl_FragColor = vec4(vColor, 1.0);
        gl_FragColor = gl_FragColor * texture2D(pointTexture, gl_PointCoord);
      }
    `,
    blending: THREE.AdditiveBlending,
    depthTest: false,
    transparent: true,
    vertexColors: true
  });
  
  particles = new THREE.Points(geometry, material);
  scene.add(particles);
  
  // Event listeners
  document.addEventListener('mousemove', onDocumentMouseMove);
  window.addEventListener('resize', onWindowResize);
  
  // Animation loop
  animate();
}

// Handle mouse movement
function onDocumentMouseMove(event) {
  mouseX = (event.clientX - windowHalfX) * 0.05;
  mouseY = (event.clientY - windowHalfY) * 0.05;
}

// Handle window resize
function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

// Animation loop
function animate() {
  requestAnimationFrame(animate);
  
  // Rotate particles
  particles.rotation.x += 0.0005;
  particles.rotation.y += 0.001;
  
  // Move particles based on mouse position
  particles.rotation.x += (mouseY - particles.rotation.x) * 0.05;
  particles.rotation.y += (mouseX - particles.rotation.y) * 0.05;
  
  renderer.render(scene, camera);
}

// GSAP animations for page elements
function initGsapAnimations() {
  // Hero section animations
  const heroTimeline = gsap.timeline({ defaults: { ease: "power3.out" } });
  
  heroTimeline.from(".hero-content h1", {
    y: 50,
    opacity: 0,
    duration: 1
  })
  .from(".hero-content p", {
    y: 30,
    opacity: 0,
    duration: 0.8
  }, "-=0.3")
  .from(".hero-content .btn", {
    y: 20,
    opacity: 0,
    duration: 0.6
  }, "-=0.2");
  
  // Animate elements when they enter the viewport
  gsap.registerPlugin(ScrollTrigger);
  
  // Animate section headings
  gsap.utils.toArray('.section-title').forEach(title => {
    gsap.from(title, {
      scrollTrigger: {
        trigger: title,
        start: "top 80%",
      },
      y: 50,
      opacity: 0,
      duration: 0.8
    });
  });
  
  // Animate module cards
  gsap.utils.toArray('.module-card').forEach((card, i) => {
    gsap.from(card, {
      scrollTrigger: {
        trigger: card,
        start: "top 85%",
      },
      y: 50,
      opacity: 0,
      duration: 0.6,
      delay: i * 0.1
    });
  });
  
  // Animate skill cards
  gsap.utils.toArray('.skill-card').forEach((card, i) => {
    gsap.from(card, {
      scrollTrigger: {
        trigger: card,
        start: "top 85%",
      },
      scale: 0.8,
      opacity: 0,
      duration: 0.5,
      delay: i * 0.05
    });
  });
}

// Initialize page transition animations
function initPageTransitions() {
  // Page transition animation
  const transitionElement = document.querySelector('.page-transition');
  
  if (transitionElement) {
    // Play exit animation when clicking on a navigation link
    document.querySelectorAll('a[data-transition]').forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const href = link.getAttribute('href');
        
        // Page exit animation
        gsap.to(transitionElement, {
          scaleY: 1,
          duration: 0.5,
          ease: "power3.inOut",
          onComplete: () => {
            // Navigate to the new page
            window.location.href = href;
          }
        });
      });
    });
    
    // Page enter animation
    gsap.from(transitionElement, {
      scaleY: 1,
      duration: 0.5,
      ease: "power3.out",
      onComplete: () => {
        transitionElement.style.transformOrigin = "top";
      }
    });
  }
}

// Mobile menu toggle
function initMobileMenu() {
  const mobileMenuToggle = document.querySelector('.mobile-menu');
  const navLinks = document.querySelector('.nav-links');
  
  if (mobileMenuToggle && navLinks) {
    mobileMenuToggle.addEventListener('click', () => {
      navLinks.classList.toggle('active');
      mobileMenuToggle.classList.toggle('active');
    });
  }
}

// Initialize everything on page load
document.addEventListener('DOMContentLoaded', () => {
  initThreeJS();
  initGsapAnimations();
  initPageTransitions();
  initMobileMenu();
});



  document.querySelectorAll('.toggle-btn').forEach(button => {
    button.addEventListener('click', function () {
      const content = this.closest('.module-section').querySelector('.module-content');
      const isHidden = content.classList.toggle('hidden');
      this.textContent = isHidden ? 'View More' : 'View Less';
    });
  });
