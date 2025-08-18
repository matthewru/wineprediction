import React, { useEffect, useImperativeHandle, useRef, useState, forwardRef } from 'react';
import { View, Platform, Text } from 'react-native';
import { GLView } from 'expo-gl';
import type { ExpoWebGLRenderingContext } from 'expo-gl';
import * as THREE from 'three';
import { Renderer as ExpoRenderer } from 'expo-three';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import { runOnJS } from 'react-native-reanimated';
import { feature, mesh as topoMesh } from 'topojson-client';
import { wineTheme } from '../constants/Colors';

const WINE_PRIMARY = 0x8b2635;
// Theme-driven colors
const THEME = {
	primary: new THREE.Color(wineTheme.colors.primary),
	background: new THREE.Color(wineTheme.colors.background),
	surface: new THREE.Color(wineTheme.colors.surface),
	text: new THREE.Color(wineTheme.colors.text),
};
// Performance/quality trade-offs for coastline and land detail
const DETAIL_SAMPLING = {
	landFillStep: 2,
	coastRingStep: 8,
	coastLineStep: 6,
	borderRingStep: 4,
	highlightRingStep: 4,
};
// Precomputed atlas data and caches to avoid repeated heavy work on navigation
let ATLAS_COUNTRIES: any = null;
let ATLAS_COUNTRIES_FC: any = null;
let WORLD_COUNTRIES: any = null;
try {
	ATLAS_COUNTRIES = require('world-atlas/countries-50m.json');
	ATLAS_COUNTRIES_FC = feature(ATLAS_COUNTRIES, ATLAS_COUNTRIES.objects.countries);
	WORLD_COUNTRIES = require('world-countries');
} catch {}
const NAME_MAP: Record<string, string> = {
	US: 'United States', Italy: 'Italy', France: 'France', Spain: 'Spain', Portugal: 'Portugal', Chile: 'Chile', Argentina: 'Argentina', Austria: 'Austria', Germany: 'Germany', Australia: 'Australia',
};
const latLonCache = new Map<string, { lat: number; lon: number }>();
const polygonsCache = new Map<string, number[][][][]>();
const geometryCache = new Map<string, number[][][]>();
const extentCache = new Map<string, { latMin: number; latMax: number; lonMin: number; lonMax: number }>();

const COUNTRY_CENTROIDS: Record<string, { lat: number; lon: number }> = {
	US: { lat: 39.8, lon: -98.6 },
	Italy: { lat: 42.5, lon: 12.5 },
	France: { lat: 46.2, lon: 2.2 },
	Spain: { lat: 40.4, lon: -3.7 },
	Portugal: { lat: 39.4, lon: -8.2 },
	Chile: { lat: -35.7, lon: -71.5 },
	Argentina: { lat: -38.4, lon: -63.6 },
	Austria: { lat: 47.5, lon: 14.5 },
	Germany: { lat: 51.0, lon: 10.0 },
	Australia: { lat: -25.0, lon: 133.8 },
};

export type GlobeSelectorHandle = {
	flyToCountry: (country: string) => void;
};

type Props = {
	countries: string[];
	selectedCountry: string;
	onCountryChange?: (country: string) => void;
	style?: any;
};

function latLonToVector3(lat: number, lon: number, radius: number): THREE.Vector3 {
	const phi = THREE.MathUtils.degToRad(90 - lat);
	const theta = THREE.MathUtils.degToRad(lon + 180);
	const x = -radius * Math.sin(phi) * Math.cos(theta);
	const z = radius * Math.sin(phi) * Math.sin(theta);
	const y = radius * Math.cos(phi);
	return new THREE.Vector3(x, y, z);
}

function computeQuaternionToFaceLatLon(lat: number, lon: number): THREE.Quaternion {
	const v = latLonToVector3(lat, lon, 1).normalize();
	const target = new THREE.Vector3(0, 0, 1);
	const dot = THREE.MathUtils.clamp(v.dot(target), -1, 1);
	if (Math.abs(dot - 1) < 1e-6) return new THREE.Quaternion();
	if (Math.abs(dot + 1) < 1e-6) return new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI);
	const axis = v.clone().cross(target).normalize();
	const angle = Math.acos(dot);
	return new THREE.Quaternion().setFromAxisAngle(axis, angle);
}

function getLatLonForCountry(shortName: string): { lat: number; lon: number } | null {
	const cached = latLonCache.get(shortName);
	if (cached) return cached;
	try {
		const common = NAME_MAP[shortName] || shortName;
		const entry = WORLD_COUNTRIES?.find((c: any) => c.name?.common === common);
		if (entry?.latlng && Array.isArray(entry.latlng) && entry.latlng.length >= 2) {
			const [lat, lon] = entry.latlng as [number, number];
			const val = { lat, lon };
			latLonCache.set(shortName, val);
			return val;
		}
	} catch {}
	const centroid = COUNTRY_CENTROIDS[shortName];
	if (centroid) {
		latLonCache.set(shortName, centroid);
		return centroid;
	}
	return null;
}

function getAtlasPolygonsForCountry(shortName: string): number[][][][] | null {
	const cached = polygonsCache.get(shortName);
	if (cached) return cached;
	try {
		const common = NAME_MAP[shortName] || shortName;
		const entry = WORLD_COUNTRIES?.find((c: any) => c.name?.common === common);
		const idStr = entry?.ccn3 || entry?.ccn2 || '';
		if (!entry || !idStr || !ATLAS_COUNTRIES_FC) return null;
		const idNum = parseInt(idStr, 10);
		const feat = ATLAS_COUNTRIES_FC.features.find((f: any) => parseInt(f.id, 10) === idNum);
		if (!feat || !feat.geometry) return null;
		let result: number[][][][] | null = null;
		if (feat.geometry.type === 'Polygon') result = [feat.geometry.coordinates as number[][][]];
		else if (feat.geometry.type === 'MultiPolygon') result = feat.geometry.coordinates as number[][][][];
		if (result) polygonsCache.set(shortName, result);
		return result;
	} catch {
		return null;
	}
}

function getAtlasGeometryForCountry(shortName: string): number[][][] | null {
	const cached = geometryCache.get(shortName);
	if (cached) return cached;
	try {
		const common = NAME_MAP[shortName] || shortName;
		const entry = WORLD_COUNTRIES?.find((c: any) => c.name?.common === common);
		const idStr = entry?.ccn3 || entry?.ccn2 || '';
		if (!entry || !idStr || !ATLAS_COUNTRIES_FC) return null;
		const idNum = parseInt(idStr, 10);
		const feat = ATLAS_COUNTRIES_FC.features.find((f: any) => parseInt(f.id, 10) === idNum);
		if (!feat || !feat.geometry) return null;
		let result: number[][][] | null = null;
		if (feat.geometry.type === 'Polygon') result = feat.geometry.coordinates as number[][][];
		else if (feat.geometry.type === 'MultiPolygon') result = (feat.geometry.coordinates as number[][][][]).flat() as number[][][];
		if (result) geometryCache.set(shortName, result);
		return result;
	} catch {
		return null;
	}
}

function getExtentForCountry(shortName: string): { latMin: number; latMax: number; lonMin: number; lonMax: number } | null {
	const cached = extentCache.get(shortName);
	if (cached) return cached;
	const geoms = getAtlasGeometryForCountry(shortName);
	if (!geoms) return null;
	let latMin = 90, latMax = -90, lonMin = 180, lonMax = -180;
	geoms.forEach((ring: number[][]) => {
		ring.forEach(([lon, lat]) => {
			if (lat < latMin) latMin = lat;
			if (lat > latMax) latMax = lat;
			if (lon < lonMin) lonMin = lon;
			if (lon > lonMax) lonMax = lon;
		});
	});
	const result = { latMin, latMax, lonMin, lonMax };
	extentCache.set(shortName, result);
	return result;
}

function suggestDistanceForExtent(extent: { latMin: number; latMax: number; lonMin: number; lonMax: number } | null): number {
	// Smaller countries => closer; larger => farther
	const minDist = 1.9;
	const maxDist = 8.5;
	if (!extent) return 5.0;
	const dLat = Math.max(0.1, Math.abs(extent.latMax - extent.latMin));
	const meanLat = (extent.latMax + extent.latMin) / 2;
	const dLon = Math.max(0.1, Math.abs(extent.lonMax - extent.lonMin) * Math.cos(THREE.MathUtils.degToRad(meanLat)));
	const angular = Math.max(dLat, dLon); // degrees spanned
	// Map 1deg -> near, 50deg -> far
	const t = THREE.MathUtils.clamp((angular - 1) / (50 - 1), 0, 1);
	return THREE.MathUtils.lerp(minDist, maxDist, t);
}

const GlobeSelectorInner = ({ countries, selectedCountry, onCountryChange, style }: Props, ref: React.Ref<GlobeSelectorHandle>) => {
	const glViewRef = useRef<GLView | null>(null);
	const glRef = useRef<ExpoWebGLRenderingContext | null>(null);
	const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
	const sceneRef = useRef<THREE.Scene | null>(null);
	const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
	const globeGroupRef = useRef<THREE.Group | null>(null);
	const highlightGroupRef = useRef<THREE.Group | null>(null);
	const highlightMaterialsRef = useRef<THREE.Material[]>([]);
	const highlightCacheRef = useRef<Map<string, { group: THREE.Group; materials: THREE.LineBasicMaterial[] }>>(new Map());
	const pulsePhaseRef = useRef(0);
	const [ready, setReady] = useState(false);
	const [viewSize, setViewSize] = useState({ width: 0, height: 0 });
	const rafRef = useRef<number | null>(null);
	const initializedRef = useRef(false);
	const bordersAddedRef = useRef(false);
	const graticuleAddedRef = useRef(false);

	const targetQuatRef = useRef(new THREE.Quaternion());
	const currentQuatRef = useRef(new THREE.Quaternion());
	const yawVelRef = useRef(0);
	const pitchVelRef = useRef(0);
	const lastTimeRef = useRef<number | null>(null);

	const currentDistanceRef = useRef(6.0);
	const targetDistanceRef = useRef(6.0);
	const pinchBaseDistanceRef = useRef(6.0);

	const RADIUS = 2.0;

	const ensureScene = async (gl: ExpoWebGLRenderingContext) => {
		if (initializedRef.current) return;
		initializedRef.current = true;
		try {
			glRef.current = gl;
			const renderer = new ExpoRenderer({ gl } as any) as unknown as THREE.WebGLRenderer;
			const initialW = (gl as any).drawingBufferWidth || viewSize.width || 300;
			const initialH = (gl as any).drawingBufferHeight || viewSize.height || 300;
			renderer.setSize(initialW, initialH);
			renderer.setClearColor('#f8f9fa');
			rendererRef.current = renderer;

			const scene = new THREE.Scene();
			scene.background = new THREE.Color('#f8f9fa');
			sceneRef.current = scene;

			const camera = new THREE.PerspectiveCamera(45, initialW / initialH, 0.1, 100);
			camera.position.set(0, 0, currentDistanceRef.current);
			camera.lookAt(0, 0, 0);
			cameraRef.current = camera;

			scene.add(new THREE.AmbientLight(0xffffff, 0.9));
			const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
			dirLight.position.set(5, 5, 5);
			scene.add(dirLight);

			const globeGroup = new THREE.Group();
			globeGroupRef.current = globeGroup;
			scene.add(globeGroup);

			highlightGroupRef.current = new THREE.Group();
			globeGroup.add(highlightGroupRef.current);

			const sphereGeom = new THREE.SphereGeometry(RADIUS, 32, 32);
			// Water sphere with slight opacity
			const sphereMat = new THREE.MeshPhongMaterial({ color: THEME.surface, shininess: 6, transparent: true, opacity: 0.5, depthTest: false, depthWrite: false });
			const sphere = new THREE.Mesh(sphereGeom, sphereMat);
			sphere.renderOrder = 600;
			globeGroup.add(sphere);

			// Initialize orientation toward selected country
			if (selectedCountry) {
				const ll = getLatLonForCountry(selectedCountry);
				if (ll) {
					const q = computeQuaternionToFaceLatLon(ll.lat, ll.lon);
					targetQuatRef.current.copy(q);
					currentQuatRef.current.copy(q);
					if (globeGroupRef.current) globeGroupRef.current.setRotationFromQuaternion(q);
					// Initial zoom suggestion
					targetDistanceRef.current = suggestDistanceForExtent(getExtentForCountry(selectedCountry));
					currentDistanceRef.current = targetDistanceRef.current;
					camera.position.z = currentDistanceRef.current;
					// Draw highlight for initial country
					drawHighlightForCountry(selectedCountry);
				}
			}

			setReady(true);

			const hasAnimLoop = typeof (renderer as any).setAnimationLoop === 'function';
			const renderFrame = () => {
				if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;
				// Apply inertial rotation
				const now = Date.now();
				const dtSec = lastTimeRef.current ? (now - lastTimeRef.current) / 1000 : 0;
				lastTimeRef.current = now;
				if ((yawVelRef.current !== 0 || pitchVelRef.current !== 0) && dtSec > 0) {
					let upAxis = new THREE.Vector3(0, 1, 0);
					let rightAxis = new THREE.Vector3(1, 0, 0);
					if (cameraRef.current) {
						upAxis = cameraRef.current.up.clone().normalize();
						const forward = new THREE.Vector3();
						cameraRef.current.getWorldDirection(forward);
						rightAxis = forward.clone().cross(upAxis).normalize();
					}
					const yawDelta = yawVelRef.current * dtSec;
					const pitchDelta = pitchVelRef.current * dtSec;
					const qYaw = new THREE.Quaternion().setFromAxisAngle(upAxis, yawDelta);
					const qPitch = new THREE.Quaternion().setFromAxisAngle(rightAxis, pitchDelta);
					targetQuatRef.current.premultiply(qYaw.multiply(qPitch)).normalize();
					const damping = Math.pow(0.985, dtSec * 60);
					yawVelRef.current *= damping;
					pitchVelRef.current *= damping;
					if (Math.abs(yawVelRef.current) < 1e-4) yawVelRef.current = 0;
					if (Math.abs(pitchVelRef.current) < 1e-4) pitchVelRef.current = 0;
				}

				// Smooth orientation
				if (globeGroupRef.current) {
					currentQuatRef.current.slerp(targetQuatRef.current, 0.5);
					globeGroupRef.current.setRotationFromQuaternion(currentQuatRef.current);
				}

				// Smooth zoom
				if (cameraRef.current) {
					currentDistanceRef.current = THREE.MathUtils.lerp(currentDistanceRef.current, targetDistanceRef.current, 0.4);
					cameraRef.current.position.z = currentDistanceRef.current;
				}

				// Pulse highlight opacity to mimic map pulse
				if (highlightMaterialsRef.current.length > 0) {
					pulsePhaseRef.current += (dtSec || 0) * 2.4; // ~2.4Hz
					const pulse = 0.6 + 0.4 * Math.sin(pulsePhaseRef.current);
					highlightMaterialsRef.current.forEach((m, idx) => {
						const base = idx === 0 ? 0.95 : 0.5; // border vs glow
						(m as THREE.LineBasicMaterial).opacity = THREE.MathUtils.clamp(base * pulse, 0.2, 1.0);
						(m as THREE.LineBasicMaterial).needsUpdate = true;
					});
				}

				rendererRef.current.render(sceneRef.current, cameraRef.current);
				if (glRef.current && (glRef.current as any).endFrameEXP) {
					(glRef.current as any).endFrameEXP();
				}
			};

			if (hasAnimLoop) {
				(renderer as any).setAnimationLoop(renderFrame);
			} else {
				const tick = () => {
					rendererRef.current && renderFrame();
					rafRef.current = requestAnimationFrame(tick);
				};
				tick();
			}

			// Defer grid and borders
			setTimeout(() => {
				try {
					// Land fill first (underneath)
					addLandFill(globeGroupRef.current!, RADIUS + 0.004);
					if (!graticuleAddedRef.current && globeGroupRef.current) {
						addGraticule(globeGroupRef.current, RADIUS + 0.003);
						graticuleAddedRef.current = true;
					}
					if (globeGroupRef.current) {
						addLandOutlines(globeGroupRef.current, RADIUS + 0.003);
					}
					if (!bordersAddedRef.current && globeGroupRef.current) {
						addCountryBorders(globeGroupRef.current, countries, RADIUS + 0.003);
						bordersAddedRef.current = true;
					}
				} catch {}
				// Prewarm caches for faster arrow presses
				try {
					countries.forEach((c) => {
						getLatLonForCountry(c);
						getAtlasGeometryForCountry(c);
						getAtlasPolygonsForCountry(c);
						getExtentForCountry(c);
					});
				} catch {}
			}, 0);
		} catch (err) {
			console.warn('Failed to initialize GL/Three scene', err);
			initializedRef.current = false;
		}
	};

	const updateSize = (width: number, height: number) => {
		setViewSize({ width, height });
		if (rendererRef.current) rendererRef.current.setSize(width, height);
		if (cameraRef.current) {
			cameraRef.current.aspect = width / height;
			cameraRef.current.updateProjectionMatrix();
		}
	};

	const drawHighlightForCountry = (shortName: string) => {
		if (!highlightGroupRef.current) return;
		// Clear current active highlight from the container
		while (highlightGroupRef.current.children.length > 0) {
			const obj = highlightGroupRef.current.children.pop();
			if (obj) obj.removeFromParent();
		}
		// Reuse cached highlight geometry if available
		const cached = highlightCacheRef.current.get(shortName);
		if (cached) {
			highlightGroupRef.current.add(cached.group);
			highlightMaterialsRef.current = cached.materials;
			pulsePhaseRef.current = 0;
			return;
		}
		const polygons = getAtlasPolygonsForCountry(shortName);
		if (!polygons) return;
		// Materials and container for new highlight
		const container = new THREE.Group();
		const borderMat = new THREE.LineBasicMaterial({ color: WINE_PRIMARY, transparent: true, opacity: 0.95, depthTest: false, depthWrite: false });
		const glowMat = new THREE.LineBasicMaterial({ color: WINE_PRIMARY, transparent: true, opacity: 0.45, depthTest: false, depthWrite: false });
		const fillMat = new THREE.MeshBasicMaterial({ color: WINE_PRIMARY, transparent: true, opacity: 0.25, depthTest: false, depthWrite: false, side: THREE.DoubleSide });
		const materials = [borderMat, glowMat] as THREE.LineBasicMaterial[];
		// Draw fill and outlines per polygon using decimated rings
		polygons.forEach((poly: number[][][]) => {
			const sampleRing = (ring: number[][], step: number) => ring.filter((_, i) => i % step === 0);
			const outer = sampleRing(poly[0], DETAIL_SAMPLING.highlightRingStep);
			const holes = poly.slice(1).map(r => sampleRing(r, DETAIL_SAMPLING.highlightRingStep));
			const outer2: THREE.Vector2[] = outer.map(([lon, lat]) => new THREE.Vector2(lon, lat));
			const holes2: THREE.Vector2[][] = holes.map(r => r.map(([lon, lat]) => new THREE.Vector2(lon, lat)));
			// Ensure rings are closed
			const closeIfNeeded = (pts: THREE.Vector2[]) => {
				if (pts.length < 2) return pts;
				const first = pts[0];
				const last = pts[pts.length - 1];
				if (first.x !== last.x || first.y !== last.y) {
					pts.push(first.clone());
				}
				return pts;
			};
			closeIfNeeded(outer2);
			holes2.forEach(h => closeIfNeeded(h));
			// Triangulate and build a unified vertex list per THREE's convention
								let triangles = THREE.ShapeUtils.triangulateShape(outer2, holes2);
					const vertices: THREE.Vector2[] = outer2.concat(...holes2);
					const positions: number[] = [];
					if (triangles.length === 0 && outer2.length >= 3) {
						for (let i = 1; i < outer2.length - 1; i++) triangles.push([0, i, i + 1]);
					}
					triangles.forEach((tri) => {
				const a = vertices[tri[0]];
				const b = vertices[tri[1]];
				const c = vertices[tri[2]];
				if (!a || !b || !c) return;
				[a, b, c].forEach((v2) => {
											const v3 = latLonToVector3(v2.y, v2.x, RADIUS + 0.006);
					positions.push(v3.x, v3.y, v3.z);
				});
			});
			if (positions.length > 0) {
				const g = new THREE.BufferGeometry();
				g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
				const mesh = new THREE.Mesh(g, fillMat);
				mesh.renderOrder = 998;
				container.add(mesh);
			}
			// Draw outlines for outer and holes
			[outer, ...holes].forEach((ring) => {
				const pos: number[] = [];
				ring.forEach(([lon, lat]) => {
					const v = latLonToVector3(lat, lon, RADIUS + 0.010);
					pos.push(v.x, v.y, v.z);
				});
				const g1 = new THREE.BufferGeometry();
				g1.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
				const loop1 = new THREE.LineLoop(g1, borderMat);
				loop1.renderOrder = 999;
				container.add(loop1);
				// Glow ring slightly above
				const pos2 = pos.slice();
				for (let i = 0; i < pos2.length; i += 3) {
					const vv = new THREE.Vector3(pos2[i], pos2[i + 1], pos2[i + 2]).normalize().multiplyScalar(RADIUS + 0.014);
					pos2[i] = vv.x; pos2[i + 1] = vv.y; pos2[i + 2] = vv.z;
				}
				const g2 = new THREE.BufferGeometry();
				g2.setAttribute('position', new THREE.Float32BufferAttribute(pos2, 3));
				const loop2 = new THREE.LineLoop(g2, glowMat);
				loop2.renderOrder = 999;
				container.add(loop2);
			});
		});
		// Attach and cache
		highlightGroupRef.current.add(container);
		highlightMaterialsRef.current = materials;
		pulsePhaseRef.current = 0;
		highlightCacheRef.current.set(shortName, { group: container, materials });
	};

	const handlePanChange = (dx: number, dy: number) => {
		const yawDelta = dx * 0.008;
		const pitchDelta = dy * 0.008;
		let upAxis = new THREE.Vector3(0, 1, 0);
		let rightAxis = new THREE.Vector3(1, 0, 0);
		if (cameraRef.current) {
			upAxis = cameraRef.current.up.clone().normalize();
			const forward = new THREE.Vector3();
			cameraRef.current.getWorldDirection(forward);
			rightAxis = forward.clone().cross(upAxis).normalize();
		}
		const qYaw = new THREE.Quaternion().setFromAxisAngle(upAxis, yawDelta);
		const qPitch = new THREE.Quaternion().setFromAxisAngle(rightAxis, pitchDelta);
		targetQuatRef.current.premultiply(qYaw.multiply(qPitch)).normalize();
	};

	const handlePanEnd = (velocityX: number, velocityY: number) => {
		const scale = 0.00004;
		yawVelRef.current = velocityX * scale;
		pitchVelRef.current = velocityY * scale;
	};

	const handlePinchBegin = () => {
		pinchBaseDistanceRef.current = currentDistanceRef.current;
	};

	const handlePinchChange = (scale: number) => {
		if (!cameraRef.current) return;
		const desired = THREE.MathUtils.clamp(pinchBaseDistanceRef.current / scale, 1.8, 9.0);
		targetDistanceRef.current = desired;
		currentDistanceRef.current = desired;
		cameraRef.current.position.z = desired;
	};

	const panGesture = Gesture.Pan()
		.minDistance(0)
		.onChange((e) => { runOnJS(handlePanChange)(e.changeX, e.changeY); })
		.onEnd((e) => { runOnJS(handlePanEnd)(e.velocityX, e.velocityY); });

	const pinchGesture = Gesture.Pinch()
		.onBegin(() => { runOnJS(handlePinchBegin)(); })
		.onChange((e) => { runOnJS(handlePinchChange)(e.scale); });

	const gestures = Gesture.Simultaneous(panGesture, pinchGesture);

	const flyTo = (country: string) => {
		const ll = getLatLonForCountry(country);
		if (!ll) return;
		const q = computeQuaternionToFaceLatLon(ll.lat, ll.lon);
		yawVelRef.current = 0;
		pitchVelRef.current = 0;
		targetQuatRef.current.copy(q);
		// Suggested zoom for country
		targetDistanceRef.current = suggestDistanceForExtent(getExtentForCountry(country));
		// Highlight
		drawHighlightForCountry(country);
	};

	useImperativeHandle(ref, () => ({ flyToCountry: flyTo }));

	useEffect(() => { if (selectedCountry) flyTo(selectedCountry); }, [selectedCountry]);

	useEffect(() => {
		return () => {
			if (rendererRef.current && (rendererRef.current as any).setAnimationLoop) {
				(rendererRef.current as any).setAnimationLoop(null);
			}
			if (rafRef.current) cancelAnimationFrame(rafRef.current);
		};
	}, []);

	if (Platform.OS === 'web') {
		return (
			<View style={[{ width: '100%', height: 500, alignItems: 'center', justifyContent: 'center', backgroundColor: wineTheme.colors.background }, style]}>
				<Text>3D globe is not supported on web. Please run on iOS/Android.</Text>
			</View>
		);
	}

	return (
		<View style={[{ width: '100%', height: 500, overflow: 'hidden' }, style]}
			onLayout={(e) => {
				const { width, height } = e.nativeEvent.layout;
				updateSize(width, height);
			}}
		>
			<GestureDetector gesture={gestures}>
				<GLView
					ref={(r: GLView | null) => { glViewRef.current = r; }}
					style={{ flex: 1 }}
					msaaSamples={2}
					onContextCreate={async (gl: ExpoWebGLRenderingContext) => { await ensureScene(gl); }}
				/>
			</GestureDetector>
			{ready ? null : (
				<View style={{ position: 'absolute', top: 8, right: 8, backgroundColor: '#00000066', paddingHorizontal: 8, paddingVertical: 4, borderRadius: 6 }}>
					<Text style={{ color: 'white', fontSize: 12 }}>Initializing…</Text>
				</View>
			)}
		</View>
	);
};

function addGraticule(group: THREE.Group, radius: number) {
	const mat = new THREE.LineBasicMaterial({ color: 0x9aa0a6, transparent: true, opacity: 0.35 });
	const addRing = (latDeg: number) => {
		const points: number[] = [];
		for (let lon = -180; lon <= 180; lon += 6) {
			const v = latLonToVector3(latDeg, lon, radius);
			points.push(v.x, v.y, v.z);
		}
		const geom = new THREE.BufferGeometry();
		geom.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
		group.add(new THREE.LineLoop(geom, mat));
	};
	const addMeridian = (lonDeg: number) => {
		const points: number[] = [];
		for (let lat = -90; lat <= 90; lat += 6) {
			const v = latLonToVector3(lat, lonDeg, radius);
			points.push(v.x, v.y, v.z);
		}
		const geom = new THREE.BufferGeometry();
		geom.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
		group.add(new THREE.Line(geom, mat));
	};
	for (let lat = -60; lat <= 60; lat += 30) addRing(lat);
	for (let lon = -150; lon <= 150; lon += 30) addMeridian(lon);
}

function addLandFill(group: THREE.Group, radius: number) {
	let world: any;
	try { world = require('world-atlas/land-50m.json'); } catch { console.log('addLandFill: failed to require world-atlas'); return; }
	const landFeat: any = feature(world, world.objects.land);
	const geometries: any[] = [];
	if (landFeat?.type === 'FeatureCollection') {
		landFeat.features?.forEach((f: any) => { if (f?.geometry) geometries.push(f.geometry); });
	} else if (landFeat?.type === 'Feature' && landFeat.geometry) {
		geometries.push(landFeat.geometry);
	} else if (landFeat && landFeat.geometry) {
		geometries.push(landFeat.geometry);
	}
	if (geometries.length === 0) { console.log('addLandFill: no geometry'); return; }
	const fillMat = new THREE.MeshBasicMaterial({ color: 'yellow', transparent: false, opacity: 1.0, depthTest: false, depthWrite: false, side: THREE.DoubleSide });

	const sampleRing = (ring: number[][], step: number) => {
		const out: number[][] = [];
		for (let i = 0; i < ring.length; i += step) out.push(ring[i]);
		return out;
	};

	// Utility: clamp latitude to avoid pole singularities
	const clampLat = (lat: number) => Math.max(-89.999, Math.min(89.999, lat));
	// Utility: unwrap longitudes so consecutive points are within 180° to avoid dateline wedges
	const unwrapRing = (ring: number[][]): number[][] => {
		if (!ring || ring.length === 0) return ring;
		const out: number[][] = [];
		let offset = 0;
		for (let i = 0; i < ring.length; i++) {
			const lon = ring[i][0];
			const lat = ring[i][1];
			if (i > 0) {
				const prevLon = out[out.length - 1][0];
				let diff = lon + offset - prevLon;
				if (diff > 180) offset -= 360;
				else if (diff < -180) offset += 360;
			}
			out.push([lon + offset, clampLat(lat)]);
		}
		return out;
	};

	// Fill a polygon by building spherical triangle fans: one for the outer ring (land), then fans for each hole (water color) to visually subtract
	const fillPolyOuterRing = (poly: number[][][]) => {
		const outerRing = sampleRing(poly[0], DETAIL_SAMPLING.landFillStep);
		const holeRings = poly.slice(1).map(r => sampleRing(r, DETAIL_SAMPLING.landFillStep));
		if (!outerRing || outerRing.length < 3) return;

		const toVec3 = (lon: number, lat: number, r: number) => latLonToVector3(lat, lon, r);
		const landRadius = radius + 0.02;
		const holeRadius = radius + 0.021;

		// Compute a safe spherical centroid near the surface for fan triangulation
		const centroid = (() => {
			const acc = new THREE.Vector3(0, 0, 0);
			outerRing.forEach(([lon, lat]) => acc.add(toVec3(lon, lat, 1)));
			if (acc.lengthSq() === 0) return new THREE.Vector3(0, 0, landRadius);
			return acc.normalize().multiplyScalar(landRadius);
		})();

		// Build fan for outer ring (land)
		const vertsOuter: THREE.Vector3[] = outerRing.map(([lon, lat]) => toVec3(lon, lat, landRadius));
		const landPositions: number[] = [];
		for (let i = 0; i < vertsOuter.length; i++) {
			const a = centroid;
			const b = vertsOuter[i];
			const c = vertsOuter[(i + 1) % vertsOuter.length];
			landPositions.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
		}
		if (landPositions.length > 0) {
			const g = new THREE.BufferGeometry();
			g.setAttribute('position', new THREE.Float32BufferAttribute(landPositions, 3));
			const mesh = new THREE.Mesh(g, fillMat);
			mesh.renderOrder = 1200;
			mesh.frustumCulled = false;
			group.add(mesh);
		}

		// Build fans for holes using water color to visually carve them out
		if (holeRings.length > 0) {
			const holeMat = new THREE.MeshBasicMaterial({ color: THEME.surface, transparent: false, opacity: 1.0, depthTest: false, depthWrite: false, side: THREE.DoubleSide });
			holeRings.forEach((ring) => {
				if (!ring || ring.length < 3) return;
				const vertsHole: THREE.Vector3[] = ring.map(([lon, lat]) => toVec3(lon, lat, holeRadius));
				const holePositions: number[] = [];
				for (let i = 0; i < vertsHole.length; i++) {
					const a = centroid.clone().setLength(holeRadius);
					const b = vertsHole[i];
					const c = vertsHole[(i + 1) % vertsHole.length];
					holePositions.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
				}
				if (holePositions.length > 0) {
					const g = new THREE.BufferGeometry();
					g.setAttribute('position', new THREE.Float32BufferAttribute(holePositions, 3));
					const mesh = new THREE.Mesh(g, holeMat);
					mesh.renderOrder = 1210;
					mesh.frustumCulled = false;
					group.add(mesh);
				}
			});
		}

		if (DEBUG_LAND) {
			try {
				const loopPos: number[] = [];
				outerRing.map(([lon, lat]) => toVec3(lon, lat, landRadius + 0.001)).forEach(v => { loopPos.push(v.x, v.y, v.z); });
				const loopGeom = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(loopPos, 3));
				const loopMat = new THREE.LineBasicMaterial({ color: 0x00ff00, transparent: true, opacity: 0.9, depthTest: true });
				const loop = new THREE.LineLoop(loopGeom, loopMat);
				loop.renderOrder = 1300;
				group.add(loop);
			} catch {}
		}
	};

	const DEBUG_LAND = false;

	const drawRings = (arr: any) => {
		const polys: [number, number][][][] = Array.isArray(arr[0][0][0]) ? (arr as any) : [arr as any];
		console.log('addLandFill: polys count', polys.length, 'first poly outer len', polys[0]?.[0]?.length ?? 0);
		polys.forEach((poly: any, idx: number) => {
			try {
				fillPolyOuterRing(poly);
				if (DEBUG_LAND) {
					const outer = poly[0];
					const verts3D: THREE.Vector3[] = sampleRing(outer, DETAIL_SAMPLING.landFillStep).map(([lon, lat]) => latLonToVector3(lat, lon, radius + 0.02));
					const loopPos: number[] = [];
					verts3D.forEach(v => { loopPos.push(v.x, v.y, v.z); });
					const loopGeom = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(loopPos, 3));
					const loopMat = new THREE.LineBasicMaterial({ color: 0x00ff00, transparent: true, opacity: 0.9, depthTest: true });
					const loop = new THREE.LineLoop(loopGeom, loopMat);
					loop.renderOrder = 1100;
					group.add(loop);
					const dbgGeom = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(verts3D.flatMap(v => [v.x, v.y, v.z]), 3));
					const dbgMat = new THREE.PointsMaterial({ color: 0x000000, size: 6, sizeAttenuation: false, depthTest: false });
					const dbgPoints = new THREE.Points(dbgGeom, dbgMat);
					dbgPoints.renderOrder = 1300;
					group.add(dbgPoints);
				}
			} catch (e) { console.log('addLandFill: poly error', idx, e); }
		});
	};

	const drawFromGeometry = (geometry: any) => {
		if (!geometry) return;
		switch (geometry.type) {
			case 'Polygon':
				fillPolyOuterRing(geometry.coordinates);
				break;
			case 'MultiPolygon':
				(geometry.coordinates as any[]).forEach((poly: any) => fillPolyOuterRing(poly));
				break;
			default:
				break;
		}
	};

	console.log('addLandFill: geometries count', geometries.length);
	geometries.forEach((g, idx) => {
		try { drawFromGeometry(g); } catch (e) { console.log('addLandFill: geometry error', idx, e); }
	});
}

function addCountryBorders(group: THREE.Group, shortNames: string[], radius: number) {
	let data: any;
	try { data = require('world-countries'); } catch { return; }
	const nameMap: Record<string, string> = {
		US: 'United States', Italy: 'Italy', France: 'France', Spain: 'Spain', Portugal: 'Portugal', Chile: 'Chile', Argentina: 'Argentina', Austria: 'Austria', Germany: 'Germany', Australia: 'Australia',
	};
	const wanted = new Set(shortNames.map(k => nameMap[k] || k));
	// Country borders for listed countries only, in theme text color
	const mat = new THREE.LineBasicMaterial({ color: THEME.text, transparent: true, opacity: 0.75 });
	const processCoords = (arr: any) => {
		const rings: [number, number][][] = Array.isArray(arr[0][0]) ? (arr as [number, number][][]) : [arr as [number, number][]];
		rings.forEach((ring: [number, number][]) => {
			// Decimate points modestly for perf
			const positions: number[] = [];
			for (let i = 0; i < ring.length; i += DETAIL_SAMPLING.borderRingStep) {
				const [lon, lat] = ring[i];
				const v = latLonToVector3(lat, lon, radius);
				positions.push(v.x, v.y, v.z);
			}
			const geom = new THREE.BufferGeometry();
			geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
			const loop1 = new THREE.LineLoop(geom, mat);
			loop1.renderOrder = 999;
			group.add(loop1);
			// subtle glow
			const pos2 = positions.slice();
			for (let i = 0; i < pos2.length; i += 3) {
				const vv = new THREE.Vector3(pos2[i], pos2[i + 1], pos2[i + 2]).normalize().multiplyScalar(radius + 0.014);
				pos2[i] = vv.x; pos2[i + 1] = vv.y; pos2[i + 2] = vv.z;
			}
			const glow = new THREE.LineBasicMaterial({ color: THEME.text, transparent: true, opacity: 0.35, depthTest: false, depthWrite: false });
			const loop2 = new THREE.LineLoop(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(pos2, 3)), glow);
			loop2.renderOrder = 999;
			group.add(loop2);
		});
	};
	data
		.filter((c: any) => wanted.has(c.name?.common))
		.forEach((country: any) => {
			const geometry = country.geometry || null;
			if (!geometry) return;
			if (geometry.type === 'Polygon') processCoords(geometry.coordinates);
			else if (geometry.type === 'MultiPolygon') geometry.coordinates.forEach((poly: any) => processCoords(poly));
		});
}

function addLandOutlines(group: THREE.Group, radius: number) {
	let world: any;
	try { world = require('world-atlas/land-50m.json'); } catch { world = null; }
	let geom: any = null;
	if (world) {
		const landFeat: any = feature(world, world.objects.land);
		geom = landFeat && landFeat.geometry ? landFeat.geometry : null;
	}
	// Land coastlines in theme text color
	const mat = new THREE.LineBasicMaterial({ color: THEME.text, transparent: true, opacity: 0.55 });

	const drawRings = (arr: any) => {
		const rings: [number, number][][] = Array.isArray(arr[0][0]) ? (arr as [number, number][][]) : [arr as [number, number][]];
		rings.forEach((ring) => {
			const positions: number[] = [];
			for (let i = 0; i < ring.length; i += DETAIL_SAMPLING.coastRingStep) {
				const [lon, lat] = ring[i];
				const v = latLonToVector3(lat, lon, radius);
				positions.push(v.x, v.y, v.z);
			}
			const g = new THREE.BufferGeometry();
			g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
			const line = new THREE.LineLoop(g, mat);
			line.renderOrder = 750;
			group.add(line);
		});
	};

	const drawLines = (arr: any) => {
		const lines: [number, number][][] = arr as [number, number][][];
		lines.forEach((line) => {
			const positions: number[] = [];
			for (let i = 0; i < line.length; i += DETAIL_SAMPLING.coastLineStep) {
				const [lon, lat] = line[i];
				const v = latLonToVector3(lat, lon, radius);
				positions.push(v.x, v.y, v.z);
			}
			const g = new THREE.BufferGeometry();
			g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
			group.add(new THREE.Line(g, mat));
		});
	};

	const drawGeom = (geometry: any) => {
		if (!geometry || !geometry.type) return;
		switch (geometry.type) {
			case 'Polygon':
				drawRings(geometry.coordinates);
				break;
			case 'MultiPolygon':
				(geometry.coordinates as any[]).forEach((poly: any) => drawRings(poly));
				break;
			case 'LineString':
				drawLines([geometry.coordinates]);
				break;
			case 'MultiLineString':
				drawLines(geometry.coordinates);
				break;
			default:
				break;
		}
	};

	if (geom) {
		drawGeom(geom);
		return;
	}
	// Fallback: derive coastlines from land mesh (exterior boundaries)
	try {
		const topo = require('world-atlas/land-50m.json');
		const coast = topoMesh(topo, topo.objects.land, (a: any, b: any) => a === b);
		if (coast && coast.type) {
			drawGeom(coast);
		}
	} catch {}
}

const GlobeSelector = forwardRef(GlobeSelectorInner);
GlobeSelector.displayName = 'GlobeSelector';

export default GlobeSelector; 