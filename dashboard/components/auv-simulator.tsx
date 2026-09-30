'use client'

import { Canvas, useFrame } from '@react-three/fiber'
import { Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'

type Point = [number, number, number]
type Simulation = { trajectory: Point[], telemetry: Array<{ alpha?: number | null }>, success: boolean, collided: boolean, energy_kj: number, steps: number }
type Obstacle = { x: number, y: number, z: number, radius: number }
type Current = [number, number, number]

const colors: Record<string, string> = { APF: '#ffb454', 'Pure DQN': '#8fa9ff', 'Pure PPO': '#c899ff', 'PPO-APF (adaptive α)': '#36dc9a', 'DQN-APF (fixed α)': '#20bfa5' }
const methods = Object.keys(colors)

function worldPoint(point: Point): Point { return [point[0] / 12 - 4.2, -(point[2] / 12 - 2.1), point[1] / 12 - 4.2] }

function Auv({ trajectory, running, resetToken }: { trajectory: Point[], running: boolean, resetToken: number }) {
  const ref = useRef<THREE.Group>(null)
  const progress = useRef(0)
  const lastReset = useRef(resetToken)

  useFrame((_, delta) => {
    if (!ref.current || trajectory.length === 0) return
    if (lastReset.current !== resetToken) {
      progress.current = 0
      lastReset.current = resetToken
    }
    if (running) progress.current = Math.min(trajectory.length - 1, progress.current + delta * 8)
    const index = Math.floor(progress.current)
    const point = trajectory[index] ?? trajectory[trajectory.length - 1]
    const next = trajectory[Math.min(index + 1, trajectory.length - 1)] ?? point
    ref.current.position.copy(new THREE.Vector3(...worldPoint(point)))
    ref.current.lookAt(new THREE.Vector3(...worldPoint(next)))
    ref.current.position.y += Math.sin(Date.now() * .004) * .025
  })

  return <group ref={ref} position={worldPoint(trajectory[0] ?? [5, 5, 5])}>
    <mesh castShadow><capsuleGeometry args={[0.26, 1.05, 8, 16]} /><meshStandardMaterial color="#d8e7eb" metalness={0.7} roughness={0.22} /></mesh>
    <mesh position={[0, 0.28, 0.1]}><boxGeometry args={[0.1, 0.05, 0.55]} /><meshStandardMaterial color="#e74f68" emissive="#e74f68" emissiveIntensity={2} /></mesh>
  </group>
}

function Obstacle({ obstacle, index }: { obstacle: Obstacle, index: number }) {
  const position = worldPoint([obstacle.x, obstacle.y, obstacle.z])
  const physicalRadius = obstacle.radius / 12
  const safetyRadius = (obstacle.radius + 7) / 12
  return <group position={position}>
    <mesh castShadow><icosahedronGeometry args={[physicalRadius, 2]} /><meshStandardMaterial color="#314d5b" emissive="#172933" emissiveIntensity={1.1} roughness={0.75} /></mesh>
    <mesh><icosahedronGeometry args={[physicalRadius * 1.04, 2]} /><meshBasicMaterial color="#ff5578" wireframe transparent opacity={0.8} /></mesh>
    <mesh><sphereGeometry args={[safetyRadius, 20, 12]} /><meshBasicMaterial color="#ff5578" wireframe transparent opacity={0.1} /></mesh>
  </group>
}

function CurrentField({ current }: { current: Current }) {
  const direction = new THREE.Vector3(current[0], -current[2], current[1]).normalize()
  const arrows = []
  for (let x = -38; x <= 38; x += 19) {
    for (let z = -32; z <= 32; z += 16) {
      arrows.push(<arrowHelper key={`${x}-${z}`} args={[direction, new THREE.Vector3(x / 12, 2.3, z / 12), 0.55, 0x42d9ed, 0.14, 0.1]} />)
    }
  }
  return <group>{arrows}</group>
}

function Scene({ running, showPaths, selectedMethod, simulations, obstacles, current, resetToken }: { running: boolean, showPaths: boolean, selectedMethod: string, simulations: Record<string, Simulation>, obstacles: Obstacle[], current: Current, resetToken: number }) {
  const selected = simulations[selectedMethod]
  return <>
    <color attach="background" args={['#07141f']} /><fog attach="fog" args={['#07141f', 8, 24]} />
    <ambientLight intensity={0.55} color="#6ca9bf" /><directionalLight position={[4, 7, 5]} intensity={2.2} color="#c9f4ff" /><pointLight position={[-4, 0, 2]} color="#26b8dc" intensity={10} distance={12} />
    <PerspectiveCamera makeDefault position={[8, 5.5, 10]} fov={42} /><OrbitControls enablePan={false} minDistance={6} maxDistance={18} target={[0, -.5, 0]} />
    <gridHelper args={[18, 18, '#173746', '#0d2531']} position={[0, 3, 0]} /><mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 2.96, 0]}><planeGeometry args={[22, 22]} /><meshStandardMaterial color="#071a27" transparent opacity={.65} /></mesh>
    <CurrentField current={current} />
    {showPaths && methods.map((method) => simulations[method] && <Line key={method} points={simulations[method].trajectory.map(worldPoint)} color={colors[method]} lineWidth={method === selectedMethod ? 2.4 : 1.3} transparent opacity={method === selectedMethod ? 1 : .35} />)}
    {obstacles.map((obstacle, index) => <Obstacle key={index} obstacle={obstacle} index={index} />)}
    <mesh position={worldPoint([90, 90, 40])}><sphereGeometry args={[.18, 16, 16]} /><meshBasicMaterial color="#4dffb2" /></mesh>
    <Auv trajectory={selected?.trajectory ?? [[5, 5, 5]]} running={running} resetToken={resetToken} />
  </>
}

export function AuvSimulator({ running, showPaths, selectedMethod, seed, resetToken, onSimulation }: { running: boolean, showPaths: boolean, selectedMethod: string, seed: number, resetToken: number, onSimulation: (method: string, data: Simulation) => void }) {
  const [simulations, setSimulations] = useState<Record<string, Simulation>>({})
  const [obstacles, setObstacles] = useState<Obstacle[]>([])
  const [current, setCurrent] = useState<Current>([0, 0, 0])
  useEffect(() => { fetch('/api/environment').then((response) => response.json()).then((data) => { setObstacles(data.obstacles ?? []); setCurrent(data.world?.current ?? [0, 0, 0]) }).catch(() => undefined) }, [])
  useEffect(() => { methods.forEach((method) => { fetch(`/api/simulation?method=${encodeURIComponent(method)}&seed=${seed}`).then((response) => response.json()).then((data) => { if (data.trajectory) { setSimulations((current) => ({ ...current, [method]: data })); onSimulation(method, data) } }).catch(() => undefined) }) }, [seed, onSimulation])
  return <div className="relative h-full min-h-[400px] w-full overflow-hidden rounded-lg border border-cyan-300/15"><Canvas shadows dpr={[1, 1.5]}><Scene running={running} showPaths={showPaths} selectedMethod={selectedMethod} simulations={simulations} obstacles={obstacles} current={current} resetToken={resetToken} /></Canvas><div className="pointer-events-none absolute left-4 top-4 font-mono text-[10px] uppercase tracking-[0.18em] text-cyan-200/65">Live 3D Occupancy Field / Real Simulation</div><div className="pointer-events-none absolute bottom-4 left-4 flex gap-4 rounded border border-white/10 bg-slate-950/70 px-3 py-2 font-mono text-[9px] text-slate-400 backdrop-blur">{methods.map((method) => <span key={method}><i className="mr-1 inline-block size-1.5 rounded-full" style={{ backgroundColor: colors[method] }} />{method}</span>)}<span><i className="mr-1 inline-block size-1.5 rounded-full bg-cyan-300" />CURRENT FIELD</span></div></div>
}
