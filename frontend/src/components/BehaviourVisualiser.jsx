import { useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'

function Car({ features }) {
  const meshRef = useRef()

  useFrame((state) => {
    if (!meshRef.current) return
    const t = state.clock.getElapsedTime()
    const brakingTilt = (features?.braking_flag || 0) * -0.07
    const laneTilt    = (features?.lane_change_flag || 0) * 0.05
    const wobble      = Math.sin(t * 3) * 0.004

    meshRef.current.rotation.x = brakingTilt + wobble
    meshRef.current.rotation.z = laneTilt
  })

  return (
    <group ref={meshRef}>
      {/* Body */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[2, 0.55, 1]} />
        <meshStandardMaterial color="#1a1a1a" metalness={0.9} roughness={0.15} />
      </mesh>

      {/* Roof */}
      <mesh position={[0, 0.52, 0]}>
        <boxGeometry args={[1.1, 0.38, 0.88]} />
        <meshStandardMaterial color="#0a0a0a" metalness={0.85} roughness={0.2} />
      </mesh>

      {/* Wheels */}
      {[[-0.65, -0.28, 0.5], [0.65, -0.28, 0.5], [-0.65, -0.28, -0.5], [0.65, -0.28, -0.5]].map(([x, y, z], i) => (
        <mesh key={i} position={[x, y, z]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.26, 0.26, 0.14, 20]} />
          <meshStandardMaterial color="#0a0a0a" metalness={0.3} roughness={0.85} />
        </mesh>
      ))}

      {/* Headlights */}
      {[[-0.28, 0, 0.52], [0.28, 0, 0.52]].map(([x, y, z], i) => (
        <mesh key={i} position={[x, y, z]}>
          <sphereGeometry args={[0.06, 8, 8]} />
          <meshStandardMaterial color="#eaeaea" emissive="#eaeaea" emissiveIntensity={1.2} />
        </mesh>
      ))}
    </group>
  )
}

export default function BehaviourVisualiser({ features }) {
  return (
    <div className="card">
      <div className="card-title">3D Behaviour Visualiser</div>
      <div className="three-wrap">
        <Canvas camera={{ position: [3, 1.8, 4], fov: 42 }}>
          <ambientLight intensity={0.28} />
          <directionalLight position={[4, 6, 4]} intensity={0.8} color="#eaeaea" />
          <pointLight position={[-3, 2, -3]} color="#2a2a2a" intensity={0.35} />
          <Car features={features} />
          <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={0.6} />
          {/* Ground */}
          <mesh position={[0, -0.56, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <planeGeometry args={[20, 20]} />
            <meshStandardMaterial color="#0a0a0a" />
          </mesh>
        </Canvas>
      </div>
    </div>
  )
}
