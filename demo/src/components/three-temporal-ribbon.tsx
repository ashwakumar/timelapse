"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { Component, type ReactNode, useMemo, useRef } from "react";
import * as THREE from "three";

type ThreeTemporalRibbonProps = {
  playing: boolean;
  urgent: boolean;
};

type CanvasBoundaryState = { failed: boolean };

class CanvasBoundary extends Component<
  { children: ReactNode },
  CanvasBoundaryState
> {
  state: CanvasBoundaryState = { failed: false };

  static getDerivedStateFromError(): CanvasBoundaryState {
    return { failed: true };
  }

  render() {
    return this.state.failed ? <StaticRibbon /> : this.props.children;
  }
}

function StaticRibbon() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none h-full min-h-20 w-full opacity-40"
      style={{
        backgroundImage:
          "radial-gradient(circle at 16% 55%, rgba(100, 220, 235, 0.5) 0 1px, transparent 2px), radial-gradient(circle at 50% 44%, rgba(100, 220, 235, 0.35) 0 1px, transparent 2px), radial-gradient(circle at 82% 48%, rgba(240, 173, 78, 0.48) 0 1px, transparent 2px)",
        backgroundSize: "17px 17px, 23px 23px, 19px 19px",
        maskImage:
          "linear-gradient(90deg, transparent, black 10%, black 90%, transparent)",
      }}
    />
  );
}

function TemporalField({ playing, urgent }: ThreeTemporalRibbonProps) {
  const historyRef = useRef<THREE.Points>(null);
  const futureRef = useRef<THREE.Points>(null);
  const boundaryRef = useRef<THREE.Group>(null);

  const { historyPositions, futurePositions, ribbonPositions } = useMemo(() => {
    const history: number[] = [];
    const future: number[] = [];
    const ribbon: number[] = [];

    for (let index = 0; index < 44; index += 1) {
      const progress = index / 43;
      const x = -3.45 + progress * 3.35;
      const y = Math.sin(progress * Math.PI * 2.2) * 0.18 - 0.08;
      history.push(x, y, (index % 3) * 0.008);
    }

    for (let index = 0; index < 58; index += 1) {
      const progress = index / 57;
      const x = 0.1 + progress * 3.45;
      const y = Math.sin(progress * Math.PI * 1.6) * 0.2 + progress * 0.12;
      future.push(x, y, ((index + 1) % 4) * 0.007);
    }

    for (let index = 0; index < 32; index += 1) {
      const progress = index / 31;
      ribbon.push(
        -3.45 + progress * 6.9,
        Math.sin(progress * Math.PI * 2) * 0.12,
        -0.03,
      );
    }

    return {
      historyPositions: new Float32Array(history),
      futurePositions: new Float32Array(future),
      ribbonPositions: new Float32Array(ribbon),
    };
  }, []);

  useFrame(({ clock }) => {
    if (!playing) return;
    const drift = Math.sin(clock.getElapsedTime() * 0.42) * 0.012;
    if (historyRef.current) historyRef.current.position.y = drift;
    if (futureRef.current) futureRef.current.position.y = -drift;
    if (boundaryRef.current) boundaryRef.current.rotation.z = drift * 0.28;
  });

  const futureColor = urgent ? "#d99455" : "#8ab8ba";

  return (
    <group>
      <group ref={boundaryRef}>
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              args={[new Float32Array([0, -0.7, 0, 0, 0.7, 0]), 3]}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#e9dcc9" transparent opacity={0.24} />
        </line>
      </group>
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[ribbonPositions, 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#7db8bc" transparent opacity={0.2} />
      </line>
      <points ref={historyRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[historyPositions, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          color="#79c7cf"
          size={0.045}
          sizeAttenuation
          transparent
          opacity={0.48}
          depthWrite={false}
        />
      </points>
      <points ref={futureRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[futurePositions, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          color={futureColor}
          size={0.05}
          sizeAttenuation
          transparent
          opacity={0.4}
          depthWrite={false}
        />
      </points>
    </group>
  );
}

export default function ThreeTemporalRibbon({
  playing,
  urgent,
}: ThreeTemporalRibbonProps) {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none h-full min-h-20 w-full overflow-hidden opacity-60"
    >
      <CanvasBoundary>
        <Canvas
          aria-hidden="true"
          camera={{ position: [0, 0, 7], fov: 45 }}
          dpr={[1, 1.5]}
          frameloop={playing ? "always" : "demand"}
          gl={{ alpha: true, antialias: true, powerPreference: "low-power" }}
          fallback={<StaticRibbon />}
        >
          <TemporalField playing={playing} urgent={urgent} />
        </Canvas>
      </CanvasBoundary>
    </div>
  );
}
