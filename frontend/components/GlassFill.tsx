import React, { useEffect, useMemo, useRef, useState } from 'react';
import { View, LayoutChangeEvent } from 'react-native';
import Svg, { Defs, ClipPath, Rect, Path } from 'react-native-svg';

export type GlassFillProps = {
  width?: number;
  durationMs?: number;
  wineColor?: string;
  maxFillRatio?: number; // 0-1, cap how high the fill can go in the bowl
};

export default function GlassFill({ width: widthProp, durationMs = 1200, wineColor = '#8B0000', maxFillRatio = 0.85 }: GlassFillProps) {
  const [containerW, setContainerW] = useState<number | null>(null);
  const [containerH, setContainerH] = useState<number | null>(null);

  const onLayout = (e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    setContainerW(width);
    setContainerH(height);
  };

  // Maintain ~1:1.4 aspect; auto-fit to container without extra scaling margins
  const computedWidth = (() => {
    if (widthProp) return widthProp;
    if (!containerW || !containerH) return 140;
    return Math.max(100, Math.floor(Math.min(containerW, containerH / 1.4)));
  })();

  const height = Math.round(computedWidth * 1.4);

  // Geometry
  const cx = Math.round(computedWidth / 2);
  const bowlHeight = Math.round(height * 0.58);
  const bowlTopY = Math.round(height * 0.06);
  const bowlBottomY = bowlTopY + bowlHeight;
  const bottomLift = Math.max(2, Math.round(height * 0.02));
  const bowlBottomYAdj = bowlBottomY - bottomLift;
  const rimWidth = Math.round(computedWidth * 0.5); // narrower top opening
  const maxWidth = Math.round(computedWidth * 0.78); // widest point
  const halfRim = Math.round(rimWidth / 2);
  const halfMax = Math.round(maxWidth / 2);
  const bottomArcWidth = Math.round(maxWidth * 0.35);
  const halfBottomArc = Math.round(bottomArcWidth / 2);

  const stemWidth = Math.max(4, Math.round(computedWidth * 0.06));
  const stemHeight = Math.round(height * 0.22);
  const stemX = Math.round(cx - stemWidth / 2);
  const stemY = bowlBottomYAdj + Math.round(height * 0.03);

  const baseWidth = Math.round(computedWidth * 0.38);
  const baseHeight = Math.max(4, Math.round(computedWidth * 0.05));
  const baseX = Math.round(cx - baseWidth / 2);
  const baseY = stemY + stemHeight + Math.round(height * 0.02);

  const [progress, setProgress] = useState(0);
  const startTsRef = useRef<number | null>(null);

  const animate = useMemo(() => {
    return (ts: number) => {
      if (startTsRef.current === null) startTsRef.current = ts;
      const elapsed = ts - startTsRef.current;
      const t = Math.min(1, elapsed / durationMs);
      const eased = 1 - Math.pow(1 - t, 3);
      setProgress(eased);
      if (t < 1) requestAnimationFrame(animate);
    };
  }, [durationMs]);

  useEffect(() => {
    setProgress(0);
    startTsRef.current = null;
    const id = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(id);
  }, [animate, computedWidth, durationMs, wineColor, maxFillRatio]);

  const cappedProgress = Math.min(1, Math.max(0, progress)) * Math.min(1, Math.max(0, maxFillRatio));
  const strokeColor = 'rgba(160,160,160,0.7)';

  // Closed bowl path for clipping (top line included to close the shape)
  const leftRimX = cx - halfRim;
  const rightRimX = cx + halfRim;
  const leftMaxX = cx - halfMax;
  const rightMaxX = cx + halfMax;
  const midY = bowlTopY + Math.round(bowlHeight * 0.55);
  const leftArcStartX = cx - halfBottomArc;
  const rightArcEndX = cx + halfBottomArc;

  const bowlClipPath = [
    `M ${leftRimX} ${bowlTopY}`,
    // Left side curve down to bottom-left arc start
    `C ${leftRimX - Math.round(halfRim * 0.3)} ${bowlTopY + Math.round(bowlHeight * 0.25)}, ${leftMaxX} ${midY}, ${leftArcStartX} ${bowlBottomYAdj}`,
    // Rounded bottom arc to bottom-right arc end
    `A ${halfBottomArc} ${halfBottomArc} 0 0 0 ${rightArcEndX} ${bowlBottomYAdj}`,
    // Right side curve back up to right rim
    `C ${rightMaxX} ${midY}, ${rightRimX + Math.round(halfRim * 0.3)} ${bowlTopY + Math.round(bowlHeight * 0.25)}, ${rightRimX} ${bowlTopY}`,
    'Z',
  ].join(' ');

  // Open-top outline path: draw sides and rounded bottom only (no top line)
  const bowlOutlinePath = [
    `M ${leftRimX} ${bowlTopY}`,
    `C ${leftRimX - Math.round(halfRim * 0.3)} ${bowlTopY + Math.round(bowlHeight * 0.25)}, ${leftMaxX} ${midY}, ${leftArcStartX} ${bowlBottomYAdj}`,
    `A ${halfBottomArc} ${halfBottomArc} 0 0 0 ${rightArcEndX} ${bowlBottomYAdj}`,
    `C ${rightMaxX} ${midY}, ${rightRimX + Math.round(halfRim * 0.3)} ${bowlTopY + Math.round(bowlHeight * 0.25)}, ${rightRimX} ${bowlTopY}`,
  ].join(' ');

  // Compute fill rect inside the bowl, capped below rim
  const bowlTopFillY = bowlTopY + Math.round(bowlHeight * (1 - maxFillRatio));
  const fillBottomY = bowlBottomYAdj;
  const currentFillY = Math.round(fillBottomY - (fillBottomY - bowlTopFillY) * cappedProgress);
  const fillHeight = Math.max(0, fillBottomY - currentFillY);

  return (
    <View style={{ flex: 1 }} onLayout={onLayout}>
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
        <Svg width={computedWidth} height={height}>
          <Defs>
            <ClipPath id="bowlClip">
              <Path d={bowlClipPath} />
            </ClipPath>
          </Defs>

          {/* Wine fill clipped to bowl, horizontal top edge */}
          <Rect
            x={leftRimX - (halfMax - halfRim)}
            y={currentFillY}
            width={maxWidth}
            height={fillHeight}
            clipPath="url(#bowlClip)"
            fill={wineColor}
            opacity={0.9}
          />

          {/* Open-top bowl outline */}
          <Path d={bowlOutlinePath} fill="none" stroke={strokeColor} strokeWidth={2} />

          {/* Stem */}
          <Rect x={stemX} y={stemY} width={stemWidth} height={stemHeight} fill="none" stroke={strokeColor} strokeWidth={2} />

          {/* Base */}
          <Rect x={baseX} y={baseY} width={baseWidth} height={baseHeight} fill="none" stroke={strokeColor} strokeWidth={2} rx={baseHeight / 2} ry={baseHeight / 2} />
        </Svg>
      </View>
    </View>
  );
} 