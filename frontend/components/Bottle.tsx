import React, { useState } from 'react';
import { View, LayoutChangeEvent } from 'react-native';
import Svg, {
  Path,
  G,
  Defs,
  ClipPath,
  Rect,
  Text as SvgText,
  TSpan,
  Image as SvgImage,
} from 'react-native-svg';

export type BottleProps = {
  color?: string;
  scale?: number;
  // dynamic label props
  title?: string;
  subtitle?: string;
  logoUri?: string;     // optional small logo for the label
  // second label props
  secondTitle?: string;
  secondSubtitle?: string;
};

export default function Bottle({
  color = '#800000',
  scale = 1,
  title = 'Gamay',
  subtitle = 'US · California · Napa Valley',
  logoUri,
  secondTitle = 'Reserve Lot',
  secondSubtitle = 'Estate Bottled',
}: BottleProps) {
  const [box, setBox] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  const onLayout = (e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    if (width !== box.w || height !== box.h) setBox({ w: width, h: height });
  };

  const viewBox = '0 0 885.82672 885.82672';
  const VB = 885.82672;
  const cx = VB / 2;
  const cy = VB / 2;
  const tx = cx - scale * cx;
  const ty = cy - scale * cy;

  // bottle path (unchanged)
  const d = `m 417.53694,781.52611 c -1.5951,-0.15106 -7.46798,-0.63351 -13.05084,-1.07214 -18.37673,-1.44377 -34.47365,-4.63394 -41.59792,-8.24408 -4.57501,-2.31833 -8.41488,-6.69852 -10.01393,-11.42302 -1.23206,-3.64018 -1.31564,-11.1894 -1.53546,-138.70834 -0.26502,-153.72857 -0.16616,-157.27001 5.31714,-190.50302 5.46385,-33.11492 9.50268,-46.28545 29.84459,-97.32236 5.40794,-13.56829 11.0717,-28.43867 12.58612,-33.04529 6.88295,-20.93679 10.36259,-51.1211 11.57196,-100.38128 1.03209,-42.03953 0.92206,-61.22351 -0.37213,-64.89165 -1.49346,-4.23295 -1.36247,-9.40472 0.36253,-14.31265 0.81471,-2.318 1.45009,-5.9844 1.45009,-8.36759 0,-5.32554 0.67712,-6.57993 4.3438,-8.04704 2.49495,-0.99828 6.4465,-1.17416 26.30409,-1.17073 13.77864,0.002 24.47029,0.31018 26.05401,0.75003 3.98981,1.1081 5.52205,3.45888 5.1023,7.82766 -0.25102,2.61219 0.098,4.71758 1.29727,7.82769 2.05957,5.34084 2.25728,12.60789 0.4373,16.06949 -1.17591,2.23668 -1.20692,5.06503 -0.48576,44.22784 0.94687,51.42081 2.49375,74.65719 6.54904,98.37814 2.61134,15.27434 4.92117,22.34338 18.83872,57.65501 19.47018,49.3996 22.91758,60.81383 28.57693,94.61855 5.53401,33.0558 5.63695,36.71071 5.37151,190.6872 -0.21978,127.51894 -0.30335,135.06816 -1.53547,138.70834 -3.84034,11.34651 -14.04946,15.53334 -47.26156,19.38221 -9.55795,1.10767 -60.1501,2.115 -68.15433,1.35703 z m 80.11762,-47.38865 c 15.98025,-1.28544 29.72679,-3.57102 31.53955,-5.24393 0.47672,-0.43994 0.72504,-35.30092 0.72504,-101.77616 l 0,-101.10703 -87.00559,7.2e-4 -87.00557,7.2e-4 -0.25489,100.59907 c -0.21207,83.69532 -0.0902,100.81117 0.72504,101.86136 1.98291,2.55427 19.71335,5.14602 43.39517,6.34329 15.85721,0.80168 85.48022,0.31914 97.88125,-0.67839 z M 451.61412,392.57343 c 7.21207,-1.02915 9.09978,-1.78146 14.65123,-5.83883 7.1176,-5.20212 11.72391,-6.44271 26.67641,-7.18466 11.94391,-0.59265 17.59088,-1.60581 18.22977,-3.27071 0.33856,-0.88224 -4.1242,-13.83446 -11.73845,-34.06861 -2.70115,-7.17795 -4.95994,-13.12465 -5.01938,-13.21488 -0.0592,-0.0902 -1.88329,0.51485 -4.05281,1.34462 -7.56762,2.89437 -16.62613,0.87141 -27.26405,-6.08857 -7.44658,-4.87209 -15.13845,-7.31508 -21.63336,-6.87097 -6.06895,0.41498 -12.70235,2.94 -19.56526,7.44758 -9.63177,6.32617 -19.0696,8.32822 -26.2714,5.57297 l -3.78395,-1.44766 -2.13779,5.54346 c -11.71368,30.37458 -15.60214,41.42464 -14.84358,42.18182 1.4379,1.43527 9.67664,2.87234 16.54485,2.8859 13.86115,0.0274 22.27514,2.17914 28.43962,7.27317 6.67209,5.5135 18.4376,7.63762 31.76815,5.73537 z M 464.32143,228.7266 c 11.06714,-1.33173 10.13331,-0.64205 10.1206,-7.47507 0,-3.25821 -0.2509,-6.29555 -0.54381,-6.74965 -0.38529,-0.59712 -2.84028,-0.65885 -8.87071,-0.2231 -11.24353,0.81247 -32.49244,0.8026 -43.11974,-0.02 -6.58078,-0.50939 -8.81382,-0.46141 -9.23325,0.19843 -0.30427,0.47867 -0.55822,3.52826 -0.56433,6.77685 l -0.0111,5.90654 6.34416,0.86346 c 3.48928,0.4749 6.83356,0.96024 7.43172,1.07856 2.89574,0.57273 33.04877,0.29352 38.44645,-0.356 z m 7.91458,-90.9285 c 1.89653,-2.28522 -1.52363,-2.58822 -29.21548,-2.58822 -14.94768,0 -27.6937,0.19804 -28.32447,0.44009 -0.63078,0.24206 -1.14687,0.88484 -1.14687,1.42844 0,1.11948 5.13168,1.33454 38.15742,1.59913 16.31194,0.13067 19.81556,-0.0194 20.5294,-0.87944 z m -0.0958,-19.08302 c -0.22689,-1.17893 -2.26697,-1.28228 -28.78926,-1.45861 -28.00822,-0.18618 -31.37285,0.0467 -30.1134,2.08455 0.22045,0.3567 13.47346,0.64288 29.77208,0.64288 27.58563,0 29.35984,-0.0773 29.13058,-1.26882 z`;

  // ===== Label geometry (user-space coords in the same viewBox) =====
  // Picked to sit on the lower “belly” of the bottle; tweak to fit your shape.
  // (Use a screenshot and nudge these numbers once or twice.)
  const label = {
    x: 363,
    y: 530,
    w: 160,
    h: 200,
    rx: 22,
    padX: 18,
    padY: 18,
  };

  // Second, shorter label below the first, same width
  const label2 = {
    x: label.x + 25,
    y: label.y - 195,
    w: label.w - 50,
    h: 40,
    rx: 16,
    padX: 16,
    padY: 12,
  };

  const svgW = Math.max(140, box.w || 0);
  const svgH = Math.max(140, box.h || 0);

  return (
    <View style={{ flex: 1 }} onLayout={onLayout}>
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
        <Svg
          viewBox={viewBox}
          width={svgW}
          height={svgH}
          preserveAspectRatio="xMidYMid slice"
        >
          {/* Apply your translate/scale ONCE to everything inside */}
          <G transform={`translate(${tx}, ${ty}) scale(${scale})`}>
            {/* Bottle */}
            <Path d={d} fill={color} stroke={color} strokeWidth={2} />

            {/* --- Label clip & contents --- */}
            <Defs>
              <ClipPath id="labelClip">
                <Rect
                  x={label.x}
                  y={label.y}
                  width={label.w}
                  height={label.h}
                  rx={label.rx}
                />
              </ClipPath>
              <ClipPath id="labelClip2">
                <Rect
                  x={label2.x}
                  y={label2.y}
                  width={label2.w}
                  height={label2.h}
                  rx={label2.rx}
                />
              </ClipPath>
            </Defs>

            {/* Draw the primary label over the bottle */}
            <G clipPath="url(#labelClip)">
              <Rect
                x={label.x}
                y={label.y}
                width={label.w}
                height={label.h}
                rx={label.rx}
                fill="#F1F3EA"
              />

              {logoUri ? (
                <SvgImage
                  href={{ uri: logoUri }}
                  x={label.x + label.padX}
                  y={label.y + label.padY}
                  width={34}
                  height={34}
                  preserveAspectRatio="xMidYMid slice"
                />
              ) : null}

              <SvgText
                x={label.x + (logoUri ? 34 + label.padX + 10 : label.padX)}
                y={label.y + label.padY + 22}
                fontSize={22}
                fontWeight="700"
                fill={color}
              >
                {title}
              </SvgText>

              <SvgText
                x={label.x + label.padX}
                y={label.y + label.padY + 22 + 24}
                fontSize={13}
                fill={color}
              >
                <TSpan>{subtitle}</TSpan>
              </SvgText>

              <Rect
                x={label.x + label.padX}
                y={label.y + label.h - label.padY - 10}
                width={label.w - 2 * label.padX}
                height={6}
                rx={3}
                fill="#D9D1CB"
              />
              <Rect
                x={label.x + label.padX}
                y={label.y + label.h - label.padY - 10}
                width={(label.w - 2 * label.padX) * 0.55}
                height={6}
                rx={3}
                fill={color}
              />
            </G>

            {/* Draw the secondary, shorter label */}
            <G clipPath="url(#labelClip2)">
              <Rect
                x={label2.x}
                y={label2.y}
                width={label2.w}
                height={label2.h}
                rx={label2.rx}
                fill="#F1F3EA"
              />

              <SvgText
                x={label2.x + label2.padX}
                y={label2.y + label2.padY + 14}
                fontSize={16}
                fontWeight="700"
                fill={color}
              >
                {secondTitle}
              </SvgText>

              <SvgText
                x={label2.x + label2.padX}
                y={label2.y + label2.padY + 14 + 18}
                fontSize={12}
                fill={color}
              >
                <TSpan>{secondSubtitle}</TSpan>
              </SvgText>
            </G>
          </G>
        </Svg>
      </View>
    </View>
  );
}
