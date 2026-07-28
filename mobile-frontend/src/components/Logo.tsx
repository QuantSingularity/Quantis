import React from "react";
import { StyleSheet, View } from "react-native";
import Svg, { Circle, Path } from "react-native-svg";
import { Text } from "react-native-paper";

interface Props {
  size?: number;
  showWordmark?: boolean;
}

const Logo: React.FC<Props> = ({ size = 32, showWordmark = true }) => (
  <View style={styles.row}>
    <View
      style={[
        styles.badge,
        {
          width: size,
          height: size,
          borderRadius: size * 0.28,
          backgroundColor: "#6366F1",
        },
      ]}
    >
      <Svg
        width={size * 0.6}
        height={size * 0.6}
        viewBox="0 0 32 32"
        fill="none"
      >
        <Path
          d="M8 21L13 12L17 18L24 8"
          stroke="white"
          strokeWidth={3}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <Circle cx={24} cy={8} r={2.4} fill="white" />
      </Svg>
    </View>
    {showWordmark && (
      <Text variant="titleLarge" style={styles.wordmark}>
        Quantis
      </Text>
    )}
  </View>
);

const styles = StyleSheet.create({
  row: { flexDirection: "row", alignItems: "center", gap: 10 },
  badge: { alignItems: "center", justifyContent: "center" },
  wordmark: { fontWeight: "800", letterSpacing: -0.5 },
});

export default Logo;
