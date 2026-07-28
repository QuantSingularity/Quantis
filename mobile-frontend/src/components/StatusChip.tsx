import React from "react";
import { Chip, useTheme } from "react-native-paper";
import { statusColor } from "../theme/tokens";

interface Props {
  status?: string | null;
}

const formatLabel = (value?: string | null): string =>
  String(value || "unknown")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

const StatusChip: React.FC<Props> = ({ status }) => {
  const theme = useTheme();
  const mode = theme.dark ? "dark" : "light";
  const color = statusColor(mode, status);

  return (
    <Chip
      compact
      style={{ backgroundColor: `${color}22`, alignSelf: "flex-start" }}
      textStyle={{ color, fontWeight: "700", fontSize: 12 }}
    >
      {formatLabel(status)}
    </Chip>
  );
};

export default StatusChip;
