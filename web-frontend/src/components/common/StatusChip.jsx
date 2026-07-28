import { Chip } from "@mui/material";

const STATUS_COLOR_MAP = {
  // Model / dataset statuses
  trained: "success",
  deployed: "success",
  ready: "success",
  active: "success",
  compliant: "success",
  approved: "success",
  completed: "success",
  training: "warning",
  processing: "warning",
  uploading: "warning",
  pending: "warning",
  under_review: "warning",
  created: "info",
  archived: "default",
  failed: "error",
  error: "error",
  rejected: "error",
  non_compliant: "error",
  locked: "error",
  suspended: "error",
  // risk levels
  low: "success",
  medium: "warning",
  high: "error",
  critical: "error",
};

const formatLabel = (value) =>
  String(value || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

const StatusChip = ({ status, size = "small", ...props }) => {
  const key = String(status || "").toLowerCase();
  const color = STATUS_COLOR_MAP[key] || "default";
  return (
    <Chip
      label={formatLabel(status) || "Unknown"}
      color={color}
      size={size}
      variant={color === "default" ? "outlined" : "filled"}
      sx={{
        fontWeight: 600,
        ...(color !== "default" && { color: "#fff" }),
      }}
      {...props}
    />
  );
};

export default StatusChip;
