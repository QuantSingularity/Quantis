import TrendingDownIcon from "@mui/icons-material/TrendingDown";
import TrendingFlatIcon from "@mui/icons-material/TrendingFlat";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import {
  Box,
  Card,
  CardContent,
  Stack,
  Typography,
  useTheme,
} from "@mui/material";

/**
 * Headline metric card: value + label + optional delta indicator + icon.
 * `delta` is a signed number (percentage points); omit for a flat/neutral card.
 */
const StatCard = ({
  label,
  value,
  delta,
  icon,
  loading = false,
  suffix = "",
}) => {
  const theme = useTheme();

  const isPositive = typeof delta === "number" && delta > 0;
  const isNegative = typeof delta === "number" && delta < 0;
  const deltaColor = isPositive
    ? theme.palette.success.main
    : isNegative
      ? theme.palette.error.main
      : theme.palette.text.secondary;

  return (
    <Card sx={{ height: "100%" }}>
      <CardContent>
        <Stack
          direction="row"
          justifyContent="space-between"
          alignItems="flex-start"
        >
          <Typography variant="body2" color="text.secondary" fontWeight={500}>
            {label}
          </Typography>
          {icon && (
            <Box
              sx={{
                width: 36,
                height: 36,
                borderRadius: 1.5,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                bgcolor: "action.hover",
                color: "primary.main",
              }}
            >
              {icon}
            </Box>
          )}
        </Stack>

        <Typography
          variant="h4"
          sx={{ fontWeight: 700, mt: 1.5, mb: delta !== undefined ? 0.5 : 0 }}
        >
          {loading ? "—" : value}
          {!loading && suffix}
        </Typography>

        {typeof delta === "number" && (
          <Stack direction="row" alignItems="center" spacing={0.5}>
            {isPositive && (
              <TrendingUpIcon sx={{ fontSize: 16, color: deltaColor }} />
            )}
            {isNegative && (
              <TrendingDownIcon sx={{ fontSize: 16, color: deltaColor }} />
            )}
            {!isPositive && !isNegative && (
              <TrendingFlatIcon sx={{ fontSize: 16, color: deltaColor }} />
            )}
            <Typography
              variant="caption"
              sx={{ color: deltaColor, fontWeight: 600 }}
            >
              {isPositive ? "+" : ""}
              {delta}%
            </Typography>
            <Typography variant="caption" color="text.secondary">
              vs last period
            </Typography>
          </Stack>
        )}
      </CardContent>
    </Card>
  );
};

export default StatCard;
