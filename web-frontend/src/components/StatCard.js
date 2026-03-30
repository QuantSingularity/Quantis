import TrendingDownIcon from "@mui/icons-material/TrendingDown";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import {
  Avatar,
  Box,
  Card,
  CardContent,
  Typography,
  useTheme,
} from "@mui/material";

const StatCard = ({ title, value, change, positive, icon, color }) => {
  const theme = useTheme();

  return (
    <Card
      sx={{
        height: "100%",
        transition: "transform 0.3s, box-shadow 0.3s",
        "&:hover": {
          transform: "translateY(-4px)",
          boxShadow: theme.shadows[6],
        },
      }}
    >
      <CardContent>
        <Box sx={{ display: "flex", justifyContent: "space-between", mb: 2 }}>
          <Typography variant="h6" color="text.secondary" fontWeight="500">
            {title}
          </Typography>
          <Avatar
            sx={{
              bgcolor: `${color}20`,
              color: color,
              width: 40,
              height: 40,
            }}
          >
            {icon}
          </Avatar>
        </Box>

        <Typography
          variant="h4"
          component="div"
          fontWeight="600"
          sx={{ mb: 1 }}
        >
          {value}
        </Typography>

        <Box sx={{ display: "flex", alignItems: "center" }}>
          {positive ? (
            <TrendingUpIcon fontSize="small" color="success" />
          ) : (
            <TrendingDownIcon fontSize="small" color="error" />
          )}
          <Typography
            variant="body2"
            sx={{
              ml: 0.5,
              color: positive
                ? theme.palette.success.main
                : theme.palette.error.main,
            }}
          >
            {change}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ ml: 0.5 }}>
            vs last month
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default StatCard;
