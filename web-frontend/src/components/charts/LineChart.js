import { Box, Typography, useTheme } from "@mui/material";
import React from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart as RechartsLineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const LineChart = ({ data, title, height = 300 }) => {
  const theme = useTheme();

  // Transform data if it's in the old format
  const transformedData = React.useMemo(() => {
    if (!data || data.length === 0) {
      return [];
    }

    // Check if data is in the old format (array of objects with id and data)
    if (data[0]?.data && Array.isArray(data[0].data)) {
      // Transform old format to new format
      const xValues = data[0].data.map((point) => point.x);
      return xValues.map((x) => {
        const point = { x };
        data.forEach((series) => {
          const dataPoint = series.data.find((p) => p.x === x);
          if (dataPoint) {
            point[series.id] = dataPoint.y;
          }
        });
        return point;
      });
    }

    // Data is already in the correct format
    return data;
  }, [data]);

  // Get series names from data
  const seriesNames = React.useMemo(() => {
    if (!transformedData || transformedData.length === 0) {
      return [];
    }
    return Object.keys(transformedData[0]).filter((key) => key !== "x");
  }, [transformedData]);

  // Color palette
  const colors = [
    theme.palette.primary.main,
    theme.palette.secondary.main,
    theme.palette.success.main,
    theme.palette.warning.main,
    theme.palette.error.main,
    theme.palette.info.main,
  ];

  if (!transformedData || transformedData.length === 0) {
    return (
      <Box
        sx={{
          height,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: theme.palette.background.default,
          borderRadius: 1,
        }}
      >
        <Typography variant="body2" color="text.secondary">
          No data available
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: "100%", height }}>
      {title && (
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
      )}
      <ResponsiveContainer width="100%" height="100%">
        <RechartsLineChart
          data={transformedData}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
          <XAxis
            dataKey="x"
            stroke={theme.palette.text.secondary}
            fontSize={12}
          />
          <YAxis stroke={theme.palette.text.secondary} fontSize={12} />
          <Tooltip
            contentStyle={{
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: theme.shape.borderRadius,
              color: theme.palette.text.primary,
            }}
          />
          <Legend />
          {seriesNames.map((seriesName, index) => (
            <Line
              key={seriesName}
              type="monotone"
              dataKey={seriesName}
              stroke={colors[index % colors.length]}
              strokeWidth={2}
              dot={{
                fill: colors[index % colors.length],
                strokeWidth: 2,
                r: 4,
              }}
              activeDot={{
                r: 6,
                stroke: colors[index % colors.length],
                strokeWidth: 2,
              }}
            />
          ))}
        </RechartsLineChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default LineChart;
