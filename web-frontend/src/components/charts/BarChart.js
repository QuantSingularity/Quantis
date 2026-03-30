import { Box, Typography, useTheme } from "@mui/material";
import React from "react";
import {
  Bar,
  CartesianGrid,
  Legend,
  BarChart as RechartsBarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const BarChart = ({
  data,
  title,
  height = 300,
  dataKey = "value",
  nameKey = "name",
}) => {
  const theme = useTheme();

  // Transform data if needed
  const transformedData = React.useMemo(() => {
    if (!data || data.length === 0) {
      return [];
    }

    // Check if data needs transformation
    if (data[0]?.feature && data[0].value !== undefined) {
      // Transform feature importance data
      return data.map((item) => ({
        name: item.feature,
        value: item.value,
      }));
    }

    // Data is already in the correct format or use as-is
    return data;
  }, [data]);

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
        <RechartsBarChart
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
            dataKey={nameKey}
            stroke={theme.palette.text.secondary}
            fontSize={12}
            angle={-45}
            textAnchor="end"
            height={60}
          />
          <YAxis stroke={theme.palette.text.secondary} fontSize={12} />
          <Tooltip
            contentStyle={{
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: theme.shape.borderRadius,
              color: theme.palette.text.primary,
            }}
            formatter={(value, name) => [
              typeof value === "number" ? value.toFixed(3) : value,
              name,
            ]}
          />
          <Legend />
          <Bar
            dataKey={dataKey}
            fill={theme.palette.primary.main}
            radius={[4, 4, 0, 0]}
          />
        </RechartsBarChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default BarChart;
