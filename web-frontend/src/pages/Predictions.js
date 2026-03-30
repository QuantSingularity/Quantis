import SendIcon from "@mui/icons-material/Send";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Divider,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Select,
  Slider,
  Typography,
  useTheme,
} from "@mui/material";
import { useState } from "react";

const Predictions = () => {
  const theme = useTheme();
  const [formData, setFormData] = useState({
    feature1: 0.5,
    feature2: 0.3,
    feature3: 0.7,
    category1: "option1",
    category2: "option2",
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSliderChange = (name) => (_e, newValue) => {
    setFormData({
      ...formData,
      [name]: newValue,
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);

    // Simulate API call
    setTimeout(() => {
      setPrediction({
        value: [0.72, 0.18, 0.1],
        confidence: 0.72,
        timestamp: new Date().toISOString(),
      });
      setLoading(false);
    }, 1000);
  };

  return (
    <Box className="fade-in">
      <Typography variant="h4" gutterBottom fontWeight="600">
        Make Predictions
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Enter feature values to generate predictions from the model
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="Input Features"
              subheader="Enter values for model features"
            />
            <Divider />
            <CardContent>
              <Box component="form" onSubmit={handleSubmit} sx={{ mt: 1 }}>
                <Typography gutterBottom>Numeric Features</Typography>

                <Box sx={{ mb: 3 }}>
                  <Typography id="feature1-label" gutterBottom>
                    Feature 1: {formData.feature1}
                  </Typography>
                  <Slider
                    name="feature1"
                    value={formData.feature1}
                    onChange={handleSliderChange("feature1")}
                    aria-labelledby="feature1-label"
                    step={0.01}
                    min={0}
                    max={1}
                    marks={[
                      { value: 0, label: "0" },
                      { value: 0.5, label: "0.5" },
                      { value: 1, label: "1" },
                    ]}
                  />
                </Box>

                <Box sx={{ mb: 3 }}>
                  <Typography id="feature2-label" gutterBottom>
                    Feature 2: {formData.feature2}
                  </Typography>
                  <Slider
                    name="feature2"
                    value={formData.feature2}
                    onChange={handleSliderChange("feature2")}
                    aria-labelledby="feature2-label"
                    step={0.01}
                    min={0}
                    max={1}
                    marks={[
                      { value: 0, label: "0" },
                      { value: 0.5, label: "0.5" },
                      { value: 1, label: "1" },
                    ]}
                  />
                </Box>

                <Box sx={{ mb: 4 }}>
                  <Typography id="feature3-label" gutterBottom>
                    Feature 3: {formData.feature3}
                  </Typography>
                  <Slider
                    name="feature3"
                    value={formData.feature3}
                    onChange={handleSliderChange("feature3")}
                    aria-labelledby="feature3-label"
                    step={0.01}
                    min={0}
                    max={1}
                    marks={[
                      { value: 0, label: "0" },
                      { value: 0.5, label: "0.5" },
                      { value: 1, label: "1" },
                    ]}
                  />
                </Box>

                <Typography gutterBottom>Categorical Features</Typography>

                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <FormControl fullWidth sx={{ mb: 2 }}>
                      <InputLabel id="category1-label">Category 1</InputLabel>
                      <Select
                        labelId="category1-label"
                        name="category1"
                        value={formData.category1}
                        label="Category 1"
                        onChange={handleChange}
                      >
                        <MenuItem value="option1">Option 1</MenuItem>
                        <MenuItem value="option2">Option 2</MenuItem>
                        <MenuItem value="option3">Option 3</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>

                  <Grid item xs={12} sm={6}>
                    <FormControl fullWidth sx={{ mb: 3 }}>
                      <InputLabel id="category2-label">Category 2</InputLabel>
                      <Select
                        labelId="category2-label"
                        name="category2"
                        value={formData.category2}
                        label="Category 2"
                        onChange={handleChange}
                      >
                        <MenuItem value="option1">Option 1</MenuItem>
                        <MenuItem value="option2">Option 2</MenuItem>
                        <MenuItem value="option3">Option 3</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                </Grid>

                <Button
                  type="submit"
                  variant="contained"
                  fullWidth
                  disabled={loading}
                  endIcon={<SendIcon />}
                  sx={{ mt: 2 }}
                >
                  {loading ? "Processing..." : "Generate Prediction"}
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={{ height: "100%" }}>
            <CardHeader
              title="Prediction Results"
              subheader="Model output based on input features"
            />
            <Divider />
            <CardContent>
              {prediction ? (
                <Box>
                  <Alert severity="success" sx={{ mb: 3 }}>
                    Prediction generated successfully!
                  </Alert>

                  <Typography variant="h6" gutterBottom>
                    Prediction Values:
                  </Typography>

                  <Box
                    sx={{
                      display: "flex",
                      justifyContent: "space-between",
                      mb: 3,
                      p: 2,
                      bgcolor: theme.palette.background.default,
                      borderRadius: 1,
                    }}
                  >
                    <Box sx={{ textAlign: "center" }}>
                      <Typography
                        variant="h4"
                        color="primary"
                        fontWeight="bold"
                      >
                        {(prediction.value[0] * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Class A
                      </Typography>
                    </Box>

                    <Box sx={{ textAlign: "center" }}>
                      <Typography variant="h4" fontWeight="bold">
                        {(prediction.value[1] * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Class B
                      </Typography>
                    </Box>

                    <Box sx={{ textAlign: "center" }}>
                      <Typography variant="h4" fontWeight="bold">
                        {(prediction.value[2] * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Class C
                      </Typography>
                    </Box>
                  </Box>

                  <Typography variant="subtitle1" gutterBottom>
                    Confidence:{" "}
                    <strong>{(prediction.confidence * 100).toFixed(1)}%</strong>
                  </Typography>

                  <Typography variant="subtitle1" gutterBottom>
                    Timestamp: {new Date(prediction.timestamp).toLocaleString()}
                  </Typography>

                  <Divider sx={{ my: 2 }} />

                  <Typography variant="body2" color="text.secondary">
                    The model predicts Class A with the highest probability.
                    This indicates that based on the provided features, the
                    outcome is most likely to be Class A.
                  </Typography>
                </Box>
              ) : (
                <Box
                  sx={{
                    height: "100%",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "center",
                    alignItems: "center",
                    py: 8,
                  }}
                >
                  <Typography
                    variant="body1"
                    color="text.secondary"
                    align="center"
                  >
                    No predictions yet
                  </Typography>
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    align="center"
                    sx={{ mt: 1 }}
                  >
                    Enter feature values and click "Generate Prediction" to see
                    results
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Predictions;
