// Registers the official in-memory mock for AsyncStorage so tests don't hit
// the native module bridge, which doesn't exist in the Jest environment.
jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);
