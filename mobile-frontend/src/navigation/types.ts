import type { NavigatorScreenParams } from "@react-navigation/native";

export type AuthStackParamList = {
  Welcome: undefined;
  Login: undefined;
  Register: undefined;
  ForgotPassword: undefined;
};

export type DashboardStackParamList = {
  DashboardHome: undefined;
};

export type DatasetsStackParamList = {
  DatasetsList: undefined;
  DatasetDetail: { datasetId: number };
};

export type ModelsStackParamList = {
  ModelsList: undefined;
  ModelDetail: { modelId: number };
};

export type PredictionsStackParamList = {
  PredictionsHome: undefined;
};

export type MoreStackParamList = {
  MoreMenu: undefined;
  Profile: undefined;
  ChangePassword: undefined;
  MfaSetup: undefined;
  ApiKeys: undefined;
  Financial: undefined;
  Notifications: undefined;
  AdminUsers: undefined;
  AdminSystem: undefined;
};

export type MainTabParamList = {
  DashboardTab: NavigatorScreenParams<DashboardStackParamList>;
  DatasetsTab: NavigatorScreenParams<DatasetsStackParamList>;
  ModelsTab: NavigatorScreenParams<ModelsStackParamList>;
  PredictionsTab: NavigatorScreenParams<PredictionsStackParamList>;
  MoreTab: NavigatorScreenParams<MoreStackParamList>;
};

export type RootStackParamList = {
  Auth: NavigatorScreenParams<AuthStackParamList>;
  Main: NavigatorScreenParams<MainTabParamList>;
};
