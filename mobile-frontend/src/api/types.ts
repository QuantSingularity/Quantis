export interface User {
  id: number;
  username: string;
  email: string;
  first_name?: string | null;
  last_name?: string | null;
  full_name?: string | null;
  phone_number?: string | null;
  timezone?: string | null;
  role: string;
  permissions: string[];
  is_active: boolean;
  is_verified: boolean;
  is_mfa_enabled: boolean;
  last_login?: string | null;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface Dataset {
  id: number;
  name: string;
  description?: string | null;
  status: string;
  row_count?: number | null;
  frequency?: string | null;
  created_at: string;
}

export interface Model {
  id: number;
  name: string;
  description?: string | null;
  model_type: string;
  status: string;
  version: string;
  dataset_id: number;
  target_column?: string | null;
  hyperparameters?: Record<string, unknown>;
  metrics?: Record<string, number> | null;
  trained_at?: string | null;
  created_at: string;
}

export interface Prediction {
  id: number;
  model_id: number;
  prediction_result: unknown;
  confidence_score?: number | null;
  status?: string;
  created_at: string;
}

export interface Transaction {
  id: number;
  amount: number;
  transaction_type: string;
  description?: string | null;
  status: string;
  risk_level?: string | null;
  compliance_status: string;
  created_at: string;
}

export interface Notification {
  id: number;
  title?: string;
  notification_type?: string;
  message: string;
  is_read: boolean;
  created_at: string;
}

export interface ApiKey {
  id: number;
  name: string;
  key_preview: string;
  is_active: boolean;
  last_used?: string | null;
  usage_count: number;
  created_at: string;
}
