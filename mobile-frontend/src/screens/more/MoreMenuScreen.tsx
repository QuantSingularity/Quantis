import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import React, { useState } from "react";
import { StyleSheet, View } from "react-native";
import {
  Avatar,
  Divider,
  List,
  Switch,
  Text,
  useTheme,
} from "react-native-paper";
import ConfirmDialog from "../../components/ConfirmDialog";
import ScreenContainer from "../../components/ScreenContainer";
import { useAuth } from "../../context/AuthContext";
import { useThemeMode } from "../../context/ThemeModeContext";
import type { MoreStackParamList } from "../../navigation/types";

type Props = NativeStackScreenProps<MoreStackParamList, "MoreMenu">;

const MoreMenuScreen: React.FC<Props> = ({ navigation }) => {
  const theme = useTheme();
  const { user, isAdmin, logout } = useAuth();
  const { mode, toggleMode } = useThemeMode();
  const [logoutOpen, setLogoutOpen] = useState(false);

  const initials = (user?.username || "?").slice(0, 2).toUpperCase();

  return (
    <ScreenContainer>
      <View style={styles.profileHeader}>
        <Avatar.Text size={56} label={initials} />
        <View style={{ marginLeft: 14 }}>
          <Text variant="titleMedium" style={{ fontWeight: "700" }}>
            {user?.full_name || user?.username}
          </Text>
          <Text
            variant="bodySmall"
            style={{ color: theme.colors.onSurfaceVariant }}
          >
            {user?.email}
          </Text>
        </View>
      </View>

      <List.Section>
        <List.Item
          title="Profile"
          left={(props) => <List.Icon {...props} icon="account-outline" />}
          onPress={() => navigation.navigate("Profile")}
        />
        <List.Item
          title="Change password"
          left={(props) => <List.Icon {...props} icon="lock-outline" />}
          onPress={() => navigation.navigate("ChangePassword")}
        />
        <List.Item
          title="Two-factor authentication"
          left={(props) => <List.Icon {...props} icon="shield-check-outline" />}
          onPress={() => navigation.navigate("MfaSetup")}
        />
        <List.Item
          title="API keys"
          left={(props) => <List.Icon {...props} icon="key-outline" />}
          onPress={() => navigation.navigate("ApiKeys")}
        />
      </List.Section>

      <Divider />

      <List.Section>
        <List.Item
          title="Financial"
          left={(props) => <List.Icon {...props} icon="bank-outline" />}
          onPress={() => navigation.navigate("Financial")}
        />
        <List.Item
          title="Notifications"
          left={(props) => <List.Icon {...props} icon="bell-outline" />}
          onPress={() => navigation.navigate("Notifications")}
        />
      </List.Section>

      {isAdmin && (
        <>
          <Divider />
          <List.Section>
            <List.Subheader>Admin</List.Subheader>
            <List.Item
              title="Users"
              left={(props) => (
                <List.Icon {...props} icon="account-group-outline" />
              )}
              onPress={() => navigation.navigate("AdminUsers")}
            />
            <List.Item
              title="System health"
              left={(props) => <List.Icon {...props} icon="heart-pulse" />}
              onPress={() => navigation.navigate("AdminSystem")}
            />
          </List.Section>
        </>
      )}

      <Divider />

      <List.Item
        title="Dark mode"
        left={(props) => <List.Icon {...props} icon="theme-light-dark" />}
        right={() => (
          <Switch value={mode === "dark"} onValueChange={toggleMode} />
        )}
      />

      <List.Item
        title="Sign out"
        titleStyle={{ color: theme.colors.error }}
        left={(props) => (
          <List.Icon {...props} icon="logout" color={theme.colors.error} />
        )}
        onPress={() => setLogoutOpen(true)}
      />

      <ConfirmDialog
        visible={logoutOpen}
        title="Sign out?"
        description="You'll need to sign in again to access your workspace."
        confirmLabel="Sign out"
        destructive
        onConfirm={async () => {
          setLogoutOpen(false);
          await logout();
        }}
        onDismiss={() => setLogoutOpen(false)}
      />
    </ScreenContainer>
  );
};

const styles = StyleSheet.create({
  profileHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 12,
  },
});

export default MoreMenuScreen;
