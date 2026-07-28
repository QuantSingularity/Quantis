import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import React, { useEffect, useState } from "react";
import { View } from "react-native";
import { Chip, IconButton, List, Text, useTheme } from "react-native-paper";
import { getErrorMessage, usersAPI } from "../../api";
import { User } from "../../api/types";
import ConfirmDialog from "../../components/ConfirmDialog";
import EmptyState from "../../components/EmptyState";
import LoadingScreen from "../../components/LoadingScreen";
import ScreenContainer from "../../components/ScreenContainer";
import { useAuth } from "../../context/AuthContext";
import type { MoreStackParamList } from "../../navigation/types";

type Props = NativeStackScreenProps<MoreStackParamList, "AdminUsers">;

const AdminUsersScreen: React.FC<Props> = () => {
  const theme = useTheme();
  const { user: currentUser } = useAuth();
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<User | null>(null);

  const load = async () => {
    try {
      const { data } = await usersAPI.list();
      setUsers(
        (data as unknown as { items?: User[] })?.items ??
          (data as User[]) ??
          [],
      );
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      await usersAPI.remove(deleteTarget.id);
      setUsers((prev) => prev.filter((u) => u.id !== deleteTarget.id));
      setDeleteTarget(null);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  if (loading) return <LoadingScreen label="Loading users…" />;

  return (
    <ScreenContainer>
      <Text
        variant="headlineSmall"
        style={{ fontWeight: "700", marginBottom: 4 }}
      >
        Users
      </Text>
      <Text
        variant="bodyMedium"
        style={{ color: theme.colors.onSurfaceVariant, marginBottom: 16 }}
      >
        Manage workspace members and their access.
      </Text>

      {error && (
        <Text
          variant="bodySmall"
          style={{ color: theme.colors.error, marginBottom: 8 }}
        >
          {error}
        </Text>
      )}

      {users.length === 0 ? (
        <EmptyState title="No users found" />
      ) : (
        users.map((u) => (
          <List.Item
            key={u.id}
            title={u.full_name || u.username}
            description={u.email}
            left={() => (
              <View style={{ justifyContent: "center" }}>
                <Chip compact textStyle={{ textTransform: "capitalize" }}>
                  {u.role}
                </Chip>
              </View>
            )}
            right={(props) =>
              u.id !== currentUser?.id ? (
                <IconButton
                  {...props}
                  icon="delete-outline"
                  onPress={() => setDeleteTarget(u)}
                />
              ) : null
            }
          />
        ))
      )}

      <ConfirmDialog
        visible={Boolean(deleteTarget)}
        title="Remove user?"
        description={`"${deleteTarget?.username}" will lose access to this workspace.`}
        confirmLabel="Remove"
        destructive
        onConfirm={handleDelete}
        onDismiss={() => setDeleteTarget(null)}
      />
    </ScreenContainer>
  );
};

export default AdminUsersScreen;
