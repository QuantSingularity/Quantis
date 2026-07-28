import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import React, { useEffect, useState } from "react";
import { StyleSheet, View } from "react-native";
import { Button, IconButton, List, Text, useTheme } from "react-native-paper";
import { getErrorMessage, notificationsAPI } from "../../api";
import { Notification } from "../../api/types";
import EmptyState from "../../components/EmptyState";
import LoadingScreen from "../../components/LoadingScreen";
import ScreenContainer from "../../components/ScreenContainer";
import type { MoreStackParamList } from "../../navigation/types";

type Props = NativeStackScreenProps<MoreStackParamList, "Notifications">;

const NotificationsScreen: React.FC<Props> = () => {
  const theme = useTheme();
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    try {
      const { data } = await notificationsAPI.list({ limit: 50 });
      setNotifications(
        (data as unknown as { items?: Notification[] })?.items ??
          (data as Notification[]) ??
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

  const markRead = async (id: number) => {
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, is_read: true } : n)),
    );
    try {
      await notificationsAPI.markRead(id);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  const markAllRead = async () => {
    setNotifications((prev) => prev.map((n) => ({ ...n, is_read: true })));
    try {
      await notificationsAPI.markAllRead();
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  const remove = async (id: number) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
    try {
      await notificationsAPI.remove(id);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  if (loading) return <LoadingScreen label="Loading notifications…" />;

  const unreadCount = notifications.filter((n) => !n.is_read).length;

  return (
    <ScreenContainer>
      <View style={styles.headerRow}>
        <Text variant="headlineSmall" style={{ fontWeight: "700" }}>
          Notifications
        </Text>
        {unreadCount > 0 && (
          <Button compact onPress={markAllRead}>
            Mark all read
          </Button>
        )}
      </View>

      {error && (
        <Text
          variant="bodySmall"
          style={{ color: theme.colors.error, marginBottom: 8 }}
        >
          {error}
        </Text>
      )}

      {notifications.length === 0 ? (
        <EmptyState
          title="No notifications"
          description="You're all caught up."
        />
      ) : (
        notifications.map((n) => (
          <List.Item
            key={n.id}
            title={n.title || n.notification_type || "Notification"}
            titleStyle={{ fontWeight: n.is_read ? "400" : "700" }}
            description={n.message}
            style={{
              backgroundColor: n.is_read
                ? "transparent"
                : theme.colors.surfaceVariant,
              borderRadius: 8,
            }}
            right={() => (
              <View style={{ flexDirection: "row" }}>
                {!n.is_read && (
                  <IconButton
                    icon="check"
                    size={18}
                    onPress={() => markRead(n.id)}
                  />
                )}
                <IconButton
                  icon="delete-outline"
                  size={18}
                  onPress={() => remove(n.id)}
                />
              </View>
            )}
          />
        ))
      )}
    </ScreenContainer>
  );
};

const styles = StyleSheet.create({
  headerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
});

export default NotificationsScreen;
