import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import React, { useState } from "react";
import { StyleSheet, View } from "react-native";
import { Button, Snackbar, Text, TextInput } from "react-native-paper";
import { authAPI, getErrorMessage } from "../../api";
import ScreenContainer from "../../components/ScreenContainer";
import { useAuth } from "../../context/AuthContext";
import type { MoreStackParamList } from "../../navigation/types";

type Props = NativeStackScreenProps<MoreStackParamList, "Profile">;

const ProfileScreen: React.FC<Props> = () => {
  const { user, refreshUser } = useAuth();
  const [firstName, setFirstName] = useState(user?.first_name || "");
  const [lastName, setLastName] = useState(user?.last_name || "");
  const [phone, setPhone] = useState(user?.phone_number || "");
  const [saving, setSaving] = useState(false);
  const [snackbar, setSnackbar] = useState<string | null>(null);

  const handleSave = async () => {
    setSaving(true);
    try {
      await authAPI.updateProfile({
        first_name: firstName,
        last_name: lastName,
        phone_number: phone,
      });
      await refreshUser();
      setSnackbar("Profile updated");
    } catch (err) {
      setSnackbar(getErrorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <ScreenContainer>
      <Text variant="headlineSmall" style={styles.title}>
        Account information
      </Text>

      <TextInput
        label="Username"
        value={user?.username || ""}
        mode="outlined"
        disabled
        style={styles.input}
      />
      <TextInput
        label="Email"
        value={user?.email || ""}
        mode="outlined"
        disabled
        style={styles.input}
      />
      <TextInput
        label="First name"
        value={firstName}
        onChangeText={setFirstName}
        mode="outlined"
        style={styles.input}
      />
      <TextInput
        label="Last name"
        value={lastName}
        onChangeText={setLastName}
        mode="outlined"
        style={styles.input}
      />
      <TextInput
        label="Phone number"
        value={phone}
        onChangeText={setPhone}
        mode="outlined"
        style={styles.input}
      />

      <Button
        mode="contained"
        onPress={handleSave}
        loading={saving}
        disabled={saving}
      >
        Save changes
      </Button>

      <Snackbar
        visible={Boolean(snackbar)}
        onDismiss={() => setSnackbar(null)}
        duration={3000}
      >
        {snackbar}
      </Snackbar>
    </ScreenContainer>
  );
};

const styles = StyleSheet.create({
  title: { fontWeight: "700", marginBottom: 16 },
  input: { marginBottom: 12 },
});

export default ProfileScreen;
