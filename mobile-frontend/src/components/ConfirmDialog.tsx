import React from "react";
import { Button, Dialog, Portal, Text } from "react-native-paper";

interface Props {
  visible: boolean;
  title: string;
  description?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  destructive?: boolean;
  loading?: boolean;
  onConfirm: () => void;
  onDismiss: () => void;
}

const ConfirmDialog: React.FC<Props> = ({
  visible,
  title,
  description,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  destructive = false,
  loading = false,
  onConfirm,
  onDismiss,
}) => (
  <Portal>
    <Dialog visible={visible} onDismiss={onDismiss}>
      <Dialog.Title>{title}</Dialog.Title>
      {description && (
        <Dialog.Content>
          <Text variant="bodyMedium">{description}</Text>
        </Dialog.Content>
      )}
      <Dialog.Actions>
        <Button onPress={onDismiss}>{cancelLabel}</Button>
        <Button
          onPress={onConfirm}
          loading={loading}
          textColor={destructive ? "#EF4444" : undefined}
        >
          {confirmLabel}
        </Button>
      </Dialog.Actions>
    </Dialog>
  </Portal>
);

export default ConfirmDialog;
