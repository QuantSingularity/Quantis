"""
Notification service for sending alerts and updates to users
"""

import logging
import smtplib
from datetime import datetime, timedelta
from email import encoders
from email.mime.base import MimeBase
from email.mime.multipart import MimeMultipart
from email.mime.text import MimeText
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from .. import models
from ..config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class NotificationService:
    """Service for handling notifications and alerts"""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_notification(
        self,
        user_id: int,
        title: str,
        message: str,
        notification_type: str = "info",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> models.Notification:
        """Create a new notification"""
        notification = models.Notification(
            user_id=user_id,
            title=title,
            message=message,
            type=notification_type,
            metadata=metadata or {},
            is_read=False,
        )
        self.db.add(notification)
        self.db.commit()
        self.db.refresh(notification)
        return notification

    def get_user_notifications(
        self, user_id: int, unread_only: bool = False, skip: int = 0, limit: int = 50
    ) -> List[models.Notification]:
        """Get notifications for a user"""
        query = self.db.query(models.Notification).filter(
            models.Notification.user_id == user_id
        )
        if unread_only:
            query = query.filter(not models.Notification.is_read)
        return (
            query.order_by(models.Notification.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def mark_as_read(self, notification_id: int, user_id: int) -> bool:
        """Mark a notification as read"""
        notification = (
            self.db.query(models.Notification)
            .filter(
                models.Notification.id == notification_id,
                models.Notification.user_id == user_id,
            )
            .first()
        )
        if notification:
            notification.is_read = True
            notification.read_at = datetime.utcnow()
            self.db.commit()
            return True
        return False

    def mark_all_as_read(self, user_id: int) -> int:
        """Mark all notifications as read for a user"""
        count = (
            self.db.query(models.Notification)
            .filter(
                models.Notification.user_id == user_id,
                not models.Notification.is_read,
            )
            .update({"is_read": True, "read_at": datetime.utcnow()})
        )
        self.db.commit()
        return count

    def delete_notification(self, notification_id: int, user_id: int) -> bool:
        """Delete a notification"""
        notification = (
            self.db.query(models.Notification)
            .filter(
                models.Notification.id == notification_id,
                models.Notification.user_id == user_id,
            )
            .first()
        )
        if notification:
            self.db.delete(notification)
            self.db.commit()
            return True
        return False

    def get_notification_stats(self, user_id: int) -> Dict[str, int]:
        """Get notification statistics for a user"""
        total = (
            self.db.query(models.Notification)
            .filter(models.Notification.user_id == user_id)
            .count()
        )
        unread = (
            self.db.query(models.Notification)
            .filter(
                models.Notification.user_id == user_id,
                not models.Notification.is_read,
            )
            .count()
        )
        return {"total": total, "unread": unread, "read": total - unread}

    def send_email_notification(
        self,
        to_email: str,
        subject: str,
        message: str,
        html_message: Optional[str] = None,
        attachments: Optional[List[str]] = None,
    ) -> bool:
        """Send email notification"""
        if not self._is_email_configured():
            logger.warning("Email not configured, skipping email notification")
            return False
        try:
            msg = MimeMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = settings.smtp_username
            msg["To"] = to_email
            text_part = MimeText(message, "plain")
            msg.attach(text_part)
            if html_message:
                html_part = MimeText(html_message, "html")
                msg.attach(html_part)
            if attachments:
                for file_path in attachments:
                    self._add_attachment(msg, file_path)
            with smtplib.SMTP(settings.smtp_server, settings.smtp_port) as server:
                if settings.smtp_use_tls:
                    server.starttls()
                server.login(settings.smtp_username, settings.smtp_password)
                server.send_message(msg)
            logger.info(f"Email sent successfully to {to_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {str(e)}")
            return False

    def _is_email_configured(self) -> bool:
        """Check if email is properly configured"""
        return all(
            [settings.smtp_server, settings.smtp_username, settings.smtp_password]
        )

    def _add_attachment(self, msg: MimeMultipart, file_path: str) -> Any:
        """Add file attachment to email"""
        try:
            with open(file_path, "rb") as attachment:
                part = MimeBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {file_path.split('/')[-1]}",
            )
            msg.attach(part)
        except Exception as e:
            logger.error(f"Failed to add attachment {file_path}: {str(e)}")

    def notify_model_training_complete(
        self, user_id: int, model_name: str, success: bool, metrics: Dict = None
    ) -> Any:
        """Notify user when model training is complete"""
        if success:
            title = f"Model '{model_name}' Training Complete"
            message = f"Your model '{model_name}' has been successfully trained."
            notification_type = "success"
            if metrics:
                message += "\n\nPerformance Metrics:\n"
                for key, value in metrics.items():
                    if isinstance(value, float):
                        message += f"- {key}: {value:.4f}\n"
                    else:
                        message += f"- {key}: {value}\n"
        else:
            title = f"Model '{model_name}' Training Failed"
            message = f"Training failed for model '{model_name}'. Please check your dataset and try again."
            notification_type = "error"
        return self.create_notification(
            user_id=user_id,
            title=title,
            message=message,
            notification_type=notification_type,
            metadata={"model_name": model_name, "metrics": metrics},
        )

    def notify_dataset_uploaded(
        self, user_id: int, dataset_name: str, row_count: int
    ) -> Any:
        """Notify user when dataset upload is complete"""
        title = f"Dataset '{dataset_name}' Uploaded"
        message = f"Your dataset '{dataset_name}' has been successfully uploaded with {row_count:,} rows."
        return self.create_notification(
            user_id=user_id,
            title=title,
            message=message,
            notification_type="success",
            metadata={"dataset_name": dataset_name, "row_count": row_count},
        )

    def notify_prediction_batch_complete(
        self, user_id: int, batch_size: int, success_count: int
    ) -> Any:
        """Notify user when batch prediction is complete"""
        title = "Batch Prediction Complete"
        message = f"Batch prediction completed: {success_count}/{batch_size} predictions successful."
        notification_type = "success" if success_count == batch_size else "warning"
        return self.create_notification(
            user_id=user_id,
            title=title,
            message=message,
            notification_type=notification_type,
            metadata={"batch_size": batch_size, "success_count": success_count},
        )

    def notify_api_key_created(self, user_id: int, key_name: str) -> Any:
        """Notify user when new API key is created"""
        title = "New API Key Created"
        message = f"A new API key '{key_name}' has been created for your account."
        return self.create_notification(
            user_id=user_id,
            title=title,
            message=message,
            notification_type="info",
            metadata={"key_name": key_name},
        )

    def notify_account_activity(
        self, user_id: int, activity: str, ip_address: str = None
    ) -> Any:
        """Notify user of account activity"""
        title = "Account Activity"
        message = f"Account activity: {activity}"
        if ip_address:
            message += f" from IP address {ip_address}"
        return self.create_notification(
            user_id=user_id,
            title=title,
            message=message,
            notification_type="info",
            metadata={"activity": activity, "ip_address": ip_address},
        )

    def send_bulk_notification(
        self,
        user_ids: List[int],
        title: str,
        message: str,
        notification_type: str = "info",
    ) -> int:
        """Send notification to multiple users"""
        notifications = []
        for user_id in user_ids:
            notification = models.Notification(
                user_id=user_id,
                title=title,
                message=message,
                type=notification_type,
                is_read=False,
            )
            notifications.append(notification)
        self.db.add_all(notifications)
        self.db.commit()
        return len(notifications)

    def send_admin_alert(self, title: str, message: str, level: str = "warning") -> Any:
        """Send alert to all admin users"""
        admin_users = (
            self.db.query(models.User)
            .filter(models.User.role == "admin", models.User.is_active)
            .all()
        )
        admin_ids = [user.id for user in admin_users]
        return self.send_bulk_notification(
            user_ids=admin_ids,
            title=f"[ADMIN ALERT] {title}",
            message=message,
            notification_type=level,
        )

    def cleanup_old_notifications(self, days: int = 30) -> int:
        """Clean up old read notifications"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        count = (
            self.db.query(models.Notification)
            .filter(
                models.Notification.is_read,
                models.Notification.read_at < cutoff_date,
            )
            .delete()
        )
        self.db.commit()
        return count
