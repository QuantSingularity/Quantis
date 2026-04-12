"""
Financial Services Module
Implements financial industry-specific functionality and compliance features
"""

import decimal
import logging
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from ..domain.models import Transaction

logger = logging.getLogger(__name__)
decimal.getcontext().prec = 28
decimal.getcontext().rounding = ROUND_HALF_UP


class TransactionType(str, Enum):
    """Transaction types for financial operations"""

    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    PAYMENT = "payment"
    REFUND = "refund"
    FEE = "fee"
    INTEREST = "interest"
    DIVIDEND = "dividend"
    ADJUSTMENT = "adjustment"


class TransactionStatus(str, Enum):
    """Transaction status enumeration"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REVERSED = "reversed"


class RiskLevel(str, Enum):
    """Risk level enumeration for transactions and users"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FinancialCalculationService:
    """Service for financial calculations with high precision"""

    @staticmethod
    def calculate_interest(
        principal: Decimal,
        rate: Decimal,
        time_periods: int,
        compound_frequency: int = 1,
    ) -> Decimal:
        """Calculate compound interest with high precision"""
        try:
            principal = Decimal(str(principal))
            rate = Decimal(str(rate))
            rate_per_period = rate / Decimal(str(compound_frequency))
            exponent = compound_frequency * time_periods
            amount = principal * (Decimal("1") + rate_per_period) ** exponent
            interest = amount - principal
            return interest.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        except Exception as e:
            logger.error(f"Error calculating interest: {e}")
            raise ValueError(f"Error in calculation: {e}")

    @staticmethod
    def calculate_present_value(
        future_value: Decimal, discount_rate: Decimal, periods: int
    ) -> Decimal:
        """Calculate present value of future cash flows"""
        try:
            future_value = Decimal(str(future_value))
            discount_rate = Decimal(str(discount_rate))
            present_value = future_value / (Decimal("1") + discount_rate) ** periods
            return present_value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        except Exception as e:
            logger.error(f"Error calculating present value: {e}")
            raise ValueError(f"Error in calculation: {e}")

    @staticmethod
    def calculate_net_present_value(
        cash_flows: List[Decimal], discount_rate: Decimal
    ) -> Decimal:
        """Calculate Net Present Value of cash flows"""
        try:
            npv = Decimal("0.00")
            discount_rate = Decimal(str(discount_rate))
            for period, cash_flow in enumerate(cash_flows):
                cash_flow = Decimal(str(cash_flow))
                pv = cash_flow / (Decimal("1") + discount_rate) ** period
                npv += pv
            return npv.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        except Exception as e:
            logger.error(f"Error calculating NPV: {e}")
            raise ValueError(f"Error in calculation: {e}")

    @staticmethod
    def calculate_internal_rate_of_return(
        cash_flows: List[Decimal], initial_guess: Decimal = Decimal("0.1")
    ) -> Optional[Decimal]:
        """Calculate Internal Rate of Return using Newton-Raphson method"""
        try:
            rate = initial_guess
            tolerance = Decimal("0.0001")
            max_iterations = 100
            for _ in range(max_iterations):
                npv = Decimal("0.00")
                npv_derivative = Decimal("0.00")
                for period, cash_flow in enumerate(cash_flows):
                    cash_flow = Decimal(str(cash_flow))
                    factor = (Decimal("1") + rate) ** period
                    npv += cash_flow / factor
                    if period > 0:
                        npv_derivative -= (
                            period * cash_flow / (Decimal("1") + rate) ** (period + 1)
                        )
                if abs(npv) < tolerance:
                    return rate.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
                if npv_derivative == 0:
                    break
                rate = rate - npv / npv_derivative
            raise ValueError("IRR calculation failed to converge.")
        except Exception as e:
            logger.error(f"Error calculating IRR: {e}")
            raise ValueError(f"Error calculating IRR: {e}")


class RiskAssessmentService:
    """Service for financial risk assessment and management"""

    def __init__(self, db: Session) -> None:
        self.db = db

    def assess_transaction_risk(
        self,
        user_id: int,
        transaction_amount: Decimal,
        transaction_type: TransactionType,
        counterparty_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assess risk level for a financial transaction"""
        try:
            risk_factors = []
            risk_score = 0
            if transaction_amount > Decimal("10000"):
                risk_factors.append("High transaction amount")
                risk_score += 30
            elif transaction_amount > Decimal("5000"):
                risk_factors.append("Medium transaction amount")
                risk_score += 15
            user_risk = self._assess_user_risk(user_id)
            risk_score += user_risk["score"]
            risk_factors.extend(user_risk["factors"])
            high_risk_types = [TransactionType.WITHDRAWAL, TransactionType.TRANSFER]
            if transaction_type in high_risk_types:
                risk_factors.append(f"High-risk transaction type: {transaction_type}")
                risk_score += 20
            if counterparty_info:
                counterparty_risk = self._assess_counterparty_risk(counterparty_info)
                risk_score += counterparty_risk["score"]
                risk_factors.extend(counterparty_risk["factors"])
            if risk_score >= 70:
                risk_level = RiskLevel.CRITICAL
            elif risk_score >= 50:
                risk_level = RiskLevel.HIGH
            elif risk_score >= 30:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            return {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "risk_factors": risk_factors,
                "requires_approval": risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL],
                "requires_additional_verification": risk_level == RiskLevel.CRITICAL,
            }
        except Exception as e:
            logger.error(f"Error assessing transaction risk: {e}")
            return {
                "risk_level": RiskLevel.HIGH,
                "risk_score": 100,
                "risk_factors": ["Risk assessment error"],
                "requires_approval": True,
                "requires_additional_verification": True,
            }

    def _assess_user_risk(self, user_id: int) -> Dict[str, Any]:
        """Assess risk based on user's transaction history"""
        try:
            recent_transactions = (
                self.db.query(Transaction)
                .filter(
                    and_(
                        Transaction.user_id == user_id,
                        Transaction.created_at
                        >= datetime.utcnow() - timedelta(days=30),
                    )
                )
                .all()
            )
            risk_score = 0
            risk_factors = []
            if len(recent_transactions) > 50:
                risk_factors.append("High transaction frequency")
                risk_score += 25
            elif len(recent_transactions) > 20:
                risk_factors.append("Medium transaction frequency")
                risk_score += 10
            failed_transactions = [
                t for t in recent_transactions if t.status == TransactionStatus.FAILED
            ]
            if recent_transactions:
                failure_rate = len(failed_transactions) / len(recent_transactions)
                if failure_rate > 0.2:
                    risk_factors.append("High transaction failure rate")
                    risk_score += 20
                elif failure_rate > 0.1:
                    risk_factors.append("Medium transaction failure rate")
                    risk_score += 10
            large_transactions = [
                t
                for t in recent_transactions
                if Decimal(str(t.amount)) > Decimal("5000")
            ]
            if len(large_transactions) > 5:
                risk_factors.append("Frequent large transactions")
                risk_score += 15
            return {"score": risk_score, "factors": risk_factors}
        except Exception as e:
            logger.error(f"Error assessing user risk: {e}")
            return {"score": 50, "factors": ["User risk assessment error"]}

    def _assess_counterparty_risk(
        self, counterparty_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risk based on counterparty information"""
        risk_score = 0
        risk_factors = []
        high_risk_countries = ["XX", "YY", "ZZ"]
        country = counterparty_info.get("country", "").upper()
        if country in high_risk_countries:
            risk_factors.append(f"High-risk country: {country}")
            risk_score += 30
        if counterparty_info.get("is_new_counterparty", False):
            risk_factors.append("New counterparty")
            risk_score += 15
        if counterparty_info.get("sanctions_hit", False):
            risk_factors.append("Potential sanctions match")
            risk_score += 50
        return {"score": risk_score, "factors": risk_factors}


class ComplianceMonitoringService:
    """Service for monitoring compliance with financial regulations"""

    def __init__(self, db: Session) -> None:
        self.db = db

    def monitor_transaction_limits(
        self,
        user_id: int,
        transaction_amount: Decimal,
        transaction_type: TransactionType,
    ) -> Dict[str, Any]:
        """Monitor transaction limits for compliance"""
        try:
            today = datetime.utcnow().date()
            daily_transactions = self.db.query(func.sum(Transaction.amount)).filter(
                and_(
                    Transaction.user_id == user_id,
                    func.date(Transaction.created_at) == today,
                    Transaction.status == TransactionStatus.COMPLETED,
                )
            ).scalar() or Decimal("0")
            daily_limit = Decimal("50000")
            month_start = datetime.utcnow().replace(day=1).date()
            monthly_transactions = self.db.query(func.sum(Transaction.amount)).filter(
                and_(
                    Transaction.user_id == user_id,
                    func.date(Transaction.created_at) >= month_start,
                    Transaction.status == TransactionStatus.COMPLETED,
                )
            ).scalar() or Decimal("0")
            monthly_limit = Decimal("500000")
            violations = []
            if daily_transactions + transaction_amount > daily_limit:
                violations.append(
                    {
                        "type": "daily_limit_exceeded",
                        "current": daily_transactions,
                        "limit": daily_limit,
                        "attempted": transaction_amount,
                    }
                )
            if monthly_transactions + transaction_amount > monthly_limit:
                violations.append(
                    {
                        "type": "monthly_limit_exceeded",
                        "current": monthly_transactions,
                        "limit": monthly_limit,
                        "attempted": transaction_amount,
                    }
                )
            return {
                "compliant": len(violations) == 0,
                "violations": violations,
                "daily_usage": {
                    "current": daily_transactions,
                    "limit": daily_limit,
                    "remaining": daily_limit - daily_transactions,
                },
                "monthly_usage": {
                    "current": monthly_transactions,
                    "limit": monthly_limit,
                    "remaining": monthly_limit - monthly_transactions,
                },
            }
        except Exception as e:
            logger.error(f"Error monitoring transaction limits: {e}")
            return {
                "compliant": False,
                "violations": [{"type": "monitoring_error", "error": str(e)}],
                "daily_usage": {
                    "current": Decimal("0"),
                    "limit": Decimal("0"),
                    "remaining": Decimal("0"),
                },
                "monthly_usage": {
                    "current": Decimal("0"),
                    "limit": Decimal("0"),
                    "remaining": Decimal("0"),
                },
            }

    def check_aml_requirements(
        self, user_id: int, transaction_amount: Decimal
    ) -> Dict[str, Any]:
        """Check Anti-Money Laundering (AML) requirements"""
        try:
            requirements = {
                "kyc_required": False,
                "enhanced_due_diligence": False,
                "suspicious_activity_report": False,
                "transaction_monitoring": True,
                "reasons": [],
            }
            if transaction_amount >= Decimal("10000"):
                requirements["kyc_required"] = True
                requirements["reasons"].append("Large transaction amount")
            if transaction_amount >= Decimal("50000"):
                requirements["enhanced_due_diligence"] = True
                requirements["reasons"].append("Very large transaction amount")
            suspicious_patterns = self._detect_suspicious_patterns(user_id)
            if suspicious_patterns:
                requirements["suspicious_activity_report"] = True
                requirements["reasons"].extend(suspicious_patterns)
            return requirements
        except Exception as e:
            logger.error(f"Error checking AML requirements: {e}")
            return {
                "kyc_required": True,
                "enhanced_due_diligence": True,
                "suspicious_activity_report": False,
                "transaction_monitoring": True,
                "reasons": ["AML check error"],
            }

    def _detect_suspicious_patterns(self, user_id: int) -> List[str]:
        """Detect suspicious transaction patterns"""
        try:
            patterns = []
            recent_transactions = (
                self.db.query(Transaction)
                .filter(
                    and_(
                        Transaction.user_id == user_id,
                        Transaction.created_at >= datetime.utcnow() - timedelta(days=7),
                    )
                )
                .all()
            )
            if len(recent_transactions) > 20:
                patterns.append("High frequency transactions")
            round_numbers = [
                t
                for t in recent_transactions
                if Decimal(str(t.amount)) % Decimal("1000") == 0
            ]
            if len(round_numbers) > 5:
                patterns.append("Multiple round number transactions")
            threshold = Decimal("9999")
            near_threshold = [
                t
                for t in recent_transactions
                if threshold - Decimal("500") <= Decimal(str(t.amount)) <= threshold
            ]
            if len(near_threshold) > 2:
                patterns.append("Multiple transactions just under reporting threshold")
            return patterns
        except Exception as e:
            logger.error(f"Error detecting suspicious patterns: {e}")
            return ["Pattern detection error"]


class FinancialReportingService:
    """Service for financial reporting and analytics"""

    def __init__(self, db: Session) -> None:
        self.db = db

    def generate_transaction_summary(
        self, user_id: int, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Generate transaction summary for a user"""
        try:
            transactions = (
                self.db.query(Transaction)
                .filter(
                    and_(
                        Transaction.user_id == user_id,
                        Transaction.created_at >= start_date,
                        Transaction.created_at <= end_date,
                    )
                )
                .all()
            )
            summary = {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
                "transaction_count": len(transactions),
                "total_volume": Decimal("0"),
                "by_type": {},
                "by_status": {},
                "average_amount": Decimal("0"),
                "largest_transaction": Decimal("0"),
                "smallest_transaction": None,
            }
            if not transactions:
                return summary
            amounts = [t.amount for t in transactions]
            summary["total_volume"] = sum(amounts)
            summary["average_amount"] = summary["total_volume"] / len(transactions)
            summary["largest_transaction"] = max(amounts)
            summary["smallest_transaction"] = min(amounts)
            for transaction in transactions:
                tx_type = transaction.transaction_type
                if tx_type not in summary["by_type"]:
                    summary["by_type"][tx_type] = {"count": 0, "volume": Decimal("0")}
                summary["by_type"][tx_type]["count"] += 1
                summary["by_type"][tx_type]["volume"] += transaction.amount
            for transaction in transactions:
                status = transaction.status
                if status not in summary["by_status"]:
                    summary["by_status"][status] = {"count": 0, "volume": Decimal("0")}
                summary["by_status"][status]["count"] += 1
                summary["by_status"][status]["volume"] += transaction.amount
            summary = self._convert_decimals_to_strings(summary)
            return summary
        except Exception as e:
            logger.error(f"Error generating transaction summary: {e}")
            return {"error": str(e)}

    def _convert_decimals_to_strings(self, obj: Any) -> Any:
        """Convert Decimal objects to strings for JSON serialization"""
        if isinstance(obj, dict):
            return {
                key: self._convert_decimals_to_strings(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_decimals_to_strings(item) for item in obj]
        elif isinstance(obj, Decimal):
            return str(obj)
        else:
            return obj


calculation_service = FinancialCalculationService()


def get_financial_services(db: Session) -> Any:
    """Get financial service instances"""
    return {
        "calculation": calculation_service,
        "risk_assessment": RiskAssessmentService(db),
        "compliance_monitoring": ComplianceMonitoringService(db),
        "reporting": FinancialReportingService(db),
    }
