"""
Financial Endpoints
Provides financial transaction and compliance endpoints
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from ..auth.auth import AuditLogger, get_current_user, require_admin
from ..core.database import get_db

# AuditLogger imported from auth
from ..domain.models import Transaction, User
from ..domain.schemas import (
    FinancialSummaryResponse,
    TransactionCreate,
    TransactionResponse,
)
from ..services.compliance_service import get_compliance_services
from ..services.financial_service import (
    TransactionStatus,
    TransactionType,
    get_financial_services,
)

router = APIRouter()
logger = logging.getLogger(__name__)
audit_logger = AuditLogger()


@router.post("/transactions", response_model=TransactionResponse)
async def create_transaction(
    transaction: TransactionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new financial transaction with risk assessment and compliance checks"""
    try:
        financial_services = get_financial_services(db)
        get_compliance_services(db)

        # Risk assessment
        risk_assessment = financial_services["risk_assessment"].assess_transaction_risk(
            user_id=current_user.id,
            transaction_amount=transaction.amount,
            transaction_type=TransactionType(transaction.transaction_type),
            counterparty_info=transaction.counterparty_info,
        )

        # Compliance monitoring
        limit_check = financial_services[
            "compliance_monitoring"
        ].monitor_transaction_limits(
            user_id=current_user.id,
            transaction_amount=transaction.amount,
            transaction_type=TransactionType(transaction.transaction_type),
        )

        # AML requirements check
        aml_check = financial_services["compliance_monitoring"].check_aml_requirements(
            user_id=current_user.id, transaction_amount=transaction.amount
        )

        # Check if transaction should be blocked
        if not limit_check["compliant"]:
            audit_logger.log_security_event(
                user_id=current_user.id,
                action="transaction_blocked",
                resource_type="transaction",
                details={
                    "reason": "limit_exceeded",
                    "violations": limit_check["violations"],
                },
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Transaction exceeds limits: {limit_check['violations']}",
            )

        # Determine transaction status based on risk assessment
        if risk_assessment["requires_additional_verification"]:
            transaction_status = TransactionStatus.PENDING
        elif risk_assessment["requires_approval"]:
            transaction_status = TransactionStatus.PROCESSING
        else:
            transaction_status = TransactionStatus.COMPLETED

        # Create transaction record
        db_transaction = Transaction(
            user_id=current_user.id,
            amount=transaction.amount,
            transaction_type=transaction.transaction_type,
            status=transaction_status.value,
            description=transaction.description,
            counterparty_info=transaction.counterparty_info,
            risk_level=risk_assessment["risk_level"].value,
            risk_score=risk_assessment["risk_score"],
            compliance_flags={
                "aml_check": aml_check,
                "risk_assessment": risk_assessment,
                "limit_check": limit_check,
            },
        )

        db.add(db_transaction)
        db.commit()
        db.refresh(db_transaction)

        # Log transaction creation
        audit_logger.log_security_event(
            user_id=current_user.id,
            action="transaction_created",
            resource_type="transaction",
            resource_id=db_transaction.id,
            details={
                "amount": str(transaction.amount),
                "type": transaction.transaction_type,
                "status": transaction_status.value,
                "risk_level": risk_assessment["risk_level"].value,
            },
        )

        return TransactionResponse(
            id=db_transaction.id,
            amount=db_transaction.amount,
            transaction_type=db_transaction.transaction_type,
            status=db_transaction.status,
            description=db_transaction.description,
            created_at=db_transaction.created_at,
            risk_level=db_transaction.risk_level,
            requires_approval=risk_assessment["requires_approval"],
            compliance_status=(
                "compliant" if limit_check["compliant"] else "non_compliant"
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating transaction: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during transaction creation",
        )


@router.get("/transactions", response_model=List[TransactionResponse])
async def get_user_transactions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    transaction_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get user's transaction history with filtering options"""
    try:
        query = db.query(Transaction).filter(Transaction.user_id == current_user.id)

        # Apply filters
        if transaction_type:
            query = query.filter(Transaction.transaction_type == transaction_type)

        if status:
            query = query.filter(Transaction.status == status)

        if start_date:
            query = query.filter(Transaction.created_at >= start_date)

        if end_date:
            query = query.filter(Transaction.created_at <= end_date)

        # Apply pagination and ordering
        transactions = (
            query.order_by(Transaction.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

        # Convert to response format
        response_transactions = []
        for transaction in transactions:
            response_transactions.append(
                TransactionResponse(
                    id=transaction.id,
                    amount=transaction.amount,
                    transaction_type=transaction.transaction_type,
                    status=transaction.status,
                    description=transaction.description,
                    created_at=transaction.created_at,
                    risk_level=transaction.risk_level,
                    requires_approval=transaction.status in ["pending", "processing"],
                    compliance_status="compliant",  # Simplified for response
                )
            )

        return response_transactions

    except Exception as e:
        logger.error(f"Error retrieving transactions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during transaction retrieval",
        )


@router.get("/transactions/{transaction_id}", response_model=TransactionResponse)
async def get_transaction(
    transaction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get specific transaction details"""
    try:
        transaction = (
            db.query(Transaction)
            .filter(
                Transaction.id == transaction_id, Transaction.user_id == current_user.id
            )
            .first()
        )

        if not transaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found"
            )

        return TransactionResponse(
            id=transaction.id,
            amount=transaction.amount,
            transaction_type=transaction.transaction_type,
            status=transaction.status,
            description=transaction.description,
            created_at=transaction.created_at,
            risk_level=transaction.risk_level,
            requires_approval=transaction.status in ["pending", "processing"],
            compliance_status="compliant",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving transaction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during transaction retrieval",
        )


@router.get("/financial-summary", response_model=FinancialSummaryResponse)
async def get_financial_summary(
    period_days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get financial summary for the user"""
    try:
        financial_services = get_financial_services(db)

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)

        summary = financial_services["reporting"].generate_transaction_summary(
            user_id=current_user.id, start_date=start_date, end_date=end_date
        )

        return FinancialSummaryResponse(**summary)

    except Exception as e:
        logger.error(f"Error generating financial summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during summary generation",
        )


@router.post("/transactions/{transaction_id}/approve")
async def approve_transaction(
    transaction_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Approve a pending transaction (admin only)"""
    try:
        transaction = (
            db.query(Transaction).filter(Transaction.id == transaction_id).first()
        )

        if not transaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found"
            )

        if transaction.status not in ["pending", "processing"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Transaction is not in a state that can be approved",
            )

        # Update transaction status
        transaction.status = TransactionStatus.COMPLETED.value
        transaction.approved_by = current_user.id
        transaction.approved_at = datetime.utcnow()

        db.commit()

        # Log approval
        audit_logger.log_security_event(
            user_id=current_user.id,
            action="transaction_approved",
            resource_type="transaction",
            resource_id=transaction.id,
            details={
                "original_status": "pending/processing",
                "new_status": "completed",
                "transaction_amount": str(transaction.amount),
            },
        )

        return {
            "message": "Transaction approved successfully",
            "transaction_id": transaction_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving transaction: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during transaction approval",
        )


@router.post("/transactions/{transaction_id}/reject")
async def reject_transaction(
    transaction_id: int,
    reason: str,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Reject a pending transaction (admin only)"""
    try:
        transaction = (
            db.query(Transaction).filter(Transaction.id == transaction_id).first()
        )

        if not transaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found"
            )

        if transaction.status not in ["pending", "processing"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Transaction is not in a state that can be rejected",
            )

        # Update transaction status
        transaction.status = TransactionStatus.FAILED.value
        transaction.rejection_reason = reason
        transaction.rejected_by = current_user.id
        transaction.rejected_at = datetime.utcnow()

        db.commit()

        # Log rejection
        audit_logger.log_security_event(
            user_id=current_user.id,
            action="transaction_rejected",
            resource_type="transaction",
            resource_id=transaction.id,
            details={"reason": reason, "transaction_amount": str(transaction.amount)},
        )

        return {
            "message": "Transaction rejected successfully",
            "transaction_id": transaction_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting transaction: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during transaction rejection",
        )


@router.get("/compliance/limits")
async def get_transaction_limits(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Get current transaction limits and usage"""
    try:
        financial_services = get_financial_services(db)

        # Get current usage for a dummy transaction to check limits
        limit_info = financial_services[
            "compliance_monitoring"
        ].monitor_transaction_limits(
            user_id=current_user.id,
            transaction_amount=Decimal("0"),  # Zero amount to just check current usage
            transaction_type=TransactionType.TRANSFER,
        )

        return {
            "daily_limits": limit_info["daily_usage"],
            "monthly_limits": limit_info["monthly_usage"],
            "compliance_status": (
                "compliant" if limit_info["compliant"] else "non_compliant"
            ),
        }

    except Exception as e:
        logger.error(f"Error retrieving transaction limits: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during limit retrieval",
        )


@router.post("/financial/calculate-interest")
async def calculate_interest(
    principal: Decimal,
    rate: Decimal,
    time_periods: int,
    compound_frequency: int = 1,
    current_user: User = Depends(get_current_user),
):
    """Calculate compound interest"""
    try:
        financial_services = get_financial_services(
            None
        )  # Calculation service doesn't need DB

        interest = financial_services["calculation"].calculate_interest(
            principal=principal,
            rate=rate,
            time_periods=time_periods,
            compound_frequency=compound_frequency,
        )

        return {
            "principal": str(principal),
            "interest_rate": str(rate),
            "time_periods": time_periods,
            "compound_frequency": compound_frequency,
            "calculated_interest": str(interest),
            "total_amount": str(principal + interest),
        }

    except Exception as e:
        logger.error(f"Error calculating interest: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during interest calculation",
        )


@router.post("/financial/calculate-npv")
async def calculate_net_present_value(
    cash_flows: List[Decimal],
    discount_rate: Decimal,
    current_user: User = Depends(get_current_user),
):
    """Calculate Net Present Value of cash flows"""
    try:
        financial_services = get_financial_services(None)

        npv = financial_services["calculation"].calculate_net_present_value(
            cash_flows=cash_flows, discount_rate=discount_rate
        )

        return {
            "cash_flows": [str(cf) for cf in cash_flows],
            "discount_rate": str(discount_rate),
            "net_present_value": str(npv),
        }

    except Exception as e:
        logger.error(f"Error calculating NPV: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during NPV calculation",
        )
