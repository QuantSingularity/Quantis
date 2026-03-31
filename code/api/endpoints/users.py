"""
User management endpoints for Quantis API
"""

import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..auth import AuditLogger, get_current_user, require_permission
from ..config import Settings, get_settings
from ..database import DataMaskingManager, get_data_masking_manager, get_db
from ..models import Permission, Role, User
from ..schemas import (
    PermissionCreate,
    PermissionResponse,
    RoleCreate,
    RoleResponse,
    UserResponse,
    UserUpdate,
)

logger = logging.getLogger(__name__)
settings: Settings = get_settings()

router = APIRouter()


# User Endpoints
@router.get("/", response_model=List[UserResponse])
@require_permission("read_users")
async def get_all_users(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    data_masking_manager: DataMaskingManager = Depends(get_data_masking_manager),
):
    """Retrieve all users (admin/privileged access only)"""
    users = db.query(User).filter(not User.is_deleted).all()

    # Apply data masking if enabled
    masked_users = []
    for user in users:
        user_dict = UserResponse.from_orm(user).dict()
        masked_users.append(data_masking_manager.mask_object(user_dict))

    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="read_all_users",
        resource_type="user",
        resource_name="all_users",
        request=request,
    )
    return masked_users


@router.get("/{user_id}", response_model=UserResponse)
@require_permission("read_user")
async def get_user_by_id(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    data_masking_manager: DataMaskingManager = Depends(get_data_masking_manager),
):
    """Retrieve a user by ID (admin/privileged access or self)"""
    user = db.query(User).filter(User.id == user_id, not User.is_deleted).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Ensure user can only access their own data unless they have admin/read_users permission
    if not (
        current_user.id == user_id or require_permission("read_all_users")(current_user)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this user's data",
        )

    user_dict = UserResponse.from_orm(user).dict()
    masked_user = data_masking_manager.mask_object(user_dict)

    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="read_user_by_id",
        resource_type="user",
        resource_id=str(user_id),
        resource_name=user.username,
        request=request,
    )
    return masked_user


@router.put("/{user_id}", response_model=UserResponse)
@require_permission("update_user")
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update a user's information (admin/privileged access or self)"""
    user = db.query(User).filter(User.id == user_id, not User.is_deleted).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Ensure user can only update their own data unless they have admin/update_users permission
    if not (
        current_user.id == user_id
        or require_permission("update_all_users")(current_user)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this user's data",
        )

    update_data = user_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(user, key, value)

    db.commit()
    db.refresh(user)

    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="update_user",
        resource_type="user",
        resource_id=str(user_id),
        resource_name=user.username,
        request=request,
    )
    return UserResponse.from_orm(user)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
@require_permission("delete_user")
async def delete_user(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Soft delete a user (admin/privileged access only)"""
    user = db.query(User).filter(User.id == user_id, not User.is_deleted).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Prevent self-deletion for admin users
    if current_user.id == user_id and require_permission("admin")(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin users cannot delete their own account.",
        )

    user.is_deleted = True
    user.deleted_at = datetime.utcnow()
    user.deleted_by_id = current_user.id
    db.commit()

    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="delete_user",
        resource_type="user",
        resource_id=str(user_id),
        resource_name=user.username,
        request=request,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# Role Endpoints
@router.post("/roles", response_model=RoleResponse, status_code=status.HTTP_201_CREATED)
@require_permission("create_role")
async def create_role(
    role_data: RoleCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new role"""
    try:
        # Check if role name already exists
        existing_role = (
            db.query(Role).filter(Role.role_name == role_data.role_name).first()
        )
        if existing_role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Role name already exists.",
            )

        # Validate permissions
        permissions = (
            db.query(Permission)
            .filter(Permission.id.in_(role_data.permission_ids))
            .all()
        )
        if len(permissions) != len(role_data.permission_ids):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="One or more permission IDs are invalid.",
            )

        new_role = Role(
            role_name=role_data.role_name,
            description=role_data.description,
            permissions=permissions,
        )
        db.add(new_role)
        db.commit()
        db.refresh(new_role)

        AuditLogger.log_event(
            db=db,
            user_id=current_user.id,
            action="create_role",
            resource_type="role",
            resource_id=str(new_role.id),
            resource_name=new_role.role_name,
            request=request,
        )
        return RoleResponse.from_orm(new_role)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role with this name already exists.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating role: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create role.",
        )


@router.get("/roles", response_model=List[RoleResponse])
@require_permission("read_roles")
async def get_all_roles(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve all roles"""
    roles = db.query(Role).all()
    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="read_all_roles",
        resource_type="role",
        resource_name="all_roles",
        request=request,
    )
    return [RoleResponse.from_orm(role) for role in roles]


@router.put("/roles/{role_id}", response_model=RoleResponse)
@require_permission("update_role")
async def update_role(
    role_id: int,
    role_update: RoleCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update an existing role"""
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Role not found."
        )

    update_data = role_update.dict(exclude_unset=True)
    if "permission_ids" in update_data:
        permissions = (
            db.query(Permission)
            .filter(Permission.id.in_(update_data["permission_ids"]))
            .all()
        )
        if len(permissions) != len(update_data["permission_ids"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="One or more permission IDs are invalid.",
            )
        role.permissions = permissions
        del update_data["permission_ids"]

    for key, value in update_data.items():
        setattr(role, key, value)

    db.commit()
    db.refresh(role)

    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="update_role",
        resource_type="role",
        resource_id=str(role_id),
        resource_name=role.role_name,
        request=request,
    )
    return RoleResponse.from_orm(role)


@router.delete("/roles/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
@require_permission("delete_role")
async def delete_role(
    role_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a role"""
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Role not found."
        )

    # Prevent deletion of roles that have associated users
    if db.query(User).filter(User.role_id == role_id).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete role: users are assigned to this role.",
        )

    db.delete(role)
    db.commit()

    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="delete_role",
        resource_type="role",
        resource_id=str(role_id),
        resource_name=role.role_name,
        request=request,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# Permission Endpoints
@router.post(
    "/permissions",
    response_model=PermissionResponse,
    status_code=status.HTTP_201_CREATED,
)
@require_permission("create_permission")
async def create_permission(
    permission_data: PermissionCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new permission"""
    try:
        new_permission = Permission(
            permission_name=permission_data.permission_name,
            description=permission_data.description,
        )
        db.add(new_permission)
        db.commit()
        db.refresh(new_permission)

        AuditLogger.log_event(
            db=db,
            user_id=current_user.id,
            action="create_permission",
            resource_type="permission",
            resource_id=str(new_permission.id),
            resource_name=new_permission.permission_name,
            request=request,
        )
        return PermissionResponse.from_orm(new_permission)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Permission with this name already exists.",
        )
    except Exception as e:
        logger.error(f"Error creating permission: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create permission.",
        )


@router.get("/permissions", response_model=List[PermissionResponse])
@require_permission("read_permissions")
async def get_all_permissions(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve all permissions"""
    permissions = db.query(Permission).all()
    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="read_all_permissions",
        resource_type="permission",
        resource_name="all_permissions",
        request=request,
    )
    return [PermissionResponse.from_orm(permission) for permission in permissions]


@router.delete("/permissions/{permission_id}", status_code=status.HTTP_204_NO_CONTENT)
@require_permission("delete_permission")
async def delete_permission(
    permission_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a permission"""
    permission = db.query(Permission).filter(Permission.id == permission_id).first()
    if not permission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Permission not found."
        )

    # Prevent deletion of permissions that are assigned to roles
    if permission.roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete permission: it is assigned to one or more roles.",
        )

    db.delete(permission)
    db.commit()

    AuditLogger.log_event(
        db=db,
        user_id=current_user.id,
        action="delete_permission",
        resource_type="permission",
        resource_id=str(permission_id),
        resource_name=permission.permission_name,
        request=request,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
