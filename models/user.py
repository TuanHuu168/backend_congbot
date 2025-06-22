# backend/models/user.py
"""
Model định nghĩa cấu trúc dữ liệu cho người dùng trong MongoDB
Bao gồm: thông tin cá nhân, avatar, trạng thái tài khoản
"""
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from .base import BaseModelWithId, PyObjectId, BaseResponse

class UserStatus(str, Enum):
    """
    Trạng thái tài khoản người dùng
    - ACTIVE: Tài khoản hoạt động bình thường
    - INACTIVE: Tài khoản tạm ngưng (có thể kích hoạt lại)
    - BANNED: Tài khoản bị cấm (cần admin can thiệp)
    """
    ACTIVE = "active"
    INACTIVE = "inactive" 
    BANNED = "banned"

class UserRole(str, Enum):
    """
    Vai trò người dùng trong hệ thống
    - USER: Người dùng thông thường
    - ADMIN: Quản trị viên hệ thống
    - MODERATOR: Điều hành viên (quyền trung gian)
    """
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"

class UserModel(BaseModelWithId):
    """
    Model chính cho người dùng với đầy đủ thông tin
    """
    # Thông tin đăng nhập cơ bản
    username: str = Field(..., min_length=3, max_length=50, description="Tên đăng nhập duy nhất")
    email: EmailStr = Field(..., description="Email người dùng (duy nhất)")
    password: str = Field(..., description="Mật khẩu đã hash")
    
    # Thông tin cá nhân
    full_name: str = Field(..., min_length=1, max_length=100, description="Họ và tên đầy đủ")
    phone_number: Optional[str] = Field(None, max_length=15, description="Số điện thoại liên hệ")
    avatar_url: Optional[str] = Field(None, description="URL ảnh đại diện người dùng")
    personal_info: Optional[str] = Field(None, max_length=500, description="Thông tin cá nhân (VD: Thương binh hạng 1/4, Con liệt sĩ, etc.)")
    
    # Quyền và trạng thái
    role: UserRole = Field(default=UserRole.USER, description="Vai trò trong hệ thống")
    status: UserStatus = Field(default=UserStatus.ACTIVE, description="Trạng thái tài khoản")
    
    # Thông tin hoạt động
    last_login_at: Optional[datetime] = Field(None, description="Lần đăng nhập cuối cùng")
    login_count: int = Field(default=0, description="Số lần đăng nhập")
    
    class Config:
        # Kế thừa config từ BaseModelWithId và thêm example
        schema_extra = {
            "example": {
                "username": "nguyen_van_a",
                "email": "nguyenvana@example.com", 
                "password": "hashed_password_here",
                "full_name": "Nguyễn Văn A",
                "phone_number": "0123456789",
                "avatar_url": "https://example.com/avatar.jpg",
                "personal_info": "Thương binh hạng 2/4, tham gia kháng chiến chống Mỹ",
                "role": "user",
                "status": "active"
            }
        }

class UserCreate(BaseModel):
    """
    Model cho việc tạo người dùng mới
    """
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6, description="Mật khẩu gốc (sẽ được hash)")
    full_name: str = Field(..., min_length=1, max_length=100)
    phone_number: Optional[str] = Field(None, max_length=15)
    avatar_url: Optional[str] = None
    personal_info: Optional[str] = Field(None, max_length=500)

class UserUpdate(BaseModel):
    """
    Model cho việc cập nhật thông tin người dùng
    """
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    phone_number: Optional[str] = Field(None, max_length=15)
    email: Optional[EmailStr] = None
    avatar_url: Optional[str] = None
    personal_info: Optional[str] = Field(None, max_length=500)
    status: Optional[UserStatus] = None
    role: Optional[UserRole] = None

class UserPasswordChange(BaseModel):
    """
    Model cho việc đổi mật khẩu
    """
    current_password: str = Field(..., description="Mật khẩu hiện tại")
    new_password: str = Field(..., min_length=6, description="Mật khẩu mới")
    confirm_password: str = Field(..., description="Xác nhận mật khẩu mới")

class UserLogin(BaseModel):
    """
    Model cho việc đăng nhập
    """
    username: str = Field(..., description="Tên đăng nhập hoặc email")
    password: str = Field(..., description="Mật khẩu")
    remember_me: bool = Field(default=False, description="Ghi nhớ đăng nhập")

class UserResponse(BaseResponse):
    """
    Model phản hồi thông tin người dùng (không bao gồm password)
    """
    data: Optional[Dict] = Field(None, description="Thông tin người dùng (không có password)")
    
    @classmethod
    def from_user_model(cls, user: UserModel, message: str = "Lấy thông tin người dùng thành công"):
        """
        Tạo UserResponse từ UserModel, loại bỏ password
        """
        user_dict = user.dict()
        user_dict.pop('password', None)  # Loại bỏ password khỏi response
        user_dict['id'] = str(user_dict.get('id', ''))  # Chuyển ObjectId thành string
        
        return cls(
            success=True,
            message=message,
            data=user_dict
        )

class Token(BaseModel):
    """
    Model cho JWT token response
    """
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Loại token")
    user_id: str = Field(..., description="ID của người dùng")
    expires_in: Optional[int] = Field(None, description="Thời gian hết hạn (seconds)")

class UserStats(BaseModel):
    """
    Model thống kê hoạt động của người dùng
    """
    total_chats: int = Field(default=0, description="Tổng số cuộc trò chuyện")
    total_messages: int = Field(default=0, description="Tổng số tin nhắn")
    days_active: int = Field(default=0, description="Số ngày hoạt động")
    last_activity: Optional[datetime] = Field(None, description="Hoạt động cuối cùng")
    favorite_topics: list = Field(default=[], description="Chủ đề yêu thích")