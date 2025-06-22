from datetime import datetime
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict
from bson import ObjectId

class PyObjectId(ObjectId):
    """
    Lớp chuyển đổi ObjectId cho Pydantic v2
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, _schema_generator):
        return {"type": "string"}

class BaseModelWithId(BaseModel):
    """
    Base model với các trường chung cho tất cả collections
    """
    model_config = ConfigDict(
        # Cho phép sử dụng alias trong field names
        populate_by_name=True,
        # Cho phép các kiểu dữ liệu tùy chỉnh như ObjectId
        arbitrary_types_allowed=True,
        # Định nghĩa cách encode JSON cho các kiểu đặc biệt
        json_encoders={
            ObjectId: str,  # Chuyển ObjectId thành string
            datetime: lambda dt: dt.isoformat(),  # Chuyển datetime thành ISO string
        }
    )
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class BaseResponse(BaseModel):
    """
    Base response model cho API responses
    Chuẩn hóa format trả về cho tất cả endpoints
    """
    model_config = ConfigDict(
        # Cấu hình JSON encoding
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        }
    )
    
    success: bool = True
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None