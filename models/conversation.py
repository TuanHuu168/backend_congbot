from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from .base import BaseModelWithId, PyObjectId, BaseResponse

class ConversationStatus(str, Enum):
    """
    Trạng thái của cuộc hội thoại
    - ACTIVE: Đang hoạt động, có thể thêm tin nhắn
    - ARCHIVED: Đã lưu trữ, không thêm tin nhắn nhưng vẫn xem được
    - DELETED: Đã xóa, ẩn khỏi danh sách người dùng
    """
    ACTIVE = "active"
    ARCHIVED = "archived" 
    DELETED = "deleted"

class ClientInfo(BaseModel):
    """
    Thông tin về thiết bị/client của người dùng khi tạo tin nhắn
    """
    platform: str = Field(..., description="Nền tảng: web, mobile, desktop")
    device_type: str = Field(..., description="Loại thiết bị: desktop, mobile, tablet")
    user_agent: Optional[str] = Field(None, description="User agent string từ browser")
    ip_address: Optional[str] = Field(None, description="IP address (cho security)")
    screen_resolution: Optional[str] = Field(None, description="Độ phân giải màn hình")

class Exchange(BaseModel):
    """
    Mô hình lưu trữ một cặp câu hỏi-trả lời trong cuộc hội thoại
    """
    exchange_id: str = Field(..., description="ID duy nhất của exchange")
    question: str = Field(..., description="Câu hỏi của người dùng")
    answer: str = Field(..., description="Câu trả lời của chatbot")
    timestamp: datetime = Field(default_factory=datetime.now, description="Thời gian tạo exchange")
    
    # Metadata về performance và context
    tokens_in_exchange: int = Field(default=0, description="Số tokens sử dụng trong exchange này")
    processing_time: float = Field(default=0.0, description="Thời gian xử lý (seconds)")
    source_documents: List[str] = Field(default=[], description="Danh sách chunk IDs được sử dụng")
    
    # Context và quality metrics  
    retrieval_score: Optional[float] = Field(None, description="Điểm số relevance của retrieval")
    confidence_score: Optional[float] = Field(None, description="Độ tin cậy của câu trả lời")
    user_feedback: Optional[Dict[str, Any]] = Field(None, description="Feedback từ người dùng")
    
    # Client information
    client_info: Optional[ClientInfo] = Field(None, description="Thông tin thiết bị client")

class ConversationModel(BaseModelWithId):
    """
    Model chính cho cuộc hội thoại
    """
    # Thông tin cơ bản
    user_id: PyObjectId = Field(..., description="ID của người dùng sở hữu cuộc trò chuyện")
    title: str = Field(default="Cuộc trò chuyện mới", max_length=200, description="Tiêu đề cuộc trò chuyện")
    summary: Optional[str] = Field(None, max_length=500, description="Tóm tắt nội dung cuộc trò chuyện")
    
    # Trạng thái và metadata
    status: ConversationStatus = Field(default=ConversationStatus.ACTIVE, description="Trạng thái cuộc trò chuyện")
    total_tokens: int = Field(default=0, description="Tổng số tokens đã sử dụng")
    total_exchanges: int = Field(default=0, description="Tổng số cặp hỏi-đáp")
    
    # Danh sách các exchanges
    exchanges: List[Exchange] = Field(default=[], description="Danh sách các cặp hỏi-đáp")
    
    # Thống kê và analytics
    average_response_time: float = Field(default=0.0, description="Thời gian phản hồi trung bình")
    topics_discussed: List[str] = Field(default=[], description="Các chủ đề đã thảo luận")
    language: str = Field(default="vi", description="Ngôn ngữ chính của cuộc trò chuyện")
    
    # Metadata bổ sung
    tags: List[str] = Field(default=[], description="Tags để phân loại cuộc trò chuyện")
    is_favorite: bool = Field(default=False, description="Người dùng có đánh dấu yêu thích không")
    last_activity_at: Optional[datetime] = Field(None, description="Thời gian hoạt động cuối")

class ConversationCreate(BaseModel):
    """
    Model cho việc tạo cuộc hội thoại mới
    """
    user_id: str = Field(..., description="ID của người dùng")
    title: str = Field(default="Cuộc trò chuyện mới", max_length=200)
    summary: Optional[str] = Field(None, max_length=500)
    tags: List[str] = Field(default=[], description="Tags ban đầu")

class ConversationUpdate(BaseModel):
    """
    Model cho việc cập nhật thông tin cuộc hội thoại
    """
    title: Optional[str] = Field(None, max_length=200)
    summary: Optional[str] = Field(None, max_length=500)
    status: Optional[ConversationStatus] = None
    tags: Optional[List[str]] = None
    is_favorite: Optional[bool] = None

class ExchangeCreate(BaseModel):
    """
    Model cho việc tạo một cặp hỏi-đáp mới
    """
    question: str = Field(..., min_length=1, description="Câu hỏi của người dùng")
    client_info: Optional[Dict[str, str]] = Field(None, description="Thông tin client")
    context_data: Optional[Dict[str, Any]] = Field(None, description="Dữ liệu context bổ sung")

class ExchangeResponse(BaseModel):
    """
    Model response khi tạo exchange thành công
    """
    exchange_id: str = Field(..., description="ID của exchange vừa tạo")
    question: str = Field(..., description="Câu hỏi")
    answer: str = Field(..., description="Câu trả lời")
    processing_time: float = Field(..., description="Thời gian xử lý")
    source_documents: List[str] = Field(default=[], description="Documents được sử dụng")
    confidence_score: Optional[float] = Field(None, description="Độ tin cậy")

class ConversationResponse(BaseResponse):
    """
    Model phản hồi khi truy vấn thông tin cuộc hội thoại
    """
    data: Optional[Dict[str, Any]] = Field(None, description="Dữ liệu cuộc trò chuyện")
    
    @classmethod
    def from_conversation_model(cls, conversation: ConversationModel, message: str = "Lấy cuộc trò chuyện thành công"):
        """
        Tạo ConversationResponse từ ConversationModel
        """
        conv_dict = conversation.dict()
        conv_dict['id'] = str(conv_dict.get('id', ''))
        conv_dict['user_id'] = str(conv_dict.get('user_id', ''))
        
        return cls(
            success=True,
            message=message,
            data=conv_dict
        )

class ConversationListResponse(BaseResponse):
    """
    Model response cho danh sách cuộc trò chuyện
    """
    data: List[Dict[str, Any]] = Field(default=[], description="Danh sách cuộc trò chuyện")
    total_count: int = Field(default=0, description="Tổng số cuộc trò chuyện")
    page: int = Field(default=1, description="Trang hiện tại")
    page_size: int = Field(default=20, description="Số items per page")
    total_pages: int = Field(default=0, description="Tổng số trang")

class ConversationStats(BaseModel):
    """
    Model thống kê cho cuộc trò chuyện
    """
    total_conversations: int = Field(default=0, description="Tổng số cuộc trò chuyện")
    active_conversations: int = Field(default=0, description="Cuộc trò chuyện đang hoạt động")
    total_exchanges: int = Field(default=0, description="Tổng số exchanges")
    average_exchanges_per_conversation: float = Field(default=0.0, description="Số exchanges trung bình")
    total_tokens_used: int = Field(default=0, description="Tổng tokens đã sử dụng")
    most_discussed_topics: List[str] = Field(default=[], description="Chủ đề được thảo luận nhiều nhất")