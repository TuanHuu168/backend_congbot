from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from .base import BaseModelWithId, PyObjectId, BaseResponse

# Import config để lấy cache TTL
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PERF_CONFIG

class CacheStatus(str, Enum):
    """
    Trạng thái của cache entry
    - VALID: Cache còn hợp lệ, có thể sử dụng
    - INVALID: Cache đã bị invalidate, cần refresh
    - EXPIRED: Cache đã hết hạn, sẽ bị xóa tự động
    """
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"

class CacheType(str, Enum):
    """
    Loại cache trong hệ thống
    - TEXT: Cache cho text queries và responses
    - VECTOR: Cache cho vector embeddings  
    - METADATA: Cache cho document metadata
    - SESSION: Cache cho session data
    """
    TEXT = "text"
    VECTOR = "vector"
    METADATA = "metadata"
    SESSION = "session"

class RelevantDocument(BaseModel):
    """
    Thông tin về tài liệu liên quan được cache
    """
    chunk_id: str = Field(..., description="ID của chunk document")
    score: float = Field(..., ge=0.0, le=1.0, description="Điểm relevance (0-1)")
    doc_id: str = Field(..., description="ID của document gốc")
    doc_type: Optional[str] = Field(None, description="Loại document (Luật, Nghị định, etc.)")
    position: int = Field(..., description="Vị trí trong kết quả search")

class CacheMetrics(BaseModel):
    """
    Metrics để đánh giá hiệu quả của cache entry
    """
    hit_count: int = Field(default=0, description="Số lần cache được sử dụng")
    miss_count: int = Field(default=0, description="Số lần cache miss")
    last_used: datetime = Field(default_factory=datetime.now, description="Lần sử dụng cuối")
    average_response_time: float = Field(default=0.0, description="Thời gian phản hồi trung bình khi dùng cache")
    cache_size_bytes: int = Field(default=0, description="Kích thước cache entry (bytes)")

class CacheModel(BaseModelWithId):
    """
    Model chính cho cache entries
    """
    # Identifiers
    cache_id: str = Field(..., description="ID duy nhất để liên kết với ChromaDB")
    cache_type: CacheType = Field(default=CacheType.TEXT, description="Loại cache")
    
    # Content data
    question_text: str = Field(..., description="Câu hỏi gốc")
    normalized_question: str = Field(..., description="Câu hỏi đã chuẩn hóa để so sánh")
    answer: str = Field(..., description="Câu trả lời được cache")
    
    # Context và retrieval info
    relevant_documents: List[RelevantDocument] = Field(default=[], description="Documents liên quan")
    context_items: List[str] = Field(default=[], description="Context text items")
    
    # Cache management
    validity_status: CacheStatus = Field(default=CacheStatus.VALID, description="Trạng thái cache")
    expires_at: datetime = Field(
        default_factory=lambda: datetime.now() + timedelta(days=PERF_CONFIG.CACHE_TTL_DAYS),
        description="Thời gian hết hạn cache"
    )
    
    # Analytics và optimization
    related_doc_ids: List[str] = Field(default=[], description="IDs của documents liên quan")
    keywords: List[str] = Field(default=[], description="Keywords để tìm kiếm cache")
    language: str = Field(default="vi", description="Ngôn ngữ của cache entry")
    
    # Metrics
    metrics: CacheMetrics = Field(default_factory=CacheMetrics, description="Metrics của cache entry")
    
    # Metadata bổ sung
    user_id: Optional[PyObjectId] = Field(None, description="User tạo cache (nếu có)")
    source_model: str = Field(default="unknown", description="Model đã tạo cache này")
    confidence_score: Optional[float] = Field(None, description="Độ tin cậy của cached answer")
    tags: List[str] = Field(default=[], description="Tags để phân loại cache")

class CacheCreate(BaseModel):
    """
    Model cho việc tạo cache entry mới
    """
    cache_id: str = Field(..., description="ID duy nhất cho cache")
    cache_type: CacheType = Field(default=CacheType.TEXT)
    question_text: str = Field(..., min_length=1)
    normalized_question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    relevant_documents: List[Dict[str, Any]] = Field(default=[])
    context_items: List[str] = Field(default=[])
    related_doc_ids: List[str] = Field(default=[])
    keywords: List[str] = Field(default=[])
    user_id: Optional[str] = None
    source_model: str = Field(default="gemini")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    expires_at: Optional[datetime] = None

class CacheUpdate(BaseModel):
    """
    Model cho việc cập nhật cache entry
    """
    validity_status: Optional[CacheStatus] = None
    answer: Optional[str] = None
    expires_at: Optional[datetime] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    tags: Optional[List[str]] = None
    
    # Update metrics
    increment_hit_count: bool = Field(default=False, description="Có tăng hit count không")
    new_response_time: Optional[float] = Field(None, description="Response time mới để cập nhật average")

class CacheQuery(BaseModel):
    """
    Model cho việc query cache
    """
    query_text: str = Field(..., min_length=1, description="Text cần tìm trong cache")
    cache_type: CacheType = Field(default=CacheType.TEXT)
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Ngưỡng similarity")
    max_results: int = Field(default=5, ge=1, le=20, description="Số kết quả tối đa")
    include_expired: bool = Field(default=False, description="Có bao gồm cache đã hết hạn không")

class CacheResponse(BaseResponse):
    """
    Model phản hồi khi truy vấn cache
    """
    data: Optional[Dict[str, Any]] = Field(None, description="Cache data")
    cache_hit: bool = Field(default=False, description="Có cache hit không")
    similarity_score: float = Field(default=0.0, description="Điểm similarity nếu có")
    
    @classmethod
    def from_cache_model(cls, cache: CacheModel, similarity_score: float = 1.0, message: str = "Cache hit"):
        """
        Tạo CacheResponse từ CacheModel
        """
        cache_dict = cache.dict()
        cache_dict['id'] = str(cache_dict.get('id', ''))
        
        return cls(
            success=True,
            message=message,
            data=cache_dict,
            cache_hit=True,
            similarity_score=similarity_score
        )

class CacheStats(BaseModel):
    """
    Model thống kê tổng thể của cache system
    """
    total_entries: int = Field(default=0, description="Tổng số cache entries")
    valid_entries: int = Field(default=0, description="Số cache entries hợp lệ")
    invalid_entries: int = Field(default=0, description="Số cache entries không hợp lệ")
    expired_entries: int = Field(default=0, description="Số cache entries đã hết hạn")
    
    # Performance metrics
    total_hits: int = Field(default=0, description="Tổng số cache hits")
    total_misses: int = Field(default=0, description="Tổng số cache misses")
    hit_rate: float = Field(default=0.0, description="Tỷ lệ cache hit")
    average_response_time: float = Field(default=0.0, description="Thời gian phản hồi trung bình")
    
    # Storage metrics
    total_size_bytes: int = Field(default=0, description="Tổng kích thước cache")
    average_size_per_entry: float = Field(default=0.0, description="Kích thước trung bình per entry")
    
    # Type breakdown
    cache_by_type: Dict[str, int] = Field(default={}, description="Số lượng cache theo từng loại")
    
    # Time-based metrics
    entries_created_today: int = Field(default=0, description="Số entries tạo hôm nay")
    entries_used_today: int = Field(default=0, description="Số entries được dùng hôm nay")
    most_popular_keywords: List[str] = Field(default=[], description="Keywords phổ biến nhất")

class CacheBatchOperation(BaseModel):
    """
    Model cho các thao tác batch trên cache
    """
    operation: str = Field(..., description="Loại operation: delete, invalidate, refresh")
    cache_ids: List[str] = Field(default=[], description="Danh sách cache IDs")
    doc_ids: List[str] = Field(default=[], description="Danh sách document IDs để invalidate")
    criteria: Optional[Dict[str, Any]] = Field(None, description="Criteria để filter cache")
    dry_run: bool = Field(default=False, description="Chỉ simulate, không thực hiện")

class CacheBatchResponse(BaseResponse):
    """
    Response cho batch operations
    """
    affected_count: int = Field(default=0, description="Số entries bị ảnh hưởng")
    operations_performed: List[str] = Field(default=[], description="Danh sách operations đã thực hiện")
    errors: List[str] = Field(default=[], description="Danh sách lỗi nếu có")
    execution_time: float = Field(default=0.0, description="Thời gian thực hiện")