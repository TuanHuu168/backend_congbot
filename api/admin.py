from fastapi import APIRouter, HTTPException, Depends, Body, UploadFile, File
from fastapi.responses import FileResponse
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import os
import json
import time
import uuid
import pandas as pd
from datetime import datetime
import shutil
from pathlib import Path
import threading
import queue
import numpy as np

# Import services
from services.retrieval_service import retrieval_service
from services.generation_service import generation_service
from services.benchmark_service import benchmark_service
from services.activity_service import activity_service, ActivityType
from database.mongodb_client import mongodb_client
from database.chroma_client import chroma_client

# Import config
from config import BENCHMARK_DIR, BENCHMARK_RESULTS_DIR, DATA_DIR
from config import EMBEDDING_MODEL_NAME

benchmark_progress = {}
benchmark_results_cache = {}

# Khởi tạo router
router = APIRouter(
    prefix="",
    tags=["admin"],
    responses={404: {"description": "Not found"}},
)

# === MODELS ===
class BenchmarkConfig(BaseModel):
    file_path: str = "benchmark.json"
    output_dir: str = "benchmark_results"

class DocumentUpload(BaseModel):
    doc_id: str
    doc_type: str
    doc_title: str
    effective_date: str
    status: str = "active"
    document_scope: str = "Quốc gia"

class SystemStatus(BaseModel):
    status: str
    message: str
    database: Dict[str, Any]
    cache_stats: Optional[Dict[str, Any]] = None

# === ENDPOINTS ===
@router.get("/status", response_model=SystemStatus)
async def get_admin_status():
    try:
        # Kiểm tra ChromaDB
        collection = chroma_client.get_collection()
        collection_count = collection.count()
        
        # Kiểm tra MongoDB
        db = mongodb_client.get_database()
        
        # Thống kê cache từ retrieval_service
        cache_stats = retrieval_service.get_cache_stats()
        
        return {
            "status": "ok", 
            "message": "Hệ thống đang hoạt động bình thường",
            "database": {
                "chromadb": {
                    "status": "connected",
                    "collection": collection.name,
                    "documents_count": collection_count
                },
                "mongodb": {
                    "status": "connected",
                    "chat_count": db.chats.count_documents({}),
                    "user_count": db.users.count_documents({})
                }
            },
            "cache_stats": cache_stats
        }
    except Exception as e:
        print(f"Error in get_admin_status: {str(e)}")
        # Chỉ log khi có lỗi
        activity_service.log_activity(
            ActivityType.SYSTEM_STATUS,
            f"Lỗi khi kiểm tra trạng thái hệ thống: {str(e)}",
            metadata={"error": str(e), "success": False}
        )
        return {
            "status": "error",
            "message": f"Hệ thống gặp sự cố: {str(e)}",
            "database": {
                "chromadb": {"status": "error"},
                "mongodb": {"status": "error"}
            },
            "cache_stats": None
        }
        
@router.get("/recent-activities")
async def get_recent_activities(limit: int = 10):
    try:
        activities = activity_service.get_recent_activities(limit)
        return {
            "activities": activities,
            "count": len(activities)
        }
    except Exception as e:
        print(f"Error in get_recent_activities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache/stats")
async def get_cache_detailed_stats():
    try:
        # Lấy thống kê cơ bản từ retrieval_service
        basic_stats = retrieval_service.get_cache_stats()
        
        db = mongodb_client.get_database()
        
        # Thống kê cache theo văn bản liên quan
        doc_stats = []
        pipeline = [
            {"$unwind": "$relatedDocIds"},
            {"$group": {"_id": "$relatedDocIds", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        doc_results = list(db.text_cache.aggregate(pipeline))
        for result in doc_results:
            doc_stats.append({
                "doc_id": result["_id"],
                "cache_count": result["count"]
            })
        
        # Thống kê cache theo hitCount
        popular_cache = list(db.text_cache.find(
            {"validityStatus": "valid"},
            {"cacheId": 1, "questionText": 1, "hitCount": 1, "_id": 0}
        ).sort("hitCount", -1).limit(5))
        
        # Thống kê cache theo thời gian tạo
        recent_cache = list(db.text_cache.find(
            {},
            {"cacheId": 1, "questionText": 1, "createdAt": 1, "_id": 0}
        ).sort("createdAt", -1).limit(5))
        
        return {
            "basic_stats": basic_stats,
            "document_distribution": doc_stats,
            "popular_cache": popular_cache,
            "recent_cache": recent_cache
        }
    except Exception as e:
        print(f"Error in get_cache_detailed_stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear-cache")
async def clear_cache():
    try:
        print("Bắt đầu xóa toàn bộ cache...")
        
        # Log thông tin trước khi xóa
        db = mongodb_client.get_database()
        mongo_before = db.text_cache.count_documents({})
        
        # Kiểm tra ChromaDB trước khi xóa
        chroma_before = 0
        try:
            cache_collection = retrieval_service.chroma.get_cache_collection()
            if cache_collection:
                chroma_before = cache_collection.count()
        except Exception as e:
            print(f"Không thể đếm ChromaDB cache: {str(e)}")
        
        print(f"Trước khi xóa - MongoDB: {mongo_before}, ChromaDB: {chroma_before}")
        
        # Gọi service để xóa toàn bộ
        deleted_count = retrieval_service.clear_all_cache()
        
        # Kiểm tra sau khi xóa
        mongo_after = db.text_cache.count_documents({})
        chroma_after = 0
        try:
            cache_collection = retrieval_service.chroma.get_cache_collection()
            if cache_collection:
                chroma_after = cache_collection.count()
        except Exception as e:
            print(f"Không thể đếm ChromaDB cache sau xóa: {str(e)}")
        
        print(f"Sau khi xóa - MongoDB: {mongo_after}, ChromaDB: {chroma_after}")
        
        # Log activity với thông tin chi tiết
        activity_service.log_activity(
            ActivityType.CACHE_CLEAR,
            f"Admin xóa toàn bộ cache: MongoDB {deleted_count} entries, ChromaDB {chroma_before - chroma_after} entries",
            metadata={
                "action": "admin_clear_all_cache",
                "mongodb_before": mongo_before,
                "mongodb_after": mongo_after,
                "mongodb_deleted": deleted_count,
                "chromadb_before": chroma_before,
                "chromadb_after": chroma_after,
                "chromadb_deleted": chroma_before - chroma_after
            }
        )
        
        return {
            "message": f"Đã xóa toàn bộ cache thành công",
            "mongodb": {
                "deleted_count": deleted_count,
                "before": mongo_before,
                "after": mongo_after
            },
            "chromadb": {
                "before": chroma_before,
                "after": chroma_after,
                "deleted": chroma_before - chroma_after
            },
            "total_deleted": deleted_count + (chroma_before - chroma_after)
        }
        
    except Exception as e:
        print(f"Error in clear_cache: {str(e)}")
        activity_service.log_activity(
            ActivityType.CACHE_CLEAR,
            f"Lỗi xóa cache: {str(e)}",
            metadata={"error": str(e), "success": False}
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear-invalid-cache")
async def clear_invalid_cache():
    try:
        print("Bắt đầu xóa cache không hợp lệ...")
        
        # Đếm cache không hợp lệ trước khi xóa
        db = mongodb_client.get_database()
        invalid_before = db.text_cache.count_documents({"validityStatus": "invalid"})
        print(f"Tìm thấy {invalid_before} cache không hợp lệ")
        
        # Gọi service để xóa
        deleted_count = retrieval_service.clear_all_invalid_cache()
        
        # Kiểm tra sau khi xóa
        invalid_after = db.text_cache.count_documents({"validityStatus": "invalid"})
        print(f"Còn lại {invalid_after} cache không hợp lệ")
        
        # Log activity
        activity_service.log_activity(
            ActivityType.CACHE_CLEAR,
            f"Admin xóa cache không hợp lệ: {deleted_count} entries",
            metadata={
                "action": "admin_clear_invalid_cache",
                "invalid_before": invalid_before,
                "invalid_after": invalid_after,
                "deleted_count": deleted_count
            }
        )
        
        return {
            "message": f"Đã xóa {deleted_count} cache entries không hợp lệ",
            "deleted_count": deleted_count,
            "verification": {
                "invalid_before": invalid_before,
                "invalid_after": invalid_after,
                "actually_deleted": invalid_before - invalid_after
            }
        }
        
    except Exception as e:
        print(f"Error in clear_invalid_cache: {str(e)}")
        activity_service.log_activity(
            ActivityType.CACHE_CLEAR,
            f"Lỗi xóa cache không hợp lệ: {str(e)}",
            metadata={"error": str(e), "success": False}
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/invalidate-cache/{doc_id}")
async def invalidate_cache(doc_id: str):
    try:
        # Sử dụng phương thức từ retrieval_service
        count = retrieval_service.invalidate_document_cache(doc_id)
        
        return {
            "message": f"Đã vô hiệu hóa {count} cache entries liên quan đến văn bản {doc_id}",
            "affected_count": count
        }
    except Exception as e:
        print(f"Error in invalidate_cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/cache/detailed-status")
async def get_cache_detailed_status():
    """Lấy thông tin chi tiết về cache trong cả MongoDB và ChromaDB"""
    try:
        # MongoDB stats
        db = mongodb_client.get_database()
        mongo_total = db.text_cache.count_documents({})
        mongo_valid = db.text_cache.count_documents({"validityStatus": "valid"})
        mongo_invalid = db.text_cache.count_documents({"validityStatus": "invalid"})
        
        # ChromaDB stats
        chroma_total = 0
        chroma_error = None
        try:
            cache_collection = retrieval_service.chroma.get_cache_collection()
            if cache_collection:
                chroma_total = cache_collection.count()
            else:
                chroma_error = "Cache collection not found"
        except Exception as e:
            chroma_error = str(e)
        
        # Retrieval service stats
        service_stats = retrieval_service.get_cache_stats()
        
        return {
            "mongodb": {
                "total": mongo_total,
                "valid": mongo_valid,
                "invalid": mongo_invalid,
                "status": "connected"
            },
            "chromadb": {
                "total": chroma_total,
                "status": "connected" if not chroma_error else "error",
                "error": chroma_error
            },
            "service_stats": service_stats,
            "sync_status": {
                "in_sync": mongo_total == chroma_total,
                "difference": abs(mongo_total - chroma_total)
            }
        }
        
    except Exception as e:
        print(f"Error getting detailed cache status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/delete-expired-cache")
async def delete_expired_cache():
    try:
        print("Admin yêu cầu xóa cache đã hết hạn...")
        
        # Đếm trước khi xóa
        db = mongodb_client.get_database()
        from datetime import datetime
        expired_before = db.text_cache.count_documents({"expiresAt": {"$lt": datetime.now()}})
        
        # Thực hiện xóa
        deleted_count = retrieval_service.delete_expired_cache()
        
        # Log activity
        activity_service.log_activity(
            ActivityType.CACHE_CLEAR,
            f"Admin xóa cache đã hết hạn: {deleted_count} entries",
            metadata={
                "action": "delete_expired_cache",
                "expired_before": expired_before,
                "deleted_count": deleted_count
            }
        )
        
        return {
            "message": f"Đã xóa {deleted_count} cache entries đã hết hạn",
            "deleted_count": deleted_count,
            "expired_found": expired_before
        }
        
    except Exception as e:
        print(f"Error in delete_expired_cache: {str(e)}")
        activity_service.log_activity(
            ActivityType.CACHE_CLEAR,
            f"Lỗi xóa expired cache: {str(e)}",
            metadata={"error": str(e), "success": False}
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search-cache/{keyword}")
async def search_cache(keyword: str, limit: int = 10):
    try:
        print(f"Admin tìm kiếm cache với từ khóa: '{keyword}'")
        
        if not keyword.strip():
            raise HTTPException(status_code=400, detail="Từ khóa không được để trống")
        
        # Gọi service search
        results = retrieval_service.search_keyword(keyword.strip(), limit)
        
        # Chuẩn bị dữ liệu trả về
        cache_results = []
        for result in results:
            try:
                # Chuyển đổi ObjectId và datetime
                processed_result = {
                    "id": str(result["_id"]),
                    "cacheId": result.get("cacheId", ""),
                    "questionText": result.get("questionText", ""),
                    "answer": result.get("answer", "")[:200] + "..." if len(result.get("answer", "")) > 200 else result.get("answer", ""),
                    "validityStatus": result.get("validityStatus", ""),
                    "hitCount": result.get("hitCount", 0),
                    "keywords": result.get("keywords", []),
                    "relatedDocIds": result.get("relatedDocIds", [])
                }
                
                # Xử lý datetime fields
                for date_field in ["createdAt", "updatedAt", "lastUsed", "expiresAt"]:
                    if date_field in result and result[date_field]:
                        processed_result[date_field] = result[date_field].isoformat()
                    else:
                        processed_result[date_field] = None
                
                cache_results.append(processed_result)
                
            except Exception as e:
                print(f"Lỗi xử lý cache result: {str(e)}")
                continue
        
        return {
            "keyword": keyword,
            "limit": limit,
            "count": len(cache_results),
            "total_found": len(results),
            "results": cache_results
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in search_cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/upload-benchmark")
async def upload_benchmark_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="Only JSON files are allowed")
        
        # Read file content
        content = await file.read()
        file_content = content.decode('utf-8')
        
        # Validate JSON format
        try:
            json_data = json.loads(file_content)
            if "benchmark" not in json_data:
                raise HTTPException(status_code=400, detail="Invalid benchmark format. Must contain 'benchmark' key")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format")
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"uploaded_benchmark_{timestamp}.json"
        
        # Save file
        saved_filename = benchmark_service.save_uploaded_benchmark(file_content, filename)
        
        return {
            "message": "Benchmark file uploaded successfully",
            "filename": saved_filename,
            "questions_count": len(json_data["benchmark"])
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error uploading benchmark file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run-benchmark")
async def run_benchmark_4models(config: BenchmarkConfig):
    try:
        benchmark_id = str(uuid.uuid4())
        
        # Log benchmark start
        activity_service.log_activity(
            ActivityType.BENCHMARK_START,
            f"Bắt đầu chạy benchmark với file {config.file_path}",
            metadata={
                "benchmark_id": benchmark_id,
                "file_path": config.file_path,
                "output_dir": config.output_dir
            }
        )
        
        progress_queue = queue.Queue()
        result_queue = queue.Queue()
        
        benchmark_progress[benchmark_id] = {
            'status': 'running',
            'progress': 0.0,
            'phase': 'starting',
            'current_step': 0,
            'total_steps': 0,
            'start_time': datetime.now().isoformat()
        }
        
        def progress_callback(progress_info):
            if isinstance(progress_info, dict):
                progress_queue.put(progress_info)
            else:
                # Backwards compatibility - nếu chỉ là số
                progress_queue.put({'progress': float(progress_info)})
        
        def run_benchmark_thread():
            try:
                stats = benchmark_service.run_benchmark(
                    benchmark_file=config.file_path,
                    progress_callback=progress_callback
                )
                result_queue.put(('success', stats))
            except Exception as e:
                result_queue.put(('error', str(e)))
        
        def monitor_progress():
            thread = threading.Thread(target=run_benchmark_thread)
            thread.start()
            
            while thread.is_alive():
                try:
                    progress_info = progress_queue.get(timeout=1)
                    if isinstance(progress_info, dict):
                        benchmark_progress[benchmark_id].update(progress_info)
                    else:
                        benchmark_progress[benchmark_id]['progress'] = float(progress_info)
                except queue.Empty:
                    continue
            
            try:
                status, result = result_queue.get(timeout=5)
                if status == 'success':
                    benchmark_progress[benchmark_id].update({
                        'status': 'completed',
                        'progress': 100.0,
                        'phase': 'completed',
                        'end_time': datetime.now().isoformat(),
                        'stats': result
                    })
                    benchmark_results_cache[benchmark_id] = result
                    
                    # Log successful completion
                    activity_service.log_activity(
                        ActivityType.BENCHMARK_COMPLETE,
                        f"Hoàn thành benchmark {benchmark_id}: {result.get('total_questions', 0)} câu hỏi",
                        metadata={
                            "benchmark_id": benchmark_id,
                            "total_questions": result.get('total_questions', 0),
                            "output_file": result.get('output_file', ''),
                            "file_path": config.file_path,
                            "duration_minutes": (datetime.now() - datetime.fromisoformat(benchmark_progress[benchmark_id]['start_time'])).total_seconds() / 60
                        }
                    )
                else:
                    benchmark_progress[benchmark_id].update({
                        'status': 'failed',
                        'phase': 'failed',
                        'end_time': datetime.now().isoformat(),
                        'error': result
                    })
                    
                    # Log failure
                    activity_service.log_activity(
                        ActivityType.BENCHMARK_FAIL,
                        f"Thất bại khi chạy benchmark {benchmark_id}: {result}",
                        metadata={
                            "benchmark_id": benchmark_id,
                            "error": result,
                            "file_path": config.file_path
                        }
                    )
            except queue.Empty:
                benchmark_progress[benchmark_id].update({
                    'status': 'failed',
                    'phase': 'timeout',
                    'end_time': datetime.now().isoformat(),
                    'error': 'Benchmark timed out'
                })
                
                # Log timeout
                activity_service.log_activity(
                    ActivityType.BENCHMARK_FAIL,
                    f"Benchmark {benchmark_id} timeout",
                    metadata={
                        "benchmark_id": benchmark_id,
                        "error": "timeout",
                        "file_path": config.file_path
                    }
                )
        
        monitor_thread = threading.Thread(target=monitor_progress)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return {
            "message": "Benchmark started",
            "benchmark_id": benchmark_id,
            "status": "running"
        }
        
    except Exception as e:
        print(f"Error starting benchmark: {str(e)}")
        # Log error
        activity_service.log_activity(
            ActivityType.BENCHMARK_FAIL,
            f"Lỗi khi khởi động benchmark: {str(e)}",
            metadata={"error": str(e), "file_path": config.file_path}
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/benchmark-progress/{benchmark_id}")
async def get_benchmark_progress(benchmark_id: str):
    try:
        if benchmark_id not in benchmark_progress:
            raise HTTPException(status_code=404, detail="Benchmark not found")
        
        progress_data = benchmark_progress[benchmark_id].copy()
        
        # Ensure all values are JSON serializable
        for key, value in progress_data.items():
            if isinstance(value, (np.floating, np.integer)):
                progress_data[key] = float(value)
        
        return progress_data
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error getting benchmark progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/benchmark-results")
async def list_benchmark_results():
    try:
        # Kiểm tra thư mục kết quả tồn tại
        if not os.path.exists(BENCHMARK_RESULTS_DIR):
            return {"results": []}
        
        # Lấy danh sách file CSV trong thư mục
        results = []
        for file in os.listdir(BENCHMARK_RESULTS_DIR):
            if file.endswith('.csv'):
                file_path = os.path.join(BENCHMARK_RESULTS_DIR, file)
                stats = {
                    "file_name": file,
                    "file_path": file_path,
                    "created_at": datetime.fromtimestamp(os.path.getctime(file_path)).strftime("%Y-%m-%d %H:%M:%S"),
                    "size_kb": round(os.path.getsize(file_path) / 1024, 2)
                }
                
                # Thử đọc file để lấy số lượng câu hỏi và thống kê
                try:
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                    if not df.empty:
                        # Đếm số câu hỏi (trừ header)
                        stats["questions_count"] = len(df)
                        
                        # Tính average scores nếu có cột SUMMARY
                        if 'STT' in df.columns:
                            summary_rows = df[df['STT'] == 'SUMMARY']
                            if not summary_rows.empty:
                                summary_row = summary_rows.iloc[0]
                                if 'current_cosine_sim' in summary_row:
                                    try:
                                        stats["avg_cosine_sim"] = float(summary_row['current_cosine_sim'])
                                    except:
                                        pass
                                if 'current_retrieval_accuracy' in summary_row:
                                    try:
                                        stats["avg_retrieval_accuracy"] = float(summary_row['current_retrieval_accuracy'])
                                    except:
                                        pass
                        else:
                            # Nếu không có SUMMARY, tính trung bình từ tất cả rows
                            if 'current_cosine_sim' in df.columns:
                                try:
                                    numeric_values = pd.to_numeric(df['current_cosine_sim'], errors='coerce')
                                    stats["avg_cosine_sim"] = float(numeric_values.mean())
                                except:
                                    pass
                except Exception as e:
                    print(f"Lỗi khi đọc file {file}: {str(e)}")
                    # Nếu không đọc được file, vẫn thêm thông tin cơ bản
                    stats["questions_count"] = "Unknown"
                
                results.append(stats)
        
        # Sắp xếp theo thời gian tạo mới nhất trước
        results.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {"results": results}
    
    except Exception as e:
        print(f"Error in list_benchmark_results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/benchmark-results/{file_name}")
async def download_benchmark_result(file_name: str):
    file_path = os.path.join(BENCHMARK_RESULTS_DIR, file_name)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file: {file_name}")
    
    return FileResponse(
        file_path, 
        media_type='text/csv;charset=utf-8',
        filename=file_name,
        headers={"Content-Disposition": f"attachment; filename={file_name}"}
    )

@router.get("/view-benchmark/{file_name}")
async def view_benchmark_content(file_name: str):
    file_path = os.path.join(BENCHMARK_RESULTS_DIR, file_name)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file: {file_name}")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # Lấy thông tin tổng quan
        total_rows = len(df)
        columns = list(df.columns)
        
        # Lọc ra dòng SUMMARY nếu có
        summary_row = None
        data_rows = df
        
        if 'STT' in df.columns:
            summary_rows = df[df['STT'] == 'SUMMARY']
            if not summary_rows.empty:
                summary_row = summary_rows.iloc[0].to_dict()
                # Lọc ra chỉ data rows (không bao gồm SUMMARY)
                data_rows = df[df['STT'] != 'SUMMARY']
        
        # Tính toán chi tiết cho từng model
        model_stats = {}
        
        # Định nghĩa các cột cho từng model
        models = {
            'current': {
                'cosine_col': 'current_cosine_sim',
                'retrieval_col': 'current_retrieval_accuracy',
                'time_col': 'current_processing_time',
                'name': 'Current System'
            },
            'langchain': {
                'cosine_col': 'langchain_cosine_sim', 
                'retrieval_col': 'langchain_retrieval_accuracy',
                'time_col': 'langchain_processing_time',
                'name': 'LangChain'
            },
            'haystack': {
                'cosine_col': 'haystack_cosine_sim',
                'retrieval_col': 'haystack_retrieval_accuracy', 
                'time_col': 'haystack_processing_time',
                'name': 'Haystack'
            },
            'chatgpt': {
                'cosine_col': 'chatgpt_cosine_sim',
                'retrieval_col': 'chatgpt_retrieval_accuracy',
                'time_col': 'chatgpt_processing_time', 
                'name': 'ChatGPT'
            }
        }
        
        for model_key, model_info in models.items():
            stats = {
                'name': model_info['name'],
                'cosine_similarity': {'avg': 0, 'min': 0, 'max': 0, 'count': 0},
                'retrieval_accuracy': {'avg': 0, 'min': 0, 'max': 0, 'count': 0},
                'processing_time': {'avg': 0, 'min': 0, 'max': 0, 'count': 0}
            }
            
            # Tính toán Cosine Similarity
            if model_info['cosine_col'] in data_rows.columns:
                cosine_values = pd.to_numeric(data_rows[model_info['cosine_col']], errors='coerce').dropna()
                if not cosine_values.empty:
                    stats['cosine_similarity'] = {
                        'avg': float(cosine_values.mean()),
                        'min': float(cosine_values.min()),
                        'max': float(cosine_values.max()),
                        'count': int(len(cosine_values))
                    }
            
            # Tính toán Retrieval Accuracy
            if model_info['retrieval_col'] in data_rows.columns:
                retrieval_values = pd.to_numeric(data_rows[model_info['retrieval_col']], errors='coerce').dropna()
                if not retrieval_values.empty:
                    stats['retrieval_accuracy'] = {
                        'avg': float(retrieval_values.mean()),
                        'min': float(retrieval_values.min()),
                        'max': float(retrieval_values.max()),
                        'count': int(len(retrieval_values))
                    }
            
            # Tính toán Processing Time
            if model_info['time_col'] in data_rows.columns:
                time_values = pd.to_numeric(data_rows[model_info['time_col']], errors='coerce').dropna()
                if not time_values.empty:
                    stats['processing_time'] = {
                        'avg': float(time_values.mean()),
                        'min': float(time_values.min()),
                        'max': float(time_values.max()),
                        'count': int(len(time_values))
                    }
            
            model_stats[model_key] = stats
        
        # Tìm model tốt nhất cho từng metric
        best_models = {
            'cosine_similarity': max(model_stats.items(), 
                                   key=lambda x: x[1]['cosine_similarity']['avg'] if x[1]['cosine_similarity']['count'] > 0 else 0),
            'retrieval_accuracy': max(model_stats.items(), 
                                    key=lambda x: x[1]['retrieval_accuracy']['avg'] if x[1]['retrieval_accuracy']['count'] > 0 else 0),
            'processing_time': min(model_stats.items(), 
                                 key=lambda x: x[1]['processing_time']['avg'] if x[1]['processing_time']['count'] > 0 else float('inf'))
        }
        
        # Lấy 5 dòng đầu làm preview
        preview_data = data_rows.head(5).to_dict('records')
        
        return {
            "file_name": file_name,
            "total_questions": int(len(data_rows)) if 'STT' in df.columns else total_rows,
            "columns": columns,
            "model_stats": model_stats,
            "best_models": {
                'cosine_similarity': {
                    'name': best_models['cosine_similarity'][1]['name'],
                    'score': best_models['cosine_similarity'][1]['cosine_similarity']['avg']
                },
                'retrieval_accuracy': {
                    'name': best_models['retrieval_accuracy'][1]['name'], 
                    'score': best_models['retrieval_accuracy'][1]['retrieval_accuracy']['avg']
                },
                'processing_time': {
                    'name': best_models['processing_time'][1]['name'],
                    'time': best_models['processing_time'][1]['processing_time']['avg']
                }
            },
            "summary_row": summary_row,
            "preview": preview_data[:3]  # Chỉ 3 dòng đầu
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể đọc file: {str(e)}")

@router.get("/download-benchmark/{filename}")
async def download_benchmark_file(filename: str):
    file_path = os.path.join(BENCHMARK_RESULTS_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    return FileResponse(
        file_path, 
        media_type='text/csv;charset=utf-8', 
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@router.get("/benchmark-files")
async def list_benchmark_files():
    try:
        files = []
        if os.path.exists(BENCHMARK_DIR):
            for filename in os.listdir(BENCHMARK_DIR):
                if filename.endswith('.json'):
                    file_path = os.path.join(BENCHMARK_DIR, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8-sig') as f:
                            data = json.load(f)
                            questions_count = len(data.get('benchmark', []))
                        
                        files.append({
                            'filename': filename,
                            'questions_count': questions_count,
                            'size': os.path.getsize(file_path),
                            'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                        })
                    except:
                        continue
        
        return {'files': files}
    except Exception as e:
        print(f"Error listing benchmark files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/invalidate-cache/{doc_id}")
async def invalidate_cache(doc_id: str):
    try:
        print(f"Admin yêu cầu vô hiệu hóa cache cho document: {doc_id}")
        count = retrieval_service.invalidate_document_cache(doc_id)
        
        return {
            "message": f"Đã vô hiệu hóa {count} cache entries liên quan đến văn bản {doc_id}",
            "affected_count": count,
            "doc_id": doc_id
        }
    except Exception as e:
        print(f"Error in invalidate_cache: {str(e)}")
        activity_service.log_activity(
            ActivityType.CACHE_INVALIDATE,
            f"Lỗi vô hiệu hóa cache cho {doc_id}: {str(e)}",
            metadata={"error": str(e), "doc_id": doc_id, "success": False}
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-document")
async def upload_document(
    metadata: DocumentUpload = Body(...),
    chunks: List[UploadFile] = File(...)
):
    try:
        # Tạo thư mục cho văn bản mới
        doc_dir = os.path.join(DATA_DIR, metadata.doc_id)
        os.makedirs(doc_dir, exist_ok=True)
        
        # Lưu các file chunk
        saved_chunks = []
        for i, chunk in enumerate(chunks):
            file_name = f"chunk_{i+1}.md"
            file_path = os.path.join(doc_dir, file_name)
            
            # Lưu file
            with open(file_path, "wb") as f:
                f.write(await chunk.read())
            
            saved_chunks.append({
                "chunk_id": f"{metadata.doc_id}_chunk_{i+1}",
                "chunk_type": "content",
                "file_path": f"/data/{metadata.doc_id}/{file_name}",
                "content_summary": f"Phần {i+1} của {metadata.doc_type} {metadata.doc_id}"
            })
        
        # Tạo metadata
        full_metadata = {
            "doc_id": metadata.doc_id,
            "doc_type": metadata.doc_type,
            "doc_title": metadata.doc_title,
            "issue_date": datetime.now().strftime("%d-%m-%Y"),
            "effective_date": metadata.effective_date,
            "expiry_date": None,
            "status": metadata.status,
            "document_scope": metadata.document_scope,
            "replaces": [],
            "replaced_by": None,
            "amends": None,
            "amended_by": None,
            "retroactive": False,
            "retroactive_date": None,
            "chunks": saved_chunks
        }
        
        # Lưu metadata
        with open(os.path.join(doc_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(full_metadata, f, ensure_ascii=False, indent=2)
        
        # Tải văn bản vào ChromaDB
        # Bổ sung sau phần này
        
        return {
            "message": f"Đã tải lên văn bản {metadata.doc_id} thành công với {len(saved_chunks)} chunks",
            "doc_id": metadata.doc_id
        }
    except Exception as e:
        print(f"Error in upload_document: {str(e)}")
        
        # Dọn dẹp nếu có lỗi
        if os.path.exists(doc_dir):
            shutil.rmtree(doc_dir)
            
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents")
async def list_documents():
    try:
        if not os.path.exists(DATA_DIR):
            return {"documents": []}
        
        documents = []
        for doc_dir in os.listdir(DATA_DIR):
            doc_path = os.path.join(DATA_DIR, doc_dir)
            metadata_path = os.path.join(doc_path, "metadata.json")
            
            if os.path.isdir(doc_path) and os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        
                    # Đếm số lượng chunk
                    chunks_count = len(metadata.get("chunks", []))
                    
                    # Thêm thông tin cơ bản
                    doc_info = {
                        "doc_id": metadata.get("doc_id", doc_dir),
                        "doc_type": metadata.get("doc_type", "Unknown"),
                        "doc_title": metadata.get("doc_title", "Unknown"),
                        "effective_date": metadata.get("effective_date", "Unknown"),
                        "status": metadata.get("status", "active"),
                        "chunks_count": chunks_count
                    }
                    
                    documents.append(doc_info)
                except Exception as e:
                    print(f"Lỗi khi đọc metadata của {doc_dir}: {str(e)}")
        
        # Sắp xếp theo ID văn bản
        documents.sort(key=lambda x: x["doc_id"])
        
        return {"documents": documents}
    except Exception as e:
        print(f"Error in list_documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    doc_dir = os.path.join(DATA_DIR, doc_id)
    metadata_path = os.path.join(doc_dir, "metadata.json")
    
    if not os.path.exists(doc_dir) or not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail=f"Không tìm thấy văn bản: {doc_id}")
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Đọc nội dung chunks
        chunks = []
        for chunk_info in metadata.get("chunks", []):
            chunk_path = os.path.join(DATA_DIR, chunk_info.get("file_path", "").replace("/data/", ""))
            if os.path.exists(chunk_path):
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    chunk_content = f.read()
                
                chunks.append({
                    "chunk_id": chunk_info.get("chunk_id"),
                    "chunk_type": chunk_info.get("chunk_type"),
                    "content_summary": chunk_info.get("content_summary"),
                    "content": chunk_content[:500] + "..." if len(chunk_content) > 500 else chunk_content
                })
        
        metadata["chunks"] = chunks
        
        return metadata
    except Exception as e:
        print(f"Error in get_document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, confirm: bool = False):
    if not confirm:
        return {"message": f"Vui lòng xác nhận việc xóa văn bản {doc_id} bằng cách gửi 'confirm: true'"}
    
    doc_dir = os.path.join(DATA_DIR, doc_id)
    
    if not os.path.exists(doc_dir):
        raise HTTPException(status_code=404, detail=f"Không tìm thấy văn bản: {doc_id}")
    
    try:
        # Vô hiệu hóa cache trước
        try:
            retrieval_service.invalidate_document_cache(doc_id)
        except Exception as e:
            print(f"Cảnh báo: Không thể vô hiệu hóa cache cho {doc_id}: {str(e)}")
        
        # Xóa thư mục văn bản
        shutil.rmtree(doc_dir)
        
        # Xóa trong ChromaDB
        # Triển khai logic để xóa văn bản trong ChromaDB
        
        return {"message": f"Đã xóa văn bản {doc_id} thành công"}
    except Exception as e:
        print(f"Error in delete_document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def get_system_statistics():
    try:
        db = mongodb_client.get_database()
        
        # Thống kê cơ bản
        total_users = db.users.count_documents({})
        total_chats = db.chats.count_documents({})
        total_exchanges = 0
        
        # Đếm tổng số exchanges
        pipeline = [
            {"$project": {"exchange_count": {"$size": {"$ifNull": ["$exchanges", []]}}}},
            {"$group": {"_id": None, "total": {"$sum": "$exchange_count"}}}
        ]
        result = list(db.chats.aggregate(pipeline))
        if result:
            total_exchanges = result[0]["total"]
        
        # Thống kê cache
        total_cache = db.text_cache.count_documents({})
        valid_cache = db.text_cache.count_documents({"validityStatus": "valid"})
        
        # Thống kê tài liệu
        document_count = 0
        chunk_count = 0
        
        if os.path.exists(DATA_DIR):
            document_count = sum(1 for item in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, item)))
            
            # Đếm số lượng chunk
            for doc_dir in os.listdir(DATA_DIR):
                doc_path = os.path.join(DATA_DIR, doc_dir)
                metadata_path = os.path.join(doc_path, "metadata.json")
                
                if os.path.isdir(doc_path) and os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            chunk_count += len(metadata.get("chunks", []))
                    except:
                        pass
        
        # Thống kê ChromaDB
        chroma_count = 0
        try:
            collection = chroma_client.get_collection()
            chroma_count = collection.count()
        except Exception as e:
            print(f"Lỗi khi lấy thông tin ChromaDB: {str(e)}")
        
        # Trả về thống kê
        return {
            "users": {
                "total": total_users
            },
            "chats": {
                "total": total_chats,
                "exchanges": total_exchanges
            },
            "cache": {
                "total": total_cache,
                "valid": valid_cache,
                "invalid": total_cache - valid_cache
            },
            "documents": {
                "total": document_count,
                "chunks": chunk_count,
                "indexed_in_chroma": chroma_count
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        print(f"Error in get_system_statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))