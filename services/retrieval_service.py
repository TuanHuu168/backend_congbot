import time
import json
import sys
import os
import re
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOP_K, MAX_TOKENS_PER_DOC, EMBEDDING_MODEL_NAME
from database.chroma_client import chroma_client
from database.mongodb_client import mongodb_client
from models.cache import CacheModel, CacheCreate, CacheStatus
from services.activity_service import activity_service, ActivityType

class RetrievalService:
    def __init__(self):
        """Khởi tạo dịch vụ retrieval"""
        self.chroma = chroma_client
        self.db = mongodb_client.get_database()
        self.text_cache_collection = self.db.text_cache
        
        # Log thông tin khởi tạo
        print(f"RetrievalService khởi tạo với embedding model: {EMBEDDING_MODEL_NAME}")
        
        # Kiểm tra cache collections
        try:
            cache_count = self.text_cache_collection.count_documents({})
            print(f"MongoDB cache collection: {cache_count} documents")
            
            cache_collection = self.chroma.get_cache_collection()
            if cache_collection:
                chroma_count = cache_collection.count()
                print(f"ChromaDB cache collection: {chroma_count} documents")
        except Exception as e:
            print(f"Lỗi kiểm tra cache: {str(e)}")
    
    def _normalize_question(self, query: str) -> str:
        """Chuẩn hóa câu hỏi để so sánh"""
        normalized = re.sub(r'[.,;:!?()"\']', '', query.lower())
        return re.sub(r'\s+', ' ', normalized).strip()
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Trích xuất từ khóa từ câu hỏi"""
        normalized = self._normalize_question(query)
        keywords = [word for word in normalized.split() if len(word) >= 3]
        return list(set(keywords))
    
    def _format_context(self, results: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Format kết quả retrieval thành context và chunks"""
        context_items = []
        retrieved_chunks = []
        
        if results.get("documents") and len(results["documents"]) > 0:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            
            for doc, meta in zip(documents, metadatas):
                # Loại bỏ prefix "passage: "
                if doc.startswith("passage: "):
                    doc = doc[9:]
                
                # Tạo thông tin nguồn
                source_info = f"(Nguồn: {meta.get('doc_type', '')} {meta.get('doc_id', '')}"
                if meta.get('effective_date'):
                    source_info += f", có hiệu lực từ {meta['effective_date']}"
                source_info += ")"
                
                # Lưu chunk_id và context
                if 'chunk_id' in meta:
                    retrieved_chunks.append(meta['chunk_id'])
                
                context_items.append(f"{doc} {source_info}")
        
        return context_items, retrieved_chunks
    
    def _check_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """Kiểm tra cache cho câu hỏi"""
        print(f"Kiểm tra cache cho: '{query}'")
        
        normalized_query = self._normalize_question(query)
        
        try:
            # 1. Exact match trong MongoDB
            exact_match = self.text_cache_collection.find_one({
                "$or": [
                    {"normalizedQuestion": normalized_query},
                    {"questionText": query}
                ],
                "validityStatus": CacheStatus.VALID
            })
            
            if exact_match:
                print("Tìm thấy exact match trong MongoDB cache")
                self._update_cache_usage(exact_match["_id"])
                return dict(exact_match)
            
            # 2. Semantic search trong ChromaDB
            print("Không có exact match, thực hiện semantic search...")
            cache_collection = self.chroma.get_cache_collection()
            
            if not cache_collection:
                print("Không thể lấy cache collection")
                return None
            
            query_text = f"query: {query}"
            cache_results = cache_collection.query(
                query_texts=[query_text],
                n_results=1,
                include=["documents", "metadatas", "distances"]
            )
            
            if (cache_results["ids"] and len(cache_results["ids"][0]) > 0 and
                cache_results["distances"] and len(cache_results["distances"][0]) > 0):
                
                cache_id = cache_results["ids"][0][0]
                distance = cache_results["distances"][0][0]
                similarity_score = 1.0 - min(distance, 1.0)
                
                print(f"ChromaDB semantic match: score={similarity_score:.4f}")
                
                # Ngưỡng tương đồng
                if similarity_score >= 0.85:
                    metadata = cache_results["metadatas"][0][0] if cache_results["metadatas"] else {}
                    if metadata.get("validityStatus") != "invalid":
                        cache_result = self.text_cache_collection.find_one({
                            "cacheId": cache_id,
                            "validityStatus": CacheStatus.VALID
                        })
                        
                        if cache_result:
                            print(f"Tìm thấy semantic match với score {similarity_score:.4f}")
                            self._update_cache_usage(cache_result["_id"])
                            return dict(cache_result)
            
            print("Không tìm thấy cache phù hợp")
            return None
            
        except Exception as e:
            print(f"Lỗi kiểm tra cache: {str(e)}")
            return None
    
    def _update_cache_usage(self, cache_id):
        """Cập nhật thống kê sử dụng cache"""
        try:
            self.text_cache_collection.update_one(
                {"_id": cache_id},
                {
                    "$inc": {"hitCount": 1},
                    "$set": {"lastUsed": datetime.now()}
                }
            )
        except Exception as e:
            print(f"Lỗi cập nhật cache usage: {str(e)}")
    
    def _extract_document_ids(self, chunk_ids: List[str]) -> List[str]:
        """Trích xuất document IDs từ chunk IDs"""
        doc_ids = []
        for chunk_id in chunk_ids:
            parts = chunk_id.split('_', 1)
            if len(parts) > 0:
                doc_id_parts = []
                for part in parts[:-1]:
                    doc_id_parts.append(part)
                doc_id = "_".join(doc_id_parts)
                if doc_id not in doc_ids:
                    doc_ids.append(doc_id)
        return doc_ids
    
    def _create_cache_entry(self, query: str, answer: str, chunks: List[str], relevance_scores: Dict[str, float]) -> str:
        """Tạo entry cache mới"""
        cache_id = f"cache_{int(time.time() * 1000)}"
        print(f"Tạo cache entry: {cache_id}")
        
        # Chuẩn bị dữ liệu
        relevant_documents = [
            {"chunkId": chunk_id, "score": relevance_scores.get(chunk_id, 0.5)}
            for chunk_id in chunks
        ]
        doc_ids = self._extract_document_ids(chunks)
        keywords = self._extract_keywords(query)
        
        cache_data = {
            "cacheId": cache_id,
            "questionText": query,
            "normalizedQuestion": self._normalize_question(query),
            "answer": answer,
            "relevantDocuments": relevant_documents,
            "validityStatus": CacheStatus.VALID,
            "relatedDocIds": doc_ids,
            "keywords": keywords,
            "hitCount": 0,
            "createdAt": datetime.now(),
            "updatedAt": datetime.now(),
            "lastUsed": datetime.now(),
            "expiresAt": None
        }
        
        try:
            # Lưu vào MongoDB
            self.text_cache_collection.insert_one(cache_data)
            print("Đã lưu cache vào MongoDB")
            
            # Lưu vào ChromaDB
            self._add_to_chroma_cache(cache_id, query, doc_ids)
            return cache_id
            
        except Exception as e:
            print(f"Lỗi tạo cache: {str(e)}")
            return ""
    
    def _add_to_chroma_cache(self, cache_id: str, query: str, related_doc_ids: List[str]) -> bool:
        """Thêm cache vào ChromaDB"""
        query_text = f"query: {query}"
        metadata = {
            "validityStatus": str(CacheStatus.VALID),
            "relatedDocIds": ",".join(related_doc_ids) if related_doc_ids else ""
        }
        
        try:
            success = self.chroma.add_documents_to_cache(
                ids=[cache_id],
                documents=[query_text],
                metadatas=[metadata]
            )
            
            if success:
                print("Đã thêm cache vào ChromaDB")
                return True
            else:
                print("Lỗi thêm cache vào ChromaDB")
                return False
                
        except Exception as e:
            print(f"Lỗi ChromaDB cache: {str(e)}")
            return False
    
    def retrieve(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """Thực hiện retrieval chính"""
        start_time = time.time()
        print(f"Bắt đầu retrieval với embedding model: {EMBEDDING_MODEL_NAME}")
        
        # Kiểm tra cache
        if use_cache:
            cache_result = self._check_cache(query)
            if cache_result:
                print("Trả về kết quả từ cache")
                retrieved_chunks = [
                    doc["chunkId"] for doc in cache_result.get("relevantDocuments", [])
                    if "chunkId" in doc
                ]
                
                return {
                    "answer": cache_result["answer"],
                    "context_items": [],
                    "retrieved_chunks": retrieved_chunks,
                    "source": "cache",
                    "cache_id": cache_result["cacheId"],
                    "execution_time": time.time() - start_time
                }
        
        # Retrieval từ ChromaDB
        print("Thực hiện retrieval từ ChromaDB")
        try:
            query_text = f"query: {query}"
            results = self.chroma.search_main(
                query_text=query_text,
                n_results=TOP_K,
                include=["documents", "metadatas", "distances"]
            )
            
            context_items, retrieved_chunks = self._format_context(results)
            
            # Tính relevance scores
            relevance_scores = {}
            if results.get("distances") and len(results["distances"]) > 0:
                distances = results["distances"][0]
                metadatas = results["metadatas"][0]
                
                for distance, meta in zip(distances, metadatas):
                    if 'chunk_id' in meta:
                        relevance_scores[meta['chunk_id']] = 1.0 - min(distance, 1.0)
            
            print(f"Retrieval hoàn tất: {len(context_items)} contexts, {len(retrieved_chunks)} chunks")
            
            return {
                "context_items": context_items,
                "retrieved_chunks": retrieved_chunks,
                "source": "chroma",
                "relevance_scores": relevance_scores,
                "execution_time": time.time() - start_time
            }
            
        except Exception as e:
            print(f"Lỗi retrieval: {str(e)}")
            return {
                "context_items": [],
                "retrieved_chunks": [],
                "source": "error",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def add_to_cache(self, query: str, answer: str, chunks: List[str], relevance_scores: Dict[str, float]) -> str:
        """Thêm kết quả vào cache"""
        print(f"Thêm kết quả vào cache cho: '{query}'")
        cache_id = self._create_cache_entry(query, answer, chunks, relevance_scores)
        if cache_id:
            print(f"Cache ID: {cache_id}")
        return cache_id
    
    def invalidate_document_cache(self, doc_id: str) -> int:
        """Vô hiệu hóa cache liên quan đến document"""
        print(f"Vô hiệu hóa cache cho document: {doc_id}")
        
        # Vô hiệu hóa trong MongoDB
        result = self.text_cache_collection.update_many(
            {"relatedDocIds": doc_id},
            {"$set": {"validityStatus": CacheStatus.INVALID}}
        )
        
        # Vô hiệu hóa trong ChromaDB
        try:
            affected_caches = list(self.text_cache_collection.find(
                {"relatedDocIds": doc_id},
                {"cacheId": 1}
            ))
            
            cache_ids = [cache["cacheId"] for cache in affected_caches]
            
            if cache_ids:
                cache_collection = self.chroma.get_cache_collection()
                if cache_collection:
                    for cache_id in cache_ids:
                        try:
                            cache_collection.update(
                                ids=[cache_id],
                                metadatas=[{"validityStatus": CacheStatus.INVALID}]
                            )
                        except Exception as e:
                            print(f"Lỗi update ChromaDB cache {cache_id}: {str(e)}")
        
        except Exception as e:
            print(f"Lỗi vô hiệu hóa ChromaDB cache: {str(e)}")
        
        # Log activity
        activity_service.log_activity(
            ActivityType.CACHE_INVALIDATE,
            f"Vô hiệu hóa cache cho document {doc_id}: {result.modified_count} entries",
            metadata={
                "doc_id": doc_id,
                "affected_count": result.modified_count,
                "action": "invalidate_document_cache"
            }
        )
        
        print(f"Đã vô hiệu hóa {result.modified_count} cache entries")
        return result.modified_count
    
    def clear_all_cache(self) -> int:
        """Xóa toàn bộ cache"""
        try:
            print("Đang xóa toàn bộ cache...")
            
            # Xóa MongoDB cache
            total_before = self.text_cache_collection.count_documents({})
            result = self.text_cache_collection.delete_many({})
            deleted_count = result.deleted_count
            print(f"Đã xóa {deleted_count} entries trong MongoDB")
            
            # Xóa ChromaDB cache
            try:
                cache_collection = self.chroma.get_cache_collection()
                if cache_collection:
                    chroma_count_before = cache_collection.count()
                    all_ids = cache_collection.get(include=[])["ids"]
                    if all_ids:
                        cache_collection.delete(ids=all_ids)
                        print(f"Đã xóa {len(all_ids)} cache từ ChromaDB")
            except Exception as ce:
                print(f"Lỗi xóa ChromaDB cache: {str(ce)}")
            
            # Log activity
            activity_service.log_activity(
                ActivityType.CACHE_CLEAR,
                f"Đã xóa toàn bộ cache: {deleted_count} entries",
                metadata={
                    "mongodb_deleted": deleted_count,
                    "mongodb_before": total_before
                }
            )
            
            return deleted_count
            
        except Exception as e:
            print(f"Lỗi xóa cache: {str(e)}")
            activity_service.log_activity(
                ActivityType.CACHE_CLEAR,
                f"Lỗi xóa cache: {str(e)}",
                metadata={"error": str(e), "success": False}
            )
            raise e
    
    def clear_all_invalid_cache(self) -> int:
        """Xóa cache không hợp lệ"""
        try:
            print("Đang xóa cache không hợp lệ...")
            
            # Lấy danh sách cache IDs không hợp lệ
            invalid_caches = list(self.text_cache_collection.find(
                {"validityStatus": "invalid"}, 
                {"cacheId": 1}
            ))
            invalid_cache_ids = [cache["cacheId"] for cache in invalid_caches if "cacheId" in cache]
            
            print(f"Tìm thấy {len(invalid_cache_ids)} cache không hợp lệ")
            
            # Xóa trong MongoDB
            result = self.text_cache_collection.delete_many({"validityStatus": "invalid"})
            deleted_count = result.deleted_count
            print(f"Đã xóa {deleted_count} cache không hợp lệ trong MongoDB")
            
            # Xóa trong ChromaDB
            if invalid_cache_ids:
                try:
                    cache_collection = self.chroma.get_cache_collection()
                    if cache_collection:
                        cache_collection.delete(ids=invalid_cache_ids)
                        print(f"Đã xóa {len(invalid_cache_ids)} cache không hợp lệ trong ChromaDB")
                except Exception as ce:
                    print(f"Lỗi xóa ChromaDB invalid cache: {str(ce)}")
            
            # Log activity
            activity_service.log_activity(
                ActivityType.CACHE_CLEAR,
                f"Đã xóa cache không hợp lệ: {deleted_count} entries",
                metadata={
                    "deleted_count": deleted_count,
                    "invalid_cache_ids": invalid_cache_ids,
                    "action": "clear_invalid_cache"
                }
            )
            
            return deleted_count
            
        except Exception as e:
            print(f"Lỗi xóa cache không hợp lệ: {str(e)}")
            activity_service.log_activity(
                ActivityType.CACHE_CLEAR,
                f"Lỗi xóa cache không hợp lệ: {str(e)}",
                metadata={"error": str(e), "success": False}
            )
            raise e
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Lấy thống kê cache"""
        try:
            total_count = self.text_cache_collection.count_documents({})
            valid_count = self.text_cache_collection.count_documents({"validityStatus": CacheStatus.VALID})
            invalid_count = self.text_cache_collection.count_documents({"validityStatus": CacheStatus.INVALID})
            
            # Tính hit rate
            hits_sum = 0
            try:
                pipeline = [{"$group": {"_id": None, "totalHits": {"$sum": "$hitCount"}}}]
                result = list(self.text_cache_collection.aggregate(pipeline))
                if result:
                    hits_sum = result[0]["totalHits"]
            except Exception as e:
                print(f"Lỗi tính hitCount: {str(e)}")
            
            hit_rate = 0
            if total_count > 0 and hits_sum > 0:
                hit_rate = hits_sum / (hits_sum + total_count)
            
            return {
                "total_count": total_count,
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "hit_rate": hit_rate
            }
            
        except Exception as e:
            print(f"Lỗi lấy cache stats: {str(e)}")
            return {
                "total_count": 0,
                "valid_count": 0,
                "invalid_count": 0,
                "hit_rate": 0,
                "error": str(e)
            }
            
    def delete_expired_cache(self) -> int:
        """Xóa tất cả cache đã hết hạn"""
        try:
            print("Đang xóa cache đã hết hạn...")
            now = datetime.now()
            
            # Tìm cache đã hết hạn
            expired_caches = list(self.text_cache_collection.find(
                {"expiresAt": {"$lt": now}},
                {"cacheId": 1}
            ))
            expired_cache_ids = [cache["cacheId"] for cache in expired_caches if "cacheId" in cache]
            
            print(f"Tìm thấy {len(expired_cache_ids)} cache đã hết hạn")
            
            # Xóa trong MongoDB
            result = self.text_cache_collection.delete_many({"expiresAt": {"$lt": now}})
            deleted_count = result.deleted_count
            
            # Xóa trong ChromaDB
            if expired_cache_ids:
                try:
                    cache_collection = self.chroma.get_cache_collection()
                    if cache_collection:
                        cache_collection.delete(ids=expired_cache_ids)
                        print(f"Đã xóa {len(expired_cache_ids)} expired cache trong ChromaDB")
                except Exception as e:
                    print(f"Lỗi xóa expired cache trong ChromaDB: {str(e)}")
            
            print(f"Đã xóa {deleted_count} cache entries đã hết hạn")
            return deleted_count
            
        except Exception as e:
            print(f"Lỗi xóa expired cache: {str(e)}")
            raise e

    def search_keyword(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Tìm kiếm cache bằng từ khóa"""
        try:
            print(f"Tìm kiếm cache với từ khóa: '{keyword}'")
            
            # Tìm kiếm bằng text search và keyword matching
            results = []
            
            # Method 1: Text search nếu có text index
            try:
                text_results = list(self.text_cache_collection.find(
                    {"$text": {"$search": keyword}, "validityStatus": CacheStatus.VALID},
                    {"score": {"$meta": "textScore"}}
                ).sort([("score", {"$meta": "textScore"})]).limit(limit))
                results.extend(text_results)
            except Exception as e:
                print(f"Text search không khả dụng: {str(e)}")
            
            # Method 2: Regex search nếu text search không hoạt động
            if not results:
                regex_pattern = {"$regex": keyword, "$options": "i"}
                regex_results = list(self.text_cache_collection.find({
                    "$or": [
                        {"questionText": regex_pattern},
                        {"normalizedQuestion": regex_pattern},
                        {"keywords": {"$in": [keyword.lower()]}}
                    ],
                    "validityStatus": CacheStatus.VALID
                }).limit(limit))
                results.extend(regex_results)
            
            # Method 3: Keyword array search
            if not results:
                keyword_results = list(self.text_cache_collection.find({
                    "keywords": {"$in": [keyword.lower()]},
                    "validityStatus": CacheStatus.VALID
                }).limit(limit))
                results.extend(keyword_results)
            
            print(f"Tìm thấy {len(results)} cache entries cho từ khóa '{keyword}'")
            return results
            
        except Exception as e:
            print(f"Lỗi tìm kiếm cache: {str(e)}")
            return []

# Singleton instance
retrieval_service = RetrievalService()