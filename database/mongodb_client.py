import pymongo
from pymongo import MongoClient
from typing import Optional, Dict, Any, List
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_CONFIG

class MongoDBClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBClient, cls).__new__(cls)
            cls._instance.client = None
            cls._instance.db = None
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if self._initialized:
            return
            
        try:
            self.client = MongoClient(
                DB_CONFIG.MONGO_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                maxPoolSize=50,
                retryWrites=True
            )
            
            self.db = self.client[DB_CONFIG.MONGO_DB_NAME]
            self.client.admin.command('ping')
            
            print(f"Đã kết nối thành công tới MongoDB: {DB_CONFIG.MONGO_DB_NAME}")
            
            self._initialized = True
            
        except Exception as e:
            print(f"Lỗi kết nối MongoDB: {str(e)}")
            self.client = None
            self.db = None
            self._initialized = False
            raise e

    def get_database(self):
        if not self._initialized:
            self.initialize()
        return self.db

    def get_collection(self, collection_name: str):
        db = self.get_database()
        if db is None:
            raise Exception("Không thể kết nối tới database")
        return db[collection_name]

    def create_indexes(self):
        if self.db is None:
            self.initialize()

        print("Bắt đầu tạo indexes cho MongoDB...")
        
        try:
            # USERS COLLECTION INDEXES
            users_collection = self.get_collection("users")
            
            users_collection.create_index([("username", 1)], unique=True, background=True)
            
            users_collection.create_index([("email", 1)], unique=True, background=True)
            
            users_collection.create_index([("status", 1)], background=True)
            users_collection.create_index([("role", 1)], background=True)
            
            users_collection.create_index([("last_login_at", -1)], background=True)

        except Exception as e:
            print(f"  Lỗi tạo indexes cho users: {str(e)}")

        try:
            # CONVERSATIONS COLLECTION INDEXES
            conversations_collection = self.get_collection("chats")
            
            conversations_collection.create_index([("user_id", 1)], background=True)
            
            conversations_collection.create_index([("user_id", 1), ("updated_at", -1)], background=True)
            
            conversations_collection.create_index([("status", 1)], background=True)
            conversations_collection.create_index([("title", "text")], background=True)
            
            conversations_collection.create_index([("created_at", -1)], background=True)

        except Exception as e:
            print(f"  Lỗi tạo indexes cho conversations: {str(e)}")

        try:
            # CACHE COLLECTION INDEXES
            cache_collection = self.get_collection("text_cache")
            
            cache_collection.create_index([("cache_id", 1)], unique=True, background=True)
            
            # Tạo text index mới với tên duy nhất
            try:
                cache_collection.create_index([
                    ("normalized_question", "text"), 
                    ("question_text", "text")
                ], background=True, name="unified_cache_text_search")
            except Exception as text_err:
                print(f"  Text index lỗi: {str(text_err)}")
            
            cache_collection.create_index([("keywords", 1)], background=True)
            cache_collection.create_index([("related_doc_ids", 1)], background=True)
            cache_collection.create_index([("validity_status", 1)], background=True)
            
            cache_collection.create_index([("expires_at", 1)], expireAfterSeconds=0, background=True)
            
            cache_collection.create_index([("cache_type", 1)], background=True)
            cache_collection.create_index([("metrics.hit_count", -1)], background=True)

        except Exception as e:
            print(f"  Lỗi tạo indexes cho cache: {str(e)}")

        try:
            # FEEDBACK COLLECTION INDEXES
            feedback_collection = self.get_collection("feedback")
            
            feedback_collection.create_index([("user_id", 1)], background=True)
            feedback_collection.create_index([("chat_id", 1)], background=True)
            feedback_collection.create_index([("timestamp", -1)], background=True)
            feedback_collection.create_index([("rating", 1)], background=True)

        except Exception as e:
            print(f"  Lỗi tạo indexes cho feedback: {str(e)}")

        try:
            # ACTIVITY LOGS COLLECTION INDEXES
            activity_logs_collection = self.get_collection("activity_logs")
            
            activity_logs_collection.create_index([("activity_type", 1)], background=True)
            activity_logs_collection.create_index([("user_id", 1)], background=True)
            activity_logs_collection.create_index([("timestamp", -1)], background=True)
            
            activity_logs_collection.create_index([("created_at", 1)], expireAfterSeconds=2592000, background=True)

        except Exception as e:
            print(f"  Lỗi tạo indexes cho activity_logs: {str(e)}")

        print("Hoàn thành tạo tất cả indexes cho MongoDB!")

    def get_collection_stats(self) -> Dict[str, Any]:
        if self.db is None:
            return {"error": "Database not connected"}

        stats = {}
        
        try:
            collection_names = self.db.list_collection_names()
            
            for collection_name in collection_names:
                try:
                    collection = self.db[collection_name]
                    doc_count = collection.count_documents({})
                    collection_stats = self.db.command("collStats", collection_name)
                    
                    stats[collection_name] = {
                        "document_count": doc_count,
                        "size_bytes": collection_stats.get("size", 0),
                        "storage_size_bytes": collection_stats.get("storageSize", 0),
                        "index_count": collection_stats.get("nindexes", 0),
                        "average_doc_size": collection_stats.get("avgObjSize", 0),
                    }
                    
                except Exception as e:
                    stats[collection_name] = {"error": str(e)}
                    
        except Exception as e:
            return {"error": f"Failed to get collection stats: {str(e)}"}
            
        return stats

    def health_check(self) -> Dict[str, Any]:
        health_info = {
            "status": "disconnected",
            "database": DB_CONFIG.MONGO_DB_NAME,
            "collections": [],
            "total_documents": 0,
            "connection_time": None
        }
        
        try:
            start_time = datetime.now()
            self.client.admin.command('ping')
            connection_time = (datetime.now() - start_time).total_seconds()
            
            collections = self.db.list_collection_names()
            total_docs = 0
            
            for collection_name in collections:
                try:
                    count = self.db[collection_name].count_documents({})
                    total_docs += count
                except:
                    pass
            
            health_info.update({
                "status": "connected",
                "collections": collections,
                "total_documents": total_docs,
                "connection_time": connection_time,
                "server_info": self.client.server_info()["version"]
            })
            
        except Exception as e:
            health_info["error"] = str(e)
            
        return health_info

    def close(self):
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self._initialized = False
            print("Đã đóng kết nối MongoDB")

    # HELPER METHODS
    def save_user(self, user_data: Dict[str, Any]) -> str:
        user_data["created_at"] = datetime.now()
        user_data["updated_at"] = datetime.now()
        
        users_collection = self.get_collection("users")
        result = users_collection.insert_one(user_data)
        return str(result.inserted_id)

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        from bson.objectid import ObjectId
        
        try:
            users_collection = self.get_collection("users")
            return users_collection.find_one({"_id": ObjectId(user_id)})
        except:
            return None

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        users_collection = self.get_collection("users")
        return users_collection.find_one({"username": username})

    def save_chat_message(self, user_id: str, query: str, answer: str, 
                         context_items: List[str] = None, retrieved_chunks: List[str] = None, 
                         performance_metrics: Dict[str, Any] = None) -> str:
        chat_data = {
            "user_id": user_id,
            "query": query,
            "answer": answer,
            "context_items": context_items or [],
            "retrieved_chunks": retrieved_chunks or [],
            "performance": performance_metrics or {},
            "timestamp": datetime.now()
        }
        
        chat_history_collection = self.get_collection("chat_history")
        result = chat_history_collection.insert_one(chat_data)
        return str(result.inserted_id)

    def get_user_chat_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        chat_history_collection = self.get_collection("chat_history")
        return list(chat_history_collection.find(
            {"user_id": user_id},
            {"query": 1, "answer": 1, "timestamp": 1}
        ).sort("timestamp", -1).limit(limit))

    def save_user_feedback(self, chat_id: str, feedback_data: Dict[str, Any]) -> str:
        feedback_data["chat_id"] = chat_id
        feedback_data["timestamp"] = datetime.now()
        
        feedback_collection = self.get_collection("user_feedback")
        result = feedback_collection.insert_one(feedback_data)
        return str(result.inserted_id)

# Khởi tạo singleton instance
mongodb_client = MongoDBClient()

def get_mongodb_client():
    return mongodb_client

def get_database():
    return mongodb_client.get_database()