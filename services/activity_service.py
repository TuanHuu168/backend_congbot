from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.mongodb_client import mongodb_client

class ActivityType(str, Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    CACHE_CLEAR = "cache_clear"
    CACHE_INVALIDATE = "cache_invalidate"
    BENCHMARK_START = "benchmark_start"
    BENCHMARK_COMPLETE = "benchmark_complete"
    BENCHMARK_FAIL = "benchmark_fail"
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_DELETE = "document_delete"
    SYSTEM_STATUS = "system_status"

class ActivityService:
    def __init__(self):
        self.db = mongodb_client.get_database()
        self.activities_collection = self.db.activity_logs
        
        # Ensure indexes
        try:
            self.activities_collection.create_index([("timestamp", -1)])
            self.activities_collection.create_index([("activity_type", 1)])
            self.activities_collection.create_index([("user_id", 1)])
        except Exception as e:
            print(f"Warning: Could not create indexes for activity_logs: {e}")
    
    def log_activity(
        self, 
        activity_type: ActivityType, 
        description: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log an activity to the database"""
        try:
            activity = {
                "activity_type": activity_type.value,
                "description": description,
                "user_id": user_id,
                "user_email": user_email,
                "metadata": metadata or {},
                "timestamp": datetime.now(),
                "created_at": datetime.now()
            }
            
            result = self.activities_collection.insert_one(activity)
            print(f"Activity logged: {activity_type.value} - {description}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error logging activity: {e}")
            return ""
    
    def get_recent_activities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent activities from the database"""
        try:
            activities = list(
                self.activities_collection
                .find({})
                .sort("timestamp", -1)
                .limit(limit)
            )
            
            # Convert ObjectId to string
            for activity in activities:
                activity["_id"] = str(activity["_id"])
                if "timestamp" in activity:
                    activity["timestamp"] = activity["timestamp"].isoformat()
                if "created_at" in activity:
                    activity["created_at"] = activity["created_at"].isoformat()
            
            return activities
        except Exception as e:
            print(f"Error getting recent activities: {e}")
            return []
    
    def get_activities_by_type(self, activity_type: ActivityType, limit: int = 5) -> List[Dict[str, Any]]:
        """Get activities filtered by type"""
        try:
            activities = list(
                self.activities_collection
                .find({"activity_type": activity_type.value})
                .sort("timestamp", -1)
                .limit(limit)
            )
            
            for activity in activities:
                activity["_id"] = str(activity["_id"])
                if "timestamp" in activity:
                    activity["timestamp"] = activity["timestamp"].isoformat()
            
            return activities
        except Exception as e:
            print(f"Error getting activities by type: {e}")
            return []
    
    def cleanup_old_activities(self, days_to_keep: int = 30) -> int:
        """Clean up old activities older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            result = self.activities_collection.delete_many(
                {"timestamp": {"$lt": cutoff_date}}
            )
            return result.deleted_count
        except Exception as e:
            print(f"Error cleaning up old activities: {e}")
            return 0

# Singleton instance
activity_service = ActivityService()