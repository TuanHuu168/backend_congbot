import chromadb
import torch
import sys
import os
import json
from typing import List, Dict, Any
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION, EMBEDDING_MODEL_NAME, USE_GPU, DATA_DIR
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class ChromaDBClient:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChromaDBClient, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.client = None
            self.main_collection = None
            self.cache_collection = None
            self.embedding_function = None
            self.persist_directory = None
            self.main_collection_name = CHROMA_COLLECTION
            self.cache_collection_name = "cache_questions"
            self.loaded_documents = 0
            self.total_chunks = 0
            self._initialized = True
            self.initialize()

    def initialize(self):
        """Khởi tạo ChromaDB client và cả 2 collections"""
        if self.client is not None:
            return
            
        try:
            os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
            self.persist_directory = CHROMA_PERSIST_DIRECTORY
            print(f"ChromaDB persist directory: {self.persist_directory}")

            if not EMBEDDING_MODEL_NAME:
                raise ValueError("EMBEDDING_MODEL_NAME is None or empty! Check your .env file")

            device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
            
            # Test model trước khi tạo embedding function
            try:
                from sentence_transformers import SentenceTransformer
                test_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                test_embedding = test_model.encode(["test"])
                print(f"Model {EMBEDDING_MODEL_NAME} loaded successfully, dimension: {test_embedding.shape[1]}")
            except Exception as e:
                print(f"Error loading model {EMBEDDING_MODEL_NAME}: {str(e)}")
                raise e
            
            self.embedding_function = SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL_NAME,
                device=device
            )
            print(f"Embedding function initialized on device: {device} using model: {EMBEDDING_MODEL_NAME}")

            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            self._create_all_collections()
                
        except Exception as e:
            print(f"Lỗi kết nối ChromaDB: {str(e)}")
            import traceback
            traceback.print_exc()
            self.client = None
            self.main_collection = None
            self.cache_collection = None
    
    def _create_all_collections(self):
        """Tạo tất cả collections cần thiết"""
        if not self.client:
            return
            
        try:
            # Main collection
            try:
                self.main_collection = self.client.get_collection(name=self.main_collection_name)
            except Exception:
                self.main_collection = self.client.create_collection(
                    name=self.main_collection_name,
                    embedding_function=self.embedding_function
                )
            
            main_count = self.main_collection.count()
            print(f"Main collection '{self.main_collection_name}' ready with {main_count} documents")
            
            # Cache collection  
            try:
                self.cache_collection = self.client.get_collection(name=self.cache_collection_name)
            except Exception:
                self.cache_collection = self.client.create_collection(
                    name=self.cache_collection_name,
                    embedding_function=self.embedding_function
                )
            
            cache_count = self.cache_collection.count()
            print(f"Cache collection '{self.cache_collection_name}' ready with {cache_count} documents")
                
        except Exception as e:
            print(f"Lỗi khi tạo collections: {str(e)}")
            import traceback
            traceback.print_exc()
            self.main_collection = None
            self.cache_collection = None

    # ==================== BASIC OPERATIONS ====================
    
    def get_client(self):
        """Lấy ChromaDB client"""
        if not self.client:
            self.initialize()
        return self.client
    
    def get_collection(self, collection_type="main"):
        """Lấy collection theo loại"""
        if not self.main_collection or not self.cache_collection:
            self.initialize()
        
        if collection_type == "main":
            return self.main_collection
        elif collection_type == "cache":
            return self.cache_collection
        else:
            return self.main_collection
    
    def get_main_collection(self):
        """Lấy main collection cho văn bản pháp luật"""
        return self.get_collection("main")
    
    def get_cache_collection(self):
        """Lấy cache collection cho cache câu hỏi"""
        return self.get_collection("cache")
    
    def add_documents_to_main(self, ids, documents, metadatas=None):
        """Thêm documents vào main collection"""
        collection = self.get_main_collection()
        if not collection:
            print("Không thể lấy main collection")
            return False
        
        try:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            return True
        except Exception as e:
            print(f"Lỗi khi thêm documents vào main collection: {str(e)}")
            return False
    
    def add_documents_to_cache(self, ids, documents, metadatas=None):
        """Thêm documents vào cache collection"""
        collection = self.get_cache_collection()
        if not collection:
            print("Không thể lấy cache collection")
            return False
        
        try:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            return True
        except Exception as e:
            print(f"Lỗi khi thêm documents vào cache collection: {str(e)}")
            return False
    
    def search_main(self, query_text, n_results=5, include=None):
        """Tìm kiếm trong main collection"""
        collection = self.get_main_collection()
        if not collection:
            print("Không thể lấy main collection để tìm kiếm")
            return None
        
        try:
            if not query_text.startswith("query:"):
                query_text = f"query: {query_text}"
                
            if include is None:
                include = ["documents", "metadatas", "distances"]
                
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=include
            )
            return results
        except Exception as e:
            print(f"Lỗi khi tìm kiếm trong main collection: {str(e)}")
            return None
    
    def search_cache(self, query_text, n_results=5, include=None):
        """Tìm kiếm trong cache collection"""
        collection = self.get_cache_collection()
        if not collection:
            print("Không thể lấy cache collection để tìm kiếm")
            return None
        
        try:
            if not query_text.startswith("query:"):
                query_text = f"query: {query_text}"
                
            if include is None:
                include = ["documents", "metadatas", "distances"]
                
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=include
            )
            return results
        except Exception as e:
            print(f"Lỗi khi tìm kiếm trong cache collection: {str(e)}")
            return None
    
    def delete_collection(self, name=None):
        """Xóa collection"""
        client = self.get_client()
        if not client:
            return False
        
        try:
            if name is None:
                name = self.main_collection_name
                
            client.delete_collection(name=name)
            if name == self.main_collection_name:
                self.main_collection = None
            elif name == self.cache_collection_name:
                self.cache_collection = None
            print(f"Đã xóa collection '{name}'")
            return True
        except Exception as e:
            print(f"Lỗi khi xóa collection {name}: {str(e)}")
            return False
    
    def clear_cache_collection(self):
        """Xóa toàn bộ cache collection và tạo lại"""
        try:
            self.delete_collection(self.cache_collection_name)
            self.cache_collection = self.client.create_collection(
                name=self.cache_collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Đã clear cache collection '{self.cache_collection_name}'")
            return True
        except Exception as e:
            print(f"Lỗi khi clear cache collection: {str(e)}")
            return False
    
    def clear_main_collection(self):
        """Xóa toàn bộ main collection và tạo lại"""
        try:
            self.delete_collection(self.main_collection_name)
            self.main_collection = self.client.create_collection(
                name=self.main_collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Đã clear main collection '{self.main_collection_name}'")
            self.loaded_documents = 0
            self.total_chunks = 0
            return True
        except Exception as e:
            print(f"Lỗi khi clear main collection: {str(e)}")
            return False
    
    def list_collections(self):
        """Liệt kê tất cả collections"""
        client = self.get_client()
        if not client:
            return []
            
        try:
            collection_names = client.list_collections()
            collections_info = []
            
            for name in collection_names:
                try:
                    col = client.get_collection(name)
                    collections_info.append({
                        "name": name, 
                        "count": col.count()
                    })
                except Exception as e:
                    print(f"Lỗi khi lấy thông tin collection {name}: {str(e)}")
                    collections_info.append({
                        "name": name, 
                        "count": "unknown"
                    })
            
            return collections_info
        except Exception as e:
            print(f"Lỗi khi liệt kê collections: {str(e)}")
            return []
    
    def get_collection_stats(self):
        """Lấy thống kê về tất cả collections"""
        stats = {
            "main_collection": {
                "name": self.main_collection_name,
                "count": 0,
                "status": "disconnected"
            },
            "cache_collection": {
                "name": self.cache_collection_name,
                "count": 0,
                "status": "disconnected"
            }
        }
        
        try:
            if self.main_collection:
                stats["main_collection"]["count"] = self.main_collection.count()
                stats["main_collection"]["status"] = "connected"
        except Exception as e:
            print(f"Lỗi khi lấy stats main collection: {str(e)}")
        
        try:
            if self.cache_collection:
                stats["cache_collection"]["count"] = self.cache_collection.count()
                stats["cache_collection"]["status"] = "connected"
        except Exception as e:
            print(f"Lỗi khi lấy stats cache collection: {str(e)}")
        
        return stats
    
    def reset_database(self):
        """Reset toàn bộ ChromaDB"""
        try:
            client = self.get_client()
            if client:
                self.delete_collection(self.main_collection_name)
                self.delete_collection(self.cache_collection_name)
                self._create_all_collections()
                self.loaded_documents = 0
                self.total_chunks = 0
                print("Đã reset ChromaDB database")
            return True
        except Exception as e:
            print(f"Lỗi khi reset database: {str(e)}")
            return False

    # ==================== DATA LOADING OPERATIONS ====================
    
    def safe_join_list(self, value, default_value=""):
        """An toàn join list, xử lý trường hợp None hoặc không phải list"""
        if value is None:
            return default_value
        elif isinstance(value, list):
            return ", ".join(str(item) for item in value if item is not None)
        elif isinstance(value, str):
            return value
        else:
            return str(value) if value is not None else default_value
    
    def read_chunk_content(self, file_path: str, doc_folder: str) -> str:
        """Đọc nội dung từ file chunk"""
        try:
            print(f"    Đang đọc file: {file_path}")
            
            if file_path.startswith("/data/"):
                relative_path = file_path[1:]
                full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), relative_path)
            elif file_path.startswith("data/"):
                full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), file_path)
            else:
                full_path = os.path.join(doc_folder, file_path)
            
            print(f"    Đường dẫn đầy đủ: {full_path}")
            
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8-sig') as f:
                    content = f.read().strip()
                    print(f"    Đọc thành công, độ dài: {len(content)} ký tự")
                    return content
            else:
                print(f"    Không tìm thấy file: {full_path}")
                return ""
            
        except Exception as e:
            print(f"    Lỗi đọc file {file_path}: {str(e)}")
            return ""
    
    def create_chunk_metadata(self, doc_metadata: Dict[str, Any], chunk_info: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
        """Tạo metadata đầy đủ cho chunk với xử lý an toàn"""
        chunks_list = doc_metadata.get("chunks", [])
        
        previous_chunk = chunks_list[chunk_index - 1]["chunk_id"] if chunk_index > 0 else None
        next_chunk = chunks_list[chunk_index + 1]["chunk_id"] if chunk_index < len(chunks_list) - 1 else None
        
        related_docs = []
        related_documents = doc_metadata.get("related_documents", [])
        if related_documents and isinstance(related_documents, list):
            for rel_doc in related_documents:
                if isinstance(rel_doc, dict):
                    doc_id = rel_doc.get('doc_id', '')
                    relationship = rel_doc.get('relationship', '')
                    if doc_id:
                        related_docs.append(f"{doc_id} ({relationship})")
        
        metadata = {
            "doc_id": str(doc_metadata.get("doc_id", "")),
            "doc_type": str(doc_metadata.get("doc_type", "")),
            "doc_title": str(doc_metadata.get("doc_title", "")),
            "issue_date": str(doc_metadata.get("issue_date", "")),
            "effective_date": str(doc_metadata.get("effective_date", "")),
            "expiry_date": str(doc_metadata.get("expiry_date", "") if doc_metadata.get("expiry_date") else ""),
            "status": str(doc_metadata.get("status", "")),
            "document_scope": str(doc_metadata.get("document_scope", "")),
            "replaces": self.safe_join_list(doc_metadata.get("replaces")),
            "replaced_by": str(doc_metadata.get("replaced_by", "") if doc_metadata.get("replaced_by") else ""),
            "amends": self.safe_join_list(doc_metadata.get("amends")),
            "amended_by": str(doc_metadata.get("amended_by", "") if doc_metadata.get("amended_by") else ""),
            "retroactive": str(doc_metadata.get("retroactive", False)),
            "retroactive_date": str(doc_metadata.get("retroactive_date", "") if doc_metadata.get("retroactive_date") else ""),
            "chunk_id": str(chunk_info.get("chunk_id", "")),
            "chunk_type": str(chunk_info.get("chunk_type", "")),
            "content_summary": str(chunk_info.get("content_summary", "")),
            "chunk_index": str(chunk_index),
            "total_chunks": str(len(chunks_list)),
            "previous_chunk": str(previous_chunk) if previous_chunk else "",
            "next_chunk": str(next_chunk) if next_chunk else "",
            "related_documents": "; ".join(related_docs),
            "related_docs_count": str(len(related_docs))
        }
        
        return metadata
    
    def process_document(self, doc_folder: str) -> bool:
        """Xử lý một document folder"""
        try:
            metadata_path = os.path.join(doc_folder, "metadata.json")
            
            if not os.path.exists(metadata_path):
                print(f"Không tìm thấy metadata.json trong {doc_folder}")
                return False
            
            with open(metadata_path, 'r', encoding='utf-8-sig') as f:
                doc_metadata = json.load(f)
            
            doc_id = doc_metadata.get("doc_id", os.path.basename(doc_folder))
            print(f"Đang xử lý document: {doc_id}")
            
            ids_to_add = []
            documents_to_add = []
            metadatas_to_add = []
            
            chunks_list = doc_metadata.get("chunks", [])
            for chunk_index, chunk_info in enumerate(chunks_list):
                chunk_id = chunk_info.get("chunk_id", "")
                file_path = chunk_info.get("file_path", "")
                
                if not chunk_id:
                    print(f"  Cảnh báo: Chunk không có chunk_id, bỏ qua")
                    continue
                
                content = self.read_chunk_content(file_path, doc_folder)
                if not content:
                    print(f"  Cảnh báo: Chunk {chunk_id} không có nội dung, bỏ qua")
                    continue
                
                document_text = f"passage: {content}"
                chunk_metadata = self.create_chunk_metadata(doc_metadata, chunk_info, chunk_index)
                
                ids_to_add.append(chunk_id)
                documents_to_add.append(document_text)
                metadatas_to_add.append(chunk_metadata)
                
                print(f"  - Chunk {chunk_index + 1}/{len(chunks_list)}: {chunk_id}")
            
            if ids_to_add:
                try:
                    success = self.add_documents_to_main(
                        ids=ids_to_add,
                        documents=documents_to_add,
                        metadatas=metadatas_to_add
                    )
                    if success:
                        print(f"Đã thêm {len(ids_to_add)} chunks cho document {doc_id}")
                        self.total_chunks += len(ids_to_add)
                        return True
                    else:
                        print(f"Lỗi khi thêm chunks vào ChromaDB cho document {doc_id}")
                        return False
                except Exception as e:
                    print(f"Lỗi khi thêm chunks vào ChromaDB: {str(e)}")
                    return False
            else:
                print(f"Không có chunk nào hợp lệ cho document {doc_id}")
                return False
                
        except Exception as e:
            print(f"Lỗi xử lý document {doc_folder}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_all_data(self) -> bool:
        """Load tất cả data từ thư mục data"""
        if not os.path.exists(DATA_DIR):
            print(f"Thư mục data không tồn tại: {DATA_DIR}")
            return False
        
        print(f"Bắt đầu load data từ: {DATA_DIR}")
        print("=" * 60)
        
        doc_folders = []
        for item in os.listdir(DATA_DIR):
            item_path = os.path.join(DATA_DIR, item)
            if os.path.isdir(item_path):
                doc_folders.append(item_path)
        
        if not doc_folders:
            print("Không tìm thấy thư mục document nào trong data")
            return False
        
        print(f"Tìm thấy {len(doc_folders)} documents")
        
        successful_docs = 0
        self.loaded_documents = 0
        self.total_chunks = 0
        
        for doc_folder in sorted(doc_folders):
            if self.process_document(doc_folder):
                successful_docs += 1
                self.loaded_documents += 1
            print("-" * 40)
        
        print("=" * 60)
        print(f"Hoàn thành load data:")
        print(f"  - Documents thành công: {successful_docs}/{len(doc_folders)}")
        print(f"  - Tổng chunks đã thêm: {self.total_chunks}")
        
        final_count = self.main_collection.count()
        print(f"  - Tổng documents trong ChromaDB: {final_count}")
        
        return successful_docs > 0
    
    def check_existing_data(self):
        """Kiểm tra data hiện có trong ChromaDB"""
        try:
            main_collection = self.get_main_collection()
            cache_collection = self.get_cache_collection()
            
            if not main_collection or not cache_collection:
                print("Không thể kết nối đến collections")
                return
            
            main_count = main_collection.count()
            cache_count = cache_collection.count()
            
            print(f"Main collection ({self.main_collection_name}) có {main_count} documents")
            print(f"Cache collection ({self.cache_collection_name}) có {cache_count} documents")
            
            if main_count > 0:
                sample_results = main_collection.get(
                    limit=5,
                    include=["documents", "metadatas"]
                )
                
                print("\nSample documents từ main collection:")
                for i, (doc_id, metadata) in enumerate(zip(sample_results["ids"], sample_results["metadatas"])):
                    print(f"{i+1}. {doc_id}")
                    print(f"   Doc: {metadata.get('doc_id', 'N/A')}")
                    print(f"   Type: {metadata.get('doc_type', 'N/A')}")
                    print(f"   Chunk: {metadata.get('chunk_type', 'N/A')}")
            
            if cache_count > 0:
                cache_sample = cache_collection.get(
                    limit=3,
                    include=["documents", "metadatas"]
                )
                
                print("\nSample documents từ cache collection:")
                for i, (doc_id, metadata) in enumerate(zip(cache_sample["ids"], cache_sample["metadatas"])):
                    print(f"{i+1}. {doc_id}")
                    print(f"   Status: {metadata.get('validityStatus', 'N/A')}")
            
        except Exception as e:
            print(f"Lỗi kiểm tra data: {str(e)}")
    
    def get_load_progress(self):
        """Lấy thông tin tiến trình load data"""
        return {
            "loaded_documents": self.loaded_documents,
            "total_chunks": self.total_chunks,
            "main_collection_count": self.main_collection.count() if self.main_collection else 0,
            "cache_collection_count": self.cache_collection.count() if self.cache_collection else 0
        }

# Singleton instance
chroma_client = ChromaDBClient()

# Helper functions
def get_chroma_client():
    """Hàm helper để lấy ChromaDB client"""
    return chroma_client

def get_collection():
    """Hàm helper để lấy main collection"""
    return chroma_client.get_main_collection()

def get_cache_collection():
    """Hàm helper để lấy cache collection"""
    return chroma_client.get_cache_collection()