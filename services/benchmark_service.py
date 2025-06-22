# backend/services/benchmark_service.py
import os
import json
import csv
import time
import uuid
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from openai import OpenAI
import sys
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEMINI_API_KEY, BENCHMARK_DIR, BENCHMARK_RESULTS_DIR, CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION, EMBEDDING_MODEL_NAME, USE_GPU
from services.retrieval_service import retrieval_service
from services.generation_service import generation_service
from database.chroma_client import chroma_client

# LangChain imports
try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Haystack imports
try:
    from haystack import Document
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.retrievers import InMemoryBM25Retriever
    from haystack.components.builders import PromptBuilder
    from haystack import Pipeline
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False

class BenchmarkService:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        genai.configure(api_key=GEMINI_API_KEY)
        self._init_langchain_client()
        
        # Prompt template giữ nguyên từ code cũ
        self.prompt_template = """
[SYSTEM INSTRUCTION]
Bạn là chuyên gia tư vấn chính sách người có công tại Việt Nam, được phát triển để cung cấp thông tin chính xác, đầy đủ và có căn cứ pháp lý rõ ràng. Nhiệm vụ của bạn là phân tích và tổng hợp thông tin từ các văn bản pháp luật để đưa ra câu trả lời hoàn chỉnh với đầy đủ thông tin cấu trúc. (không được thêm emoji hay sticker gì)

### NGUYÊN TẮC XỬ LÝ THÔNG TIN
1. **Phân tích toàn diện**: Đọc kỹ TẤT CẢ đoạn văn bản được cung cấp, không bỏ sót thông tin nào.
2. **Tổng hợp logic**: Kết hợp thông tin từ nhiều văn bản theo thứ tự ưu tiên:
   - Văn bản mới nhất > văn bản cũ hơn
   - Văn bản cấp cao hơn > văn bản cấp thấp hơn (Luật > Nghị định > Thông tư > Quyết định)
   - Văn bản chuyên biệt > văn bản tổng quát
   - Văn bản còn hiệu lực > văn bản đã hết hiệu lực
3. **Xử lý mâu thuẫn**: Khi có thông tin khác nhau, nêu rõ sự khác biệt và giải thích căn cứ áp dụng.

### CẤU TRÚC CÂU TRẢ LỜI BẮT BUỘC - 17 THÀNH PHẦN
**Format chính**: "Theo [tên văn bản + số hiệu + điều khoản cụ thể], thì [nội dung trả lời chi tiết]."

**Cấu trúc hoàn chỉnh** - BẮT BUỘC bao gồm các thông tin sau (nếu có trong văn bản):
1. **Câu trả lời trực tiếp** với trích dẫn văn bản đầy đủ
2. **17 thành phần thông tin cấu trúc**:
   - **Mã định danh**: Mã thủ tục hành chính, mã văn bản, mã chương mục
   - **Loại văn bản/chính sách**: Phân loại (trợ cấp, thủ tục, ưu đãi, điều dưỡng...)
   - **Số liệu/mức tiền**: Tất cả con số, tỷ lệ, hệ số, giới hạn CHÍNH XÁC
   - **Đối tượng**: Phân loại chi tiết đối tượng áp dụng và điều kiện
   - **Điều kiện/yêu cầu**: Tất cả tiêu chí, hạn chế, điều kiện loại trừ
   - **Thủ tục/hồ sơ**: Quy trình và thành phần hồ sơ đầy đủ
   - **Thời hạn**: Mọi loại thời hạn (xử lý, nộp, hiệu lực, thanh toán...)
   - **Cơ quan/tổ chức**: Tất cả cơ quan liên quan và phân cấp thẩm quyền
   - **Địa điểm/phạm vi**: Nơi thực hiện và phạm vi áp dụng địa lý
   - **Phí/lệ phí**: Mức thu, công thức tính, miễn phí (nếu có)
   - **Văn bản pháp luật**: Số hiệu đầy đủ, ngày ban hành, cơ quan
   - **Ngày tháng**: Các mốc thời gian quan trọng (ban hành, hiệu lực, hết hạn...)
   - **Trạng thái văn bản**: Hiệu lực, thay thế, bãi bỏ, sửa đổi, bổ sung
   - **Mức độ dịch vụ công**: DVC cấp 2/3/4, thực hiện qua DVCTT
   - **Nguồn kinh phí**: Ngân sách trung ương/địa phương, phương thức đảm bảo
   - **Phương thức thực hiện**: Trực tiếp, qua bưu điện, trực tuyến, tần suất
   - **Kết quả/sản phẩm**: Loại giấy tờ, chứng nhận, thẻ nhận được

### TEMPLATE CÂU TRẢ LỜI CHI TIẾT ĐẦY ĐỦ
"Theo [văn bản + điều khoản], thì [câu trả lời trực tiếp].

**Thông tin chi tiết:**
- **Mã định danh**: [Mã thủ tục/chương mục] (nếu có)
- **Loại chính sách**: [Phân loại cụ thể]
- **Mức tiền/Tỷ lệ**: [Số liệu chính xác với đơn vị]
- **Đối tượng**: [Phân loại chi tiết và điều kiện]
- **Điều kiện/Yêu cầu**: [Tất cả tiêu chí và hạn chế]
- **Thủ tục/Hồ sơ**: [Quy trình và thành phần hồ sơ]
- **Thời hạn**: [Xử lý/Nộp/Hiệu lực/Thanh toán...]
- **Cơ quan thực hiện**: [Tiếp nhận/Xử lý/Quyết định]
- **Địa điểm/Phạm vi**: [Nơi thực hiện và áp dụng]
- **Phí/Lệ phí**: [Mức thu và cách tính] (nếu có)
- **Văn bản pháp luật**: [Số hiệu + ngày + cơ quan ban hành]
- **Ngày tháng quan trọng**: [Ban hành/Hiệu lực/Hết hạn...]
- **Trạng thái văn bản**: [Có hiệu lực/Thay thế/Bãi bỏ...]
- **Mức độ dịch vụ công**: [DVC cấp X, DVCTT có/không]
- **Nguồn kinh phí**: [Ngân sách nào đảm bảo]
- **Phương thức**: [Trực tiếp/Trực tuyến/Qua bưu điện...]
- **Kết quả nhận được**: [Loại giấy tờ/chứng nhận]

*Lưu ý đặc biệt*: [Các ngoại lệ, điều kiện đặc biệt, thay đổi gần đây]"

### QUY TẮC TRÌNH BÀY BẮT BUỘC
1. **Độ chính xác tuyệt đối**: Giữ nguyên TẤT CẢ con số, mã số, ngày tháng
2. **Trích dẫn đầy đủ**: Số hiệu văn bản + ngày ban hành + cơ quan + điều khoản
3. **Phân loại rõ ràng**: Dùng **in đậm** cho từng loại thông tin trong 17 trường
4. **Cấu trúc logic**: Từ tổng quát đến chi tiết, từ chính đến phụ
5. **Ngôn ngữ dễ hiểu**: Giải thích thuật ngữ pháp lý phức tạp
6. **Không thêm thông tin ngoài**: Chỉ sử dụng thông tin trong đoạn văn bản cung cấp
7. **Không sử dụng emoji/sticker**: Trả lời bằng văn bản thuần túy
8, **Trả lời tự nhiên**: Không sử dụng ngôn ngữ máy móc, câu trả lời phải tự nhiên và dễ hiểu

[USER QUERY]
{question}

[CONTEXT]
{context}
"""

        # Entity extraction prompt mới theo format JSON
        self.entity_extraction_prompt = """
Bạn là chuyên gia phân tích văn bản pháp luật về chính sách người có công tại Việt Nam.
Nhiệm vụ của bạn là trích xuất TOÀN BỘ thông tin có cấu trúc từ câu trả lời về chính sách để phục vụ đánh giá độ chính xác của hệ thống RAG.

NGUYÊN TẮC TRÍCH XUẤT:
- Trích xuất TẤT CẢ thông tin, không bỏ sót
- Giữ nguyên con số, ký hiệu, mã số CHÍNH XÁC
- Chuẩn hóa format nhưng không thay đổi nội dung
- Nếu có nhiều giá trị cho cùng 1 field, liệt kê tất cả

YÊU CẦU FORMAT OUTPUT:
- Trả về JSON với structure rõ ràng
- Mỗi field là array để chứa nhiều giá trị
- Sử dụng key tiếng Anh, value tiếng Việt nguyên gốc
- Nếu không có thông tin, để array rỗng []
- Đặc biệt chú ý: KHÔNG làm tròn số, KHÔNG thay đổi format tiền tệ

Câu trả lời cần phân tích:
{answer_text}

JSON:
{{
  "ma_dinh_danh": [],
  "loai_van_ban_chinh_sach": [],
  "so_lieu_muc_tien": [],
  "doi_tuong": [],
  "dieu_kien_yeu_cau": [],
  "thu_tuc_ho_so": [],
  "thoi_han": [],
  "co_quan_to_chuc": [],
  "dia_diem_pham_vi": [],
  "phi_le_phi": [],
  "van_ban_phap_luat": [],
  "ngay_thang": [],
  "trang_thai_van_ban": [],
  "muc_do_dich_vu_cong": [],
  "nguon_kinh_phi": [],
  "phuong_thuc_thuc_hien": [],
  "ket_qua_san_pham": []
}}
"""

        # Định nghĩa fields cần exact match vs similarity
        self.exact_match_fields = {
            "ma_dinh_danh", "so_lieu_muc_tien", "ngay_thang", 
            "phi_le_phi", "van_ban_phap_luat"
        }
        
        self.similarity_fields = {
            "loai_van_ban_chinh_sach", "doi_tuong", "dieu_kien_yeu_cau",
            "thu_tuc_ho_so", "thoi_han", "co_quan_to_chuc", "dia_diem_pham_vi",
            "trang_thai_van_ban", "muc_do_dich_vu_cong", "nguon_kinh_phi",
            "phuong_thuc_thuc_hien", "ket_qua_san_pham"
        }

    def _init_langchain_client(self):
        """Khởi tạo ChromaDB client cho LangChain"""
        try:
            self.langchain_chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
            device = "cuda" if USE_GPU and self._check_gpu() else "cpu"
            self.langchain_embedding_function = SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL_NAME, device=device
            )
            print(f"LangChain ChromaDB client initialized: {CHROMA_PERSIST_DIRECTORY}")
        except Exception as e:
            print(f"Lỗi khởi tạo LangChain ChromaDB client: {str(e)}")
            self.langchain_chroma_client = None
            self.langchain_embedding_function = None

    def _check_gpu(self):
        """Kiểm tra GPU availability"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj

    def extract_entities(self, answer_text):
        """Trích xuất entities từ câu trả lời sử dụng Gemini với format JSON mới"""
        try:
            time.sleep(0.5)  # Delay để tránh quá tải API
            
            model = genai.GenerativeModel('gemini-2.0-flash')
            prompt = self.entity_extraction_prompt.format(answer_text=answer_text)
            
            response = model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Tìm JSON trong response
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                entities = json.loads(json_str)
                
                # Đảm bảo có đủ các key theo format mới
                standard_entities = {}
                for key in ["ma_dinh_danh", "loai_van_ban_chinh_sach", "so_lieu_muc_tien", 
                           "doi_tuong", "dieu_kien_yeu_cau", "thu_tuc_ho_so", "thoi_han",
                           "co_quan_to_chuc", "dia_diem_pham_vi", "phi_le_phi", 
                           "van_ban_phap_luat", "ngay_thang", "trang_thai_van_ban",
                           "muc_do_dich_vu_cong", "nguon_kinh_phi", "phuong_thuc_thuc_hien",
                           "ket_qua_san_pham"]:
                    standard_entities[key] = entities.get(key, [])
                
                return standard_entities
            else:
                raise ValueError("Không tìm thấy JSON trong response")
                
        except Exception as e:
            print(f"Lỗi khi trích xuất entities: {str(e)}")
            # Return empty structure
            return {key: [] for key in ["ma_dinh_danh", "loai_van_ban_chinh_sach", 
                   "so_lieu_muc_tien", "doi_tuong", "dieu_kien_yeu_cau", "thu_tuc_ho_so",
                   "thoi_han", "co_quan_to_chuc", "dia_diem_pham_vi", "phi_le_phi",
                   "van_ban_phap_luat", "ngay_thang", "trang_thai_van_ban",
                   "muc_do_dich_vu_cong", "nguon_kinh_phi", "phuong_thuc_thuc_hien",
                   "ket_qua_san_pham"]}

    def _normalize_for_exact_match(self, values):
        """Chuẩn hóa giá trị để so sánh exact match"""
        normalized = []
        for value in values:
            if isinstance(value, str):
                # Loại bỏ khoảng trắng, chuyển thường, chuẩn hóa format số
                norm = re.sub(r'\s+', '', value.lower())
                norm = re.sub(r'\.(?=\d{3})', '', norm)  # 5.500.000 -> 5500000
                norm = re.sub(r',(?=\d{3})', '', norm)   # 5,500,000 -> 5500000
                normalized.append(norm)
            else:
                normalized.append(str(value))
        return set(normalized)

    def _calculate_field_similarity(self, values1, values2, field_name):
        """Tính similarity cho một field cụ thể"""
        if not values1 and not values2:
            return 1.0  # Cả hai đều rỗng
        
        if not values1 or not values2:
            return 0.0  # Một bên rỗng
        
        if field_name in self.exact_match_fields:
            # So sánh chính xác 100%
            norm1 = self._normalize_for_exact_match(values1)
            norm2 = self._normalize_for_exact_match(values2)
            
            if norm1 == norm2:
                return 1.0
            else:
                # Tính overlap ratio
                intersection = len(norm1.intersection(norm2))
                union = len(norm1.union(norm2))
                return intersection / union if union > 0 else 0.0
        
        else:
            # So sánh bằng cosine similarity
            text1 = " ".join([str(v) for v in values1])
            text2 = " ".join([str(v) for v in values2])
            
            try:
                embeddings = self.embedding_model.encode([text1, text2])
                cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return float(cosine_sim)
            except:
                return 0.0

    def calculate_entity_similarity(self, entities1, entities2):
        """Tính độ tương đồng giữa 2 bộ entities với logic tối ưu"""
        if not entities1 or not entities2:
            return 0.0
        
        total_score = 0.0
        field_count = 0
        
        # Tính similarity cho từng field
        all_fields = self.exact_match_fields.union(self.similarity_fields)
        
        for field in all_fields:
            values1 = entities1.get(field, [])
            values2 = entities2.get(field, [])
            
            field_score = self._calculate_field_similarity(values1, values2, field)
            total_score += field_score
            field_count += 1
        
        return total_score / field_count if field_count > 0 else 0.0

    def calculate_cosine_similarity(self, generated_answer, reference_answer):
        """Tính cosine similarity giữa câu trả lời generated và reference"""
        if isinstance(reference_answer, dict):
            ref_text = reference_answer.get("current_citation", str(reference_answer))
        else:
            ref_text = str(reference_answer)
        
        gen_text = str(generated_answer)
        
        try:
            embeddings = self.embedding_model.encode([gen_text, ref_text])
            cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(cosine_sim), ref_text
        except Exception as e:
            print(f"Error calculating cosine similarity: {str(e)}")
            return 0.0, ref_text

    def evaluate_retrieval_accuracy(self, retrieved_chunks, benchmark_chunks):
        """Đánh giá độ chính xác của retrieval"""
        if not benchmark_chunks:
            return 1.0, []
        
        clean_retrieved = [chunk.split(' (doc:')[0].strip() if '(' in chunk and 'doc:' in chunk 
                          else chunk.strip() for chunk in retrieved_chunks if chunk]
        
        found = sum(1 for benchmark_chunk in benchmark_chunks 
                   if any(benchmark_chunk == retrieved_chunk or 
                         benchmark_chunk in retrieved_chunk or 
                         retrieved_chunk in benchmark_chunk 
                         for retrieved_chunk in clean_retrieved))
        
        return float(found / len(benchmark_chunks)), []

    def process_current_system(self, question):
        """Xử lý câu hỏi bằng hệ thống hiện tại"""
        start_time = time.time()
        try:
            retrieval_result = retrieval_service.retrieve(question, use_cache=False)
            context_items = retrieval_result.get("context_items", [])
            retrieved_chunks = retrieval_result.get("retrieved_chunks", [])
            
            time.sleep(0.5)
            
            if context_items:
                generation_result = generation_service.generate_answer(question, use_cache=False)
                answer = generation_result.get("answer", "")
            else:
                answer = "Tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong cơ sở dữ liệu."
            
            return answer, retrieved_chunks, time.time() - start_time
        except Exception as e:
            return f"ERROR: {str(e)}", [], time.time() - start_time

    def process_langchain(self, question):
        """Xử lý câu hỏi bằng LangChain"""
        start_time = time.time()
        
        if not LANGCHAIN_AVAILABLE or not self.langchain_chroma_client:
            return "LangChain not available", [], time.time() - start_time
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain.prompts import ChatPromptTemplate
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
            
            embedding_function = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cuda' if USE_GPU and self._check_gpu() else 'cpu'}
            )
            
            db = Chroma(
                client=self.langchain_chroma_client,
                collection_name=CHROMA_COLLECTION,
                embedding_function=embedding_function
            )
            
            results = db.similarity_search_with_relevance_scores(question, k=5)
            
            time.sleep(0.5)
            
            if not results or results[0][1] < 0.7:
                return "No relevant documents found", [], time.time() - start_time
            
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
            prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
            prompt = prompt_template.format(context=context_text, question=question)
            
            model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)
            response = model.invoke(prompt)
            answer = str(response.content)
            
            retrieved_chunks = [f"{doc.metadata.get('chunk_id', f'unknown_chunk_{i}')} (doc: {doc.metadata.get('doc_id', 'unknown_doc')}, score: {float(score):.3f})" 
                              for i, (doc, score) in enumerate(results)]
            
            return answer, retrieved_chunks, time.time() - start_time
            
        except Exception as e:
            return f"ERROR: {str(e)}", [], time.time() - start_time

    def process_haystack(self, question):
        """Xử lý câu hỏi bằng Haystack"""
        start_time = time.time()
        
        if not HAYSTACK_AVAILABLE:
            return "Haystack not available", [], time.time() - start_time
        
        try:
            from haystack import Document
            from haystack.document_stores.in_memory import InMemoryDocumentStore
            from haystack.components.retrievers import InMemoryBM25Retriever
            
            # Load documents
            data_dir = "data"
            all_docs = []
            
            if os.path.exists(data_dir):
                for subdir in [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]:
                    metadata_path = os.path.join(subdir, "metadata.json")
                    doc_id = os.path.basename(subdir)
                    
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r", encoding="utf-8-sig") as f:
                            metadata = json.load(f)
                        
                        for chunk in metadata.get("chunks", []):
                            chunk_id = chunk.get("chunk_id", "unknown")
                            file_path = chunk.get("file_path", "")
                            
                            if file_path.startswith("/data/") or file_path.startswith("data/"):
                                abs_file_path = os.path.join(subdir, os.path.basename(file_path))
                            else:
                                abs_file_path = os.path.join(subdir, f"chunk_{len(all_docs)+1}.md")
                            
                            if os.path.exists(abs_file_path):
                                with open(abs_file_path, "r", encoding="utf-8-sig") as f:
                                    content = f.read()
                                doc = Document(content=content, meta={"doc_id": doc_id, "chunk_id": chunk_id})
                                all_docs.append(doc)
            
            if not all_docs:
                return "No documents found for Haystack", [], time.time() - start_time
            
            document_store = InMemoryDocumentStore()
            document_store.write_documents(all_docs)
            retriever = InMemoryBM25Retriever(document_store, top_k=5)
            
            retrieved_docs = retriever.run(query=question)
            
            time.sleep(0.5)
            
            chunk_ids = [f"{doc.meta.get('chunk_id', 'unknown')} (doc: {doc.meta.get('doc_id', 'unknown')}, type: BM25)" 
                        for doc in retrieved_docs["documents"]]
            context_text = "\n\n---\n\n".join([doc.content for doc in retrieved_docs["documents"]])
            
            prompt = self.prompt_template.format(context=context_text, question=question)
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            answer = response.text
            
            return answer, chunk_ids, time.time() - start_time
            
        except Exception as e:
            return f"ERROR: {str(e)}", [], time.time() - start_time

    def process_chatgpt(self, question):
        """Xử lý câu hỏi bằng ChatGPT"""
        start_time = time.time()
        
        try:
            retrieval_result = retrieval_service.retrieve(question, use_cache=False)
            context_items = retrieval_result.get("context_items", [])
            
            time.sleep(0.5)
            
            if not context_items:
                return "No relevant context found", [], time.time() - start_time
            
            context_text = "\n\n---\n\n".join(context_items)
            prompt = self.prompt_template.format(context=context_text, question=question)
            
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                return "OpenAI API key not configured", [], time.time() - start_time
            
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            return answer, [], time.time() - start_time
            
        except Exception as e:
            return f"ERROR: {str(e)}", [], time.time() - start_time

    def save_uploaded_benchmark(self, file_content, filename):
        """Save uploaded benchmark file"""
        try:
            os.makedirs(BENCHMARK_DIR, exist_ok=True)
            file_path = os.path.join(BENCHMARK_DIR, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
            return filename
        except Exception as e:
            raise Exception(f"Failed to save benchmark file: {str(e)}")

    def run_benchmark(self, benchmark_file="benchmark.json", progress_callback=None):
        """Chạy benchmark so sánh 4 models với entity extraction tối ưu"""
        try:
            benchmark_path = os.path.join(BENCHMARK_DIR, benchmark_file)
            if not os.path.exists(benchmark_path):
                raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")
            
            with open(benchmark_path, "r", encoding="utf-8-sig") as f:
                benchmark_data = json.load(f).get("benchmark", [])
            
            if not benchmark_data:
                raise ValueError("No benchmark questions found")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"benchmark_4models_{timestamp}.csv"
            output_path = os.path.join(BENCHMARK_RESULTS_DIR, output_file)
            
            os.makedirs(BENCHMARK_RESULTS_DIR, exist_ok=True)
            
            results = []
            total_questions = len(benchmark_data)
            
            # Bước 1: Trích xuất entities từ benchmark
            print("Bước 1: Đang trích xuất entities từ benchmark answers...")
            benchmark_entities = []
            for i, entry in enumerate(benchmark_data):
                if progress_callback:
                    progress_callback({
                        'phase': 'extracting_benchmark_entities',
                        'current_step': i + 1,
                        'total_steps': total_questions,
                        'progress': (i + 1) / total_questions * 10
                    })
                
                expected = entry.get("ground_truth", entry.get("answer", ""))
                benchmark_text = expected.get("current_citation", str(expected)) if isinstance(expected, dict) else str(expected)
                
                entities = self.extract_entities(benchmark_text)
                benchmark_entities.append(entities)
                time.sleep(0.3)
            
            # Bước 2: Chạy benchmark cho 4 models
            print("Bước 2: Đang chạy benchmark cho 4 models...")
            
            with open(output_path, "w", encoding="utf-8-sig", newline="") as csvfile:
                writer = csv.writer(csvfile)
                
                # Header
                writer.writerow([
                    "STT", "question", "benchmark_answer", 
                    "current_answer", "current_cosine_sim", "current_entity_sim", "current_retrieval_accuracy", "current_processing_time",
                    "langchain_answer", "langchain_cosine_sim", "langchain_entity_sim", "langchain_retrieval_accuracy", "langchain_processing_time",
                    "haystack_answer", "haystack_cosine_sim", "haystack_entity_sim", "haystack_retrieval_accuracy", "haystack_processing_time",
                    "chatgpt_answer", "chatgpt_cosine_sim", "chatgpt_entity_sim", "chatgpt_processing_time",
                    "benchmark_chunks"
                ])
                
                for i, entry in enumerate(benchmark_data, start=1):
                    question = entry["question"]
                    expected = entry.get("ground_truth", entry.get("answer", ""))
                    benchmark_chunks = entry.get("contexts", [])
                    benchmark_entity = benchmark_entities[i-1]
                    
                    if progress_callback:
                        progress_callback({
                            'phase': 'processing_models',
                            'current_step': i,
                            'total_steps': total_questions,
                            'progress': 10 + (i / total_questions * 85)
                        })
                    
                    print(f"Processing question {i}/{total_questions}: {question[:50]}...")
                    
                    # Process với tất cả 4 models
                    models_data = {}
                    
                    # Current System
                    if progress_callback:
                        progress_callback({'phase': 'current_system', 'current_step': i, 'total_steps': total_questions, 'progress': 10 + (i-1) / total_questions * 85 + 85/total_questions * 0.1})
                    
                    current_answer, current_chunks, current_time = self.process_current_system(question)
                    current_cosine_sim, benchmark_text = self.calculate_cosine_similarity(current_answer, expected)
                    current_retrieval_acc, _ = self.evaluate_retrieval_accuracy(current_chunks, benchmark_chunks)
                    current_entities = self.extract_entities(current_answer)
                    current_entity_sim = self.calculate_entity_similarity(benchmark_entity, current_entities)
                    
                    models_data['current'] = {
                        'answer': current_answer, 'cosine_sim': current_cosine_sim, 'entity_sim': current_entity_sim,
                        'retrieval_acc': current_retrieval_acc, 'time': current_time
                    }
                    
                    # LangChain
                    if progress_callback:
                        progress_callback({'phase': 'langchain', 'current_step': i, 'total_steps': total_questions, 'progress': 10 + (i-1) / total_questions * 85 + 85/total_questions * 0.3})
                    
                    langchain_answer, langchain_chunks, langchain_time = self.process_langchain(question)
                    langchain_cosine_sim, _ = self.calculate_cosine_similarity(langchain_answer, expected)
                    langchain_retrieval_acc, _ = self.evaluate_retrieval_accuracy(langchain_chunks, benchmark_chunks)
                    langchain_entities = self.extract_entities(langchain_answer)
                    langchain_entity_sim = self.calculate_entity_similarity(benchmark_entity, langchain_entities)
                    
                    models_data['langchain'] = {
                        'answer': langchain_answer, 'cosine_sim': langchain_cosine_sim, 'entity_sim': langchain_entity_sim,
                        'retrieval_acc': langchain_retrieval_acc, 'time': langchain_time
                    }
                    
                    # Haystack
                    if progress_callback:
                        progress_callback({'phase': 'haystack', 'current_step': i, 'total_steps': total_questions, 'progress': 10 + (i-1) / total_questions * 85 + 85/total_questions * 0.6})
                    
                    haystack_answer, haystack_chunks, haystack_time = self.process_haystack(question)
                    haystack_cosine_sim, _ = self.calculate_cosine_similarity(haystack_answer, expected)
                    haystack_retrieval_acc, _ = self.evaluate_retrieval_accuracy(haystack_chunks, benchmark_chunks)
                    haystack_entities = self.extract_entities(haystack_answer)
                    haystack_entity_sim = self.calculate_entity_similarity(benchmark_entity, haystack_entities)
                    
                    models_data['haystack'] = {
                        'answer': haystack_answer, 'cosine_sim': haystack_cosine_sim, 'entity_sim': haystack_entity_sim,
                        'retrieval_acc': haystack_retrieval_acc, 'time': haystack_time
                    }
                    
                    # ChatGPT
                    if progress_callback:
                        progress_callback({'phase': 'chatgpt', 'current_step': i, 'total_steps': total_questions, 'progress': 10 + (i-1) / total_questions * 85 + 85/total_questions * 0.9})
                    
                    chatgpt_answer, _, chatgpt_time = self.process_chatgpt(question)
                    chatgpt_cosine_sim, _ = self.calculate_cosine_similarity(chatgpt_answer, expected)
                    chatgpt_entities = self.extract_entities(chatgpt_answer)
                    chatgpt_entity_sim = self.calculate_entity_similarity(benchmark_entity, chatgpt_entities)
                    
                    models_data['chatgpt'] = {
                        'answer': chatgpt_answer, 'cosine_sim': chatgpt_cosine_sim, 'entity_sim': chatgpt_entity_sim,
                        'time': chatgpt_time
                    }
                    
                    # Ghi kết quả
                    writer.writerow([
                        i, question, benchmark_text,
                        models_data['current']['answer'], f"{models_data['current']['cosine_sim']:.4f}", f"{models_data['current']['entity_sim']:.4f}", f"{models_data['current']['retrieval_acc']:.4f}", f"{models_data['current']['time']:.3f}",
                        models_data['langchain']['answer'], f"{models_data['langchain']['cosine_sim']:.4f}", f"{models_data['langchain']['entity_sim']:.4f}", f"{models_data['langchain']['retrieval_acc']:.4f}", f"{models_data['langchain']['time']:.3f}",
                        models_data['haystack']['answer'], f"{models_data['haystack']['cosine_sim']:.4f}", f"{models_data['haystack']['entity_sim']:.4f}", f"{models_data['haystack']['retrieval_acc']:.4f}", f"{models_data['haystack']['time']:.3f}",
                        models_data['chatgpt']['answer'], f"{models_data['chatgpt']['cosine_sim']:.4f}", f"{models_data['chatgpt']['entity_sim']:.4f}", f"{models_data['chatgpt']['time']:.3f}",
                        " | ".join(benchmark_chunks)
                    ])
                    
                    results.append(models_data)
                    time.sleep(1)  # Delay giữa các câu hỏi
                
                # Tính toán và ghi thống kê cuối
                if progress_callback:
                    progress_callback({'phase': 'finalizing', 'current_step': total_questions, 'total_steps': total_questions, 'progress': 95})
                
                # Tính averages
                averages = {}
                for model in ['current', 'langchain', 'haystack', 'chatgpt']:
                    avg_cosine = sum(r[model]['cosine_sim'] for r in results) / len(results)
                    avg_entity = sum(r[model]['entity_sim'] for r in results) / len(results)
                    avg_time = sum(r[model]['time'] for r in results) / len(results)
                    
                    averages[f'{model}_avg_cosine'] = float(avg_cosine)
                    averages[f'{model}_avg_entity'] = float(avg_entity)
                    averages[f'{model}_avg_time'] = float(avg_time)
                    
                    if model in ['current', 'langchain', 'haystack']:
                        avg_retrieval = sum(r[model]['retrieval_acc'] for r in results) / len(results)
                        averages[f'{model}_avg_retrieval'] = float(avg_retrieval)
                
                # Ghi dòng SUMMARY
                writer.writerow([
                    "SUMMARY", f"Average results from {total_questions} questions", "Statistical Summary",
                    "See individual results above", f"{averages['current_avg_cosine']:.4f}", f"{averages['current_avg_entity']:.4f}", f"{averages['current_avg_retrieval']:.4f}", f"{averages['current_avg_time']:.3f}",
                    "See individual results above", f"{averages['langchain_avg_cosine']:.4f}", f"{averages['langchain_avg_entity']:.4f}", f"{averages['langchain_avg_retrieval']:.4f}", f"{averages['langchain_avg_time']:.3f}",
                    "See individual results above", f"{averages['haystack_avg_cosine']:.4f}", f"{averages['haystack_avg_entity']:.4f}", f"{averages['haystack_avg_retrieval']:.4f}", f"{averages['haystack_avg_time']:.3f}",
                    "See individual results above", f"{averages['chatgpt_avg_cosine']:.4f}", f"{averages['chatgpt_avg_entity']:.4f}", f"{averages['chatgpt_avg_time']:.3f}",
                    "All benchmark chunks across questions"
                ])
            
            # Return statistics
            stats = {**averages, 'total_questions': int(total_questions), 'output_file': output_file}
            stats = self._convert_numpy_types(stats)
            
            if progress_callback:
                progress_callback({'phase': 'completed', 'current_step': total_questions, 'total_steps': total_questions, 'progress': 100})
            
            return stats
            
        except Exception as e:
            raise Exception(f"Benchmark failed: {str(e)}")

# Singleton instance
benchmark_service = BenchmarkService()