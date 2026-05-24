# ==============================================================================
# @title CELL 4 (ULTIMATE): COMPLETE METADATA - DocumentProcessor + UnifiedChunker
# ==============================================================================

"""
CELL 4 ULTIMATE - Kết hợp HOÀN CHỈNH metadata từ CẢ HAI phiên bản

METADATA HOÀN CHỈNH (15 FIELDS):
 FROM DOCUMENT PROCESSOR (7 fields):
   - filename: Tên file PDF
   - category: Danh mục từ folder
   - parent_folder: Thư mục cha
   - folder_path: Đường dẫn đầy đủ
   - year: Năm từ filename
   - doc_type: Loại văn bản (quy_che, quy_dinh, etc.)
   - extension: Phần mở rộng file

 FROM UNIFIED CHUNKER (9 fields):
   - chapter: Chương (VD: "Chương I")
   - article: Điều (VD: "Điều 12")
   - article_title: Tiêu đề điều (VD: "Cảnh báo học vụ")
   - section: Khoản (VD: "Khoản 1")
   - point: Điểm (VD: "Điểm a")
   - level: Cấp độ (1=Điều, 2=Khoản, 3=Điểm)
   - page: Số trang
   - hierarchical_path: Đường dẫn phân cấp
   - article_key: Key để map với full article

 PERSISTENCE:
   - JSON storage (human-readable)
   - FAISS index storage
   - Incremental processing
   - Change detection

CÁCH DÙNG:
1. Chạy Cell này lần đầu → Process tất cả PDF với metadata đầy đủ
2. Lần sau chạy lại → Tự động skip file đã process
3. Metadata đầy đủ nhất cho filtering, reranking, và display
"""

print("="*70)
print(" CELL 4 ULTIMATE: COMPLETE METADATA")
print("="*70)

import os
import re
import fitz  # PyMuPDF
import faiss
import numpy as np
import unicodedata
from typing import List, Dict, Tuple
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import json
import hashlib
from pathlib import Path


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Kiem tra xem Cell 1 da chay chua
if 'DOCUMENTS_PATH' not in globals() or 'PROCESSED_DATA_PATH' not in globals():
    print("CANH BAO: Chua load config, dang load...")
    from config import DOCUMENTS_PATH, PROCESSED_DATA_PATH
    globals()['DOCUMENTS_PATH'] = DOCUMENTS_PATH
    globals()['PROCESSED_DATA_PATH'] = PROCESSED_DATA_PATH

# Paths
FOLDER_PATH = DOCUMENTS_PATH
OUTPUT_DIR = PROCESSED_DATA_PATH

# Processing parameters
MAX_CHUNK_SIZE = 300
MIN_CHUNK_SIZE = 50
PATTERN = "*.pdf"
RECURSIVE = True
FORCE_REPROCESS = False
CHUNK_SCHEMA_VERSION = 8
FAISS_INDEX_FILENAME = f"faiss_index_b_v{CHUNK_SCHEMA_VERSION}.bin"
DOCUMENT_METADATA_FILENAME = "document.json"
CHAPTERS_FILENAME = "chapters.json"
ARTICLES_FILENAME = "articles.json"
SECTIONS_FILENAME = "sections.json"
CHUNKS_FILENAME = "chunks.json"

print(f"\nConfiguration:")
print(f"   • Folder: {FOLDER_PATH}")
print(f"   • Output: {OUTPUT_DIR}")
print(f"   • Max chunk size: {MAX_CHUNK_SIZE}")
print(f"   • Min chunk size: {MIN_CHUNK_SIZE}")
print(f"   • Pattern: {PATTERN}")
print(f"   • Recursive: {RECURSIVE}")
print(f"   • Force reprocess: {FORCE_REPROCESS}")
print(f"   • Chunk schema version: {CHUNK_SCHEMA_VERSION}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# COMPLETE METADATA STRUCTURE
# ==============================================================================

@dataclass
class CompleteMetadata:
    """Complete metadata combining both approaches"""
    # From DocumentProcessor (7 fields)
    doc_id: str
    filename: str
    category: str
    parent_folder: str
    folder_path: str
    year: int
    doc_type: str
    extension: str
    decision_number: str
    decision_code: str
    document_title: str
    
    # From UnifiedChunker (9 fields)
    chapter: str
    article: str
    article_title: str
    section: str
    point: str
    level: int
    page: int
    hierarchical_path: str
    article_key: str


# ==============================================================================
# TEXT REPRESENTATION HELPERS
# ==============================================================================

BM25_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def normalize_identifier_text(text: str) -> str:
    """Normalize Vietnamese text into a stable identifier token."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("đ", "d").replace("Đ", "D")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def compose_decision_reference(metadata: Dict) -> str:
    """Compose decision reference from number/code if both exist."""
    decision_number = str(metadata.get("decision_number", "") or "").strip()
    decision_code = str(metadata.get("decision_code", "") or "").strip()
    if decision_number and decision_code:
        return f"{decision_number}/{decision_code}"
    return ""


def build_doc_id(filename: str) -> str:
    stem = Path(filename).stem
    normalized = normalize_identifier_text(stem)
    return f"doc_{normalized}" if normalized else "doc_unknown"


def build_chapter_id(doc_id: str, chapter: str) -> str:
    chapter_slug = normalize_identifier_text(chapter)
    return f"{doc_id}::chapter::{chapter_slug or 'unknown'}"


def build_article_id(doc_id: str, article: str) -> str:
    article_slug = normalize_identifier_text(article)
    return f"{doc_id}::article::{article_slug or 'unknown'}"


def build_section_id(doc_id: str, article: str, section: str) -> str:
    article_slug = normalize_identifier_text(article)
    section_slug = normalize_identifier_text(section)
    return f"{doc_id}::article::{article_slug or 'unknown'}::section::{section_slug or 'unknown'}"


def build_hierarchical_path(article: str, article_title: str = "", section: str = "", point: str = "") -> str:
    """Build a stable hierarchical path string for legal chunks."""
    path_parts = [article]
    if article_title:
        path_parts[0] += f": {article_title}"
    if section:
        path_parts.append(section)
    if point:
        path_parts.append(point)
    return " > ".join(part for part in path_parts if part)


def scope_metadata(metadata: Dict, scope: str = "chunk") -> Dict:
    """Normalize metadata to article / section / chunk scope."""
    scoped = dict(metadata or {})

    if scope == "article":
        scoped["section"] = ""
        scoped["point"] = ""
        scoped["level"] = 1
    elif scope == "section":
        scoped["point"] = ""
        scoped["level"] = 2 if scoped.get("section") else scoped.get("level", 1)

    scoped["hierarchical_path"] = build_hierarchical_path(
        scoped.get("article", ""),
        scoped.get("article_title", ""),
        scoped.get("section", ""),
        scoped.get("point", "")
    )
    return scoped


def build_contextualized_text(text: str, metadata: Dict) -> str:
    """Build deterministic context-rich text for indexing and reranking."""
    metadata = metadata or {}
    header_parts = []

    filename = (metadata.get("filename", "") or "").replace(".pdf", "").strip()
    if filename:
        header_parts.append(filename)
    decision_reference = compose_decision_reference(metadata)
    if decision_reference:
        header_parts.append(f"So QD: {decision_reference}")
    document_title = (metadata.get("document_title", "") or "").strip()
    if document_title and document_title.lower() not in filename.lower():
        header_parts.append(document_title)
    if metadata.get("chapter"):
        header_parts.append(metadata["chapter"])
    if metadata.get("hierarchical_path"):
        header_parts.append(metadata["hierarchical_path"])
    elif metadata.get("article"):
        header_parts.append(
            build_hierarchical_path(
                metadata.get("article", ""),
                metadata.get("article_title", ""),
                metadata.get("section", ""),
                metadata.get("point", "")
            )
        )
    if metadata.get("row_label"):
        header_parts.append(str(metadata.get("row_label", "")).strip())

    header = "\n".join(part for part in header_parts if part).strip()
    body = (text or "").strip()

    if header and body:
        return f"{header}\n\n{body}"
    return header or body


def tokenize_for_bm25(text: str) -> List[str]:
    """Consistent tokenizer for BM25 over Vietnamese legal text."""
    return [token for token in BM25_TOKEN_PATTERN.findall((text or "").lower()) if token]


# ==============================================================================
# METADATA EXTRACTOR (From DocumentProcessor)
# ==============================================================================

class MetadataExtractor:
    """Extract metadata from folder structure and filename"""
    
    @staticmethod
    def extract_from_path(file_path: str, base_folder: str) -> Dict:
        """Extract metadata from file path"""
        path_obj = Path(file_path)
        
        # Basic info
        filename = path_obj.name
        extension = path_obj.suffix
        
        # Folder structure
        relative_path = path_obj.relative_to(base_folder)
        folder_path = str(relative_path.parent)
        
        # Category from immediate parent folder
        category = relative_path.parent.name if relative_path.parent.name != '.' else 'Unknown'
        
        # Parent folder (grandparent)
        parent_folder = relative_path.parent.parent.name if len(relative_path.parent.parts) > 1 else 'Root'
        
        # Extract year from filename (flexible pattern)
        # Match 4-digit year starting with 19 or 20, anywhere in filename
        year_match = re.search(r'(19|20)\d{2}', filename)
        year = int(year_match.group(0)) if year_match else None
        
        # Extract doc type from filename
        doc_type = MetadataExtractor._extract_doc_type(filename)
        decision_metadata = MetadataExtractor._extract_decision_metadata(filename)
        
        return {
            'filename': filename,
            'category': category,
            'parent_folder': parent_folder,
            'folder_path': folder_path,
            'year': year,
            'doc_type': doc_type,
            'extension': extension,
            **decision_metadata
        }
    
    @staticmethod
    def _extract_doc_type(filename: str) -> str:
        """Extract document type from filename"""
        filename_lower = filename.lower()
        
        if 'quy che' in filename_lower or 'quy chế' in filename_lower:
            return 'quy_che'
        elif 'quy dinh' in filename_lower or 'quy định' in filename_lower:
            return 'quy_dinh'
        elif 'huong dan' in filename_lower or 'hướng dẫn' in filename_lower:
            return 'huong_dan'
        elif 'thong tu' in filename_lower or 'thông tư' in filename_lower:
            return 'thong_tu'
        elif 'quyet dinh' in filename_lower or 'quyết định' in filename_lower:
            return 'quyet_dinh'
        elif 'chuan' in filename_lower or 'chuẩn' in filename_lower:
            return 'chuan'
        else:
            return 'other'

    @staticmethod
    def _extract_decision_metadata(filename: str) -> Dict:
        """Extract decision number/reference from structured filename.

        Example:
            2021_1410_QĐ_ĐHSP_Quy chế đào tạo.pdf
            -> decision_number=1410
            -> decision_code=QĐ-ĐHSP
            -> decision_reference=1410/QĐ-ĐHSP
            -> document_title=Quy chế đào tạo
        """
        stem = Path(filename).stem.strip()
        parts = [part.strip() for part in re.split(r"[_]+", stem) if part and part.strip()]

        year_index = None
        for idx, part in enumerate(parts):
            if re.fullmatch(r"(19|20)\d{2}", part):
                year_index = idx
                break

        search_start = year_index + 1 if year_index is not None else 0
        decision_index = None
        decision_number = ""
        for idx in range(search_start, len(parts)):
            if re.fullmatch(r"\d{1,6}", parts[idx]):
                decision_index = idx
                decision_number = parts[idx]
                break

        code_parts = []
        title_start = None
        if decision_index is not None:
            for idx in range(decision_index + 1, len(parts)):
                normalized_part = unicodedata.normalize("NFKD", parts[idx])
                normalized_part = "".join(ch for ch in normalized_part if not unicodedata.combining(ch))
                normalized_part = normalized_part.upper().replace("Đ", "D")
                if re.fullmatch(r"[A-Z0-9]+", normalized_part):
                    code_parts.append(parts[idx].upper())
                    continue
                title_start = idx
                break
            if title_start is None:
                title_start = decision_index + 1 + len(code_parts)
        else:
            title_start = search_start

        decision_code = "-".join(code_parts)
        document_title = " ".join(parts[title_start:]).strip() if title_start is not None else ""

        return {
            "decision_number": decision_number,
            "decision_code": decision_code,
            "document_title": document_title
        }


# ==============================================================================
# UNIFIED CHUNKER (From UnifiedChunker)
# ==============================================================================

class UnifiedChunker:
    """Hierarchical chunking with detailed metadata"""
    
    def __init__(self, max_chunk_size: int = 300, min_chunk_size: int = 50):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # Regex patterns
        self.chapter_pattern = r'(Chương\s+(?:[IVX]+|\d+)[^\n]*)'
        self.article_pattern = r'(Điều\s+\d+)\.?\s*([^\n]*)'
        self.section_pattern = r'^\s*(\d+)\.\s+'
        self.point_pattern = r'^\s*([a-đ])\)\s+'
    
    def chunk_document(self, full_text: str, article_page_map: Dict[str, int]) -> Tuple[List[Dict], Dict[str, Dict]]:
        """Main chunking function"""
        chunks = []
        article_full_text_map = {}
        
        # Split by Điều
        article_segments = re.split(r'(?=Điều\s+\d+\.)', full_text)
        current_chapter = "Quy định chung"
        
        for segment in article_segments:
            segment = segment.strip()
            if len(segment) < self.min_chunk_size:
                continue
            
            # Update chapter
            chapter_match = re.search(self.chapter_pattern, segment, re.IGNORECASE)
            if chapter_match:
                current_chapter = chapter_match.group(1).strip()
                segment = segment.replace(chapter_match.group(0), "").strip()
            
            # Parse article header
            article_match = re.search(self.article_pattern, segment)
            if not article_match:
                continue
            
            article_num = article_match.group(1)
            article_title = article_match.group(2).strip()
            article_full = f"{article_num}: {article_title}" if article_title else article_num
            
            # Store full text
            article_full_text_map[article_num] = {
                "title": article_full,
                "chapter": current_chapter,
                "content": segment,
                "page": article_page_map.get(article_num, 1)
            }
            
            # Hierarchical chunking
            article_chunks = self._chunk_article_hierarchical(
                segment, current_chapter, article_num, article_title,
                article_page_map.get(article_num, 1)
            )
            
            chunks.extend(article_chunks)
        
        return chunks, article_full_text_map

    
    def _chunk_article_hierarchical(self, text: str, chapter: str, article_num: str,
                                    article_title: str, page: int) -> List[Dict]:
        """Chunk một Điều theo hierarchy"""
        chunks = []
        
        # Split by Khoản
        section_splits = re.split(r'(?=\n\s*\d+\.\s)', text)
        
        if len(section_splits) <= 1:
            return self._create_simple_chunk(text, chapter, article_num, article_title, page)
        
        for section_text in section_splits:
            section_text = section_text.strip()
            if len(section_text) < self.min_chunk_size:
                continue
            
            # Detect Khoản
            section_match = re.match(self.section_pattern, section_text, re.MULTILINE)
            section_num = section_match.group(1) if section_match else ""
            
            # Split by Điểm
            point_splits = re.split(r'(?=\n\s*[a-đ]\)\s)', section_text)
            
            if len(point_splits) <= 1:
                # No Điểm
                chunks.append(self._create_chunk_dict(
                    text=section_text,
                    chapter=chapter,
                    article=article_num,
                    article_title=article_title,
                    section=f"Khoản {section_num}" if section_num else "",
                    point="",
                    level=2,
                    page=page
                ))
            else:
                # Has Điểm
                for point_text in point_splits:
                    point_text = point_text.strip()
                    if len(point_text) < self.min_chunk_size:
                        continue
                    
                    point_match = re.match(self.point_pattern, point_text, re.MULTILINE)
                    point_char = point_match.group(1) if point_match else ""

                    table_chunks = self._create_duration_table_row_chunks(
                        point_text=point_text,
                        chapter=chapter,
                        article_num=article_num,
                        article_title=article_title,
                        section_num=section_num,
                        point_char=point_char,
                        page=page
                    )
                    if table_chunks:
                        chunks.extend(table_chunks)
                    else:
                        chunks.append(self._create_chunk_dict(
                            text=point_text,
                            chapter=chapter,
                            article=article_num,
                            article_title=article_title,
                            section=f"Khoản {section_num}" if section_num else "",
                            point=f"Điểm {point_char}" if point_char else "",
                            level=3,
                            page=page
                        ))
        
        return chunks

    def _is_duration_line(self, text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text or "").strip().lower()
        return bool(re.fullmatch(r"\d+(?:[.,]\d+)?\s+năm học", normalized))

    def _create_duration_table_row_chunks(
        self,
        point_text: str,
        chapter: str,
        article_num: str,
        article_title: str,
        section_num: str,
        point_char: str,
        page: int
    ) -> List[Dict]:
        section_label = f"Khoản {section_num}" if section_num else ""
        point_label = f"Điểm {point_char}" if point_char else ""
        point_norm = (point_char or "").lower()
        text_norm = normalize_identifier_text(point_text)

        if article_num != "Điều 3" or section_label != "Khoản 6" or point_norm not in {"a", "b"}:
            return []
        if "thoi_gian_hoc_tap_toi_da" not in text_norm or "nam_hoc" not in text_norm:
            return []

        scope_label = "hình thức đào tạo chính quy" if point_norm == "a" else "hình thức đào tạo vừa làm vừa học"
        program_scope = "chinh_quy" if point_norm == "a" else "vlvh"

        raw_lines = [re.sub(r"\s+", " ", line).strip() for line in (point_text or "").splitlines()]
        lines = [line for line in raw_lines if line]
        if not lines:
            return []

        def _build_row_chunk(program_name: str, standard_duration: str, max_duration: str) -> Dict:
            row_text = (
                f"{scope_label}: {program_name}. "
                f"Thời gian học tập chuẩn {standard_duration}; "
                f"thời gian học tập tối đa {max_duration}."
            )
            return self._create_chunk_dict(
                text=row_text,
                chapter=chapter,
                article=article_num,
                article_title=article_title,
                section=section_label,
                point=point_label,
                level=3,
                page=page,
                extra_metadata={
                    "row_label": program_name,
                    "program_scope": program_scope,
                    "table_kind": "duration_limit",
                    "standard_duration": standard_duration,
                    "max_duration": max_duration,
                }
            )

        def _format_duration_token(token: str) -> str:
            match = re.fullmatch(r"(\d+)(?:_(\d+))?_nam_hoc", token or "")
            if not match:
                return token.replace("_", " ").strip()
            integer = match.group(1)
            decimal = match.group(2)
            if decimal:
                return f"{integer},{decimal} năm học"
            return f"{integer} năm học"

        if len(lines) == 1:
            inline_text = lines[0]
            intro_text = re.split(r"Chương trình đào tạo", inline_text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            intro_text = intro_text.rstrip(":;,.") + ":"

            normalized_inline = normalize_identifier_text(inline_text)
            normalized_inline = re.sub(r"^[a-z]_", "", normalized_inline).strip("_")
            split_token = "nhu_sau"
            if split_token in normalized_inline:
                normalized_inline = normalized_inline.split(split_token, 1)[1].strip()
            header_token = "chuong_trinh_dao_tao_thoi_gian_hoc_tap_chuan_thoi_gian_hoc_tap_toi_da"
            if normalized_inline.startswith(header_token):
                normalized_inline = normalized_inline[len(header_token):].strip("_")

            known_program_rows = [
                ("Đào tạo đại học cấp bằng thứ nhất", "dao_tao_dai_hoc_cap_bang_thu_nhat"),
                ("Đào tạo liên thông từ trình độ cao đẳng lên trình độ đại học", "dao_tao_lien_thong_tu_trinh_do_cao_dang_len_trinh_do_dai_hoc"),
                ("Đào tạo liên thông từ trình độ trung cấp lên trình độ đại học", "dao_tao_lien_thong_tu_trinh_do_trung_cap_len_trinh_do_dai_hoc"),
                ("Đào tạo liên thông trình độ đại học đối với người đã có một bằng đại học", "dao_tao_lien_thong_trinh_do_dai_hoc_doi_voi_nguoi_da_co_mot_bang_dai_hoc"),
            ]

            row_chunks = []
            row_positions = []
            for display_name, normalized_name in known_program_rows:
                pos = normalized_inline.find(normalized_name)
                if pos >= 0:
                    row_positions.append((pos, display_name, normalized_name))

            row_positions.sort(key=lambda item: item[0])
            for idx, (pos, display_name, normalized_name) in enumerate(row_positions):
                next_pos = row_positions[idx + 1][0] if idx + 1 < len(row_positions) else len(normalized_inline)
                row_segment = normalized_inline[pos:next_pos].strip()
                duration_matches = re.findall(r"\d+(?:_\d+)?_nam_hoc", row_segment)
                if len(duration_matches) < 2:
                    continue

                standard_duration = _format_duration_token(duration_matches[0])
                max_duration = _format_duration_token(duration_matches[1])
                row_chunks.append(_build_row_chunk(display_name, standard_duration, max_duration))

            if row_chunks:
                intro_chunk = self._create_chunk_dict(
                    text=intro_text,
                    chapter=chapter,
                    article=article_num,
                    article_title=article_title,
                    section=section_label,
                    point=point_label,
                    level=3,
                    page=page,
                    extra_metadata={
                        "program_scope": program_scope,
                        "table_kind": "duration_limit_intro",
                    }
                )
                return [intro_chunk, *row_chunks]

        filtered_lines = []
        for idx, line in enumerate(lines):
            if idx == 0:
                filtered_lines.append(line)
                continue
            line_norm = normalize_identifier_text(line)
            if line_norm in {
                "chuong trinh dao tao",
                "thoi gian hoc tap chuan",
                "thoi gian hoc tap toi da",
                "thoi gian",
                "hoc tap chuan",
                "hoc tap toi da",
            }:
                continue
            filtered_lines.append(line)

        intro_lines = []
        row_lines = []
        first_duration_idx = None
        for idx, line in enumerate(filtered_lines):
            if self._is_duration_line(line):
                first_duration_idx = idx
                break
        if first_duration_idx is None or first_duration_idx == 0:
            return []

        intro_lines = filtered_lines[:max(1, first_duration_idx - 1)]
        row_lines = filtered_lines[first_duration_idx - 1:]

        row_chunks = []
        i = 0
        while i < len(row_lines):
            program_parts = []
            while i < len(row_lines) and not self._is_duration_line(row_lines[i]):
                program_parts.append(row_lines[i])
                i += 1

            if not program_parts or i + 1 >= len(row_lines):
                break

            standard_duration = row_lines[i]
            max_duration = row_lines[i + 1]
            if not self._is_duration_line(standard_duration) or not self._is_duration_line(max_duration):
                i += 1
                continue

            i += 2
            program_name = " ".join(program_parts).strip(" -")
            if not program_name:
                continue

            row_chunks.append(_build_row_chunk(program_name, standard_duration, max_duration))

        if not row_chunks:
            return []

        intro_text = "\n".join(intro_lines).strip()
        intro_chunk = self._create_chunk_dict(
            text=intro_text,
            chapter=chapter,
            article=article_num,
            article_title=article_title,
            section=section_label,
            point=point_label,
            level=3,
            page=page,
            extra_metadata={
                "program_scope": program_scope,
                "table_kind": "duration_limit_intro",
            }
        )
        return [intro_chunk, *row_chunks]
    
    def _create_simple_chunk(self, text: str, chapter: str, article_num: str,
                            article_title: str, page: int) -> List[Dict]:
        """Fallback: chunk đơn giản"""
        return [self._create_chunk_dict(
            text=text,
            chapter=chapter,
            article=article_num,
            article_title=article_title,
            section="",
            point="",
            level=1,
            page=page
        )]
    
    def _create_chunk_dict(self, text: str, chapter: str, article: str, article_title: str,
                          section: str, point: str, level: int, page: int,
                          extra_metadata: Dict = None) -> Dict:
        """Create chunk dictionary"""
        text = self._clean_text(text)
        
        # Build hierarchical path
        path_parts = [article]
        if article_title:
            path_parts[0] += f": {article_title}"
        if section:
            path_parts.append(section)
        if point:
            path_parts.append(point)
        
        hierarchical_path = " > ".join(path_parts)
        
        content_metadata = {
            "chapter": chapter,
            "article": article,
            "article_title": article_title,
            "section": section,
            "point": point,
            "level": level,
            "page": page,
            "hierarchical_path": hierarchical_path,
            "article_key": article
        }
        if extra_metadata:
            content_metadata.update(extra_metadata)

        return {
            "text": text,
            "content_metadata": content_metadata
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean text - chỉ xóa khoảng trắng thừa, giữ nguyên tất cả nội dung bao gồm số"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


# ==============================================================================
# PERSISTENCE MANAGER
# ==============================================================================

class PersistenceManager:
    """Manage persistence with complete metadata"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.registry_file = os.path.join(output_dir, "registry.json")
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def should_process(self, file_path: str, force: bool = False) -> bool:
        if force:
            return True
        
        filename = os.path.basename(file_path)
        
        if filename not in self.registry:
            return True

        stored_schema_version = self.registry[filename].get('schema_version')
        if stored_schema_version != CHUNK_SCHEMA_VERSION:
            return True
        
        current_hash = self._calculate_file_hash(file_path)
        stored_hash = self.registry[filename].get('hash', '')
        
        return current_hash != stored_hash

    def _doc_dir(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename.replace('.pdf', ''))

    def save_chunks(self, filename: str, chunks: List[Dict], file_hash: str):
        doc_dir = self._doc_dir(filename)
        os.makedirs(doc_dir, exist_ok=True)

        chunks_file = os.path.join(doc_dir, CHUNKS_FILENAME)
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        for stale_file in [
            os.path.join(doc_dir, DOCUMENT_METADATA_FILENAME),
            os.path.join(doc_dir, CHAPTERS_FILENAME),
            os.path.join(doc_dir, ARTICLES_FILENAME),
            os.path.join(doc_dir, SECTIONS_FILENAME),
        ]:
            if os.path.exists(stale_file):
                os.remove(stale_file)

        self.registry[filename] = {
            'hash': file_hash,
            'chunks_count': len(chunks),
            'processed_at': str(np.datetime64('now')),
            'schema_version': CHUNK_SCHEMA_VERSION
        }
        self._save_registry()

        print(f"    Saved legacy schema v3 chunks to {chunks_file} ({len(chunks)})")

    def save_document_bundle(
        self,
        filename: str,
        document_record: Dict,
        chapter_records: List[Dict],
        article_records: List[Dict],
        section_records: List[Dict],
        chunk_records: List[Dict],
        file_hash: str
    ):
        doc_dir = self._doc_dir(filename)
        os.makedirs(doc_dir, exist_ok=True)

        document_file = os.path.join(doc_dir, DOCUMENT_METADATA_FILENAME)
        chapters_file = os.path.join(doc_dir, CHAPTERS_FILENAME)
        articles_file = os.path.join(doc_dir, ARTICLES_FILENAME)
        sections_file = os.path.join(doc_dir, SECTIONS_FILENAME)
        chunks_file = os.path.join(doc_dir, CHUNKS_FILENAME)

        with open(document_file, 'w', encoding='utf-8') as f:
            json.dump(document_record, f, ensure_ascii=False, indent=2)
        with open(chapters_file, 'w', encoding='utf-8') as f:
            json.dump(chapter_records, f, ensure_ascii=False, indent=2)
        with open(articles_file, 'w', encoding='utf-8') as f:
            json.dump(article_records, f, ensure_ascii=False, indent=2)
        with open(sections_file, 'w', encoding='utf-8') as f:
            json.dump(section_records, f, ensure_ascii=False, indent=2)
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_records, f, ensure_ascii=False, indent=2)

        self.registry[filename] = {
            'hash': file_hash,
            'doc_id': document_record.get('doc_id', ''),
            'chunks_count': len(chunk_records),
            'chapters_count': len(chapter_records),
            'articles_count': len(article_records),
            'sections_count': len(section_records),
            'processed_at': str(np.datetime64('now')),
            'schema_version': CHUNK_SCHEMA_VERSION
        }
        self._save_registry()

        print(f"    Saved normalized schema v4 bundle to {doc_dir}")
        print(f"      • document: {document_file}")
        print(f"      • chapters: {chapters_file} ({len(chapter_records)})")
        print(f"      • articles: {articles_file} ({len(article_records)})")
        print(f"      • sections: {sections_file} ({len(section_records)})")
        print(f"      • chunks: {chunks_file} ({len(chunk_records)})")

    def _load_json_file(self, file_path: str, default):
        if not os.path.exists(file_path):
            return default
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_document_bundle(self, filename: str) -> Dict:
        doc_dir = self._doc_dir(filename)
        document_file = os.path.join(doc_dir, DOCUMENT_METADATA_FILENAME)
        if not os.path.exists(document_file):
            return {}

        return {
            'document': self._load_json_file(document_file, {}),
            'chapters': self._load_json_file(os.path.join(doc_dir, CHAPTERS_FILENAME), []),
            'articles': self._load_json_file(os.path.join(doc_dir, ARTICLES_FILENAME), []),
            'sections': self._load_json_file(os.path.join(doc_dir, SECTIONS_FILENAME), []),
            'chunks': self._load_json_file(os.path.join(doc_dir, CHUNKS_FILENAME), [])
        }

    def load_chunks(self, filename: str) -> List[Dict]:
        bundle = self.load_document_bundle(filename)
        if bundle:
            document_record = dict(bundle.get('document', {}) or {})
            chapter_map = {record.get('chapter_id', ''): record for record in bundle.get('chapters', [])}
            article_map = {record.get('article_id', ''): record for record in bundle.get('articles', [])}
            section_map = {record.get('section_id', ''): record for record in bundle.get('sections', [])}
            enriched_chunks = []

            for chunk_record in bundle.get('chunks', []):
                chunk_metadata = dict(chunk_record.get('metadata', {}) or {})
                chunk_doc_id = chunk_record.get('doc_id') or document_record.get('doc_id', '')
                chapter_id = chunk_record.get('chapter_id', '')
                article_id = chunk_record.get('article_id', '')
                section_id = chunk_record.get('section_id', '')

                enriched_metadata = {
                    **document_record,
                    **chunk_metadata,
                    'doc_id': chunk_doc_id,
                    'chapter_id': chapter_id,
                    'article_id': article_id,
                    'section_id': section_id,
                    'article_key': chunk_metadata.get('article', '')
                }

                raw_text = chunk_record.get('text_raw', chunk_record.get('text', ''))
                contextualized_text = chunk_record.get('text_contextualized', raw_text)

                parent_chapter_text = contextualized_text
                if chapter_id and chapter_id in chapter_map:
                    chapter_record = chapter_map[chapter_id]
                    chapter_metadata = {
                        **document_record,
                        'chapter': chapter_record.get('chapter', ''),
                        'article': '',
                        'article_title': '',
                        'section': '',
                        'point': '',
                        'level': 0,
                        'page': chapter_record.get('page_start', 0),
                        'hierarchical_path': chapter_record.get('chapter', '')
                    }
                    parent_chapter_text = build_contextualized_text(
                        chapter_record.get('chapter_text', raw_text),
                        chapter_metadata
                    )

                parent_section_text = contextualized_text
                if section_id and section_id in section_map:
                    section_record = section_map[section_id]
                    section_metadata = {
                        **document_record,
                        'chapter': section_record.get('chapter', ''),
                        'article': section_record.get('article', ''),
                        'article_title': section_record.get('article_title', ''),
                        'section': section_record.get('section', ''),
                        'point': '',
                        'level': 2,
                        'page': section_record.get('page_start', 0),
                        'hierarchical_path': build_hierarchical_path(
                            section_record.get('article', ''),
                            section_record.get('article_title', ''),
                            section_record.get('section', '')
                        )
                    }
                    parent_section_text = build_contextualized_text(section_record.get('section_text', raw_text), section_metadata)

                parent_article_text = contextualized_text
                if article_id and article_id in article_map:
                    article_record = article_map[article_id]
                    article_metadata = {
                        **document_record,
                        'chapter': article_record.get('chapter', ''),
                        'article': article_record.get('article', ''),
                        'article_title': article_record.get('article_title', ''),
                        'section': '',
                        'point': '',
                        'level': 1,
                        'page': article_record.get('page_start', 0),
                        'hierarchical_path': build_hierarchical_path(
                            article_record.get('article', ''),
                            article_record.get('article_title', '')
                        )
                    }
                    parent_article_text = build_contextualized_text(article_record.get('article_text', raw_text), article_metadata)

                enriched_chunks.append({
                    'chunk_id': chunk_record.get('chunk_id', ''),
                    'text': raw_text,
                    'text_raw': raw_text,
                    'text_contextualized': contextualized_text,
                    'parent_chapter_text': parent_chapter_text,
                    'parent_section_text': parent_section_text,
                    'parent_article_text': parent_article_text,
                    'metadata': enriched_metadata
                })

            return enriched_chunks

        # Legacy schema fallback
        doc_dir = self._doc_dir(filename)
        chunks_file = os.path.join(doc_dir, CHUNKS_FILENAME)
        if not os.path.exists(chunks_file):
            return []
        with open(chunks_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_faiss_index(self, index, filename: str = 'faiss_index.bin'):
        index_file = os.path.join(self.output_dir, filename)
        faiss.write_index(index, index_file)
        print(f"    Saved FAISS index to {index_file}")
    
    def load_faiss_index(self, filename: str = 'faiss_index.bin'):
        index_file = os.path.join(self.output_dir, filename)
        if not os.path.exists(index_file):
            return None
        return faiss.read_index(index_file)


# ==============================================================================
# ULTIMATE DOCUMENT PROCESSOR
# ==============================================================================

class UltimateDocumentProcessor:
    """Process documents with COMPLETE metadata"""
    
    def __init__(self, embedder, output_dir: str, max_chunk_size: int = 300, min_chunk_size: int = 50):
        self.embedder = embedder
        self.output_dir = output_dir
        self.chunker = UnifiedChunker(max_chunk_size, min_chunk_size)
        self.persistence = PersistenceManager(output_dir)
        self.metadata_extractor = MetadataExtractor()
        self.documents_reprocessed = False
    
    def process_file(self, file_path: str, base_folder: str, force: bool = False) -> List[Dict]:
        """Process single PDF file with complete metadata"""
        filename = os.path.basename(file_path)
        
        # Check if should process
        if not self.persistence.should_process(file_path, force):
            print(f"⏭  Skip: {filename} (already processed)")
            return self.persistence.load_chunks(filename)
        
        print(f"⏳ Processing: {filename}")
        
        # Extract file-level metadata (from DocumentProcessor)
        file_metadata = self.metadata_extractor.extract_from_path(file_path, base_folder)
        
        # Read PDF
        full_text = ""
        article_page_map = {}
        
        try:
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                full_text += text
                
                # Map articles to pages
                matches = re.findall(r'(Điều \d+)', text)
                for m in matches:
                    if m not in article_page_map:
                        article_page_map[m] = page_num + 1
            
            doc.close()
        except Exception as e:
            print(f"    Error reading PDF: {e}")
            return []
        
        # Chunk with content metadata (from UnifiedChunker)
        base_chunks, article_full_text_map = self.chunker.chunk_document(full_text, article_page_map)

        doc_id = build_doc_id(filename)

        section_text_map = {}
        article_text_map = {}

        for chunk in base_chunks:
            content_metadata = dict(chunk['content_metadata'])
            article_key = content_metadata.get('article_key', content_metadata.get('article', ''))
            article_text_map[article_key] = article_full_text_map.get(article_key, {}).get('content', chunk['text'])

            section = content_metadata.get('section', '')
            if section:
                section_key = (article_key, section)
                section_text_map.setdefault(section_key, []).append(chunk['text'])

        complete_chunks = []
        for chunk in base_chunks:
            content_metadata = dict(chunk['content_metadata'])
            article_key = content_metadata.get('article_key', content_metadata.get('article', ''))
            section = content_metadata.get('section', '')

            complete_metadata = {
                'doc_id': doc_id,
                **file_metadata,
                **content_metadata,
                'article_key': article_key,
                'source_path': str(Path(file_path).relative_to(base_folder)).replace("\\", "/")
            }

            decision_reference = compose_decision_reference(complete_metadata)
            if decision_reference:
                complete_metadata['decision_reference'] = decision_reference

            contextualized_text = build_contextualized_text(chunk['text'], complete_metadata)

            if section:
                section_key = (article_key, section)
                section_text = "\n".join(section_text_map.get(section_key, [chunk['text']]))
                section_metadata = {
                    **file_metadata,
                    **scope_metadata(content_metadata, "section"),
                    'doc_id': doc_id
                }
                parent_section_text = build_contextualized_text(section_text, section_metadata)
            else:
                parent_section_text = contextualized_text

            article_text = article_text_map.get(article_key, chunk['text'])
            article_metadata = {
                **file_metadata,
                **scope_metadata(content_metadata, "article"),
                'doc_id': doc_id
            }
            parent_article_text = build_contextualized_text(article_text, article_metadata)

            complete_chunks.append({
                'chunk_id': f"{doc_id}::chunk::{len(complete_chunks):04d}",
                'text': chunk['text'],
                'text_raw': chunk['text'],
                'text_contextualized': contextualized_text,
                'parent_section_text': parent_section_text,
                'parent_article_text': parent_article_text,
                'metadata': complete_metadata
            })
        
        file_hash = self.persistence._calculate_file_hash(file_path)
        self.persistence.save_chunks(
            filename=filename,
            chunks=complete_chunks,
            file_hash=file_hash
        )
        self.documents_reprocessed = True
        
        print(f"    Processed: {len(complete_chunks)} chunks")
        
        return self.persistence.load_chunks(filename)
    
    def process_folder(self, folder_path: str, pattern: str = "*.pdf", 
                      recursive: bool = True, force: bool = False) -> Dict[str, List[Dict]]:
        """Process all PDFs in folder"""
        path_obj = Path(folder_path)
        self.documents_reprocessed = False
        
        if recursive:
            pdf_files = list(path_obj.rglob(pattern))
        else:
            pdf_files = list(path_obj.glob(pattern))
        
        print(f"\n Found {len(pdf_files)} PDF files")
        
        all_chunks = {}
        for pdf_file in pdf_files:
            chunks = self.process_file(str(pdf_file), folder_path, force)
            all_chunks[pdf_file.name] = chunks
        
        return all_chunks


# ==============================================================================
# STEP 1: INITIALIZE PROCESSOR
# ==============================================================================

print("\n" + "="*70)
print(" STEP 1: Initialize Ultimate Document Processor")
print("="*70)

# Kiểm tra embedder từ Cell 3
if 'embedder' not in globals():
    print(" LỖI: Biến 'embedder' chưa được định nghĩa!")
    print(" Hãy chạy CELL 3 trước để load models")
    raise NameError("Please run CELL 3 first to load embedder model")

processor = UltimateDocumentProcessor(
    embedder=embedder,
    output_dir=OUTPUT_DIR,
    max_chunk_size=MAX_CHUNK_SIZE,
    min_chunk_size=MIN_CHUNK_SIZE
)

print(" Ultimate Document Processor initialized")

# ==============================================================================
# STEP 2: PROCESS DOCUMENTS
# ==============================================================================

print("\n" + "="*70)
print(" STEP 2: Process Documents with Complete Metadata")
print("="*70)

if not os.path.exists(FOLDER_PATH):
    print(f" Error: Folder not found: {FOLDER_PATH}")
else:
    print(f" Folder found: {FOLDER_PATH}")
    
    all_chunks_dict = processor.process_folder(
        folder_path=FOLDER_PATH,
        pattern=PATTERN,
        recursive=RECURSIVE,
        force=FORCE_REPROCESS  # Fixed: force instead of force_reprocess
    )
    
    print(f"\n Processed {len(all_chunks_dict)} documents")

# ==============================================================================
# STEP 3: LOAD ALL CHUNKS
# ==============================================================================

print("\n" + "="*70)
print(" STEP 3: Load All Chunks")
print("="*70)

# Load all chunks from registry
all_chunks = []
for filename in processor.persistence.registry.keys():
    chunks = processor.persistence.load_chunks(filename)
    all_chunks.extend(chunks)

print(f"Loaded {len(all_chunks)} chunks from {len(processor.persistence.registry)} documents")

if len(all_chunks) == 0:
    print("\n" + "="*70)
    print("CANH BAO: KHONG CO DOCUMENTS NAO!")
    print("="*70)
    print("\nVui long:")
    print("1. Dat cac file PDF/TXT vao thu muc: documents/")
    print("2. Chay lai chatbot")
    print("\nHe thong se tiep tuc nhung khong the tra loi cau hoi.")
    print("="*70)

# ==============================================================================
# STEP 4: CREATE EMBEDDINGS & FAISS INDEX
# ==============================================================================

print("\n" + "="*70)
print(" STEP 4: Create Embeddings & FAISS Index")
print("="*70)

# Try to load existing index
faiss_index_b = processor.persistence.load_faiss_index(FAISS_INDEX_FILENAME)

if faiss_index_b is None or FORCE_REPROCESS or processor.documents_reprocessed:
    print("Creating embeddings...")
    
    vectors = []
    for chunk in tqdm(all_chunks, desc="Embedding"):
        embedding_text = chunk.get('text_contextualized') or chunk.get('text') or ""
        vec = embedder.encode(embedding_text, convert_to_tensor=True).cpu().numpy()
        vectors.append(vec)
    
    if len(vectors) == 0:
        print("CANH BAO: Khong co chunks nao de embedding!")
        print("Vui long dat cac file PDF/TXT vao thu muc documents/")
        # Tao FAISS index rong
        faiss_index_b = faiss.IndexFlatIP(768)  # Default dimension cho BGE-M3
    else:
        vectors = np.array(vectors, dtype='float32')
        
        print(f"Created {len(vectors)} embeddings")
        
        # Create FAISS index
        print("Building FAISS index...")
        faiss_index_b = faiss.IndexFlatIP(vectors.shape[1])
        faiss.normalize_L2(vectors)
        faiss_index_b.add(vectors)
        
        print(f"FAISS index created: {faiss_index_b.ntotal} vectors")
    
    # Save
    processor.persistence.save_faiss_index(faiss_index_b, FAISS_INDEX_FILENAME)
else:
    print(f"Loaded FAISS index from cache: {faiss_index_b.ntotal} vectors")

# ==============================================================================
# STEP 5: CREATE BM25 INDEX
# ==============================================================================

print("\n" + "="*70)
print("STEP 5: Create BM25 Index")
print("="*70)

if len(all_chunks) == 0:
    print("CANH BAO: Khong co chunks, tao BM25 index rong")
    tokenized_chunks_raw = []
    tokenized_chunks_contextualized = []
    # Tao BM25 voi 1 document gia de tranh loi division by zero
    bm25_index = BM25Okapi([["placeholder"]])
    bm25_index_raw = bm25_index
    bm25_index_contextualized = bm25_index
else:
    print("Creating BM25 index...")
    tokenized_chunks_raw = [tokenize_for_bm25(chunk.get('text_raw') or chunk.get('text') or "") for chunk in all_chunks]
    tokenized_chunks_contextualized = [
        tokenize_for_bm25(chunk.get('text_contextualized') or chunk.get('text') or "")
        for chunk in all_chunks
    ]
    bm25_index_raw = BM25Okapi(tokenized_chunks_raw)
    bm25_index_contextualized = BM25Okapi(tokenized_chunks_contextualized)
    bm25_index = bm25_index_contextualized
    print(f"BM25 raw index created with {len(tokenized_chunks_raw)} documents")
    print(f"BM25 contextualized index created with {len(tokenized_chunks_contextualized)} documents")


# ==============================================================================
# STEP 6: PREPARE VARIABLES FOR CELL 5+
# ==============================================================================

print("\n" + "="*70)
print(" STEP 6: Prepare Variables for Cell 5+")
print("="*70)

# Convert to Chunk objects for compatibility
from dataclasses import dataclass as dc

@dc
class Chunk:
    text: str
    metadata: Dict
    text_raw: str = ""
    text_contextualized: str = ""
    parent_section_text: str = ""
    parent_article_text: str = ""
    chunk_id: str = ""

chunks = []
for i, chunk_dict in enumerate(all_chunks):
    chunk_obj = Chunk(
        text=chunk_dict.get('text_raw', chunk_dict.get('text', '')),
        metadata=chunk_dict['metadata'],
        text_raw=chunk_dict.get('text_raw', chunk_dict.get('text', '')),
        text_contextualized=chunk_dict.get('text_contextualized', chunk_dict.get('text', '')),
        parent_section_text=chunk_dict.get('parent_section_text', chunk_dict.get('text_contextualized', chunk_dict.get('text', ''))),
        parent_article_text=chunk_dict.get('parent_article_text', chunk_dict.get('text_contextualized', chunk_dict.get('text', ''))),
        chunk_id=f"chunk_{i}"
    )
    chunks.append(chunk_obj)

# Create article full text map
article_full_text_map = {}
for chunk in chunks:
    article = chunk.metadata.get('article_id') or chunk.metadata.get('article_key', chunk.metadata.get('article', chunk.metadata.get('filename', 'Unknown')))
    
    if article not in article_full_text_map:
        article_full_text_map[article] = []
    
    article_full_text_map[article].append(chunk.text_raw or chunk.text)

# Merge
for article, texts in article_full_text_map.items():
    article_full_text_map[article] = "\n\n".join(texts)

# Create faiss_index_a (same as b for compatibility)
faiss_index_a = faiss_index_b

print(f"\n Exported variables:")
print(f"   • chunks: {len(chunks)} Chunk objects")
print(f"   • faiss_index_a: {faiss_index_a.ntotal} vectors")
print(f"   • faiss_index_b: {faiss_index_b.ntotal} vectors")
print(f"   • bm25_index_raw: {len(tokenized_chunks_raw)} documents")
print(f"   • bm25_index_contextualized: {len(tokenized_chunks_contextualized)} documents")
print(f"   • bm25_index: {len(tokenized_chunks_contextualized)} documents")
print(f"   • article_full_text_map: {len(article_full_text_map)} articles")

# ==============================================================================
# STEP 7: SAMPLE OUTPUT & COMPLETE METADATA DISPLAY
# ==============================================================================

print("\n" + "="*70)
print(" STEP 7: Sample Chunks with Runtime Metadata (Legacy Schema)")
print("="*70)

if len(chunks) > 0:
    print(f"\n Showing first 2 chunks in legacy full-chunk format:\n")
    
    for i, chunk in enumerate(chunks[:2]):
        print(f"{'='*70}")
        print(f"Chunk #{i+1}")
        print(f"{'='*70}")
        print(f" Text: {chunk.text[:120]}...")
        print(f"\n RUNTIME-ENRICHED METADATA:")
        print(f"\n     DOCUMENT-LEVEL:")
        print(f"      • doc_id            : {chunk.metadata.get('doc_id', 'N/A')}")
        print(f"      • filename          : {chunk.metadata.get('filename', 'N/A')}")
        print(f"      • category          : {chunk.metadata.get('category', 'N/A')}")
        print(f"      • parent_folder     : {chunk.metadata.get('parent_folder', 'N/A')}")
        print(f"      • folder_path       : {chunk.metadata.get('folder_path', 'N/A')}")
        print(f"      • year              : {chunk.metadata.get('year', 'N/A')}")
        print(f"      • doc_type          : {chunk.metadata.get('doc_type', 'N/A')}")
        print(f"      • extension         : {chunk.metadata.get('extension', 'N/A')}")
        print(f"      • decision_number   : {chunk.metadata.get('decision_number', 'N/A')}")
        print(f"      • decision_code     : {chunk.metadata.get('decision_code', 'N/A')}")
        print(f"      • document_title    : {chunk.metadata.get('document_title', 'N/A')}")
        print(f"\n    CHUNK-LEVEL:")
        print(f"      • chapter_id        : {chunk.metadata.get('chapter_id', 'N/A')}")
        print(f"      • article_id        : {chunk.metadata.get('article_id', 'N/A')}")
        print(f"      • section_id        : {chunk.metadata.get('section_id', 'N/A')}")
        print(f"      • chapter           : {chunk.metadata.get('chapter', 'N/A')}")
        print(f"      • article           : {chunk.metadata.get('article', 'N/A')}")
        print(f"      • article_title     : {chunk.metadata.get('article_title', 'N/A')}")
        print(f"      • section           : {chunk.metadata.get('section', 'N/A')}")
        print(f"      • point             : {chunk.metadata.get('point', 'N/A')}")
        print(f"      • level             : {chunk.metadata.get('level', 'N/A')}")
        print(f"      • page              : {chunk.metadata.get('page', 'N/A')}")
        print(f"      • hierarchical_path : {chunk.metadata.get('hierarchical_path', 'N/A')}")
        print()


# ==============================================================================
# STEP 8: METADATA STATISTICS
# ==============================================================================

print("\n" + "="*70)
print(" STEP 8: Complete Metadata Statistics")
print("="*70)

# Statistics
doc_types = {}
years = {}
categories = {}
levels = {}
chapters = {}

for chunk in chunks:
    # Doc type
    doc_type = chunk.metadata.get('doc_type', 'unknown')
    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    # Year
    year = chunk.metadata.get('year', 'unknown')
    years[year] = years.get(year, 0) + 1
    
    # Category
    category = chunk.metadata.get('category', 'unknown')
    categories[category] = categories.get(category, 0) + 1
    
    # Level
    level = chunk.metadata.get('level', 'unknown')
    levels[level] = levels.get(level, 0) + 1
    
    # Chapter
    chapter = chunk.metadata.get('chapter', 'unknown')
    chapters[chapter] = chapters.get(chapter, 0) + 1

print("\n Chunks by Document Type (from filename):")
for doc_type, count in sorted(doc_types.items(), key=lambda x: x[1], reverse=True):
    print(f"   • {doc_type:15s}: {count:4d} chunks")

print("\n Chunks by Year (from filename):")
def _year_sort_key(item):
    year = item[0]
    if isinstance(year, int):
        return (0, year)
    if isinstance(year, str) and year.isdigit():
        return (0, int(year))
    return (1, str(year or "unknown"))


for year, count in sorted(years.items(), key=_year_sort_key):
    print(f"   • {str(year):15s}: {count:4d} chunks")

print("\n Chunks by Category (from folder):")
for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"   • {category:30s}: {count:4d} chunks")

print("\n Chunks by Level (from content structure):")
for level in sorted(levels.keys()):
    level_name = {1: "Điều", 2: "Khoản", 3: "Điểm"}.get(level, "Unknown")
    print(f"   • Level {level} ({level_name:5s}): {levels[level]:4d} chunks")

print(f"\n Chunks by Chapter (from content): {len(chapters)} chapters")
for chapter, count in sorted(chapters.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"   • {chapter:30s}: {count:4d} chunks")

# ==============================================================================
# COMPLETION
# ==============================================================================

print("\n" + "="*70)
print(" CELL 4 ULTIMATE COMPLETE - READY FOR RETRIEVAL!")
print("="*70)

print("\n Available Variables:")
print("   • chunks: List of Chunk objects with legacy full metadata")
print("   • faiss_index_a: FAISS index (structural)")
print("   • faiss_index_b: FAISS index (contextualized chunks)")
print("   • bm25_index_raw: BM25 index for raw chunk text")
print("   • bm25_index_contextualized: BM25 index for contextualized chunk text")
print("   • bm25_index: Default BM25 alias (contextualized)")
print("   • article_full_text_map: Dict mapping article_id -> full text")
print("   • processor: UltimateDocumentProcessor instance")

print("\n Legacy on-disk layout:")
print("   • chunks.json    - Full chunk records with duplicated metadata")
print("\n Runtime chunk metadata:")
print("   • document-level + content-level metadata already stored inside each chunk")

print("\n Benefits:")
print("    Legacy compatibility - Matches the earlier chunk format")
print("    Simple storage - Everything kept in chunks.json")
print("    File-level metadata - Stored directly in each chunk")
print("    Content-level metadata - Stored directly in each chunk")
print("    Hierarchical structure - Điều > Khoản > Điểm context preserved")
print("    Persistence - Fast reload (10 sec)")
print("    Change detection - Only process new/changed files")
print("    Compatible - Works with Cell 5+ directly")

print("\n Next: Run Cell 5 for retrieval with COMPLETE metadata!")
print("="*70)
