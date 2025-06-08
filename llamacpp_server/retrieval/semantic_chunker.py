"""Семантическая разбивка текста на чанки."""

import re
import uuid
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import structlog
from sklearn.metrics.pairwise import cosine_similarity

logger = structlog.get_logger(__name__)


class SemanticChunker:
    """Семантическая разбивка текста на чанки."""
    
    def __init__(
        self,
        embedding_model: SentenceTransformer,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 100,
        similarity_threshold: float = 0.7,
        overlap_sentences: int = 2
    ):
        self.embedding_model = embedding_model
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.overlap_sentences = overlap_sentences
        
    def _protect_urls_and_emails(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Экранировать URL и email адреса."""
        protected_text = text
        protected_entities = {}
        
        # Паттерны для URL и email
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # Находим и заменяем URL
        for match in re.finditer(url_pattern, text):
            url = match.group()
            placeholder = f"__URL_{uuid.uuid4().hex[:8]}__"
            protected_entities[placeholder] = url
            protected_text = protected_text.replace(url, placeholder, 1)
        
        # Находим и заменяем email
        for match in re.finditer(email_pattern, text):
            email = match.group()
            placeholder = f"__EMAIL_{uuid.uuid4().hex[:8]}__"
            protected_entities[placeholder] = email
            protected_text = protected_text.replace(email, placeholder, 1)
            
        logger.debug("Защищены сущности", 
                    urls_count=len([k for k in protected_entities.keys() if k.startswith("__URL_")]),
                    emails_count=len([k for k in protected_entities.keys() if k.startswith("__EMAIL_")]))
        
        return protected_text, protected_entities
    
    def _restore_protected_entities(self, text: str, protected_entities: Dict[str, str]) -> str:
        """Восстановить экранированные URL и email."""
        for placeholder, original in protected_entities.items():
            text = text.replace(placeholder, original)
        return text
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Разбить текст на предложения."""
        # Улучшенная регулярка для разбиения на предложения
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-ZА-Я])|(?<=\n\n)|(?<=\n(?=[A-ZА-Я]))'
        sentences = re.split(sentence_pattern, text)
        
        # Очищаем и фильтруем предложения
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Минимальная длина предложения
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _calculate_semantic_similarity(self, sentences: List[str]) -> np.ndarray:
        """Вычислить семантическое сходство между предложениями."""
        if len(sentences) < 2:
            return np.array([[1.0]])
        
        # Создаем эмбеддинги для предложений
        embeddings = self.embedding_model.encode(sentences, convert_to_tensor=False)
        embeddings = np.array(embeddings)
        
        # Вычисляем косинусное сходство
        similarity_matrix = cosine_similarity(embeddings)
        
        return similarity_matrix
    
    def _find_semantic_boundaries(self, sentences: List[str], similarity_matrix: np.ndarray) -> List[int]:
        """Найти границы семантических блоков."""
        boundaries = [0]  # Начало всегда граница
        
        for i in range(1, len(sentences)):
            # Сравниваем текущее предложение с предыдущим
            prev_similarity = similarity_matrix[i-1, i] if i > 0 else 1.0
            
            # Если сходство ниже порога - это граница нового блока
            if prev_similarity < self.similarity_threshold:
                boundaries.append(i)
                logger.debug("Найдена семантическая граница", 
                           position=i, 
                           similarity=prev_similarity,
                           sentence_preview=sentences[i][:100])
        
        boundaries.append(len(sentences))  # Конец всегда граница
        return boundaries
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Объединить слишком маленькие чанки."""
        merged_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            if len(current_chunk) + len(chunk) <= self.max_chunk_size:
                current_chunk = (current_chunk + " " + chunk).strip()
            else:
                if current_chunk:
                    merged_chunks.append(current_chunk)
                current_chunk = chunk
        
        if current_chunk:
            merged_chunks.append(current_chunk)
        
        return merged_chunks
    
    def _split_large_chunks(self, chunks: List[str]) -> List[str]:
        """Разделить слишком большие чанки."""
        final_chunks = []
        
        for chunk in chunks:
            if len(chunk) <= self.max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Разбиваем большой чанк на части
                sentences = self._split_into_sentences(chunk)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                        current_chunk = (current_chunk + " " + sentence).strip()
                    else:
                        if current_chunk:
                            final_chunks.append(current_chunk)
                        current_chunk = sentence
                
                if current_chunk:
                    final_chunks.append(current_chunk)
        
        return final_chunks
    
    def _add_sentence_overlap(self, chunks: List[str], sentences: List[str]) -> List[str]:
        """Добавить перекрытие предложений между чанками."""
        if len(chunks) <= 1 or self.overlap_sentences == 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # Первый чанк остается без изменений
                overlapped_chunks.append(chunk)
            else:
                # Находим последние предложения предыдущего чанка
                prev_chunk = chunks[i-1]
                prev_sentences = self._split_into_sentences(prev_chunk)
                
                # Берем последние N предложений для перекрытия
                overlap_sentences = prev_sentences[-self.overlap_sentences:] if len(prev_sentences) >= self.overlap_sentences else prev_sentences
                overlap_text = " ".join(overlap_sentences)
                
                # Добавляем перекрытие к текущему чанку
                overlapped_chunk = overlap_text + " " + chunk
                overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    async def chunk_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Разбить текст на семантические чанки."""
        logger.info("🧠 Начинаем семантическую разбивку текста", 
                   text_length=len(text),
                   max_chunk_size=self.max_chunk_size)
        
        if not text or len(text) < self.min_chunk_size:
            logger.warning("Текст слишком короткий для разбивки", length=len(text))
            return []
        
        # Защищаем URL и email
        protected_text, protected_entities = self._protect_urls_and_emails(text)
        
        # Разбиваем на предложения
        sentences = self._split_into_sentences(protected_text)
        logger.debug("Текст разбит на предложения", count=len(sentences))
        
        if len(sentences) < 2:
            # Если предложений мало, возвращаем как один чанк
            restored_text = self._restore_protected_entities(protected_text, protected_entities)
            return [{
                "content": restored_text,
                "metadata": metadata or {},
                "chunk_type": "single"
            }]
        
        # Вычисляем семантическое сходство
        similarity_matrix = self._calculate_semantic_similarity(sentences)
        
        # Находим границы семантических блоков
        boundaries = self._find_semantic_boundaries(sentences, similarity_matrix)
        logger.debug("Найдены семантические границы", boundaries=boundaries)
        
        # Создаем чанки по границам
        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk_sentences = sentences[start:end]
            chunk_text = " ".join(chunk_sentences)
            chunks.append(chunk_text)
        
        logger.debug("Созданы первичные чанки", count=len(chunks))
        
        # Объединяем маленькие чанки
        chunks = self._merge_small_chunks(chunks)
        logger.debug("После объединения маленьких чанков", count=len(chunks))
        
        # Разделяем большие чанки
        chunks = self._split_large_chunks(chunks)
        logger.debug("После разделения больших чанков", count=len(chunks))
        
        # Добавляем перекрытие предложений
        chunks = self._add_sentence_overlap(chunks, sentences)
        logger.debug("После добавления перекрытия", count=len(chunks))
        
        # Восстанавливаем защищенные сущности и создаем финальные чанки
        final_chunks = []
        for i, chunk in enumerate(chunks):
            restored_chunk = self._restore_protected_entities(chunk, protected_entities)
            
            chunk_metadata = (metadata or {}).copy()
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_type": "semantic",
                "chunk_size": len(restored_chunk)
            })
            
            final_chunks.append({
                "content": restored_chunk,
                "metadata": chunk_metadata
            })
        
        logger.info("✅ Семантическая разбивка завершена", 
                   original_length=len(text),
                   chunks_count=len(final_chunks),
                   avg_chunk_size=sum(len(chunk["content"]) for chunk in final_chunks) // len(final_chunks))
        
        return final_chunks 