"""Семантическая разбивка текста на чанки."""

import re
import uuid
from datetime import datetime
from typing import Any

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = structlog.get_logger(__name__)


class SemanticChunker:
    """Семантическая разбивка текста на чанки с современными техниками."""

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 100,
        similarity_threshold: float = 0.7,
        overlap_sentences: int = 2,
        window_size: int = 3,  # Размер окна для анализа связности
        adaptive_threshold: bool = True,  # Адаптивный порог
        hierarchical_chunking: bool = True  # Иерархическая разбивка для больших документов
    ):
        self.embedding_model = embedding_model
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.overlap_sentences = overlap_sentences
        self.window_size = window_size
        self.adaptive_threshold = adaptive_threshold
        self.hierarchical_chunking = hierarchical_chunking

    def _protect_urls_and_emails(self, text: str) -> tuple[str, dict[str, str]]:
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

    def _restore_protected_entities(self, text: str, protected_entities: dict[str, str]) -> str:
        """Восстановить экранированные URL и email."""
        for placeholder, original in protected_entities.items():
            text = text.replace(placeholder, original)
        return text

    def _split_into_sentences(self, text: str) -> list[str]:
        """Разбить текст на предложения с улучшенной логикой."""
        # Используем более простые паттерны без lookbehind переменной длины
        sentences = []
        
        # Сначала разбиваем по абзацам
        paragraphs = re.split(r'\n\s*\n', text)
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            # Разбиваем каждый абзац на предложения
            # Паттерн для предложений с фиксированным lookbehind
            sentence_pattern = r'(?<=[.!?])\s+(?=[A-ZА-Я])'
            para_sentences = re.split(sentence_pattern, paragraph)
            
            # Дополнительная разбивка по структурным элементам
            final_sentences = []
            for sent in para_sentences:
                sent = sent.strip()
                if not sent:
                    continue
                    
                # Разбиваем по спискам и заголовкам
                if '\n' in sent:
                    # Разбиваем по переносам строк для заголовков и списков
                    lines = sent.split('\n')
                    current_sentence = ""
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            if current_sentence:
                                final_sentences.append(current_sentence)
                                current_sentence = ""
                            continue
                            
                        # Проверяем на заголовки (начинаются с заглавной буквы и короткие)
                        if (re.match(r'^[A-ZА-Я]', line) and 
                            len(line) < 100 and 
                            not line.endswith(('.', '!', '?'))):
                            if current_sentence:
                                final_sentences.append(current_sentence)
                                current_sentence = ""
                            final_sentences.append(line)
                        # Проверяем на элементы списков
                        elif re.match(r'^[-•*]\s+', line) or re.match(r'^\d+\.\s+', line):
                            if current_sentence:
                                final_sentences.append(current_sentence)
                                current_sentence = ""
                            final_sentences.append(line)
                        else:
                            # Обычная строка - добавляем к текущему предложению
                            if current_sentence:
                                current_sentence += " " + line
                            else:
                                current_sentence = line
                    
                    if current_sentence:
                        final_sentences.append(current_sentence)
                else:
                    final_sentences.append(sent)
            
            sentences.extend(final_sentences)

        # Очищаем и фильтруем предложения
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:  # Уменьшил минимальную длину
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _calculate_semantic_similarity(self, sentences: list[str]) -> np.ndarray:
        """Вычислить семантическое сходство между предложениями."""
        if len(sentences) < 2:
            return np.array([[1.0]])

        # Создаем эмбеддинги для предложений батчами для эффективности
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch, 
                convert_to_tensor=False,
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings)
        
        # Вычисляем косинусное сходство
        similarity_matrix = cosine_similarity(embeddings)

        return similarity_matrix

    def _calculate_adaptive_threshold(self, similarity_scores: list[float]) -> float:
        """Вычислить адаптивный порог на основе распределения сходства."""
        if not similarity_scores:
            return self.similarity_threshold
            
        scores_array = np.array(similarity_scores)
        
        # Используем перцентиль + статистику для адаптивного порога
        percentile_25 = np.percentile(scores_array, 25)
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        
        # Адаптивный порог: между 25-м перцентилем и средним минус стандартное отклонение
        adaptive_threshold = max(
            percentile_25,
            mean_score - 0.5 * std_score,
            0.3  # Минимальный порог
        )
        
        logger.debug("Адаптивный порог",
                    original_threshold=self.similarity_threshold,
                    adaptive_threshold=adaptive_threshold,
                    mean_similarity=mean_score,
                    std_similarity=std_score)
        
        return min(adaptive_threshold, self.similarity_threshold)

    def _find_semantic_boundaries_with_window(
        self, 
        sentences: list[str], 
        similarity_matrix: np.ndarray
    ) -> list[int]:
        """Найти границы семантических блоков с использованием скользящего окна."""
        boundaries = [0]  # Начало всегда граница
        
        if len(sentences) < 2:
            boundaries.append(len(sentences))
            return boundaries

        # Собираем все значения сходства для адаптивного порога
        similarity_scores = []
        for i in range(1, len(sentences)):
            if i < len(similarity_matrix) and i-1 < len(similarity_matrix):
                similarity_scores.append(similarity_matrix[i-1, i])

        # Определяем порог
        threshold = (
            self._calculate_adaptive_threshold(similarity_scores) 
            if self.adaptive_threshold 
            else self.similarity_threshold
        )

        # Анализируем с помощью скользящего окна
        for i in range(self.window_size, len(sentences) - self.window_size):
            # Анализируем связность в окне
            window_similarities = []
            
            # Сходство в пределах окна
            for j in range(max(0, i - self.window_size), min(len(sentences), i + self.window_size + 1)):
                if j != i and j < len(similarity_matrix) and i < len(similarity_matrix[j]):
                    window_similarities.append(similarity_matrix[j, i])
            
            if not window_similarities:
                continue
                
            avg_window_similarity = np.mean(window_similarities)
            
            # Сходство с предыдущим предложением
            prev_similarity = similarity_matrix[i-1, i] if i > 0 else 1.0
            
            # Критерий разрыва: низкое сходство с предыдущим И низкое среднее в окне
            if (prev_similarity < threshold and 
                avg_window_similarity < threshold * 1.1):  # Немного более мягкий критерий для окна
                
                boundaries.append(i)
                logger.debug("Найдена семантическая граница (окно)",
                           position=i,
                           prev_similarity=prev_similarity,
                           window_similarity=avg_window_similarity,
                           threshold=threshold,
                           sentence_preview=sentences[i][:100])

        boundaries.append(len(sentences))  # Конец всегда граница
        return boundaries

    def _hierarchical_split(self, text: str) -> list[str]:
        """Иерархическая разбивка для очень больших документов."""
        if len(text) <= self.max_chunk_size * 3:  # Не слишком большой
            return [text]
            
        # Сначала разбиваем по крупным структурным элементам
        major_delimiters = [
            r'\n#{1,3}\s+.+\n',  # Заголовки
            r'\n\n={3,}\n',      # Разделители
            r'\n\n-{3,}\n',      # Другие разделители
            r'\n\n\*{3,}\n',     # Звездочки
        ]
        
        parts = [text]
        for delimiter in major_delimiters:
            new_parts = []
            for part in parts:
                split_parts = re.split(delimiter, part)
                new_parts.extend([p for p in split_parts if p.strip()])
            parts = new_parts
            
        # Фильтруем слишком маленькие части
        filtered_parts = []
        current_part = ""
        
        for part in parts:
            if len(current_part) + len(part) <= self.max_chunk_size * 2:
                current_part = (current_part + "\n\n" + part).strip()
            else:
                if current_part:
                    filtered_parts.append(current_part)
                current_part = part
                
        if current_part:
            filtered_parts.append(current_part)
            
        logger.debug("Иерархическая разбивка",
                    original_length=len(text),
                    parts_count=len(filtered_parts),
                    avg_part_size=sum(len(p) for p in filtered_parts) // len(filtered_parts) if filtered_parts else 0)
        
        return filtered_parts

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """Объединить слишком маленькие чанки с улучшенной логикой."""
        if not chunks:
            return []
            
        merged_chunks = []
        current_chunk = ""

        for chunk in chunks:
            # Проверяем, можем ли объединить
            combined_length = len(current_chunk) + len(chunk)
            
            if combined_length <= self.max_chunk_size:
                # Объединяем с умным разделителем
                separator = "\n\n" if current_chunk and not current_chunk.endswith('\n') else " "
                current_chunk = (current_chunk + separator + chunk).strip()
            else:
                # Сохраняем текущий чанк если он не пустой
                if current_chunk:
                    merged_chunks.append(current_chunk)
                current_chunk = chunk

        # Добавляем последний чанк
        if current_chunk:
            merged_chunks.append(current_chunk)

        # Финальная проверка: объединяем очень маленькие последние чанки
        if (len(merged_chunks) > 1 and 
            len(merged_chunks[-1]) < self.min_chunk_size and
            len(merged_chunks[-2]) + len(merged_chunks[-1]) <= self.max_chunk_size):
            
            last_chunk = merged_chunks.pop()
            merged_chunks[-1] = merged_chunks[-1] + "\n\n" + last_chunk
            
        return merged_chunks

    def _split_large_chunks(self, chunks: list[str]) -> list[str]:
        """Разделить слишком большие чанки с сохранением семантики."""
        final_chunks = []

        for chunk in chunks:
            if len(chunk) <= self.max_chunk_size:
                final_chunks.append(chunk)
                continue
                
            # Для больших чанков применяем рекурсивную семантическую разбивку
            logger.debug("Разбиваем большой чанк", size=len(chunk))
            
            # Разбиваем на предложения и ищем семантические границы
            sentences = self._split_into_sentences(chunk)
            if len(sentences) <= 1:
                # Если не можем разбить на предложения, делаем принудительное разделение
                words = chunk.split()
                chunk_size_words = self.max_chunk_size // 7  # Примерно 7 символов на слово
                
                for i in range(0, len(words), chunk_size_words):
                    word_chunk = " ".join(words[i:i + chunk_size_words])
                    final_chunks.append(word_chunk)
            else:
                # Семантическая разбивка предложений
                similarity_matrix = self._calculate_semantic_similarity(sentences)
                boundaries = self._find_semantic_boundaries_with_window(sentences, similarity_matrix)
                
                # Создаем чанки из предложений
                for i in range(len(boundaries) - 1):
                    start_idx = boundaries[i]
                    end_idx = boundaries[i + 1]
                    sentence_chunk = " ".join(sentences[start_idx:end_idx])
                    
                    # Если чанк все еще слишком большой, разбиваем принудительно
                    if len(sentence_chunk) > self.max_chunk_size:
                        sub_sentences = sentences[start_idx:end_idx]
                        current_sub_chunk = ""
                        
                        for sentence in sub_sentences:
                            if len(current_sub_chunk) + len(sentence) <= self.max_chunk_size:
                                current_sub_chunk = (current_sub_chunk + " " + sentence).strip()
                            else:
                                if current_sub_chunk:
                                    final_chunks.append(current_sub_chunk)
                                current_sub_chunk = sentence
                                
                        if current_sub_chunk:
                            final_chunks.append(current_sub_chunk)
                    else:
                        final_chunks.append(sentence_chunk)

        return final_chunks

    def _add_semantic_overlap(self, chunks: list[str], sentences: list[str]) -> list[str]:
        """Добавить семантическое перекрытие между чанками."""
        if len(chunks) <= 1 or self.overlap_sentences == 0:
            return chunks

        overlapped_chunks = []
        sentence_to_chunk = {}  # Маппинг предложений к чанкам
        
        # Создаем маппинг предложений к чанкам
        sentence_idx = 0
        for chunk_idx, chunk in enumerate(chunks):
            chunk_sentences = self._split_into_sentences(chunk)
            for sent in chunk_sentences:
                if sentence_idx < len(sentences):
                    sentence_to_chunk[sentence_idx] = chunk_idx
                    sentence_idx += 1

        for i, chunk in enumerate(chunks):
            if i == 0:
                # Первый чанк остается без изменений
                overlapped_chunks.append(chunk)
            else:
                # Ищем предложения для перекрытия из предыдущего чанка
                prev_chunk_sentences = self._split_into_sentences(chunks[i-1])
                
                # Берем последние N предложений для семантического перекрытия
                overlap_count = min(self.overlap_sentences, len(prev_chunk_sentences))
                overlap_sentences = prev_chunk_sentences[-overlap_count:] if overlap_count > 0 else []
                
                if overlap_sentences:
                    overlap_text = " ".join(overlap_sentences)
                    # Умное объединение с учетом контекста
                    separator = "\n\n" if chunk.startswith(chunk.split()[0].upper()) else " "
                    overlapped_chunk = overlap_text + separator + chunk
                    overlapped_chunks.append(overlapped_chunk)
                else:
                    overlapped_chunks.append(chunk)

        return overlapped_chunks

    def _analyze_chunk_quality(self, chunks: list[str]) -> dict[str, Any]:
        """Расширенный анализ качества разбивки на чанки."""
        if not chunks:
            return {"error": "Нет чанков для анализа"}

        sizes = [len(chunk) for chunk in chunks]
        word_counts = [len(chunk.split()) for chunk in chunks]
        
        # Основная статистика
        total_chunks = len(chunks)
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        min_size = min(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 0
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0

        # Анализ проблемных чанков
        small_chunks = [s for s in sizes if s < self.min_chunk_size]
        large_chunks = [s for s in sizes if s > self.max_chunk_size * 1.2]
        
        # Анализ распределения размеров
        size_std = np.std(sizes) if len(sizes) > 1 else 0
        size_variance = np.var(sizes) if len(sizes) > 1 else 0
        
        # Оценка качества (0-100)
        quality_score = 100
        
        # Штрафы за проблемы
        if small_chunks:
            quality_score -= len(small_chunks) * 10
        if large_chunks:
            quality_score -= len(large_chunks) * 15
        if size_std > avg_size * 0.5:  # Высокая вариативность размеров
            quality_score -= 20
            
        quality_score = max(0, quality_score)

        analysis = {
            "total_chunks": total_chunks,
            "size_stats": {
                "average": round(avg_size, 1),
                "min": min_size,
                "max": max_size,
                "std_dev": round(size_std, 1),
                "variance": round(size_variance, 1)
            },
            "word_stats": {
                "average_words": round(avg_words, 1),
                "min_words": min(word_counts) if word_counts else 0,
                "max_words": max(word_counts) if word_counts else 0
            },
            "quality_metrics": {
                "score": round(quality_score, 1),
                "small_chunks_count": len(small_chunks),
                "large_chunks_count": len(large_chunks),
                "size_consistency": "good" if size_std < avg_size * 0.3 else "needs_improvement"
            },
            "distribution": {
                "size_quartiles": {
                    "q1": round(np.percentile(sizes, 25), 1) if sizes else 0,
                    "q2": round(np.percentile(sizes, 50), 1) if sizes else 0,
                    "q3": round(np.percentile(sizes, 75), 1) if sizes else 0
                }
            }
        }

        # Рекомендации
        recommendations = []
        if len(small_chunks) > total_chunks * 0.2:
            recommendations.append("Увеличьте min_chunk_size или уменьшите similarity_threshold")
        if len(large_chunks) > total_chunks * 0.1:
            recommendations.append("Уменьшите max_chunk_size или активируйте hierarchical_chunking")
        if size_std > avg_size * 0.5:
            recommendations.append("Настройте адаптивный порог для более равномерной разбивки")
            
        analysis["recommendations"] = recommendations

        return analysis

    def _extract_key_terms(self, text: str, max_terms: int = 10) -> list[str]:
        """Извлечь ключевые термины из текста."""
        # Удаляем знаки препинания и приводим к нижнему регистру
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Извлекаем слова длиной от 4 символов
        words = re.findall(r'\b\w{4,}\b', clean_text)
        
        # Подсчитываем частоту слов
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Сортируем по частоте и берем топ терминов
        sorted_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        key_terms = [term for term, freq in sorted_terms[:max_terms] if freq > 1]
        
        return key_terms

    def _create_enhanced_metadata(self, content: str, base_metadata: dict[str, Any] | None) -> dict[str, Any]:
        """Создать расширенные метаданные для чанка."""
        metadata = (base_metadata or {}).copy()
        
        # Базовые метрики контента
        metadata.update({
            "content_length": len(content),
            "word_count": len(content.split()),
            "sentence_count": len(self._split_into_sentences(content)),
            "processing_timestamp": datetime.now().isoformat(),
        })
        
        # Ключевые термины
        key_terms = self._extract_key_terms(content)
        metadata["key_terms"] = key_terms
        
        # Анализ типа контента
        content_lower = content.lower()
        content_features = {
            "has_urls": bool(re.search(r'https?://', content)),
            "has_emails": bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)),
            "has_code": bool(re.search(r'[{}()[\]<>]', content)) and content.count('{') > 2,
            "has_numbers": bool(re.search(r'\d+', content)),
            "is_list_like": content.count('\n-') > 2 or content.count('•') > 2,
        }
        metadata["content_features"] = content_features
        
        # Языковые особенности (простая эвристика)
        russian_chars = len(re.findall(r'[а-яё]', content_lower))
        english_chars = len(re.findall(r'[a-z]', content_lower))
        total_letters = russian_chars + english_chars
        
        if total_letters > 0:
            metadata["language_mix"] = {
                "russian_ratio": round(russian_chars / total_letters, 2),
                "english_ratio": round(english_chars / total_letters, 2),
                "primary_language": "russian" if russian_chars > english_chars else "english"
            }
        
        return metadata

    def _create_contextual_text(self, content: str, metadata: dict[str, Any]) -> str:
        """Создать контекстуальный текст для лучшего поиска."""
        contextual_parts = []
        
        # Добавляем название документа/страницы
        title = metadata.get("title")
        if title:
            contextual_parts.append(f"page_title: {title}")
        
        # Добавляем полный URL источника (без обрезания)
        source = metadata.get("source")
        if source:
            contextual_parts.append(f"page_url: {source}")
        
        # Добавляем категорию/тип
        category = metadata.get("category") or metadata.get("type")
        if category:
            contextual_parts.append(f"Категория: {category}")
        
        # Добавляем раздел/главу
        section = metadata.get("section") or metadata.get("chapter")
        if section:
            contextual_parts.append(f"Раздел: {section}")
        
        # Собираем контекстуальный текст
        if contextual_parts:
            context_header = " | ".join(contextual_parts)
            return f"{context_header}\n\n{content}"
        
        return content

    async def chunk_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Разбить текст на семантические чанки с современными техниками."""
        if not text or len(text.strip()) < self.min_chunk_size:
            logger.warning("Текст слишком короткий для разбивки", length=len(text) if text else 0)
            return []

        logger.info("🧠 Начинаем продвинутую семантическую разбивку",
                   text_length=len(text),
                   similarity_threshold=self.similarity_threshold,
                   adaptive_threshold=self.adaptive_threshold,
                   hierarchical_chunking=self.hierarchical_chunking,
                   window_size=self.window_size)

        start_time = datetime.now()

        # Иерархическая разбивка для очень больших документов
        if self.hierarchical_chunking and len(text) > self.max_chunk_size * 3:
            logger.info("Применяем иерархическую разбивку для большого документа")
            text_parts = self._hierarchical_split(text)
            
            all_result_chunks = []
            for part_idx, part in enumerate(text_parts):
                logger.debug(f"Обрабатываем часть {part_idx + 1}/{len(text_parts)}", size=len(part))
                
                # Рекурсивно обрабатываем каждую часть БЕЗ иерархической разбивки
                # чтобы избежать бесконечной рекурсии
                part_chunker = SemanticChunker(
                    embedding_model=self.embedding_model,
                    max_chunk_size=self.max_chunk_size,
                    min_chunk_size=self.min_chunk_size,
                    similarity_threshold=self.similarity_threshold,
                    overlap_sentences=self.overlap_sentences,
                    window_size=self.window_size,
                    adaptive_threshold=self.adaptive_threshold,
                    hierarchical_chunking=False  # ОТКЛЮЧАЕМ для избежания рекурсии
                )
                part_chunks = await part_chunker.chunk_text(part, metadata)
                
                # Обновляем метаданные для частей
                for chunk in part_chunks:
                    chunk["metadata"].update({
                        "hierarchical_part": part_idx,
                        "total_hierarchical_parts": len(text_parts)
                    })
                
                all_result_chunks.extend(part_chunks)
            
            logger.info("✅ Иерархическая разбивка завершена", 
                       total_chunks=len(all_result_chunks),
                       hierarchical_parts=len(text_parts))
            return all_result_chunks

        # Защищаем URL и email
        protected_text, protected_entities = self._protect_urls_and_emails(text)
        logger.debug("Защищены сущности", entities_count=len(protected_entities))

        # Разбиваем на предложения
        sentences = self._split_into_sentences(protected_text)
        logger.debug("Разбито на предложения", sentences_count=len(sentences))

        if len(sentences) <= 1:
            # Если предложений мало, возвращаем весь текст как один чанк
            logger.info("Мало предложений, возвращаем как один чанк")
            restored_text = self._restore_protected_entities(text, protected_entities)
            enhanced_metadata = self._create_enhanced_metadata(restored_text, metadata)
            contextual_text = self._create_contextual_text(restored_text, enhanced_metadata)
            
            return [{
                "content": contextual_text,
                "metadata": enhanced_metadata
            }]

        # Вычисляем семантическое сходство
        similarity_matrix = self._calculate_semantic_similarity(sentences)
        logger.debug("Создана матрица сходства", shape=similarity_matrix.shape)

        # Находим границы семантических блоков с окном
        boundaries = self._find_semantic_boundaries_with_window(sentences, similarity_matrix)
        logger.debug("Найдены семантические границы", boundaries=boundaries, 
                    boundaries_count=len(boundaries)-2)  # -2 так как первая и последняя - это начало и конец

        # Создаем чанки из предложений
        chunks = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)
            chunks.append(chunk_text)

        logger.debug("Созданы предварительные чанки", count=len(chunks))

        # Объединяем маленькие чанки
        chunks = self._merge_small_chunks(chunks)
        logger.debug("После объединения маленьких", count=len(chunks))

        # Разделяем большие чанки
        chunks = self._split_large_chunks(chunks)
        logger.debug("После разделения больших", count=len(chunks))

        # Добавляем семантическое перекрытие
        chunks = self._add_semantic_overlap(chunks, sentences)
        logger.debug("После добавления семантического перекрытия", count=len(chunks))

        # Восстанавливаем защищенные сущности
        chunks = [self._restore_protected_entities(chunk, protected_entities) for chunk in chunks]

        # Анализируем качество
        quality_analysis = self._analyze_chunk_quality(chunks)
        
        processing_time = (datetime.now() - start_time).total_seconds()

        logger.info("✅ Продвинутая семантическая разбивка завершена",
                   chunks_created=len(chunks),
                   processing_time_seconds=round(processing_time, 2),
                   quality_score=quality_analysis.get("quality_metrics", {}).get("score", 0),
                   avg_chunk_size=quality_analysis.get("size_stats", {}).get("average", 0),
                   size_consistency=quality_analysis.get("quality_metrics", {}).get("size_consistency", "unknown"))

        # Логируем рекомендации из анализа качества
        recommendations = quality_analysis.get("recommendations", [])
        if recommendations:
            logger.warning("Рекомендации по улучшению качества чанкинга", recommendations=recommendations)

        # Создаем финальные чанки с метаданными
        result_chunks = []
        for i, chunk in enumerate(chunks):
            # Создаем расширенные метаданные
            enhanced_metadata = self._create_enhanced_metadata(chunk, metadata)
            enhanced_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_type": "semantic_advanced",
                "chunking_config": {
                    "similarity_threshold": self.similarity_threshold,
                    "adaptive_threshold": self.adaptive_threshold,
                    "window_size": self.window_size,
                    "hierarchical_chunking": self.hierarchical_chunking,
                    "overlap_sentences": self.overlap_sentences
                },
                "processing_time": processing_time,
                "quality_analysis": quality_analysis if i == 0 else None,  # Добавляем анализ только к первому чанку
            })

            # Создаем контекстуальный текст
            contextual_text = self._create_contextual_text(chunk, enhanced_metadata)
            
            result_chunks.append({
                "content": contextual_text,
                "metadata": enhanced_metadata
            })

        return result_chunks
