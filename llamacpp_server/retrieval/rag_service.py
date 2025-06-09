"""RAG сервис для контекстного поиска."""

from typing import List
import structlog
from pathlib import Path

from .protocols import (
    EmbeddingServiceProtocol, 
    VectorStoreProtocol,
    RAGServiceProtocol
)
from ..config.settings import get_settings
from ..prompts.templates import RAG_TEMPLATES


logger = structlog.get_logger(__name__)


class RAGService:
    """Сервис для Retrieval-Augmented Generation."""
    
    def __init__(
        self,
        embedding_service: EmbeddingServiceProtocol,
        vector_store: VectorStoreProtocol
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        settings = get_settings()
        self._index_path = Path(settings.faiss_index_path)
        self._search_k = settings.rag_search_k
        self._max_context_length = settings.rag_max_context
        
        # Инициализируем компоненты
        self._embedding_model = None
        self._faiss_index = None
        self._documents = []
        
    async def search_relevant_context(
        self, query: str, k: int = 5
    ) -> List[str]:
        """Найти релевантный контекст для запроса."""
        logger.info("🔍 RAG поиск контекста", query_preview=query[:100], k=k)
        
        try:
            # Получаем эмбеддинг запроса
            logger.debug("Создание эмбеддинга для запроса")
            query_embedding = await self._embedding_service.embed_text(query)
            logger.debug("Эмбеддинг создан", embedding_dim=len(query_embedding))
            
            # Ищем релевантные документы
            logger.debug("Поиск в векторном хранилище")
            search_results = await self._vector_store.search(query_embedding, k=k)
            logger.info("Результаты поиска получены", found_docs=len(search_results))
            
            if not search_results:
                logger.warning("❌ Не найдено релевантных документов")
                return []
            
            # Форматируем контекст для модели с обязательными источниками
            context_docs = []
            total_length = 0
            
            # Группируем уникальные источники для финального списка
            unique_sources = set()
            
            for i, result in enumerate(search_results):
                document = result.document
                content = document.content
                score = result.score
                metadata = document.metadata
                
                title = metadata.get('title', 'Без названия')
                source_url = metadata.get('source', '')
                chunk_type = metadata.get('chunk_type', 'unknown')
                chunk_index = metadata.get('chunk_index', 0)
                total_chunks = metadata.get('total_chunks', 1)
                
                # Строгий формат: page_url + page_name + page_content 
                if source_url:
                    unique_sources.add(source_url)
                    formatted_doc = f"page_url: {source_url}\npage_name: {title}\npage_content: \"{content}\"\n"
                else:
                    # Если нет URL, используем fallback
                    formatted_doc = f"page_url: [источник неизвестен]\npage_name: {title}\npage_content: \"{content}\"\n"
                
                logger.debug("Обработка документа для контекста", 
                           doc_id=document.id,
                           title=title,
                           score=score,
                           chunk_type=chunk_type,
                           chunk_index=chunk_index,
                           source_url=source_url,
                           content_len=len(content),
                           formatted_len=len(formatted_doc))
                
                # Проверяем лимит длины
                if total_length + len(formatted_doc) > self._max_context_length:
                    remaining = self._max_context_length - total_length
                    if remaining > 100:  # Минимальный размер
                        # Обрезаем контент, сохраняя формат page_url + page_name + page_content
                        url_part = f"page_url: {source_url}\n" if source_url else "page_url: [источник неизвестен]\n"
                        name_part = f"page_name: {title}\n"
                        content_space = remaining - len(url_part) - len(name_part) - len('page_content: ""\n') - 20
                        
                        if content_space > 50:
                            truncated_content = content[:content_space] + "...[ОБРЕЗАНО]"
                            formatted_doc = f"{url_part}{name_part}page_content: \"{truncated_content}\"\n"
                            
                            context_docs.append(formatted_doc)
                            total_length += len(formatted_doc)
                            logger.debug("Документ обрезан", remaining_chars=remaining, final_size=len(formatted_doc))
                    break
                
                context_docs.append(formatted_doc)
                total_length += len(formatted_doc)
                
                logger.debug("Документ добавлен в контекст",
                           doc_index=i,
                           doc_length=len(formatted_doc),
                           total_length=total_length,
                           score=score)
            
            # Создаем итоговый список источников для логирования
            sources_list = list(unique_sources)
            
            logger.info("✅ RAG контекст найден", 
                       documents=len(context_docs), 
                       total_chars=total_length,
                       unique_sources=len(sources_list),
                       sources=sources_list,
                       scores=[r.score for r in search_results[:len(context_docs)]])
            
            return context_docs
            
        except Exception as e:
            logger.error("❌ Ошибка RAG поиска", error=str(e), exc_info=True)
            return []
    
    async def enhance_prompt_with_context(
        self, original_prompt: str, context: List[str]
    ) -> str:
        """Улучшить промпт контекстной информацией."""
        if not context:
            logger.debug("Нет контекста для улучшения промпта")
            return original_prompt
        
        # Объединяем контекст с разделителями
        context_text = "\n\n".join(context)
        
        # Получаем настройки для выбора стиля промпта
        settings = get_settings()
        prompt_style = settings.rag_prompt_style
        
        # Выбираем шаблон на основе настройки
        if prompt_style == "simple_citation":
            template_key = "simple_citation_prompt"
        else:
            template_key = "citation_focused_prompt"
        
        # Используем выбранный шаблон
        enhanced_prompt = RAG_TEMPLATES[template_key].format(
            context=context_text,
            question=original_prompt
        )
        
        logger.debug("Промпт улучшен контекстом", 
                    original_len=len(original_prompt),
                    enhanced_len=len(enhanced_prompt),
                    context_docs=len(context),
                    prompt_style=prompt_style,
                    template_used=template_key)
        
        # Детальное логирование для диагностики
        logger.debug("🔍 RAG Enhanced Prompt Preview",
                    enhanced_preview=enhanced_prompt[:800] + "..." if len(enhanced_prompt) > 800 else enhanced_prompt)
        
        return enhanced_prompt

    async def search_relevant_context_with_citations(
        self, query: str, k: int = 5
    ) -> List[str]:
        """Найти релевантный контекст для запроса с акцентом на цитирование."""
        logger.info("🔍 RAG поиск контекста с цитированием", query_preview=query[:100], k=k)
        
        try:
            # Получаем эмбеддинг запроса  
            query_embedding = await self._embedding_service.embed_text(query)
            logger.debug("Эмбеддинг создан", embedding_dim=len(query_embedding))
            
            # Ищем релевантные документы
            search_results = await self._vector_store.search(query_embedding, k=k)
            logger.info("Результаты поиска получены", found_docs=len(search_results))
            
            if not search_results:
                logger.warning("❌ Не найдено релевантных документов")
                return []
            
            # Форматируем контекст с особым акцентом на источники
            context_docs = []
            total_length = 0
            unique_sources = set()
            
            for i, result in enumerate(search_results):
                document = result.document
                content = document.content
                score = result.score
                metadata = document.metadata
                
                title = metadata.get('title', 'Без названия')
                source_url = metadata.get('source', '')
                chunk_type = metadata.get('chunk_type', 'unknown')
                
                # Усиленный формат для цитирования
                if source_url:
                    unique_sources.add((title, source_url))
                    formatted_doc = f"""Документ: {title}
URL: {source_url}
Содержание: {content}

"""
                else:
                    formatted_doc = f"""Документ: {title}
URL: [источник неизвестен]
Содержание: {content}

"""
                
                # Проверяем лимит длины
                if total_length + len(formatted_doc) > self._max_context_length:
                    remaining = self._max_context_length - total_length
                    if remaining > 150:  # Минимальный размер для полезного контента
                        # Сохраняем важные части
                        header = f"Документ: {title}\nURL: {source_url}\n"
                        footer = f"\n\n"
                        content_space = remaining - len(header) - len(footer) - len('Содержание: ') - 20
                        
                        if content_space > 80:
                            truncated_content = content[:content_space] + "...[ОБРЕЗАНО]"
                            formatted_doc = f'{header}Содержание: {truncated_content}{footer}'
                            
                            context_docs.append(formatted_doc)
                            total_length += len(formatted_doc)
                            if source_url:
                                unique_sources.add((title, source_url))
                    break
                
                context_docs.append(formatted_doc)
                total_length += len(formatted_doc)
                
            sources_list = [f"{title} - {url}" for title, url in unique_sources]
            
            logger.info("✅ RAG контекст с цитированием готов", 
                       documents=len(context_docs), 
                       total_chars=total_length,
                       unique_sources=len(sources_list),
                       sources=sources_list)
            
            return context_docs
            
        except Exception as e:
            logger.error("❌ Ошибка RAG поиска с цитированием", error=str(e), exc_info=True)
            return []

    async def create_citation_focused_prompt(
        self, original_prompt: str, context: List[str]
    ) -> str:
        """Создать промпт с фокусом на цитирование источников."""
        if not context:
            logger.debug("Нет контекста для citation-focused промпта")
            return original_prompt
        
        # Объединяем контекст
        context_text = "\n".join(context)
        
        # Используем новый шаблон для цитирования
        enhanced_prompt = RAG_TEMPLATES["citation_focused_prompt"].format(
            context=context_text,
            question=original_prompt
        )
        
        logger.debug("Citation-focused промпт создан", 
                    original_len=len(original_prompt),
                    enhanced_len=len(enhanced_prompt),
                    context_docs=len(context),
                    template_used="citation_focused_prompt")
        
        return enhanced_prompt 