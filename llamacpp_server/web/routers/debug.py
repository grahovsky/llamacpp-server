"""Роутер для отладки и диагностики."""

import os
from typing import Dict, Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel

from ...config import get_settings
from ...domain.protocols import LlamaServiceProtocol

router = APIRouter()
logger = structlog.get_logger(__name__)


class GenerationParams(BaseModel):
    """Модель для настройки параметров генерации."""
    temperature: Optional[float] = None
    repeat_penalty: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None


class TestPrompt(BaseModel):
    """Модель для тестирования промптов."""
    prompt: str
    params: Optional[GenerationParams] = None
    stream: bool = False


class DebugInfo(BaseModel):
    """Информация для отладки."""
    message: str
    current_params: Dict[str, Any]
    model_info: Dict[str, Any]
    rag_status: Dict[str, Any]


async def get_llama_service(request: Request) -> LlamaServiceProtocol:
    """Получить LLama сервис из DI контейнера.""" 
    return request.app.state.container.llama_service()


@router.get("/debug/info", response_model=DebugInfo)
async def get_debug_info(
    llama_service: LlamaServiceProtocol = Depends(get_llama_service)
):
    """Получить отладочную информацию о системе."""
    settings = get_settings()
    
    # Информация о модели
    model_info = {
        "model_path": str(settings.model_path),
        "model_exists": settings.model_path.exists() if settings.model_path else False,
        "model_size": os.path.getsize(settings.model_path) if settings.model_path and settings.model_path.exists() else 0,
        "n_ctx": settings.n_ctx,
        "n_gpu_layers": settings.n_gpu_layers,
        "ready": await llama_service.is_ready()
    }
    
    # Текущие параметры генерации
    current_params = {
        "temperature": settings.temperature,
        "repeat_penalty": settings.repeat_penalty,
        "top_p": settings.top_p,
        "top_k": settings.top_k,
        "max_response_tokens": settings.max_response_tokens,
        "max_history_tokens": settings.max_history_tokens
    }
    
    # Статус RAG
    rag_status = {
        "enabled": settings.enable_rag,
        "data_dir": str(settings.data_dir),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "max_retrieved_chunks": settings.max_retrieved_chunks
    }
    
    return DebugInfo(
        message="Система готова к отладке",
        current_params=current_params,
        model_info=model_info,
        rag_status=rag_status
    )


@router.post("/debug/test-prompt")
async def test_prompt(
    test_data: TestPrompt,
    llama_service: LlamaServiceProtocol = Depends(get_llama_service)
):
    """Тестирование промпта с указанными параметрами."""
    try:
        from ...domain.models import ChatCompletionRequest, ChatMessage
        
        # Создаем сообщения
        messages = [ChatMessage(role="user", content=test_data.prompt)]
        
        # Создаем запрос с тестовыми параметрами
        request = ChatCompletionRequest(
            messages=messages,
            model="test-model",
            stream=test_data.stream,
            temperature=test_data.params.temperature if test_data.params else None,
            repeat_penalty=test_data.params.repeat_penalty if test_data.params else None,
            top_p=test_data.params.top_p if test_data.params else None,
            top_k=test_data.params.top_k if test_data.params else None,
            max_tokens=test_data.params.max_tokens if test_data.params else None
        )
        
        logger.info("🧪 Тестирование промпта", 
                   prompt_preview=test_data.prompt[:100],
                   params=test_data.params.dict() if test_data.params else None)
        
        if test_data.stream:
            # Для стриминга возвращаем генератор
            async def stream_response():
                async for chunk in llama_service.chat_completion_stream(request):
                    yield chunk
            return stream_response()
        else:
            # Обычный ответ
            response = await llama_service.chat_completion(request)
            return response
            
    except Exception as e:
        logger.error("❌ Ошибка тестирования промпта", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка тестирования: {str(e)}")


@router.post("/debug/analyze-loop")
async def analyze_loop_potential(
    test_data: TestPrompt
):
    """Анализ потенциала зацикливания для промпта."""
    from ...llama.loop_detector import LoopDetector
    
    try:
        # Создаем детектор для анализа
        detector = LoopDetector(max_repeats=2, min_repeat_length=5)
        
        # Разбиваем промпт на части для анализа
        words = test_data.prompt.split()
        chunk_size = 10
        chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        
        potential_issues = []
        
        # Проверяем каждый chunk
        for i, chunk in enumerate(chunks):
            if detector.add_chunk(chunk):
                potential_issues.append(f"Потенциальное зацикливание в части {i+1}: {chunk[:50]}...")
        
        # Проверяем повторяющиеся фразы
        prompt_lower = test_data.prompt.lower()
        repeat_phrases = [
            "пожалуйста",
            "подождите", 
            "проверить",
            "информация",
            "ответ",
            "вопрос"
        ]
        
        phrase_counts = {}
        for phrase in repeat_phrases:
            count = prompt_lower.count(phrase)
            if count > 1:
                phrase_counts[phrase] = count
        
        stats = detector.get_statistics()
        
        return {
            "prompt_analysis": {
                "length": len(test_data.prompt),
                "word_count": len(words),
                "chunks_analyzed": len(chunks)
            },
            "potential_issues": potential_issues,
            "repeated_phrases": phrase_counts,
            "detector_stats": stats,
            "recommendations": _get_loop_prevention_recommendations(test_data.prompt, phrase_counts)
        }
        
    except Exception as e:
        logger.error("❌ Ошибка анализа зацикливания", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")


@router.post("/debug/emergency-stop")
async def emergency_stop():
    """Экстренная остановка всех генераций (заглушка)."""
    logger.warning("🚨 Запрос экстренной остановки генерации")
    
    # TODO: Реализовать механизм остановки активных генераций
    # Пока что просто логируем
    
    return {
        "message": "Экстренная остановка инициирована",
        "status": "acknowledged",
        "note": "Механизм остановки активных генераций в разработке"
    }


@router.get("/debug/generation-stats")
async def get_generation_stats():
    """Получить статистику генераций."""
    # TODO: Реализовать сбор статистики
    return {
        "total_requests": "unknown",
        "active_streams": "unknown", 
        "loops_detected": "unknown",
        "average_response_time": "unknown",
        "note": "Статистика в разработке"
    }


def _get_loop_prevention_recommendations(prompt: str, phrase_counts: dict) -> list[str]:
    """Получить рекомендации по предотвращению зацикливания."""
    recommendations = []
    
    if len(prompt.split()) < 10:
        recommendations.append("Промпт слишком короткий - добавьте больше контекста")
    
    if phrase_counts:
        recommendations.append(f"Обнаружены повторяющиеся фразы: {list(phrase_counts.keys())}")
        recommendations.append("Рекомендуется увеличить repeat_penalty до 1.15-1.2")
    
    if "подождите" in phrase_counts or "проверить" in phrase_counts:
        recommendations.append("Обнаружены фразы, часто вызывающие зацикливание")
        recommendations.append("Рекомендуется пересмотреть системный промпт")
    
    if not recommendations:
        recommendations.append("Промпт выглядит безопасным для генерации")
    
    return recommendations 