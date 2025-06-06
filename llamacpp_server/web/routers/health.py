"""Health check роутер."""

from fastapi import APIRouter, Depends, Request

from ...domain.protocols import LlamaServiceProtocol


router = APIRouter()


async def get_llama_service(request: Request) -> LlamaServiceProtocol:
    """Получить LLama сервис из DI контейнера."""
    return request.app.state.container.llama_service()


@router.get("")
async def health_check(llama_service: LlamaServiceProtocol = Depends(get_llama_service)):
    """Проверка здоровья сервиса."""
    is_ready = await llama_service.is_ready()
    
    return {
        "status": "healthy" if is_ready else "unhealthy",
        "model_ready": is_ready,
    } 