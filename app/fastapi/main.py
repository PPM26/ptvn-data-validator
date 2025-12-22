from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.fastapi.api.state import AppState, load_prompts
from app.services.fixer_service import FixerService
from app.services.llm_service import LLMService
from app.services.ragflow_service import RagFlowService

from app.fastapi.api.routers.health import router as health_router
from app.fastapi.api.routers.item import router as item_router
from app.fastapi.api.routers.category import router as category_router
from app.fastapi.api.routers.spec import router as spec_router
from app.fastapi.api.routers.pipeline import router as pipeline_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    prompts = load_prompts()

    # Singletons (shared across routers)
    llm = LLMService()
    rag = RagFlowService()
    fixer = FixerService()

    # store in app.state
    app.state.app_state = AppState(
        prompts=prompts,
        llm=llm,
        rag=rag,
        fixer=fixer,
    )

    yield


app = FastAPI(
    title="Data Fixer and Validator",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health_router, tags=["health"])
app.include_router(item_router, prefix="/item", tags=["item"])
app.include_router(category_router, prefix="/category", tags=["category"])
app.include_router(spec_router, prefix="/spec", tags=["spec"])
app.include_router(pipeline_router, prefix="/pipeline", tags=["pipeline"])
