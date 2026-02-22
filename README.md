# PTVN Data Validator

## Data Validation Services
Automated Item Prediction · RAG Category Fixing · Specification Validation · Multi-Item Detection · FastAPI Microservices

This project provides an intelligent data validation and fixing system that leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to automatically validate, fix, and enhance product catalog data.

## Configuration Settings
When running the data validator, you can configure the following parameters in your `.env` file:

### 1. LLM Parameter
- **`MODEL_NAME`**: Configure the Qwen3-VL-30B model.

### 2. RAG Settings
- **`TOP_K`**: Number of similar patterns to retrieve: `30`
- **`RAGFLOW_SIMILARITY_THRESHOLD`**: Minimum similarity score: `0.2`
- **`RAGFLOW_VECTOR_SIMILARITY_WEIGHT`**: Weight for vector similarity: `0.3`

## System Overview
![System Architecture](system_architecture_image/system_architecture.jpg)
![Workflow](system_architecture_image/workflow_no_fix_category.jpg)

The system processes product descriptions and specifications to:
1. **Predict items** from product descriptions using LLM
2. **Fix categories** based on RAG-retrieved patterns
3. **Validate and correct specifications** using multi-stage LLM pipelines
4. **Extract structured data** from unstructured text
5. **Remove duplicate items** from specifications
6. **Ensure data quality** through automated validation


### Core Components
- **FastAPI Server**: REST API endpoints for data processing
- **Fixer Service**: Orchestrates the data fixing workflow
- **LLM Service**: Handles LLM interactions for prediction and validation
- **RagFlow Service**: Retrieves similar patterns from knowledge base
- **Pipeline**: Batch processing with async concurrency control

## Project Structure
```text
data_validation/
├── app/
│   ├── fastapi/           # FastAPI application
│   │   ├── api/          # API routes and models
│   │   └── main.py       # FastAPI app configuration
│   ├── pipeline/         # Batch processing pipeline
│   ├── prompts/          # LLM prompt templates
│   ├── services/         # Core services
│   │   ├── fixer_service.py    # Main fixing workflow
│   │   ├── llm_service.py      # LLM interactions
│   │   └── ragflow_service.py  # RAG operations
│   └── utils/            # Utility functions
├── demo_dataset/         # Sample datasets
├── output/               # Output directory
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Docker Compose setup
├── requirements.txt      # Python dependencies
├── run_fastapi.py        # Example batch processing script
└── README.md
```

## Workflow: Data Fixing Pipeline
Pipeline Order
1. **Item Prediction**: Predict item name from description using LLM
2. **RAG Query**: Retrieve similar patterns from RagFlow using predicted item
3. **Category Fix**: Fix category using RAG-retrieved categories
4. **Spec Fix**: Fix specifications using description and patterns
5. **Multi-Item Removal**: Remove duplicate items from specifications
6. **Spec Validation**: Final validation of specifications against description
7. **Item Extraction**: Extract final item from validated specifications

## FastAPI Endpoints

### Main Pipeline Endpoints
- POST `/pipeline/fix-batch`

### Validation & Fixing Endpoints
- POST `/item/predict`
- POST `/category/fix`
- POST `/spec/fix`

### Health Monitoring
- GET `/health`

#### Key Behavior:
- Automatic item prediction from descriptions
- Category correction using domain knowledge
- Specification validation and fixing
- Multi-item detection and removal
- Async batch processing with configurable concurrency
- Timeout safeguards for LLM calls

### API Documentation
Once the server is running, visit:
- **Swagger UI**: http://localhost:5500/docs
- **ReDoc**: http://localhost:5500/redoc

## Prompt Logic
The fixing logic uses LLM prompts located in `app/prompts/` to perform its tasks:
- **Item Prediction**: Extracts the core product name
- **Category Fixing**: Determines the correct category from RAG suggestions
- **Specification Validation**: Verifies and corrects specs against the description
- **Multi-Item Removal**: Detects and cleans up mixed item specifications

## Run FastAPI Local
```bash
uvicorn app.fastapi.main:app --port 5500 --reload
```

### Manual Testing Steps
1. **Start the API Server**
   Start the server locally using uvicorn as shown above.
2. **Run Batch Processing Script** (Async)
   Execute the included sample script:
   ```bash
   python run_fastapi.py
   ```
   This script:
   - Loads test data from `demo_dataset/test_set.csv`
   - Processes rows in batches of 50 against `/pipeline/fix-batch`
   - Saves results to `output/results.csv`
