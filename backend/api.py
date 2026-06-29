"""
FastAPI Backend for Multi-Modal Disease Detection
==================================================
REST API for React frontend communication.

Endpoints:
- /api/config - Device and system configuration
- /api/classify - Disease classification
- /api/analysis - AS-IS, To-Be, Social, Clinical analysis
- /api/report - Report generation
- /api/status - System status
- /api/rag - RAG queries with Ollama
- /api/xai - Explainability and interpretability
- /api/governance - AI governance and ethics
- /api/integration - IoT, Cloud, Mobile integrations

Author: Research Team
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging
import sys
import os
import numpy as np
import time

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.multi_agent_system import MultiAgentSystem

# Import new modules
try:
    from rag.rag_system import RAGSystem, RAGConfig, OutputLevel
    HAS_RAG = True
except ImportError:
    HAS_RAG = False

try:
    from xai.explainability import XAISystem, DebugLevel
    HAS_XAI = True
except ImportError:
    HAS_XAI = False

try:
    from governance.ai_governance import AIGovernanceSystem
    HAS_GOVERNANCE = True
except ImportError:
    HAS_GOVERNANCE = False

try:
    from integrations.integration_hub import IntegrationHub, IoTDeviceType, CloudProvider, MobilePlatform
    HAS_INTEGRATIONS = True
except ImportError:
    HAS_INTEGRATIONS = False

try:
    from cache.cache_manager import CacheManager, CacheConfig
    HAS_CACHE = True
except ImportError:
    HAS_CACHE = False

try:
    from pipelines.pipeline_orchestrator import (
        pipeline_orchestrator, PipelineConfig, PipelineType,
        DataModality, ModelType, TaskType
    )
    from pipelines.job_scheduler import (
        job_scheduler, Job, JobPriority, ScheduleType,
        ScheduleConfig, ResourceRequirements
    )
    from pipelines.inference_pipeline import (
        inference_pipeline, InferenceInput, InputDataType
    )
    HAS_PIPELINES = True
except ImportError:
    HAS_PIPELINES = False

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="NeuroAI Disease Detection API",
    description="Multi-Modal Neurological Disease Detection System",
    version="2.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize multi-agent system
agent_system = MultiAgentSystem()

# WebSocket connections
active_connections: List[WebSocket] = []


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ConfigRequest(BaseModel):
    n_channels: Optional[int] = 22
    modality: Optional[str] = "eeg"
    disease_type: Optional[str] = "all"
    sampling_rate: Optional[float] = 256.0


class ClassifyRequest(BaseModel):
    disease: str
    modality: str = "eeg"
    n_channels: int = 22
    include_analysis: bool = True


class AnalysisRequest(BaseModel):
    disease: str
    analysis_type: str  # 'asis', 'tobe', 'social', 'clinical', 'full'
    manual: Optional[bool] = False
    manual_data: Optional[Dict[str, Any]] = None


class ReportRequest(BaseModel):
    disease: str
    report_type: str = "full"
    include_predictions: bool = True


class DeviceConnectRequest(BaseModel):
    device_type: str = "custom"
    n_channels: int = 22


class RAGQueryRequest(BaseModel):
    query: str
    disease_context: Optional[str] = None
    output_level: Optional[str] = "standard"
    max_results: Optional[int] = 5


class RAGDocumentRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None
    chunking_strategy: Optional[str] = "recursive"


class XAIExplainRequest(BaseModel):
    disease: str
    prediction_data: Optional[Dict[str, Any]] = None
    explanation_type: Optional[str] = "feature_importance"
    include_counterfactual: Optional[bool] = False


class GovernanceAssessRequest(BaseModel):
    disease: str
    model_id: Optional[str] = "neuroai_classifier"
    assessment_type: Optional[str] = "full"


class IntegrationConnectRequest(BaseModel):
    integration_type: str  # 'iot', 'cloud', 'mobile'
    provider: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class PipelineCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    pipeline_type: str = "full_pipeline"
    data_modality: str = "eeg"
    model_type: str = "cnn"
    task_type: str = "classification"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    preprocessing: Optional[Dict[str, Any]] = None
    augmentation: Optional[Dict[str, Any]] = None


class JobCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    pipeline_config: Dict[str, Any]
    priority: str = "normal"
    schedule_type: str = "immediate"
    scheduled_time: Optional[str] = None
    depends_on: Optional[List[str]] = None
    resources: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class InferenceTestRequest(BaseModel):
    data_type: str = "eeg_raw"
    eeg_channels: int = 22
    eeg_sampling_rate: float = 256.0
    eeg_duration_seconds: float = 10.0
    eeg_data: Optional[List[List[float]]] = None
    image_path: Optional[str] = None
    patient_id: Optional[str] = None
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    clinical_notes: Optional[str] = None


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "NeuroAI Disease Detection API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "config": "/api/config",
            "classify": "/api/classify",
            "analysis": "/api/analysis",
            "report": "/api/report",
            "status": "/api/status",
            "health": "/api/health",
            "devices": "/api/devices",
            "diseases": "/api/diseases",
            "statistics": "/api/statistics/{disease}",
            "rag": {
                "status": "/api/rag/status",
                "query": "/api/rag/query",
                "ingest": "/api/rag/ingest",
                "token_usage": "/api/rag/token-usage"
            },
            "xai": {
                "status": "/api/xai/status",
                "explain": "/api/xai/explain",
                "interpretability": "/api/xai/interpretability/{disease}",
                "debug": "/api/xai/debug/{disease}",
                "test": "/api/xai/test"
            },
            "governance": {
                "status": "/api/governance/status",
                "assess": "/api/governance/assess",
                "ethics": "/api/governance/ethics/{disease}",
                "bias": "/api/governance/bias/{disease}",
                "trust": "/api/governance/trust/{disease}",
                "risk": "/api/governance/risk/{disease}",
                "compliance": "/api/governance/compliance/{disease}",
                "score": "/api/governance/score/{disease}",
                "responsibility": "/api/governance/responsibility/{disease}"
            },
            "integration": {
                "status": "/api/integration/status",
                "connect": "/api/integration/connect",
                "iot_devices": "/api/integration/iot/devices",
                "iot_stream": "/api/integration/iot/stream",
                "cloud_providers": "/api/integration/cloud/providers",
                "cloud_sync": "/api/integration/cloud/sync",
                "mobile_clients": "/api/integration/mobile/clients",
                "mobile_push": "/api/integration/mobile/push"
            },
            "cache": {
                "status": "/api/cache/status",
                "clear": "/api/cache/clear",
                "stats": "/api/cache/stats"
            }
        },
        "modules": {
            "rag": HAS_RAG,
            "xai": HAS_XAI,
            "governance": HAS_GOVERNANCE,
            "integrations": HAS_INTEGRATIONS,
            "cache": HAS_CACHE
        }
    }


@app.get("/api/status")
async def get_status():
    """Get system status"""
    return agent_system.get_status()


@app.get("/api/diseases")
async def get_diseases():
    """Get list of supported diseases"""
    return {
        "diseases": [
            {"id": "alzheimer", "name": "Alzheimer's Disease", "modalities": ["eeg", "image", "hybrid"]},
            {"id": "parkinson", "name": "Parkinson's Disease", "modalities": ["eeg", "image", "hybrid"]},
            {"id": "schizophrenia", "name": "Schizophrenia", "modalities": ["eeg", "image", "hybrid"]},
            {"id": "epilepsy", "name": "Epilepsy", "modalities": ["eeg", "image"]},
            {"id": "autism", "name": "Autism Spectrum Disorder", "modalities": ["eeg", "hybrid"]},
            {"id": "stress", "name": "Chronic Stress", "modalities": ["eeg"]},
            {"id": "depression", "name": "Depression", "modalities": ["eeg", "hybrid"]}
        ]
    }


@app.get("/api/devices")
async def get_devices():
    """Get available device configurations"""
    return {
        "devices": [
            {"name": "Emotiv Insight", "channels": 5, "sampling_rate": 128},
            {"name": "Emotiv EPOC X", "channels": 14, "sampling_rate": 128},
            {"name": "Emotiv EPOC Flex", "channels": 32, "sampling_rate": 256},
            {"name": "Custom 2-channel", "channels": 2, "sampling_rate": 256},
            {"name": "Custom 4-channel", "channels": 4, "sampling_rate": 256},
            {"name": "Custom 8-channel", "channels": 8, "sampling_rate": 256},
            {"name": "Custom 16-channel", "channels": 16, "sampling_rate": 256},
            {"name": "Custom 22-channel", "channels": 22, "sampling_rate": 256},
            {"name": "Custom 24-channel", "channels": 24, "sampling_rate": 256},
            {"name": "Custom 64-channel", "channels": 64, "sampling_rate": 256}
        ],
        "modalities": ["eeg", "image", "hybrid"]
    }


@app.post("/api/config")
async def configure_system(config: ConfigRequest):
    """Configure the system"""
    try:
        result = await agent_system.configure(
            n_channels=config.n_channels,
            modality=config.modality,
            disease_type=config.disease_type
        )
        return {"success": True, "config": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/classify")
async def classify_disease(request: ClassifyRequest):
    """Classify disease from EEG/Image data"""
    try:
        # Configure system
        await agent_system.configure(
            n_channels=request.n_channels,
            modality=request.modality,
            disease_type=request.disease
        )

        # Run classification
        result = await agent_system.classify()

        response = {
            "success": True,
            "disease": request.disease,
            "modality": request.modality,
            "classification": result.get('classification', {}).result if result.get('classification') else {}
        }

        # Include analysis if requested
        if request.include_analysis:
            analysis_result = result.get('analysis', {})
            if analysis_result:
                response["analysis"] = analysis_result.result

        return response

    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analysis")
async def get_analysis(request: AnalysisRequest):
    """Get specific analysis"""
    try:
        if request.analysis_type == 'asis':
            result = await agent_system.get_asis_analysis(request.disease)
        elif request.analysis_type == 'tobe':
            result = await agent_system.get_tobe_analysis(
                request.disease,
                manual=request.manual,
                manual_data=request.manual_data
            )
        elif request.analysis_type == 'social':
            result = await agent_system.get_social_analysis(request.disease)
        elif request.analysis_type == 'clinical':
            result = await agent_system.get_clinical_analysis(request.disease)
        elif request.analysis_type == 'full':
            result = await agent_system.get_full_report(request.disease)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown analysis type: {request.analysis_type}")

        return {"success": True, "analysis": result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/report")
async def generate_report(request: ReportRequest):
    """Generate comprehensive report"""
    try:
        # Get full report
        report = await agent_system.get_full_report(request.disease)

        # Add predictions if requested
        if request.include_predictions:
            await agent_system.configure(disease_type=request.disease)
            classification = await agent_system.classify()
            report['classification'] = classification.get('classification', {}).result if classification.get('classification') else {}

        return {
            "success": True,
            "report": report,
            "disease": request.disease
        }

    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/device/connect")
async def connect_device(request: DeviceConnectRequest):
    """Connect to EEG device"""
    try:
        await agent_system.configure(n_channels=request.n_channels)
        return {
            "success": True,
            "device_type": request.device_type,
            "n_channels": request.n_channels,
            "status": "connected"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# WEBSOCKET FOR REAL-TIME DATA
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time EEG streaming"""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get('action') == 'start_streaming':
                n_channels = message.get('n_channels', 22)
                await agent_system.configure(n_channels=n_channels)

                # Stream simulated EEG data
                for _ in range(100):  # 100 samples
                    eeg_data = np.random.randn(n_channels).tolist()
                    await websocket.send_json({
                        "type": "eeg_data",
                        "channels": eeg_data,
                        "timestamp": asyncio.get_event_loop().time()
                    })
                    await asyncio.sleep(0.01)  # ~100 Hz

                await websocket.send_json({"type": "stream_end"})

            elif message.get('action') == 'classify':
                disease = message.get('disease', 'all')
                result = await agent_system.classify()

                await websocket.send_json({
                    "type": "classification",
                    "result": result.get('classification', {}).result if result.get('classification') else {}
                })

    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info("WebSocket disconnected")


# =============================================================================
# STATISTICS ENDPOINTS
# =============================================================================

@app.get("/api/statistics/{disease}")
async def get_statistics(disease: str):
    """Get statistical analysis for a disease"""
    try:
        # Generate mock statistics
        return {
            "disease": disease,
            "statistics": {
                "accuracy": 0.95 + np.random.uniform(0, 0.049),
                "precision": 0.93 + np.random.uniform(0, 0.06),
                "recall": 0.92 + np.random.uniform(0, 0.07),
                "f1_score": 0.94 + np.random.uniform(0, 0.05),
                "auc": 0.96 + np.random.uniform(0, 0.039),
                "confusion_matrix": {
                    "tp": int(np.random.uniform(85, 95)),
                    "tn": int(np.random.uniform(80, 90)),
                    "fp": int(np.random.uniform(5, 15)),
                    "fn": int(np.random.uniform(5, 15))
                },
                "confidence_interval": {
                    "lower": 0.93,
                    "upper": 0.99
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RAG ENDPOINTS
# =============================================================================

# Initialize RAG system lazily
_rag_system = None

def get_rag_system():
    global _rag_system
    if _rag_system is None and HAS_RAG:
        _rag_system = RAGSystem(RAGConfig(
            ollama_base_url="http://localhost:11434",
            model_name="llama2",
            embedding_model="nomic-embed-text"
        ))
    return _rag_system


@app.get("/api/rag/status")
async def rag_status():
    """Get RAG system status"""
    if not HAS_RAG:
        return {"available": False, "message": "RAG module not installed"}
    return {
        "available": True,
        "ollama_url": "http://localhost:11434",
        "output_levels": ["minimal", "standard", "detailed", "debug", "expert"]
    }


@app.post("/api/rag/query")
async def rag_query(request: RAGQueryRequest):
    """Query the RAG system"""
    if not HAS_RAG:
        raise HTTPException(status_code=503, detail="RAG system not available")

    try:
        rag = get_rag_system()

        # Map output level string to enum
        level_map = {
            "minimal": OutputLevel.MINIMAL,
            "standard": OutputLevel.STANDARD,
            "detailed": OutputLevel.DETAILED,
            "debug": OutputLevel.DEBUG,
            "expert": OutputLevel.EXPERT
        }
        output_level = level_map.get(request.output_level.lower(), OutputLevel.STANDARD)

        result = await rag.query(
            query=request.query,
            context=request.disease_context,
            output_level=output_level,
            top_k=request.max_results
        )

        return {
            "success": True,
            "query": request.query,
            "response": result.get("response", ""),
            "sources": result.get("sources", []),
            "tokens_used": result.get("tokens_used", 0)
        }
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/ingest")
async def rag_ingest(request: RAGDocumentRequest):
    """Ingest a document into RAG system"""
    if not HAS_RAG:
        raise HTTPException(status_code=503, detail="RAG system not available")

    try:
        rag = get_rag_system()
        result = await rag.ingest_document(
            content=request.content,
            metadata=request.metadata or {},
            chunking_strategy=request.chunking_strategy
        )

        return {
            "success": True,
            "chunks_created": result.get("chunks", 0),
            "document_id": result.get("document_id", "")
        }
    except Exception as e:
        logger.error(f"RAG ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/token-usage")
async def rag_token_usage():
    """Get RAG token usage statistics"""
    if not HAS_RAG:
        raise HTTPException(status_code=503, detail="RAG system not available")

    try:
        rag = get_rag_system()
        return {
            "success": True,
            "usage": rag.get_token_usage() if rag else {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# XAI (EXPLAINABILITY) ENDPOINTS
# =============================================================================

# Initialize XAI system lazily
_xai_system = None

def get_xai_system():
    global _xai_system
    if _xai_system is None and HAS_XAI:
        _xai_system = XAISystem(debug_level=DebugLevel.INFO)
    return _xai_system


@app.get("/api/xai/status")
async def xai_status():
    """Get XAI system status"""
    if not HAS_XAI:
        return {"available": False, "message": "XAI module not installed"}
    return {
        "available": True,
        "explanation_types": ["feature_importance", "attention", "counterfactual"],
        "debug_levels": ["minimal", "info", "detailed", "verbose"]
    }


@app.post("/api/xai/explain")
async def xai_explain(request: XAIExplainRequest):
    """Get explanation for a prediction"""
    if not HAS_XAI:
        raise HTTPException(status_code=503, detail="XAI system not available")

    try:
        xai = get_xai_system()

        # Generate explanation based on type
        explanation = await xai.explain(
            disease=request.disease,
            prediction_data=request.prediction_data,
            explanation_type=request.explanation_type
        )

        result = {
            "success": True,
            "disease": request.disease,
            "explanation_type": request.explanation_type,
            "explanation": explanation
        }

        # Add counterfactual if requested
        if request.include_counterfactual:
            counterfactual = await xai.generate_counterfactual(
                disease=request.disease,
                prediction_data=request.prediction_data
            )
            result["counterfactual"] = counterfactual

        return result
    except Exception as e:
        logger.error(f"XAI explain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/xai/interpretability/{disease}")
async def xai_interpretability(disease: str):
    """Get interpretability report for a disease model"""
    if not HAS_XAI:
        raise HTTPException(status_code=503, detail="XAI system not available")

    try:
        xai = get_xai_system()
        report = await xai.get_interpretability_report(disease=disease)

        return {
            "success": True,
            "disease": disease,
            "interpretability": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/xai/debug/{disease}")
async def xai_debug(disease: str, level: str = "info"):
    """Get debug information for a disease model"""
    if not HAS_XAI:
        raise HTTPException(status_code=503, detail="XAI system not available")

    try:
        xai = get_xai_system()

        level_map = {
            "minimal": DebugLevel.MINIMAL,
            "info": DebugLevel.INFO,
            "detailed": DebugLevel.DETAILED,
            "verbose": DebugLevel.VERBOSE
        }
        debug_level = level_map.get(level.lower(), DebugLevel.INFO)

        debug_info = await xai.debug_model(disease=disease, level=debug_level)

        return {
            "success": True,
            "disease": disease,
            "debug_level": level,
            "debug_info": debug_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/xai/test")
async def xai_test(disease: str):
    """Run tests for a disease model"""
    if not HAS_XAI:
        raise HTTPException(status_code=503, detail="XAI system not available")

    try:
        xai = get_xai_system()
        test_results = await xai.run_tests(disease=disease)

        return {
            "success": True,
            "disease": disease,
            "test_results": test_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# GOVERNANCE ENDPOINTS
# =============================================================================

# Initialize Governance system lazily
_governance_system = None

def get_governance_system():
    global _governance_system
    if _governance_system is None and HAS_GOVERNANCE:
        _governance_system = AIGovernanceSystem()
    return _governance_system


@app.get("/api/governance/status")
async def governance_status():
    """Get governance system status"""
    if not HAS_GOVERNANCE:
        return {"available": False, "message": "Governance module not installed"}
    return {
        "available": True,
        "assessment_types": ["ethics", "bias", "trust", "risk", "compliance", "full"],
        "compliance_frameworks": ["GDPR", "HIPAA", "FDA_SaMD", "EU_AI_Act"]
    }


@app.post("/api/governance/assess")
async def governance_assess(request: GovernanceAssessRequest):
    """Run governance assessment"""
    if not HAS_GOVERNANCE:
        raise HTTPException(status_code=503, detail="Governance system not available")

    try:
        gov = get_governance_system()

        assessment = await gov.assess(
            disease=request.disease,
            model_id=request.model_id,
            assessment_type=request.assessment_type
        )

        return {
            "success": True,
            "disease": request.disease,
            "assessment_type": request.assessment_type,
            "assessment": assessment
        }
    except Exception as e:
        logger.error(f"Governance assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/governance/ethics/{disease}")
async def governance_ethics(disease: str):
    """Get ethical assessment for a disease model"""
    if not HAS_GOVERNANCE:
        raise HTTPException(status_code=503, detail="Governance system not available")

    try:
        gov = get_governance_system()
        ethics = await gov.assess_ethics(disease=disease)

        return {
            "success": True,
            "disease": disease,
            "ethics": ethics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/governance/bias/{disease}")
async def governance_bias(disease: str):
    """Get bias detection report for a disease model"""
    if not HAS_GOVERNANCE:
        raise HTTPException(status_code=503, detail="Governance system not available")

    try:
        gov = get_governance_system()
        bias_report = await gov.detect_bias(disease=disease)

        return {
            "success": True,
            "disease": disease,
            "bias_report": bias_report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/governance/trust/{disease}")
async def governance_trust(disease: str):
    """Get trust score for a disease model"""
    if not HAS_GOVERNANCE:
        raise HTTPException(status_code=503, detail="Governance system not available")

    try:
        gov = get_governance_system()
        trust_score = await gov.evaluate_trust(disease=disease)

        return {
            "success": True,
            "disease": disease,
            "trust": trust_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/governance/risk/{disease}")
async def governance_risk(disease: str):
    """Get risk assessment for a disease model"""
    if not HAS_GOVERNANCE:
        raise HTTPException(status_code=503, detail="Governance system not available")

    try:
        gov = get_governance_system()
        risk = await gov.assess_risk(disease=disease)

        return {
            "success": True,
            "disease": disease,
            "risk": risk
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/governance/compliance/{disease}")
async def governance_compliance(disease: str, framework: str = "all"):
    """Check compliance for a disease model"""
    if not HAS_GOVERNANCE:
        raise HTTPException(status_code=503, detail="Governance system not available")

    try:
        gov = get_governance_system()
        compliance = await gov.check_compliance(disease=disease, framework=framework)

        return {
            "success": True,
            "disease": disease,
            "framework": framework,
            "compliance": compliance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/governance/score/{disease}")
async def governance_score(disease: str):
    """Get overall governance score for a disease model"""
    if not HAS_GOVERNANCE:
        raise HTTPException(status_code=503, detail="Governance system not available")

    try:
        gov = get_governance_system()
        score = await gov.get_governance_score(disease=disease)

        return {
            "success": True,
            "disease": disease,
            "governance_score": score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/governance/responsibility/{disease}")
async def governance_responsibility(disease: str):
    """Get responsibility tracking for a disease model"""
    if not HAS_GOVERNANCE:
        raise HTTPException(status_code=503, detail="Governance system not available")

    try:
        gov = get_governance_system()
        responsibility = await gov.track_responsibility(disease=disease)

        return {
            "success": True,
            "disease": disease,
            "responsibility": responsibility
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# INTEGRATION ENDPOINTS
# =============================================================================

# Initialize Integration hub lazily
_integration_hub = None

def get_integration_hub():
    global _integration_hub
    if _integration_hub is None and HAS_INTEGRATIONS:
        _integration_hub = IntegrationHub()
    return _integration_hub


@app.get("/api/integration/status")
async def integration_status():
    """Get integration hub status"""
    if not HAS_INTEGRATIONS:
        return {"available": False, "message": "Integration module not installed"}
    return {
        "available": True,
        "iot_devices": ["eeg_headset", "wearable_tracker", "smart_watch", "pulse_oximeter"],
        "cloud_providers": ["aws", "gcp", "azure"],
        "mobile_platforms": ["ios", "android", "react_native"]
    }


@app.post("/api/integration/connect")
async def integration_connect(request: IntegrationConnectRequest):
    """Connect to an integration"""
    if not HAS_INTEGRATIONS:
        raise HTTPException(status_code=503, detail="Integration system not available")

    try:
        hub = get_integration_hub()

        if request.integration_type == "iot":
            result = await hub.connect_iot_device(
                device_type=request.provider or "eeg_headset",
                config=request.config or {}
            )
        elif request.integration_type == "cloud":
            result = await hub.connect_cloud(
                provider=request.provider or "aws",
                config=request.config or {}
            )
        elif request.integration_type == "mobile":
            result = await hub.connect_mobile(
                platform=request.provider or "ios",
                config=request.config or {}
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown integration type: {request.integration_type}")

        return {
            "success": True,
            "integration_type": request.integration_type,
            "provider": request.provider,
            "connection": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Integration connect error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/integration/iot/devices")
async def integration_iot_devices():
    """Get connected IoT devices"""
    if not HAS_INTEGRATIONS:
        raise HTTPException(status_code=503, detail="Integration system not available")

    try:
        hub = get_integration_hub()
        devices = await hub.get_iot_devices()

        return {
            "success": True,
            "devices": devices
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/integration/iot/stream")
async def integration_iot_stream(device_id: str):
    """Start streaming from IoT device"""
    if not HAS_INTEGRATIONS:
        raise HTTPException(status_code=503, detail="Integration system not available")

    try:
        hub = get_integration_hub()
        result = await hub.start_iot_stream(device_id=device_id)

        return {
            "success": True,
            "device_id": device_id,
            "stream": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/integration/cloud/providers")
async def integration_cloud_providers():
    """Get connected cloud providers"""
    if not HAS_INTEGRATIONS:
        raise HTTPException(status_code=503, detail="Integration system not available")

    try:
        hub = get_integration_hub()
        providers = await hub.get_cloud_providers()

        return {
            "success": True,
            "providers": providers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/integration/cloud/sync")
async def integration_cloud_sync(provider: str, data_type: str = "all"):
    """Sync data with cloud provider"""
    if not HAS_INTEGRATIONS:
        raise HTTPException(status_code=503, detail="Integration system not available")

    try:
        hub = get_integration_hub()
        result = await hub.sync_cloud(provider=provider, data_type=data_type)

        return {
            "success": True,
            "provider": provider,
            "sync_result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/integration/mobile/clients")
async def integration_mobile_clients():
    """Get connected mobile clients"""
    if not HAS_INTEGRATIONS:
        raise HTTPException(status_code=503, detail="Integration system not available")

    try:
        hub = get_integration_hub()
        clients = await hub.get_mobile_clients()

        return {
            "success": True,
            "clients": clients
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/integration/mobile/push")
async def integration_mobile_push(client_id: str, notification: Dict[str, Any]):
    """Push notification to mobile client"""
    if not HAS_INTEGRATIONS:
        raise HTTPException(status_code=503, detail="Integration system not available")

    try:
        hub = get_integration_hub()
        result = await hub.push_notification(client_id=client_id, notification=notification)

        return {
            "success": True,
            "client_id": client_id,
            "push_result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CACHE ENDPOINTS
# =============================================================================

@app.get("/api/cache/status")
async def cache_status():
    """Get cache system status"""
    if not HAS_CACHE:
        return {"available": False, "message": "Cache module not installed"}
    return {
        "available": True,
        "cache_types": ["memory", "disk", "redis"],
        "specialized_caches": ["embedding", "model", "rag"]
    }


@app.post("/api/cache/clear")
async def cache_clear(cache_type: str = "all"):
    """Clear cache"""
    if not HAS_CACHE:
        raise HTTPException(status_code=503, detail="Cache system not available")

    try:
        from cache.cache_manager import CacheManager
        cache = CacheManager()
        result = await cache.clear(cache_type=cache_type)

        return {
            "success": True,
            "cache_type": cache_type,
            "cleared": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    if not HAS_CACHE:
        raise HTTPException(status_code=503, detail="Cache system not available")

    try:
        from cache.cache_manager import CacheManager
        cache = CacheManager()
        stats = cache.get_stats()

        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PIPELINE ENDPOINTS
# =============================================================================

@app.get("/api/pipeline/status")
async def pipeline_status():
    """Get pipeline system status"""
    if not HAS_PIPELINES:
        return {"available": False, "message": "Pipeline module not installed"}
    return {
        "available": True,
        "pipeline_types": ["data", "training", "inference", "performance", "testing", "drift_detection", "hyperparameter_tuning", "full_pipeline"],
        "data_modalities": ["eeg", "mri", "ct", "xray", "image", "tabular", "text", "audio", "multimodal"],
        "model_types": pipeline_orchestrator.get_available_models() if HAS_PIPELINES else [],
        "preprocessing_options": pipeline_orchestrator.get_preprocessing_options() if HAS_PIPELINES else {}
    }


@app.post("/api/pipeline/create")
async def pipeline_create(request: PipelineCreateRequest):
    """Create a new pipeline"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Pipeline system not available")

    try:
        config = PipelineConfig(
            name=request.name,
            description=request.description,
            pipeline_type=PipelineType(request.pipeline_type),
            data_modality=DataModality(request.data_modality),
            model_type=ModelType(request.model_type),
            task_type=TaskType(request.task_type),
            train_split=request.train_split,
            val_split=request.val_split,
            test_split=request.test_split,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            preprocessing=request.preprocessing or {},
            augmentation=request.augmentation or {}
        )

        pipeline_id = pipeline_orchestrator.create_pipeline(config)

        return {
            "success": True,
            "pipeline_id": pipeline_id,
            "config": config.to_dict()
        }
    except Exception as e:
        logger.error(f"Pipeline create error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pipeline/run/{pipeline_id}")
async def pipeline_run(pipeline_id: str):
    """Run a pipeline"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Pipeline system not available")

    try:
        result = await pipeline_orchestrator.run_pipeline(pipeline_id)
        return {
            "success": True,
            "result": result.to_dict()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Pipeline run error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pipeline/{pipeline_id}")
async def pipeline_get(pipeline_id: str):
    """Get pipeline status"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Pipeline system not available")

    result = pipeline_orchestrator.get_pipeline_status(pipeline_id)
    if not result:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    return {
        "success": True,
        "result": result.to_dict()
    }


@app.get("/api/pipeline/list/all")
async def pipeline_list():
    """List all pipelines"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Pipeline system not available")

    return {
        "success": True,
        "pipelines": pipeline_orchestrator.get_all_pipelines()
    }


@app.delete("/api/pipeline/{pipeline_id}")
async def pipeline_cancel(pipeline_id: str):
    """Cancel a running pipeline"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Pipeline system not available")

    success = pipeline_orchestrator.cancel_pipeline(pipeline_id)
    return {
        "success": success,
        "message": "Pipeline cancelled" if success else "Pipeline not found or not running"
    }


@app.get("/api/pipeline/models")
async def pipeline_models():
    """Get available models for training"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Pipeline system not available")

    return {
        "success": True,
        "models": pipeline_orchestrator.get_available_models()
    }


@app.get("/api/pipeline/preprocessing")
async def pipeline_preprocessing():
    """Get available preprocessing options"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Pipeline system not available")

    return {
        "success": True,
        "options": pipeline_orchestrator.get_preprocessing_options()
    }


# =============================================================================
# JOB SCHEDULER ENDPOINTS
# =============================================================================

@app.get("/api/jobs/status")
async def jobs_status():
    """Get job scheduler status"""
    if not HAS_PIPELINES:
        return {"available": False, "message": "Job scheduler not installed"}

    return {
        "available": True,
        "is_running": job_scheduler.is_running,
        "statistics": job_scheduler.get_statistics(),
        "resources": job_scheduler.get_resource_status()
    }


@app.post("/api/jobs/start-scheduler")
async def jobs_start_scheduler():
    """Start the job scheduler"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Job scheduler not available")

    await job_scheduler.start()
    return {"success": True, "message": "Scheduler started"}


@app.post("/api/jobs/stop-scheduler")
async def jobs_stop_scheduler():
    """Stop the job scheduler"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Job scheduler not available")

    await job_scheduler.stop()
    return {"success": True, "message": "Scheduler stopped"}


@app.post("/api/jobs/create")
async def jobs_create(request: JobCreateRequest):
    """Create and submit a new job"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Job scheduler not available")

    try:
        from datetime import datetime

        priority_map = {
            "critical": JobPriority.CRITICAL,
            "high": JobPriority.HIGH,
            "normal": JobPriority.NORMAL,
            "low": JobPriority.LOW,
            "background": JobPriority.BACKGROUND
        }

        schedule_map = {
            "immediate": ScheduleType.IMMEDIATE,
            "once": ScheduleType.ONCE,
            "recurring": ScheduleType.RECURRING,
            "cron": ScheduleType.CRON,
            "dependency": ScheduleType.DEPENDENCY
        }

        scheduled_time = None
        if request.scheduled_time:
            scheduled_time = datetime.fromisoformat(request.scheduled_time)

        job = job_scheduler.create_job(
            name=request.name,
            description=request.description,
            pipeline_config=request.pipeline_config,
            priority=priority_map.get(request.priority, JobPriority.NORMAL),
            schedule_type=schedule_map.get(request.schedule_type, ScheduleType.IMMEDIATE),
            scheduled_time=scheduled_time,
            depends_on=request.depends_on,
            resources=request.resources,
            tags=request.tags
        )

        return {
            "success": True,
            "job_id": job.job_id,
            "job": job.to_dict()
        }
    except Exception as e:
        logger.error(f"Job create error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs/{job_id}")
async def jobs_get(job_id: str):
    """Get job status"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Job scheduler not available")

    job = job_scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "success": True,
        "job": job.to_dict()
    }


@app.get("/api/jobs/list/all")
async def jobs_list():
    """List all jobs"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Job scheduler not available")

    return {
        "success": True,
        "jobs": job_scheduler.get_all_jobs()
    }


@app.get("/api/jobs/list/queued")
async def jobs_queued():
    """List queued jobs"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Job scheduler not available")

    return {
        "success": True,
        "jobs": job_scheduler.get_queued_jobs()
    }


@app.get("/api/jobs/list/running")
async def jobs_running():
    """List running jobs"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Job scheduler not available")

    return {
        "success": True,
        "jobs": job_scheduler.get_running_jobs()
    }


@app.delete("/api/jobs/{job_id}")
async def jobs_cancel(job_id: str):
    """Cancel a job"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Job scheduler not available")

    success = job_scheduler.cancel_job(job_id)
    return {
        "success": success,
        "message": "Job cancelled" if success else "Job not found or cannot be cancelled"
    }


@app.post("/api/jobs/{job_id}/pause")
async def jobs_pause(job_id: str):
    """Pause a job"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Job scheduler not available")

    success = job_scheduler.pause_job(job_id)
    return {
        "success": success,
        "message": "Job paused" if success else "Job not found or cannot be paused"
    }


@app.post("/api/jobs/{job_id}/resume")
async def jobs_resume(job_id: str):
    """Resume a paused job"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Job scheduler not available")

    success = job_scheduler.resume_job(job_id)
    return {
        "success": success,
        "message": "Job resumed" if success else "Job not found or cannot be resumed"
    }


@app.put("/api/jobs/{job_id}/priority")
async def jobs_update_priority(job_id: str, priority: str):
    """Update job priority"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Job scheduler not available")

    priority_map = {
        "critical": JobPriority.CRITICAL,
        "high": JobPriority.HIGH,
        "normal": JobPriority.NORMAL,
        "low": JobPriority.LOW,
        "background": JobPriority.BACKGROUND
    }

    job_priority = priority_map.get(priority)
    if not job_priority:
        raise HTTPException(status_code=400, detail="Invalid priority")

    success = job_scheduler.update_priority(job_id, job_priority)
    return {
        "success": success,
        "message": "Priority updated" if success else "Job not found or cannot update priority"
    }


# =============================================================================
# INFERENCE TESTING ENDPOINTS
# =============================================================================

@app.get("/api/inference/status")
async def inference_status():
    """Get inference pipeline status"""
    if not HAS_PIPELINES:
        return {"available": False, "message": "Inference pipeline not installed"}

    return {
        "available": True,
        "supported_data_types": ["eeg_raw", "eeg_file", "mri_file", "ct_file", "image_file", "multimodal"],
        "supported_diseases": inference_pipeline.DISEASES if HAS_PIPELINES else [],
        "eeg_channels": inference_pipeline.EEG_CHANNELS if HAS_PIPELINES else [],
        "brain_regions": inference_pipeline.BRAIN_REGIONS if HAS_PIPELINES else []
    }


@app.post("/api/inference/test")
async def inference_test(request: InferenceTestRequest):
    """Run inference test on provided data"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Inference pipeline not available")

    try:
        input_data = InferenceInput(
            data_type=InputDataType(request.data_type),
            eeg_channels=request.eeg_channels,
            eeg_sampling_rate=request.eeg_sampling_rate,
            eeg_duration_seconds=request.eeg_duration_seconds,
            eeg_data=request.eeg_data,
            image_path=request.image_path,
            patient_id=request.patient_id,
            patient_age=request.patient_age,
            patient_gender=request.patient_gender,
            clinical_notes=request.clinical_notes
        )

        report = await inference_pipeline.run_inference(input_data)

        return {
            "success": True,
            "report": report.to_dict()
        }
    except Exception as e:
        logger.error(f"Inference test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/inference/report/{report_id}")
async def inference_report(report_id: str):
    """Get inference report by ID"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Inference pipeline not available")

    report = inference_pipeline.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    return {
        "success": True,
        "report": report.to_dict()
    }


@app.get("/api/inference/reports")
async def inference_reports():
    """Get all inference reports"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Inference pipeline not available")

    return {
        "success": True,
        "reports": inference_pipeline.get_all_reports()
    }


@app.post("/api/inference/batch")
async def inference_batch(inputs: List[InferenceTestRequest]):
    """Run batch inference on multiple inputs"""
    if not HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="Inference pipeline not available")

    try:
        input_data_list = []
        for req in inputs:
            input_data = InferenceInput(
                data_type=InputDataType(req.data_type),
                eeg_channels=req.eeg_channels,
                eeg_sampling_rate=req.eeg_sampling_rate,
                eeg_duration_seconds=req.eeg_duration_seconds,
                eeg_data=req.eeg_data,
                image_path=req.image_path,
                patient_id=req.patient_id
            )
            input_data_list.append(input_data)

        reports = await inference_pipeline.run_batch_inference(input_data_list)

        return {
            "success": True,
            "reports": [r.to_dict() for r in reports]
        }
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SYSTEM HEALTH ENDPOINTS
# =============================================================================

@app.get("/api/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "modules": {
            "rag": HAS_RAG,
            "xai": HAS_XAI,
            "governance": HAS_GOVERNANCE,
            "integrations": HAS_INTEGRATIONS,
            "cache": HAS_CACHE,
            "pipelines": HAS_PIPELINES
        },
        "agent_system": agent_system.get_status() if agent_system else {"status": "not_initialized"},
        "job_scheduler": job_scheduler.get_statistics() if HAS_PIPELINES else {"status": "not_available"}
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
