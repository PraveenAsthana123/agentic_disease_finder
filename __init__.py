"""
Neurological Disease Detection using Agentic AI with MCP
=========================================================

A comprehensive AI system for detecting neurological diseases:
- Alzheimer's Disease
- Parkinson's Disease
- Schizophrenia

Features:
- Model Context Protocol (MCP) for AI agent integration
- Agent-to-Agent (A2A) communication
- Multi-modal data analysis (MRI, EEG, Voice, Gait)
- Deep learning models (CNN3D, LSTM, EEGNet, GraphNet)
- Comprehensive evaluation metrics

Author: Research Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Research Team"

# Core imports — each subpackage wrapped so a missing/aspirational symbol in one
# module never breaks importing the whole package (and pytest can always collect).
# Whatever is actually implemented is exported; the rest is silently skipped.
try:
    from .agents import (
        BaseAgent,
        AgentState,
        AgentMessage,
        MessageBus,
        AgentOrchestrator,
        AlzheimerDetectionAgent,
        ParkinsonDetectionAgent,
        SchizophreniaDetectionAgent
    )
except ImportError:
    pass

try:
    from .mcp import (
        MCPServer,
        MCPClient,
        MCPTool,
        MCPToolParameter,
        MCPResource,
        MCPAgentOrchestrator,
        NeuroDiseaseTools,
        JSONRPCRequest,
        JSONRPCResponse,
        JSONRPCError,
        ToolCall,
        create_neuro_disease_mcp_server
    )
except ImportError:
    pass

try:
    from .portal import (
        ModelControlPortal,
        ModelRegistry,
        ModelInfo
    )
except ImportError:
    pass

try:
    # NOTE: BrainConnectivityGNN, MultiModalFusion, create_*_model factories are
    # aspirational (not yet in models/). Only implemented models are exported.
    from .models import (
        AlzheimerCNN3D,
        ParkinsonLSTM,
        SchizophreniaEEGNet,
    )
except ImportError:
    pass

try:
    from .data_loaders import (
        ADNIDataLoader,
        PPMIDataLoader,
        COBREDataLoader,
        MultiDatasetLoader
    )
except ImportError:
    pass

try:
    from .evaluation import (
        ModelEvaluator,
        DiseaseModelScorer,
        ClassificationMetrics,
        evaluate_all_diseases
    )
except ImportError:
    pass

try:
    from .utils import (
        NeuroDiseaseVisualizer
    )
except ImportError:
    pass

__all__ = [
    # Version
    '__version__',
    '__author__',

    # Agents
    'BaseAgent',
    'AgentState',
    'AgentMessage',
    'MessageBus',
    'AgentOrchestrator',
    'AlzheimerDetectionAgent',
    'ParkinsonDetectionAgent',
    'SchizophreniaDetectionAgent',

    # MCP
    'MCPServer',
    'MCPClient',
    'MCPTool',
    'MCPToolParameter',
    'MCPResource',
    'MCPAgentOrchestrator',
    'NeuroDiseaseTools',
    'JSONRPCRequest',
    'JSONRPCResponse',
    'JSONRPCError',
    'ToolCall',
    'create_neuro_disease_mcp_server',

    # Portal
    'ModelControlPortal',
    'ModelRegistry',
    'ModelInfo',

    # Models
    'AlzheimerCNN3D',
    'ParkinsonLSTM',
    'SchizophreniaEEGNet',

    # Data Loaders
    'ADNIDataLoader',
    'PPMIDataLoader',
    'COBREDataLoader',
    'MultiDatasetLoader',

    # Evaluation
    'ModelEvaluator',
    'DiseaseModelScorer',
    'ClassificationMetrics',
    'evaluate_all_diseases',

    # Utils
    'NeuroDiseaseVisualizer'
]
