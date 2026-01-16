"""
MCP Server for Neurological Disease Detection
==============================================
Model Context Protocol (MCP) server implementation for the
Agentic AI neurological disease detection system.

Based on Anthropic's MCP specification:
- JSON-RPC 2.0 protocol
- Tool discovery via list_tools()
- Structured tool execution
- Session state management

Author: Research Team
Project: Neurological Disease Detection using Agentic AI with MCP
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import uuid
import sys

logger = logging.getLogger(__name__)


# ============================================================================
# MCP Protocol Types (JSON-RPC 2.0)
# ============================================================================

class MCPMessageType(Enum):
    """MCP message types"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 Request"""
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    jsonrpc: str = "2.0"

    def to_dict(self) -> Dict:
        return {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
            "params": self.params,
            "id": self.id
        }


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 Response"""
    result: Any
    id: str
    jsonrpc: str = "2.0"

    def to_dict(self) -> Dict:
        return {
            "jsonrpc": self.jsonrpc,
            "result": self.result,
            "id": self.id
        }


@dataclass
class JSONRPCError:
    """JSON-RPC 2.0 Error"""
    code: int
    message: str
    data: Any = None
    id: str = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> Dict:
        error = {
            "jsonrpc": self.jsonrpc,
            "error": {
                "code": self.code,
                "message": self.message
            },
            "id": self.id
        }
        if self.data:
            error["error"]["data"] = self.data
        return error


# Standard JSON-RPC error codes
class ErrorCode:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


# ============================================================================
# MCP Tool Definition
# ============================================================================

@dataclass
class MCPToolParameter:
    """Tool parameter definition with JSON Schema"""
    name: str
    type: str
    description: str
    required: bool = True
    enum: List[str] = None
    default: Any = None

    def to_json_schema(self) -> Dict:
        schema = {
            "type": self.type,
            "description": self.description
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class MCPTool:
    """
    MCP Tool Definition

    Represents a callable tool that agents can discover and execute.
    Follows the MCP specification for tool definitions.
    """
    name: str
    description: str
    parameters: List[MCPToolParameter] = field(default_factory=list)
    handler: Callable = None
    category: str = "general"
    version: str = "1.0.0"

    def to_dict(self) -> Dict:
        """Convert to MCP tool format"""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


# ============================================================================
# MCP Resource Definition
# ============================================================================

@dataclass
class MCPResource:
    """MCP Resource - represents data that can be read by agents"""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"

    def to_dict(self) -> Dict:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type
        }


# ============================================================================
# MCP Server Implementation
# ============================================================================

class MCPServer:
    """
    Model Context Protocol Server

    Implements the MCP specification for AI agent tool integration.
    Provides:
    - Tool discovery (list_tools)
    - Tool execution (call_tool)
    - Resource management (list_resources, read_resource)
    - Session state management

    Transport: stdio (standard input/output) or HTTP
    """

    def __init__(self, name: str = "neuro-disease-mcp",
                 version: str = "1.0.0"):
        """
        Initialize MCP Server

        Parameters
        ----------
        name : str
            Server name
        version : str
            Server version
        """
        self.name = name
        self.version = version

        # Tool registry
        self.tools: Dict[str, MCPTool] = {}

        # Resource registry
        self.resources: Dict[str, MCPResource] = {}

        # Session state
        self.sessions: Dict[str, Dict] = {}

        # Method handlers
        self.methods: Dict[str, Callable] = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_list_tools,
            "tools/call": self._handle_call_tool,
            "resources/list": self._handle_list_resources,
            "resources/read": self._handle_read_resource,
            "ping": self._handle_ping,
            "shutdown": self._handle_shutdown
        }

        # Server state
        self.initialized = False
        self.running = False

        logger.info(f"MCP Server '{name}' v{version} created")

    def register_tool(self, tool: MCPTool):
        """
        Register a tool with the server

        Parameters
        ----------
        tool : MCPTool
            Tool to register
        """
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def register_resource(self, resource: MCPResource):
        """Register a resource"""
        self.resources[resource.uri] = resource
        logger.info(f"Registered resource: {resource.uri}")

    async def handle_message(self, message: str) -> str:
        """
        Handle incoming JSON-RPC message

        Parameters
        ----------
        message : str
            JSON-RPC request string

        Returns
        -------
        response : str
            JSON-RPC response string
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            return json.dumps(JSONRPCError(
                code=ErrorCode.PARSE_ERROR,
                message=f"Parse error: {str(e)}"
            ).to_dict())

        # Validate JSON-RPC format
        if "jsonrpc" not in data or data["jsonrpc"] != "2.0":
            return json.dumps(JSONRPCError(
                code=ErrorCode.INVALID_REQUEST,
                message="Invalid JSON-RPC version"
            ).to_dict())

        method = data.get("method")
        params = data.get("params", {})
        request_id = data.get("id")

        # Find and execute handler
        if method in self.methods:
            try:
                result = await self.methods[method](params)
                return json.dumps(JSONRPCResponse(
                    result=result,
                    id=request_id
                ).to_dict())
            except Exception as e:
                logger.error(f"Error handling {method}: {e}")
                return json.dumps(JSONRPCError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=str(e),
                    id=request_id
                ).to_dict())
        else:
            return json.dumps(JSONRPCError(
                code=ErrorCode.METHOD_NOT_FOUND,
                message=f"Method not found: {method}",
                id=request_id
            ).to_dict())

    # ========================================================================
    # MCP Method Handlers
    # ========================================================================

    async def _handle_initialize(self, params: Dict) -> Dict:
        """
        Handle initialization request

        Returns server capabilities and protocol version.
        """
        self.initialized = True

        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"subscribe": True, "listChanged": True},
                "prompts": {"listChanged": True}
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }

    async def _handle_list_tools(self, params: Dict) -> Dict:
        """
        Handle tools/list request

        Returns all available tools with their schemas.
        This is the primary discovery mechanism for agents.
        """
        tools_list = [tool.to_dict() for tool in self.tools.values()]

        return {
            "tools": tools_list
        }

    async def _handle_call_tool(self, params: Dict) -> Dict:
        """
        Handle tools/call request

        Executes a tool and returns the result.
        """
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool = self.tools[tool_name]

        if tool.handler is None:
            raise ValueError(f"Tool {tool_name} has no handler")

        # Execute tool handler
        try:
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(**arguments)
            else:
                result = tool.handler(**arguments)

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, default=str)
                    }
                ],
                "isError": False
            }
        except Exception as e:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: {str(e)}"
                    }
                ],
                "isError": True
            }

    async def _handle_list_resources(self, params: Dict) -> Dict:
        """Handle resources/list request"""
        resources_list = [res.to_dict() for res in self.resources.values()]
        return {"resources": resources_list}

    async def _handle_read_resource(self, params: Dict) -> Dict:
        """Handle resources/read request"""
        uri = params.get("uri")

        if uri not in self.resources:
            raise ValueError(f"Unknown resource: {uri}")

        # Return resource content (would be implemented per resource)
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": self.resources[uri].mime_type,
                    "text": "{}"  # Placeholder
                }
            ]
        }

    async def _handle_ping(self, params: Dict) -> Dict:
        """Handle ping request"""
        return {"status": "pong", "timestamp": datetime.now().isoformat()}

    async def _handle_shutdown(self, params: Dict) -> Dict:
        """Handle shutdown request"""
        self.running = False
        return {"status": "shutting_down"}

    # ========================================================================
    # Server Transport (stdio)
    # ========================================================================

    async def run_stdio(self):
        """
        Run server using stdio transport

        Reads JSON-RPC requests from stdin, writes responses to stdout.
        This is the standard MCP transport mechanism.
        """
        self.running = True
        logger.info(f"MCP Server running on stdio")

        while self.running:
            try:
                # Read line from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )

                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                # Handle message
                response = await self.handle_message(line)

                # Write response to stdout
                print(response, flush=True)

            except Exception as e:
                logger.error(f"Error in stdio loop: {e}")

        logger.info("MCP Server stopped")


# ============================================================================
# Neurological Disease MCP Tools
# ============================================================================

class NeuroDiseaseTools:
    """
    MCP Tools for Neurological Disease Detection

    Provides tool definitions and handlers for:
    - Alzheimer's disease analysis
    - Parkinson's disease analysis
    - Schizophrenia analysis
    - Multi-disease ensemble analysis
    """

    def __init__(self):
        self.tools: List[MCPTool] = []
        self._create_tools()

    def _create_tools(self):
        """Create all disease detection tools"""

        # ====================================================================
        # Alzheimer's Disease Tools
        # ====================================================================

        self.tools.append(MCPTool(
            name="analyze_alzheimer_mri",
            description="Analyze MRI brain scan for Alzheimer's disease markers including hippocampal atrophy, cortical thinning, and ventricular enlargement",
            parameters=[
                MCPToolParameter("patient_id", "string", "Unique patient identifier"),
                MCPToolParameter("mri_data_path", "string", "Path to MRI data file or base64 encoded data"),
                MCPToolParameter("analysis_type", "string", "Type of analysis",
                                enum=["volumetric", "cortical_thickness", "full"], default="full"),
            ],
            handler=self._analyze_alzheimer_mri,
            category="alzheimer"
        ))

        self.tools.append(MCPTool(
            name="assess_cognitive_status",
            description="Assess cognitive status using clinical scores (MMSE, CDR, ADAS-Cog) for Alzheimer's staging",
            parameters=[
                MCPToolParameter("patient_id", "string", "Unique patient identifier"),
                MCPToolParameter("mmse_score", "number", "Mini-Mental State Examination score (0-30)"),
                MCPToolParameter("cdr_score", "number", "Clinical Dementia Rating (0, 0.5, 1, 2, 3)"),
                MCPToolParameter("adas_cog", "number", "ADAS-Cog score", required=False),
                MCPToolParameter("age", "number", "Patient age in years"),
            ],
            handler=self._assess_cognitive_status,
            category="alzheimer"
        ))

        self.tools.append(MCPTool(
            name="predict_alzheimer_stage",
            description="Predict Alzheimer's disease stage (CN/MCI/AD) using combined features",
            parameters=[
                MCPToolParameter("patient_id", "string", "Patient identifier"),
                MCPToolParameter("features", "object", "Combined imaging and clinical features"),
            ],
            handler=self._predict_alzheimer_stage,
            category="alzheimer"
        ))

        # ====================================================================
        # Parkinson's Disease Tools
        # ====================================================================

        self.tools.append(MCPTool(
            name="analyze_voice_parkinson",
            description="Analyze voice recording for Parkinson's disease markers including jitter, shimmer, and HNR",
            parameters=[
                MCPToolParameter("patient_id", "string", "Patient identifier"),
                MCPToolParameter("audio_path", "string", "Path to audio file or base64 encoded audio"),
                MCPToolParameter("sample_rate", "number", "Audio sample rate in Hz", default=16000),
            ],
            handler=self._analyze_voice_parkinson,
            category="parkinson"
        ))

        self.tools.append(MCPTool(
            name="analyze_gait_parkinson",
            description="Analyze gait sensor data for Parkinson's disease including stride length, cadence, and freezing",
            parameters=[
                MCPToolParameter("patient_id", "string", "Patient identifier"),
                MCPToolParameter("sensor_data_path", "string", "Path to gait sensor data"),
                MCPToolParameter("sensor_type", "string", "Type of sensor",
                                enum=["accelerometer", "gyroscope", "combined"], default="combined"),
            ],
            handler=self._analyze_gait_parkinson,
            category="parkinson"
        ))

        self.tools.append(MCPTool(
            name="calculate_updrs",
            description="Calculate UPDRS (Unified Parkinson's Disease Rating Scale) motor score",
            parameters=[
                MCPToolParameter("patient_id", "string", "Patient identifier"),
                MCPToolParameter("motor_assessments", "object", "Motor assessment scores"),
            ],
            handler=self._calculate_updrs,
            category="parkinson"
        ))

        self.tools.append(MCPTool(
            name="analyze_datscan",
            description="Analyze DaTscan SPECT imaging for dopaminergic deficit",
            parameters=[
                MCPToolParameter("patient_id", "string", "Patient identifier"),
                MCPToolParameter("datscan_path", "string", "Path to DaTscan image"),
            ],
            handler=self._analyze_datscan,
            category="parkinson"
        ))

        # ====================================================================
        # Schizophrenia Tools
        # ====================================================================

        self.tools.append(MCPTool(
            name="analyze_eeg_schizophrenia",
            description="Analyze EEG data for schizophrenia biomarkers including gamma oscillations, P300, and MMN",
            parameters=[
                MCPToolParameter("patient_id", "string", "Patient identifier"),
                MCPToolParameter("eeg_data_path", "string", "Path to EEG data file"),
                MCPToolParameter("sampling_rate", "number", "EEG sampling rate in Hz", default=256),
                MCPToolParameter("channels", "number", "Number of EEG channels", default=64),
            ],
            handler=self._analyze_eeg_schizophrenia,
            category="schizophrenia"
        ))

        self.tools.append(MCPTool(
            name="analyze_fmri_connectivity",
            description="Analyze fMRI functional connectivity for schizophrenia patterns",
            parameters=[
                MCPToolParameter("patient_id", "string", "Patient identifier"),
                MCPToolParameter("fmri_path", "string", "Path to fMRI data"),
                MCPToolParameter("atlas", "string", "Brain atlas for parcellation",
                                enum=["AAL", "Schaefer", "Gordon"], default="AAL"),
            ],
            handler=self._analyze_fmri_connectivity,
            category="schizophrenia"
        ))

        self.tools.append(MCPTool(
            name="calculate_panss",
            description="Calculate PANSS (Positive and Negative Syndrome Scale) score",
            parameters=[
                MCPToolParameter("patient_id", "string", "Patient identifier"),
                MCPToolParameter("symptom_ratings", "object", "Symptom ratings dictionary"),
            ],
            handler=self._calculate_panss,
            category="schizophrenia"
        ))

        # ====================================================================
        # Ensemble/Multi-Disease Tools
        # ====================================================================

        self.tools.append(MCPTool(
            name="multi_disease_screening",
            description="Perform comprehensive screening for multiple neurological diseases (Alzheimer's, Parkinson's, Schizophrenia)",
            parameters=[
                MCPToolParameter("patient_id", "string", "Patient identifier"),
                MCPToolParameter("patient_data", "object", "Complete patient data including imaging and clinical"),
                MCPToolParameter("diseases", "array", "List of diseases to screen for",
                                default=["alzheimer", "parkinson", "schizophrenia"]),
            ],
            handler=self._multi_disease_screening,
            category="ensemble"
        ))

        self.tools.append(MCPTool(
            name="get_diagnosis_report",
            description="Generate comprehensive diagnosis report with recommendations",
            parameters=[
                MCPToolParameter("patient_id", "string", "Patient identifier"),
                MCPToolParameter("analysis_results", "object", "Results from disease analyses"),
            ],
            handler=self._get_diagnosis_report,
            category="ensemble"
        ))

    # ========================================================================
    # Tool Handlers
    # ========================================================================

    async def _analyze_alzheimer_mri(self, patient_id: str, mri_data_path: str,
                                     analysis_type: str = "full") -> Dict:
        """Analyze MRI for Alzheimer's markers"""
        import numpy as np

        # Simulated analysis results
        results = {
            "patient_id": patient_id,
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "findings": {
                "hippocampal_volume": {
                    "left": round(np.random.uniform(2800, 4200), 1),
                    "right": round(np.random.uniform(2900, 4300), 1),
                    "unit": "mm³",
                    "percentile": round(np.random.uniform(15, 85), 1),
                    "status": np.random.choice(["normal", "mild_atrophy", "moderate_atrophy"])
                },
                "cortical_thickness": {
                    "mean": round(np.random.uniform(2.0, 3.0), 2),
                    "temporal": round(np.random.uniform(1.8, 2.8), 2),
                    "parietal": round(np.random.uniform(2.0, 3.0), 2),
                    "frontal": round(np.random.uniform(2.2, 3.2), 2),
                    "unit": "mm"
                },
                "ventricular_volume": {
                    "value": round(np.random.uniform(20, 60), 1),
                    "unit": "ml",
                    "status": np.random.choice(["normal", "enlarged"])
                },
                "white_matter_lesions": {
                    "fazekas_score": np.random.randint(0, 4),
                    "volume": round(np.random.uniform(0, 15), 1)
                }
            },
            "risk_score": round(np.random.uniform(0.1, 0.9), 3),
            "confidence": round(np.random.uniform(0.75, 0.95), 3)
        }

        return results

    async def _assess_cognitive_status(self, patient_id: str, mmse_score: float,
                                       cdr_score: float, age: float,
                                       adas_cog: float = None) -> Dict:
        """Assess cognitive status from clinical scores"""

        # Determine stage based on scores
        if mmse_score >= 27 and cdr_score == 0:
            stage = "CN"
            stage_description = "Cognitively Normal"
        elif mmse_score >= 21 and cdr_score <= 0.5:
            stage = "MCI"
            stage_description = "Mild Cognitive Impairment"
        else:
            stage = "AD"
            stage_description = "Alzheimer's Disease"

        return {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "scores": {
                "mmse": mmse_score,
                "cdr": cdr_score,
                "adas_cog": adas_cog
            },
            "assessment": {
                "stage": stage,
                "description": stage_description,
                "confidence": round(0.7 + (30 - mmse_score) * 0.01, 3)
            },
            "risk_factors": {
                "age_risk": "elevated" if age > 65 else "normal",
                "cognitive_decline_rate": "unknown"
            }
        }

    async def _predict_alzheimer_stage(self, patient_id: str, features: Dict) -> Dict:
        """Predict Alzheimer's stage from combined features"""
        import numpy as np

        stages = ["CN", "MCI", "AD"]
        probabilities = np.random.dirichlet([2, 3, 2])

        predicted_stage = stages[np.argmax(probabilities)]

        return {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "prediction": {
                "stage": predicted_stage,
                "probabilities": {
                    "CN": round(float(probabilities[0]), 4),
                    "MCI": round(float(probabilities[1]), 4),
                    "AD": round(float(probabilities[2]), 4)
                },
                "confidence": round(float(np.max(probabilities)), 4)
            },
            "model_info": {
                "model_type": "ensemble",
                "version": "1.0.0"
            }
        }

    async def _analyze_voice_parkinson(self, patient_id: str, audio_path: str,
                                       sample_rate: int = 16000) -> Dict:
        """Analyze voice for Parkinson's markers"""
        import numpy as np

        return {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "voice_features": {
                "jitter_percent": round(np.random.uniform(0.1, 2.5), 4),
                "jitter_abs": round(np.random.uniform(10, 100) * 1e-6, 8),
                "shimmer_percent": round(np.random.uniform(1, 12), 4),
                "shimmer_db": round(np.random.uniform(0.1, 1.5), 4),
                "hnr": round(np.random.uniform(15, 35), 2),
                "nhr": round(np.random.uniform(0.01, 0.1), 4),
                "dfa": round(np.random.uniform(0.5, 0.8), 4),
                "rpde": round(np.random.uniform(0.3, 0.6), 4),
                "ppe": round(np.random.uniform(0.1, 0.3), 4),
                "f0_mean": round(np.random.uniform(100, 200), 2),
                "f0_std": round(np.random.uniform(5, 40), 2)
            },
            "assessment": {
                "parkinson_probability": round(np.random.uniform(0.2, 0.8), 4),
                "voice_quality": np.random.choice(["normal", "mild_dysphonia", "moderate_dysphonia"]),
                "confidence": round(np.random.uniform(0.7, 0.95), 4)
            }
        }

    async def _analyze_gait_parkinson(self, patient_id: str, sensor_data_path: str,
                                      sensor_type: str = "combined") -> Dict:
        """Analyze gait for Parkinson's markers"""
        import numpy as np

        return {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "gait_features": {
                "stride_length": round(np.random.uniform(0.8, 1.4), 3),
                "stride_time": round(np.random.uniform(0.9, 1.3), 3),
                "cadence": round(np.random.uniform(90, 130), 1),
                "gait_speed": round(np.random.uniform(0.7, 1.4), 3),
                "step_width": round(np.random.uniform(0.08, 0.15), 3),
                "swing_time": round(np.random.uniform(0.35, 0.45), 3),
                "stance_time": round(np.random.uniform(0.55, 0.70), 3),
                "double_support": round(np.random.uniform(0.1, 0.25), 3),
                "stride_variability": round(np.random.uniform(0.02, 0.12), 4),
                "asymmetry_index": round(np.random.uniform(0, 0.2), 4)
            },
            "freezing_analysis": {
                "fog_episodes": np.random.randint(0, 5),
                "fog_duration_total": round(np.random.uniform(0, 30), 1),
                "fog_probability": round(np.random.uniform(0, 0.5), 4)
            },
            "assessment": {
                "gait_impairment": np.random.choice(["none", "mild", "moderate", "severe"]),
                "parkinson_probability": round(np.random.uniform(0.2, 0.8), 4),
                "confidence": round(np.random.uniform(0.7, 0.92), 4)
            }
        }

    async def _calculate_updrs(self, patient_id: str, motor_assessments: Dict) -> Dict:
        """Calculate UPDRS score"""
        import numpy as np

        # Default scores if not provided
        defaults = {
            'speech': 1, 'facial_expression': 1, 'tremor_at_rest': 2,
            'action_tremor': 1, 'rigidity': 2, 'finger_tapping': 2,
            'hand_movements': 2, 'leg_agility': 1, 'arising_from_chair': 1,
            'posture': 1, 'gait': 2, 'postural_stability': 1, 'bradykinesia': 2
        }

        scores = {k: motor_assessments.get(k, v) for k, v in defaults.items()}
        total_score = sum(scores.values())

        # Severity classification
        if total_score < 10:
            severity = "minimal"
        elif total_score < 20:
            severity = "mild"
        elif total_score < 40:
            severity = "moderate"
        else:
            severity = "severe"

        return {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "updrs_part_iii": {
                "subscores": scores,
                "total_score": total_score,
                "max_score": 52,
                "severity": severity
            },
            "interpretation": {
                "motor_impairment_level": severity,
                "dominant_symptoms": ["bradykinesia", "tremor"] if scores.get('bradykinesia', 0) > 1 else ["rigidity"]
            }
        }

    async def _analyze_datscan(self, patient_id: str, datscan_path: str) -> Dict:
        """Analyze DaTscan imaging"""
        import numpy as np

        return {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "sbr_values": {
                "left_putamen": round(np.random.uniform(1.0, 3.0), 3),
                "right_putamen": round(np.random.uniform(1.0, 3.0), 3),
                "left_caudate": round(np.random.uniform(1.5, 3.5), 3),
                "right_caudate": round(np.random.uniform(1.5, 3.5), 3),
                "putamen_caudate_ratio": round(np.random.uniform(0.6, 1.0), 3),
                "asymmetry_index": round(np.random.uniform(0, 0.3), 4)
            },
            "assessment": {
                "dopaminergic_deficit": np.random.choice([True, False]),
                "pattern": np.random.choice(["normal", "pd_pattern", "atypical"]),
                "confidence": round(np.random.uniform(0.8, 0.95), 4)
            }
        }

    async def _analyze_eeg_schizophrenia(self, patient_id: str, eeg_data_path: str,
                                         sampling_rate: int = 256,
                                         channels: int = 64) -> Dict:
        """Analyze EEG for schizophrenia biomarkers"""
        import numpy as np

        return {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "spectral_features": {
                "delta_power": round(np.random.uniform(0.1, 0.3), 4),
                "theta_power": round(np.random.uniform(0.1, 0.25), 4),
                "alpha_power": round(np.random.uniform(0.15, 0.35), 4),
                "beta_power": round(np.random.uniform(0.1, 0.2), 4),
                "gamma_power": round(np.random.uniform(0.02, 0.12), 4),
                "gamma_40hz": round(np.random.uniform(0.01, 0.08), 4)
            },
            "erp_features": {
                "p300_amplitude": round(np.random.uniform(3, 15), 2),
                "p300_latency": round(np.random.uniform(280, 400), 1),
                "mmn_amplitude": round(np.random.uniform(2, 8), 2),
                "mmn_latency": round(np.random.uniform(150, 250), 1),
                "p50_ratio": round(np.random.uniform(0.3, 1.0), 3)
            },
            "connectivity": {
                "gamma_phase_locking": round(np.random.uniform(0.3, 0.8), 4),
                "frontal_coherence": round(np.random.uniform(0.2, 0.6), 4)
            },
            "assessment": {
                "schizophrenia_probability": round(np.random.uniform(0.2, 0.8), 4),
                "biomarker_abnormalities": np.random.randint(0, 5),
                "confidence": round(np.random.uniform(0.7, 0.9), 4)
            }
        }

    async def _analyze_fmri_connectivity(self, patient_id: str, fmri_path: str,
                                         atlas: str = "AAL") -> Dict:
        """Analyze fMRI functional connectivity"""
        import numpy as np

        return {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "global_metrics": {
                "global_efficiency": round(np.random.uniform(0.3, 0.6), 4),
                "local_efficiency": round(np.random.uniform(0.5, 0.8), 4),
                "modularity": round(np.random.uniform(0.3, 0.6), 4),
                "small_worldness": round(np.random.uniform(1.0, 3.0), 4)
            },
            "network_connectivity": {
                "dmn_strength": round(np.random.uniform(0.2, 0.6), 4),
                "salience_network": round(np.random.uniform(0.2, 0.5), 4),
                "executive_network": round(np.random.uniform(0.2, 0.5), 4),
                "dmn_salience_coupling": round(np.random.uniform(-0.2, 0.3), 4)
            },
            "hub_analysis": {
                "hub_disruption_index": round(np.random.uniform(0, 1), 4),
                "affected_hubs": ["PCC", "mPFC"] if np.random.random() > 0.5 else []
            },
            "assessment": {
                "connectivity_pattern": np.random.choice(["normal", "sz_pattern", "atypical"]),
                "schizophrenia_probability": round(np.random.uniform(0.2, 0.8), 4),
                "confidence": round(np.random.uniform(0.7, 0.88), 4)
            }
        }

    async def _calculate_panss(self, patient_id: str, symptom_ratings: Dict) -> Dict:
        """Calculate PANSS score"""
        import numpy as np

        # Default ratings
        positive_items = ['delusions', 'disorganization', 'hallucinations',
                         'excitement', 'grandiosity', 'suspiciousness', 'hostility']
        negative_items = ['blunted_affect', 'emotional_withdrawal', 'poor_rapport',
                         'passive_withdrawal', 'abstract_thinking', 'spontaneity',
                         'stereotyped_thinking']

        positive_score = sum(symptom_ratings.get(item, np.random.randint(1, 5))
                            for item in positive_items)
        negative_score = sum(symptom_ratings.get(item, np.random.randint(1, 5))
                            for item in negative_items)
        general_score = symptom_ratings.get('general_total', np.random.randint(20, 50))

        total = positive_score + negative_score + general_score

        # Severity
        if total < 58:
            severity = "mild"
        elif total < 75:
            severity = "moderate"
        elif total < 95:
            severity = "marked"
        else:
            severity = "severe"

        return {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "panss_scores": {
                "positive_scale": positive_score,
                "negative_scale": negative_score,
                "general_scale": general_score,
                "total_score": total
            },
            "interpretation": {
                "severity": severity,
                "predominant_symptoms": "positive" if positive_score > negative_score else "negative"
            }
        }

    async def _multi_disease_screening(self, patient_id: str, patient_data: Dict,
                                       diseases: List[str] = None) -> Dict:
        """Comprehensive multi-disease screening"""
        import numpy as np

        diseases = diseases or ["alzheimer", "parkinson", "schizophrenia"]

        results = {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "screening_results": {}
        }

        for disease in diseases:
            prob = round(np.random.uniform(0.1, 0.7), 4)
            results["screening_results"][disease] = {
                "probability": prob,
                "risk_level": "high" if prob > 0.5 else "medium" if prob > 0.3 else "low",
                "confidence": round(np.random.uniform(0.7, 0.95), 4)
            }

        # Determine primary concern
        max_prob = 0
        primary = None
        for disease, data in results["screening_results"].items():
            if data["probability"] > max_prob:
                max_prob = data["probability"]
                primary = disease

        results["summary"] = {
            "primary_concern": primary,
            "highest_probability": max_prob,
            "requires_followup": max_prob > 0.4
        }

        return results

    async def _get_diagnosis_report(self, patient_id: str,
                                    analysis_results: Dict) -> Dict:
        """Generate comprehensive diagnosis report"""

        recommendations = []

        for disease, result in analysis_results.get("screening_results", {}).items():
            if result.get("risk_level") == "high":
                recommendations.append(f"Urgent {disease} specialist consultation recommended")
            elif result.get("risk_level") == "medium":
                recommendations.append(f"Follow-up {disease} assessment in 3-6 months")

        if not recommendations:
            recommendations.append("Continue routine monitoring")
            recommendations.append("Annual neurological checkup recommended")

        return {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "report_type": "comprehensive_neurological_assessment",
            "analysis_summary": analysis_results,
            "recommendations": recommendations,
            "disclaimer": "This AI-assisted analysis should be reviewed by a qualified healthcare professional"
        }

    def get_all_tools(self) -> List[MCPTool]:
        """Return all registered tools"""
        return self.tools


# ============================================================================
# Create and Run MCP Server
# ============================================================================

def create_neuro_disease_mcp_server() -> MCPServer:
    """Create configured MCP server for neurological disease detection"""

    server = MCPServer(
        name="neuro-disease-detection",
        version="1.0.0"
    )

    # Register tools
    tools = NeuroDiseaseTools()
    for tool in tools.get_all_tools():
        server.register_tool(tool)

    # Register resources
    server.register_resource(MCPResource(
        uri="neuro://models/alzheimer",
        name="Alzheimer's Detection Models",
        description="Available models for Alzheimer's disease detection"
    ))

    server.register_resource(MCPResource(
        uri="neuro://models/parkinson",
        name="Parkinson's Detection Models",
        description="Available models for Parkinson's disease detection"
    ))

    server.register_resource(MCPResource(
        uri="neuro://models/schizophrenia",
        name="Schizophrenia Detection Models",
        description="Available models for Schizophrenia detection"
    ))

    return server


async def main():
    """Main entry point for MCP server"""
    print("=" * 70)
    print("  MCP Server for Neurological Disease Detection")
    print("  Model Context Protocol Implementation")
    print("=" * 70)

    server = create_neuro_disease_mcp_server()

    print(f"\nServer: {server.name} v{server.version}")
    print(f"Tools registered: {len(server.tools)}")
    print(f"Resources registered: {len(server.resources)}")

    print("\nAvailable Tools:")
    for tool_name, tool in server.tools.items():
        print(f"  • {tool_name}: {tool.description[:60]}...")

    # Demo: Test tool execution
    print("\n" + "=" * 70)
    print("  DEMO: Testing MCP Tool Execution")
    print("=" * 70)

    # Test multi-disease screening
    request = JSONRPCRequest(
        method="tools/call",
        params={
            "name": "multi_disease_screening",
            "arguments": {
                "patient_id": "DEMO_001",
                "patient_data": {"age": 72, "clinical_data": {}},
                "diseases": ["alzheimer", "parkinson", "schizophrenia"]
            }
        }
    )

    response = await server.handle_message(json.dumps(request.to_dict()))
    result = json.loads(response)

    print("\nMulti-Disease Screening Result:")
    print(json.dumps(json.loads(result['result']['content'][0]['text']), indent=2))

    print("\n" + "=" * 70)
    print("  MCP Server Ready")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
