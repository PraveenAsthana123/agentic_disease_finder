"""
MCP Client for Agentic AI Orchestration
========================================
Model Context Protocol client implementation for connecting
AI agents to the neurological disease detection MCP server.

Author: Research Team
Project: Neurological Disease Detection using Agentic AI with MCP
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from .mcp_server import (
    MCPServer, MCPTool, JSONRPCRequest, JSONRPCResponse,
    create_neuro_disease_mcp_server
)

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Record of a tool call"""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: float = 0
    success: bool = True
    error: Optional[str] = None


class MCPClient:
    """
    MCP Client for AI Agent Integration

    Connects to MCP servers and provides:
    - Tool discovery
    - Tool execution
    - Session management
    - Result caching
    """

    def __init__(self, client_id: str = None):
        """
        Initialize MCP Client

        Parameters
        ----------
        client_id : str
            Unique client identifier
        """
        self.client_id = client_id or str(uuid.uuid4())
        self.server: Optional[MCPServer] = None
        self.available_tools: Dict[str, MCPTool] = {}
        self.call_history: List[ToolCall] = []
        self.initialized = False

        logger.info(f"MCP Client {self.client_id} created")

    async def connect(self, server: MCPServer):
        """
        Connect to MCP server

        Parameters
        ----------
        server : MCPServer
            MCP server instance
        """
        self.server = server

        # Initialize connection
        request = JSONRPCRequest(
            method="initialize",
            params={"clientInfo": {"name": "neuro-agent", "version": "1.0.0"}}
        )

        response = await self._send_request(request)

        if response and "capabilities" in response:
            self.initialized = True
            logger.info(f"Connected to MCP server: {server.name}")

            # Discover available tools
            await self._discover_tools()
        else:
            raise ConnectionError("Failed to initialize MCP connection")

    async def _send_request(self, request: JSONRPCRequest) -> Optional[Dict]:
        """Send request to server and parse response"""
        if not self.server:
            raise RuntimeError("Not connected to server")

        response_str = await self.server.handle_message(
            json.dumps(request.to_dict())
        )
        response = json.loads(response_str)

        if "error" in response:
            logger.error(f"MCP Error: {response['error']}")
            return None

        return response.get("result")

    async def _discover_tools(self):
        """Discover available tools from server"""
        request = JSONRPCRequest(method="tools/list", params={})
        result = await self._send_request(request)

        if result and "tools" in result:
            self.available_tools = {}
            for tool_def in result["tools"]:
                tool = MCPTool(
                    name=tool_def["name"],
                    description=tool_def["description"],
                    parameters=[]  # Simplified
                )
                self.available_tools[tool.name] = tool

            logger.info(f"Discovered {len(self.available_tools)} tools")

    def list_tools(self) -> List[Dict]:
        """List available tools"""
        return [
            {"name": t.name, "description": t.description}
            for t in self.available_tools.values()
        ]

    def get_tools_by_category(self, category: str) -> List[str]:
        """Get tool names by category"""
        category_map = {
            "alzheimer": ["analyze_alzheimer_mri", "assess_cognitive_status", "predict_alzheimer_stage"],
            "parkinson": ["analyze_voice_parkinson", "analyze_gait_parkinson", "calculate_updrs", "analyze_datscan"],
            "schizophrenia": ["analyze_eeg_schizophrenia", "analyze_fmri_connectivity", "calculate_panss"],
            "ensemble": ["multi_disease_screening", "get_diagnosis_report"]
        }
        return category_map.get(category.lower(), [])

    async def call_tool(self, tool_name: str, **arguments) -> Dict:
        """
        Call a tool on the MCP server

        Parameters
        ----------
        tool_name : str
            Name of the tool to call
        **arguments : dict
            Tool arguments

        Returns
        -------
        result : dict
            Tool execution result
        """
        if tool_name not in self.available_tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        start_time = datetime.now()

        request = JSONRPCRequest(
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments
            }
        )

        result = await self._send_request(request)

        duration = (datetime.now() - start_time).total_seconds() * 1000

        # Parse result
        if result and "content" in result:
            content = result["content"][0]
            if content["type"] == "text":
                try:
                    parsed_result = json.loads(content["text"])
                except json.JSONDecodeError:
                    parsed_result = {"raw": content["text"]}

                # Record call
                self.call_history.append(ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=parsed_result,
                    duration_ms=duration,
                    success=not result.get("isError", False)
                ))

                return parsed_result

        return {"error": "No result returned"}

    async def ping(self) -> bool:
        """Ping the server"""
        request = JSONRPCRequest(method="ping", params={})
        result = await self._send_request(request)
        return result is not None and result.get("status") == "pong"


class MCPAgentOrchestrator:
    """
    Agentic AI Orchestrator using MCP

    Coordinates multiple AI agents through MCP protocol for
    comprehensive neurological disease analysis.
    """

    def __init__(self):
        self.server = create_neuro_disease_mcp_server()
        self.client = MCPClient()
        self.initialized = False

        # Agent state
        self.current_patient = None
        self.analysis_results = {}

    async def initialize(self):
        """Initialize the orchestrator"""
        await self.client.connect(self.server)
        self.initialized = True
        logger.info("MCP Agent Orchestrator initialized")

    async def analyze_patient(self, patient_id: str, patient_data: Dict,
                             diseases: List[str] = None) -> Dict:
        """
        Run comprehensive patient analysis using MCP tools

        Parameters
        ----------
        patient_id : str
            Patient identifier
        patient_data : dict
            Patient data including imaging and clinical data
        diseases : list
            Diseases to screen for

        Returns
        -------
        results : dict
            Complete analysis results
        """
        if not self.initialized:
            await self.initialize()

        diseases = diseases or ["alzheimer", "parkinson", "schizophrenia"]
        self.current_patient = patient_id
        self.analysis_results = {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "disease_analyses": {}
        }

        # Run multi-disease screening first
        screening = await self.client.call_tool(
            "multi_disease_screening",
            patient_id=patient_id,
            patient_data=patient_data,
            diseases=diseases
        )
        self.analysis_results["screening"] = screening

        # Detailed analysis for each disease
        for disease in diseases:
            self.analysis_results["disease_analyses"][disease] = \
                await self._analyze_disease(disease, patient_id, patient_data)

        # Generate final report
        report = await self.client.call_tool(
            "get_diagnosis_report",
            patient_id=patient_id,
            analysis_results=self.analysis_results
        )
        self.analysis_results["report"] = report

        return self.analysis_results

    async def _analyze_disease(self, disease: str, patient_id: str,
                               patient_data: Dict) -> Dict:
        """Run disease-specific analysis"""
        results = {}

        if disease == "alzheimer":
            # MRI analysis
            if "mri_path" in patient_data:
                results["mri_analysis"] = await self.client.call_tool(
                    "analyze_alzheimer_mri",
                    patient_id=patient_id,
                    mri_data_path=patient_data.get("mri_path", ""),
                    analysis_type="full"
                )

            # Cognitive assessment
            clinical = patient_data.get("clinical_data", {})
            if "mmse" in clinical:
                results["cognitive_assessment"] = await self.client.call_tool(
                    "assess_cognitive_status",
                    patient_id=patient_id,
                    mmse_score=clinical.get("mmse", 25),
                    cdr_score=clinical.get("cdr", 0.5),
                    age=clinical.get("age", 70)
                )

            # Stage prediction
            results["stage_prediction"] = await self.client.call_tool(
                "predict_alzheimer_stage",
                patient_id=patient_id,
                features=patient_data
            )

        elif disease == "parkinson":
            # Voice analysis
            if "audio_path" in patient_data:
                results["voice_analysis"] = await self.client.call_tool(
                    "analyze_voice_parkinson",
                    patient_id=patient_id,
                    audio_path=patient_data.get("audio_path", "")
                )

            # Gait analysis
            if "gait_data_path" in patient_data:
                results["gait_analysis"] = await self.client.call_tool(
                    "analyze_gait_parkinson",
                    patient_id=patient_id,
                    sensor_data_path=patient_data.get("gait_data_path", "")
                )

            # UPDRS calculation
            motor = patient_data.get("motor_assessments", {})
            results["updrs"] = await self.client.call_tool(
                "calculate_updrs",
                patient_id=patient_id,
                motor_assessments=motor
            )

        elif disease == "schizophrenia":
            # EEG analysis
            if "eeg_path" in patient_data:
                results["eeg_analysis"] = await self.client.call_tool(
                    "analyze_eeg_schizophrenia",
                    patient_id=patient_id,
                    eeg_data_path=patient_data.get("eeg_path", "")
                )

            # fMRI connectivity
            if "fmri_path" in patient_data:
                results["connectivity_analysis"] = await self.client.call_tool(
                    "analyze_fmri_connectivity",
                    patient_id=patient_id,
                    fmri_path=patient_data.get("fmri_path", "")
                )

            # PANSS calculation
            symptoms = patient_data.get("symptom_ratings", {})
            results["panss"] = await self.client.call_tool(
                "calculate_panss",
                patient_id=patient_id,
                symptom_ratings=symptoms
            )

        return results

    def get_tool_statistics(self) -> Dict:
        """Get statistics on tool usage"""
        stats = {
            "total_calls": len(self.client.call_history),
            "by_tool": {},
            "avg_duration_ms": 0,
            "success_rate": 0
        }

        if self.client.call_history:
            durations = []
            successes = 0

            for call in self.client.call_history:
                stats["by_tool"][call.tool_name] = \
                    stats["by_tool"].get(call.tool_name, 0) + 1
                durations.append(call.duration_ms)
                if call.success:
                    successes += 1

            stats["avg_duration_ms"] = sum(durations) / len(durations)
            stats["success_rate"] = successes / len(self.client.call_history)

        return stats


async def demo():
    """Demonstrate MCP Agentic AI"""
    print("=" * 70)
    print("  MCP Agentic AI - Neurological Disease Detection Demo")
    print("=" * 70)

    # Create orchestrator
    orchestrator = MCPAgentOrchestrator()
    await orchestrator.initialize()

    print(f"\nAvailable MCP Tools: {len(orchestrator.client.available_tools)}")
    for tool in orchestrator.client.list_tools()[:5]:
        print(f"  • {tool['name']}")
    print("  ...")

    # Demo patient data
    patient_data = {
        "mri_path": "/data/patient001/mri.nii",
        "audio_path": "/data/patient001/voice.wav",
        "eeg_path": "/data/patient001/eeg.edf",
        "clinical_data": {
            "age": 72,
            "mmse": 24,
            "cdr": 0.5
        },
        "motor_assessments": {
            "tremor_at_rest": 2,
            "bradykinesia": 2,
            "rigidity": 1
        },
        "symptom_ratings": {}
    }

    print("\n" + "-" * 70)
    print("  Running Comprehensive Analysis...")
    print("-" * 70)

    results = await orchestrator.analyze_patient(
        patient_id="DEMO_001",
        patient_data=patient_data
    )

    # Print results
    print("\n📊 SCREENING RESULTS:")
    screening = results.get("screening", {})
    for disease, data in screening.get("screening_results", {}).items():
        prob = data.get("probability", 0)
        risk = data.get("risk_level", "unknown")
        print(f"  • {disease.capitalize()}: {prob:.1%} probability ({risk} risk)")

    print("\n📋 RECOMMENDATIONS:")
    report = results.get("report", {})
    for rec in report.get("recommendations", []):
        print(f"  → {rec}")

    # Tool statistics
    stats = orchestrator.get_tool_statistics()
    print(f"\n📈 TOOL STATISTICS:")
    print(f"  Total MCP tool calls: {stats['total_calls']}")
    print(f"  Average duration: {stats['avg_duration_ms']:.1f}ms")
    print(f"  Success rate: {stats['success_rate']:.1%}")

    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo())
