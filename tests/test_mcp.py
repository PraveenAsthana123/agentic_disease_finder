"""
Unit Tests for MCP (Model Context Protocol) Module
"""

import pytest
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.mcp_server import (
    MCPServer,
    MCPTool,
    MCPToolParameter,
    NeuroDiseaseTools,
    JSONRPCRequest,
    JSONRPCResponse,
    create_neuro_disease_mcp_server
)
from mcp.mcp_client import MCPClient, MCPAgentOrchestrator, ToolCall


class TestMCPServer:
    """Test MCP Server"""

    @pytest.fixture
    def server(self):
        return create_neuro_disease_mcp_server()

    def test_server_creation(self, server):
        assert server is not None
        assert server.name == "neuro-disease-mcp"
        assert len(server.tools) > 0

    def test_tool_list(self, server):
        tools = server.tools
        assert 'analyze_alzheimer_mri' in tools
        assert 'analyze_voice_parkinson' in tools
        assert 'analyze_eeg_schizophrenia' in tools

    @pytest.mark.asyncio
    async def test_initialize(self, server):
        request = JSONRPCRequest(
            method="initialize",
            params={"clientInfo": {"name": "test", "version": "1.0"}}
        )
        response = await server.handle_message(json.dumps(request.to_dict()))
        result = json.loads(response)

        assert 'result' in result
        assert 'capabilities' in result['result']

    @pytest.mark.asyncio
    async def test_list_tools(self, server):
        # Initialize first
        init_request = JSONRPCRequest(method="initialize", params={})
        await server.handle_message(json.dumps(init_request.to_dict()))

        # List tools
        request = JSONRPCRequest(method="tools/list", params={})
        response = await server.handle_message(json.dumps(request.to_dict()))
        result = json.loads(response)

        assert 'result' in result
        assert 'tools' in result['result']
        assert len(result['result']['tools']) >= 12

    @pytest.mark.asyncio
    async def test_ping(self, server):
        request = JSONRPCRequest(method="ping", params={})
        response = await server.handle_message(json.dumps(request.to_dict()))
        result = json.loads(response)

        assert result['result']['status'] == 'pong'


class TestMCPClient:
    """Test MCP Client"""

    @pytest.fixture
    def client(self):
        return MCPClient("test-client")

    @pytest.fixture
    def server(self):
        return create_neuro_disease_mcp_server()

    @pytest.mark.asyncio
    async def test_client_creation(self, client):
        assert client is not None
        assert client.client_id == "test-client"

    @pytest.mark.asyncio
    async def test_connect(self, client, server):
        await client.connect(server)

        assert client.initialized
        assert len(client.available_tools) > 0

    @pytest.mark.asyncio
    async def test_list_tools(self, client, server):
        await client.connect(server)
        tools = client.list_tools()

        assert len(tools) >= 12
        assert all('name' in t and 'description' in t for t in tools)

    @pytest.mark.asyncio
    async def test_ping(self, client, server):
        await client.connect(server)
        result = await client.ping()

        assert result is True

    @pytest.mark.asyncio
    async def test_call_tool(self, client, server):
        await client.connect(server)

        result = await client.call_tool(
            "multi_disease_screening",
            patient_id="TEST001",
            patient_data={"age": 70},
            diseases=["alzheimer"]
        )

        assert 'screening_results' in result or 'patient_id' in result


class TestMCPAgentOrchestrator:
    """Test MCP Agent Orchestrator"""

    @pytest.fixture
    def orchestrator(self):
        return MCPAgentOrchestrator()

    @pytest.mark.asyncio
    async def test_initialization(self, orchestrator):
        await orchestrator.initialize()

        assert orchestrator.initialized
        assert len(orchestrator.client.available_tools) > 0

    @pytest.mark.asyncio
    async def test_analyze_patient(self, orchestrator):
        await orchestrator.initialize()

        patient_data = {
            "mri_path": "/test/mri.nii",
            "clinical_data": {"age": 72, "mmse": 24}
        }

        results = await orchestrator.analyze_patient(
            patient_id="TEST001",
            patient_data=patient_data,
            diseases=["alzheimer"]
        )

        assert 'patient_id' in results
        assert 'screening' in results

    @pytest.mark.asyncio
    async def test_tool_statistics(self, orchestrator):
        await orchestrator.initialize()

        # Make some calls
        await orchestrator.client.call_tool(
            "multi_disease_screening",
            patient_id="TEST",
            patient_data={},
            diseases=["alzheimer"]
        )

        stats = orchestrator.get_tool_statistics()

        assert 'total_calls' in stats
        assert stats['total_calls'] >= 1


class TestNeuroDiseaseTools:
    """Test NeuroDiseaseTools implementation"""

    @pytest.fixture
    def tools(self):
        return NeuroDiseaseTools()

    @pytest.mark.asyncio
    async def test_analyze_alzheimer_mri(self, tools):
        result = await tools.analyze_alzheimer_mri(
            patient_id="TEST001",
            mri_data_path="/test/mri.nii",
            analysis_type="full"
        )

        assert 'patient_id' in result
        assert 'analysis_type' in result
        assert 'biomarkers' in result

    @pytest.mark.asyncio
    async def test_analyze_voice_parkinson(self, tools):
        result = await tools.analyze_voice_parkinson(
            patient_id="TEST001",
            audio_path="/test/voice.wav"
        )

        assert 'patient_id' in result
        assert 'voice_features' in result

    @pytest.mark.asyncio
    async def test_analyze_eeg_schizophrenia(self, tools):
        result = await tools.analyze_eeg_schizophrenia(
            patient_id="TEST001",
            eeg_data_path="/test/eeg.edf"
        )

        assert 'patient_id' in result
        assert 'spectral_features' in result

    @pytest.mark.asyncio
    async def test_multi_disease_screening(self, tools):
        result = await tools.multi_disease_screening(
            patient_id="TEST001",
            patient_data={"age": 70},
            diseases=["alzheimer", "parkinson", "schizophrenia"]
        )

        assert 'patient_id' in result
        assert 'screening_results' in result


class TestJSONRPC:
    """Test JSON-RPC protocol implementation"""

    def test_request_creation(self):
        request = JSONRPCRequest(
            method="test_method",
            params={"key": "value"}
        )

        assert request.jsonrpc == "2.0"
        assert request.method == "test_method"
        assert request.params == {"key": "value"}

    def test_request_to_dict(self):
        request = JSONRPCRequest(method="test", params={})
        d = request.to_dict()

        assert 'jsonrpc' in d
        assert 'method' in d
        assert 'id' in d

    def test_response_creation(self):
        response = JSONRPCResponse(
            result={"status": "ok"},
            id="123"
        )

        assert response.result == {"status": "ok"}
        assert response.id == "123"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
