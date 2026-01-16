"""
MCP (Model Context Protocol) Module for Neurological Disease Detection

Implements Anthropic's MCP specification for Agentic AI:
- JSON-RPC 2.0 protocol
- Tool discovery and execution
- Agent-to-Agent communication
- Session state management
"""

from .mcp_server import (
    MCPServer,
    MCPTool,
    MCPToolParameter,
    MCPResource,
    NeuroDiseaseTools,
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCError,
    create_neuro_disease_mcp_server
)

from .mcp_client import (
    MCPClient,
    MCPAgentOrchestrator,
    ToolCall
)

__all__ = [
    # Server
    'MCPServer',
    'MCPTool',
    'MCPToolParameter',
    'MCPResource',
    'NeuroDiseaseTools',
    'JSONRPCRequest',
    'JSONRPCResponse',
    'JSONRPCError',
    'create_neuro_disease_mcp_server',
    # Client
    'MCPClient',
    'MCPAgentOrchestrator',
    'ToolCall'
]
