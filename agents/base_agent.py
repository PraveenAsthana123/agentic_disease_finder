"""
Base Agent Module for Neurological Disease Detection
=====================================================
Provides the foundational agent architecture for the Agentic AI system.
Implements core agent capabilities, state management, and A2A communication.

Author: Research Team
Project: Neurological Disease Detection using Agentic AI
"""

import uuid
import json
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from datetime import datetime
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent lifecycle states"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    TERMINATED = "terminated"


class MessageType(Enum):
    """Types of inter-agent messages"""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"
    COMMAND = "command"
    DATA = "data"
    RESULT = "result"
    ERROR = "error"
    REGISTER = "register"
    DEREGISTER = "deregister"


@dataclass
class AgentMessage:
    """
    Standard message format for Agent-to-Agent (A2A) communication

    Attributes
    ----------
    id : str
        Unique message identifier
    sender_id : str
        ID of sending agent
    receiver_id : str
        ID of receiving agent (or 'broadcast')
    message_type : MessageType
        Type of message
    action : str
        Action to perform
    payload : dict
        Message data
    timestamp : str
        ISO format timestamp
    correlation_id : str
        ID linking related messages
    priority : int
        Message priority (0-10)
    """
    sender_id: str
    receiver_id: str
    message_type: MessageType
    action: str
    payload: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    correlation_id: Optional[str] = None
    priority: int = 5

    def to_dict(self) -> Dict:
        """Convert message to dictionary"""
        return {
            'id': self.id,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'message_type': self.message_type.value,
            'action': self.action,
            'payload': self.payload,
            'timestamp': self.timestamp,
            'correlation_id': self.correlation_id,
            'priority': self.priority
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentMessage':
        """Create message from dictionary"""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            sender_id=data['sender_id'],
            receiver_id=data['receiver_id'],
            message_type=MessageType(data['message_type']),
            action=data['action'],
            payload=data.get('payload', {}),
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            correlation_id=data.get('correlation_id'),
            priority=data.get('priority', 5)
        )


@dataclass
class AgentCapability:
    """Defines an agent capability"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]


class BaseAgent(ABC):
    """
    Abstract Base Agent for Neurological Disease Detection System

    This class provides the foundation for all specialized agents in the system.
    It implements core functionality for:
    - State management
    - Message handling
    - A2A communication
    - Task execution

    Parameters
    ----------
    agent_id : str
        Unique identifier for the agent
    agent_name : str
        Human-readable name
    agent_type : str
        Type/category of agent
    """

    def __init__(self, agent_id: str = None, agent_name: str = "BaseAgent",
                 agent_type: str = "generic"):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.state = AgentState.IDLE

        # Message queues
        self.inbox = queue.PriorityQueue()
        self.outbox = queue.Queue()

        # Capabilities
        self.capabilities: Dict[str, AgentCapability] = {}

        # Action handlers
        self.action_handlers: Dict[str, Callable] = {}

        # Message bus reference (set by orchestrator)
        self.message_bus = None

        # State history
        self.state_history: List[Dict] = []

        # Metrics
        self.metrics = {
            'messages_received': 0,
            'messages_sent': 0,
            'tasks_completed': 0,
            'errors': 0,
            'start_time': None,
            'last_activity': None
        }

        # Running flag
        self._running = False
        self._thread = None

        # Register default actions
        self._register_default_actions()

    def _register_default_actions(self):
        """Register default action handlers"""
        self.register_action('ping', self._handle_ping)
        self.register_action('status', self._handle_status)
        self.register_action('capabilities', self._handle_capabilities)
        self.register_action('shutdown', self._handle_shutdown)

    def register_action(self, action: str, handler: Callable):
        """Register an action handler"""
        self.action_handlers[action] = handler
        logger.debug(f"Agent {self.agent_id}: Registered action '{action}'")

    def register_capability(self, capability: AgentCapability):
        """Register a capability"""
        self.capabilities[capability.name] = capability

    def set_state(self, new_state: AgentState):
        """Update agent state with history tracking"""
        old_state = self.state
        self.state = new_state
        self.state_history.append({
            'from': old_state.value,
            'to': new_state.value,
            'timestamp': datetime.now().isoformat()
        })
        logger.info(f"Agent {self.agent_id}: State changed {old_state.value} -> {new_state.value}")

    def start(self):
        """Start the agent"""
        if self._running:
            return

        self._running = True
        self.set_state(AgentState.INITIALIZING)

        # Initialize agent
        self.initialize()

        # Start processing thread
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

        self.metrics['start_time'] = datetime.now().isoformat()
        self.set_state(AgentState.READY)
        logger.info(f"Agent {self.agent_id} ({self.agent_name}) started")

    def stop(self):
        """Stop the agent"""
        self._running = False
        self.set_state(AgentState.TERMINATED)
        if self._thread:
            self._thread.join(timeout=5)
        self.cleanup()
        logger.info(f"Agent {self.agent_id} ({self.agent_name}) stopped")

    def _process_loop(self):
        """Main processing loop"""
        while self._running:
            try:
                # Get message from inbox (with timeout)
                try:
                    priority, message = self.inbox.get(timeout=0.1)
                    self._handle_message(message)
                except queue.Empty:
                    pass

            except Exception as e:
                logger.error(f"Agent {self.agent_id} error: {e}")
                self.metrics['errors'] += 1

    def _handle_message(self, message: AgentMessage):
        """Process incoming message"""
        self.metrics['messages_received'] += 1
        self.metrics['last_activity'] = datetime.now().isoformat()

        logger.debug(f"Agent {self.agent_id} received: {message.action}")

        self.set_state(AgentState.PROCESSING)

        try:
            # Find handler for action
            handler = self.action_handlers.get(message.action)

            if handler:
                result = handler(message)

                # Send response if needed
                if message.message_type == MessageType.REQUEST:
                    response = AgentMessage(
                        sender_id=self.agent_id,
                        receiver_id=message.sender_id,
                        message_type=MessageType.RESPONSE,
                        action=f"{message.action}_response",
                        payload={'result': result},
                        correlation_id=message.id
                    )
                    self.send_message(response)
            else:
                logger.warning(f"Agent {self.agent_id}: No handler for action '{message.action}'")

        except Exception as e:
            logger.error(f"Agent {self.agent_id} error handling {message.action}: {e}")
            self.metrics['errors'] += 1

            # Send error response
            if message.message_type == MessageType.REQUEST:
                error_response = AgentMessage(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type=MessageType.ERROR,
                    action=f"{message.action}_error",
                    payload={'error': str(e)},
                    correlation_id=message.id
                )
                self.send_message(error_response)

        finally:
            self.set_state(AgentState.READY)

    def receive_message(self, message: AgentMessage):
        """Add message to inbox"""
        # Use negative priority for max-heap behavior (higher priority first)
        self.inbox.put((-message.priority, message))

    def send_message(self, message: AgentMessage):
        """Send message via message bus"""
        self.metrics['messages_sent'] += 1

        if self.message_bus:
            self.message_bus.route_message(message)
        else:
            self.outbox.put(message)
            logger.warning(f"Agent {self.agent_id}: No message bus, message queued")

    def broadcast(self, action: str, payload: Dict[str, Any]):
        """Broadcast message to all agents"""
        message = AgentMessage(
            sender_id=self.agent_id,
            receiver_id='broadcast',
            message_type=MessageType.BROADCAST,
            action=action,
            payload=payload
        )
        self.send_message(message)

    def request(self, receiver_id: str, action: str, payload: Dict[str, Any],
                priority: int = 5) -> str:
        """Send request to another agent"""
        message = AgentMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=MessageType.REQUEST,
            action=action,
            payload=payload,
            priority=priority
        )
        self.send_message(message)
        return message.id

    # Default action handlers
    def _handle_ping(self, message: AgentMessage) -> Dict:
        """Handle ping request"""
        return {'status': 'pong', 'agent_id': self.agent_id}

    def _handle_status(self, message: AgentMessage) -> Dict:
        """Handle status request"""
        return self.get_status()

    def _handle_capabilities(self, message: AgentMessage) -> Dict:
        """Handle capabilities request"""
        return {
            'agent_id': self.agent_id,
            'capabilities': {k: v.__dict__ for k, v in self.capabilities.items()}
        }

    def _handle_shutdown(self, message: AgentMessage) -> Dict:
        """Handle shutdown request"""
        self.stop()
        return {'status': 'shutting_down'}

    def get_status(self) -> Dict:
        """Get agent status"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'agent_type': self.agent_type,
            'state': self.state.value,
            'metrics': self.metrics,
            'capabilities': list(self.capabilities.keys())
        }

    @abstractmethod
    def initialize(self):
        """Initialize agent resources - must be implemented by subclass"""
        pass

    @abstractmethod
    def cleanup(self):
        """Cleanup agent resources - must be implemented by subclass"""
        pass

    @abstractmethod
    def process_data(self, data: Any) -> Dict[str, Any]:
        """Process data - must be implemented by subclass"""
        pass


class MessageBus:
    """
    Central Message Bus for Agent-to-Agent Communication

    Routes messages between agents and manages subscriptions.
    """

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # action -> [agent_ids]
        self.message_log: List[Dict] = []
        self._lock = threading.Lock()

    def register_agent(self, agent: BaseAgent):
        """Register an agent with the message bus"""
        with self._lock:
            self.agents[agent.agent_id] = agent
            agent.message_bus = self
            logger.info(f"MessageBus: Registered agent {agent.agent_id}")

    def deregister_agent(self, agent_id: str):
        """Remove an agent from the message bus"""
        with self._lock:
            if agent_id in self.agents:
                self.agents[agent_id].message_bus = None
                del self.agents[agent_id]
                logger.info(f"MessageBus: Deregistered agent {agent_id}")

    def subscribe(self, agent_id: str, action: str):
        """Subscribe agent to specific action broadcasts"""
        with self._lock:
            if action not in self.subscriptions:
                self.subscriptions[action] = []
            if agent_id not in self.subscriptions[action]:
                self.subscriptions[action].append(agent_id)

    def unsubscribe(self, agent_id: str, action: str):
        """Unsubscribe agent from action"""
        with self._lock:
            if action in self.subscriptions:
                self.subscriptions[action] = [
                    aid for aid in self.subscriptions[action] if aid != agent_id
                ]

    def route_message(self, message: AgentMessage):
        """Route message to appropriate agent(s)"""
        # Log message
        self.message_log.append({
            'message': message.to_dict(),
            'routed_at': datetime.now().isoformat()
        })

        if message.receiver_id == 'broadcast':
            # Broadcast to all agents except sender
            with self._lock:
                for agent_id, agent in self.agents.items():
                    if agent_id != message.sender_id:
                        agent.receive_message(message)

            # Also send to subscribers of this action
            subscribers = self.subscriptions.get(message.action, [])
            for agent_id in subscribers:
                if agent_id != message.sender_id and agent_id in self.agents:
                    self.agents[agent_id].receive_message(message)
        else:
            # Direct message
            with self._lock:
                if message.receiver_id in self.agents:
                    self.agents[message.receiver_id].receive_message(message)
                else:
                    logger.warning(f"MessageBus: Unknown receiver {message.receiver_id}")

    def get_statistics(self) -> Dict:
        """Get message bus statistics"""
        return {
            'registered_agents': len(self.agents),
            'total_messages': len(self.message_log),
            'subscriptions': {k: len(v) for k, v in self.subscriptions.items()}
        }


class AgentOrchestrator:
    """
    Orchestrator for Managing Multiple Agents

    Coordinates agent lifecycle, task distribution, and result aggregation.
    """

    def __init__(self):
        self.message_bus = MessageBus()
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue = queue.Queue()
        self.results: Dict[str, Any] = {}
        self._running = False

    def register_agent(self, agent: BaseAgent):
        """Register and start an agent"""
        self.agents[agent.agent_id] = agent
        self.message_bus.register_agent(agent)
        agent.start()

    def deregister_agent(self, agent_id: str):
        """Stop and remove an agent"""
        if agent_id in self.agents:
            self.agents[agent_id].stop()
            self.message_bus.deregister_agent(agent_id)
            del self.agents[agent_id]

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)

    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """Get all agents of a specific type"""
        return [a for a in self.agents.values() if a.agent_type == agent_type]

    def broadcast_to_all(self, action: str, payload: Dict):
        """Send message to all agents"""
        message = AgentMessage(
            sender_id='orchestrator',
            receiver_id='broadcast',
            message_type=MessageType.COMMAND,
            action=action,
            payload=payload
        )
        self.message_bus.route_message(message)

    def request_from_agent(self, agent_id: str, action: str,
                          payload: Dict) -> Optional[str]:
        """Send request to specific agent"""
        if agent_id not in self.agents:
            logger.error(f"Unknown agent: {agent_id}")
            return None

        message = AgentMessage(
            sender_id='orchestrator',
            receiver_id=agent_id,
            message_type=MessageType.REQUEST,
            action=action,
            payload=payload
        )
        self.message_bus.route_message(message)
        return message.id

    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all agents"""
        return {aid: agent.get_status() for aid, agent in self.agents.items()}

    def shutdown_all(self):
        """Shutdown all agents"""
        logger.info("Orchestrator: Shutting down all agents")
        for agent_id in list(self.agents.keys()):
            self.deregister_agent(agent_id)
        self._running = False

    def start(self):
        """Start the orchestrator"""
        self._running = True
        logger.info("Orchestrator started")

    def stop(self):
        """Stop the orchestrator"""
        self.shutdown_all()
        logger.info("Orchestrator stopped")


if __name__ == "__main__":
    # Demo
    print("Base Agent Module Demo")
    print("=" * 50)

    # Create a simple test agent
    class TestAgent(BaseAgent):
        def initialize(self):
            self.register_capability(AgentCapability(
                name="echo",
                description="Echo back received data",
                input_schema={"data": "string"},
                output_schema={"echo": "string"}
            ))
            self.register_action('echo', self._handle_echo)

        def cleanup(self):
            pass

        def process_data(self, data):
            return {"processed": data}

        def _handle_echo(self, message):
            return {"echo": message.payload.get('data', '')}

    # Create orchestrator
    orchestrator = AgentOrchestrator()
    orchestrator.start()

    # Create and register test agents
    agent1 = TestAgent(agent_name="Agent1", agent_type="test")
    agent2 = TestAgent(agent_name="Agent2", agent_type="test")

    orchestrator.register_agent(agent1)
    orchestrator.register_agent(agent2)

    print(f"\nRegistered agents: {list(orchestrator.agents.keys())}")

    # Get status
    import time
    time.sleep(0.5)

    status = orchestrator.get_all_status()
    for aid, s in status.items():
        print(f"\n{s['agent_name']}:")
        print(f"  State: {s['state']}")
        print(f"  Capabilities: {s['capabilities']}")

    # Cleanup
    orchestrator.stop()
    print("\nDemo complete!")
