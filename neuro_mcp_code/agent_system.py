"""
Agent Chaining System with Evaluation Metrics
Implements agent-to-agent communication, protocols, and output evaluation
"""

import time
import asyncio
import concurrent.futures
from typing import List, Dict, Optional, Callable
from datetime import datetime
import json
from enum import Enum
import ollama


class AgentProtocol(Enum):
    """Agent communication protocols"""
    SEQUENTIAL = "sequential"  # Agents run one after another
    PARALLEL = "parallel"      # Agents run concurrently
    CONDITIONAL = "conditional"  # Agents run based on conditions
    ITERATIVE = "iterative"    # Agents loop until condition met


class Agent:
    """Base Agent class with retry and error handling"""

    def __init__(self,
                 name: str,
                 model: str,
                 role: str,
                 instructions: str,
                 max_retries: int = 3,
                 fallback_model: Optional[str] = None):
        """
        Initialize Agent

        Args:
            name: Agent name
            model: Ollama model to use
            role: Agent's role description
            instructions: System instructions for the agent
            max_retries: Maximum retry attempts on failure
            fallback_model: Fallback model if primary fails
        """
        self.name = name
        self.model = model
        self.role = role
        self.instructions = instructions
        self.max_retries = max_retries
        self.fallback_model = fallback_model
        self.execution_history = []

    def execute(self, input_data: str, context: Optional[Dict] = None, retry_count: int = 0) -> Dict:
        """
        Execute agent task

        Args:
            input_data: Input for the agent
            context: Additional context

        Returns:
            Dictionary with output and metadata
        """
        start_time = time.time()

        try:
            # Build prompt
            prompt = f"""Role: {self.role}
Instructions: {self.instructions}

Context: {json.dumps(context) if context else 'None'}

Task: {input_data}

Please provide a structured response."""

            # Call Ollama
            response = ollama.generate(
                model=self.model,
                prompt=prompt
            )

            elapsed_time = time.time() - start_time

            result = {
                'agent': self.name,
                'output': response['response'],
                'elapsed_time': elapsed_time,
                'success': True,
                'model': self.model,
                'timestamp': datetime.now().isoformat()
            }

            # Track execution
            self.execution_history.append(result)

            return result

        except Exception as e:
            elapsed_time = time.time() - start_time

            # Retry logic
            if retry_count < self.max_retries:
                print(f"Agent {self.name} failed (attempt {retry_count + 1}/{self.max_retries}). Retrying...")
                time.sleep(2 ** retry_count)  # Exponential backoff
                return self.execute(input_data, context, retry_count + 1)

            # Try fallback model
            if self.fallback_model and retry_count >= self.max_retries:
                print(f"Agent {self.name} switching to fallback model: {self.fallback_model}")
                original_model = self.model
                self.model = self.fallback_model
                try:
                    result = self.execute(input_data, context, retry_count=999)  # Skip retry for fallback
                    self.model = original_model
                    result['used_fallback'] = True
                    return result
                except:
                    self.model = original_model

            result = {
                'agent': self.name,
                'output': '',
                'error': str(e),
                'elapsed_time': elapsed_time,
                'success': False,
                'retry_count': retry_count,
                'timestamp': datetime.now().isoformat()
            }
            self.execution_history.append(result)
            return result


class AgentChain:
    """Agent chaining system with different protocols"""

    def __init__(self, protocol: AgentProtocol = AgentProtocol.SEQUENTIAL):
        """
        Initialize Agent Chain

        Args:
            protocol: Communication protocol to use
        """
        self.protocol = protocol
        self.agents = []
        self.execution_log = []

    def add_agent(self, agent: Agent):
        """Add agent to the chain"""
        self.agents.append(agent)

    def execute_sequential(self, initial_input: str, context: Optional[Dict] = None) -> Dict:
        """
        Execute agents sequentially

        Args:
            initial_input: Initial input for first agent
            context: Shared context

        Returns:
            Final result and execution log
        """
        current_input = initial_input
        context = context or {}

        for agent in self.agents:
            # Execute agent
            result = agent.execute(current_input, context)

            # Log execution
            self.execution_log.append(result)

            if not result['success']:
                return {
                    'success': False,
                    'error': f"Agent {agent.name} failed: {result.get('error')}",
                    'execution_log': self.execution_log
                }

            # Output becomes input for next agent
            current_input = result['output']

            # Update context
            context[f'{agent.name}_output'] = result['output']

        return {
            'success': True,
            'final_output': current_input,
            'execution_log': self.execution_log,
            'context': context
        }

    def execute_conditional(self,
                           initial_input: str,
                           condition_fn: Callable,
                           context: Optional[Dict] = None) -> Dict:
        """
        Execute agents conditionally

        Args:
            initial_input: Initial input
            condition_fn: Function that determines which agent to run
            context: Shared context

        Returns:
            Result and execution log
        """
        current_input = initial_input
        context = context or {}

        for agent in self.agents:
            # Check condition
            should_run = condition_fn(agent, current_input, context)

            if not should_run:
                self.execution_log.append({
                    'agent': agent.name,
                    'skipped': True,
                    'reason': 'Condition not met'
                })
                continue

            # Execute agent
            result = agent.execute(current_input, context)
            self.execution_log.append(result)

            if not result['success']:
                return {
                    'success': False,
                    'error': result.get('error'),
                    'execution_log': self.execution_log
                }

            current_input = result['output']
            context[f'{agent.name}_output'] = result['output']

        return {
            'success': True,
            'final_output': current_input,
            'execution_log': self.execution_log
        }

    def execute_parallel(self, initial_input: str, context: Optional[Dict] = None) -> Dict:
        """
        Execute agents in parallel

        Args:
            initial_input: Initial input for all agents
            context: Shared context

        Returns:
            Combined results from all agents
        """
        context = context or {}
        results = []

        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            # Submit all agents
            future_to_agent = {
                executor.submit(agent.execute, initial_input, context): agent
                for agent in self.agents
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.execution_log.append(result)

                    # Add to shared context
                    context[f'{agent.name}_output'] = result['output']
                except Exception as e:
                    error_result = {
                        'agent': agent.name,
                        'success': False,
                        'error': str(e)
                    }
                    results.append(error_result)
                    self.execution_log.append(error_result)

        # Combine outputs
        successful_results = [r for r in results if r['success']]
        combined_output = "\n\n".join([r['output'] for r in successful_results])

        return {
            'success': all(r['success'] for r in results),
            'final_output': combined_output,
            'individual_results': results,
            'execution_log': self.execution_log,
            'context': context
        }

    def execute_iterative(self,
                         initial_input: str,
                         condition_fn: Callable,
                         max_iterations: int = 10,
                         context: Optional[Dict] = None) -> Dict:
        """
        Execute agents iteratively until condition met

        Args:
            initial_input: Initial input
            condition_fn: Function that returns True when done
            max_iterations: Maximum iterations to prevent infinite loops
            context: Shared context

        Returns:
            Final result after iterations
        """
        current_input = initial_input
        context = context or {}
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            context['iteration'] = iteration

            # Execute all agents in sequence
            for agent in self.agents:
                result = agent.execute(current_input, context)
                self.execution_log.append(result)

                if not result['success']:
                    return {
                        'success': False,
                        'error': f"Agent {agent.name} failed at iteration {iteration}",
                        'execution_log': self.execution_log,
                        'iterations': iteration
                    }

                current_input = result['output']
                context[f'{agent.name}_output'] = result['output']

            # Check if condition is met
            if condition_fn(current_input, context):
                return {
                    'success': True,
                    'final_output': current_input,
                    'execution_log': self.execution_log,
                    'iterations': iteration,
                    'context': context
                }

        return {
            'success': False,
            'error': f'Max iterations ({max_iterations}) reached without meeting condition',
            'final_output': current_input,
            'execution_log': self.execution_log,
            'iterations': iteration
        }

    def execute(self, initial_input: str, **kwargs) -> Dict:
        """Execute chain based on protocol"""
        if self.protocol == AgentProtocol.SEQUENTIAL:
            return self.execute_sequential(initial_input, kwargs.get('context'))
        elif self.protocol == AgentProtocol.CONDITIONAL:
            return self.execute_conditional(
                initial_input,
                kwargs.get('condition_fn', lambda a, i, c: True),
                kwargs.get('context')
            )
        elif self.protocol == AgentProtocol.PARALLEL:
            return self.execute_parallel(initial_input, kwargs.get('context'))
        elif self.protocol == AgentProtocol.ITERATIVE:
            return self.execute_iterative(
                initial_input,
                kwargs.get('condition_fn', lambda o, c: True),
                kwargs.get('max_iterations', 10),
                kwargs.get('context')
            )
        else:
            raise ValueError(f"Protocol {self.protocol} not supported")


class OutputEvaluator:
    """Evaluate agent outputs"""

    def __init__(self, model: str = "llama3.2"):
        """
        Initialize evaluator

        Args:
            model: Ollama model for evaluation
        """
        self.model = model

    def evaluate_quality(self, output: str, criteria: Dict) -> Dict:
        """
        Evaluate output quality

        Args:
            output: Output to evaluate
            criteria: Evaluation criteria

        Returns:
            Evaluation scores
        """
        start_time = time.time()

        # Build evaluation prompt
        criteria_str = "\n".join([f"- {k}: {v}" for k, v in criteria.items()])

        prompt = f"""Evaluate the following output based on these criteria:

{criteria_str}

Output to evaluate:
{output}

For each criterion, provide a score from 1-10 and a brief explanation.
Return your evaluation in JSON format with this structure:
{{
    "criterion_name": {{
        "score": <1-10>,
        "explanation": "<reason>"
    }}
}}"""

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt
            )

            elapsed_time = time.time() - start_time

            # Try to parse JSON from response
            try:
                # Extract JSON from response
                response_text = response['response']
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                json_str = response_text[start_idx:end_idx]
                scores = json.loads(json_str)
            except:
                scores = {'raw_response': response['response']}

            return {
                'success': True,
                'scores': scores,
                'elapsed_time': elapsed_time,
                'overall_score': self._calculate_overall_score(scores)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _calculate_overall_score(self, scores: Dict) -> float:
        """Calculate overall score from individual scores"""
        if not scores or 'raw_response' in scores:
            return 0.0

        total = 0
        count = 0

        for criterion, data in scores.items():
            if isinstance(data, dict) and 'score' in data:
                total += data['score']
                count += 1

        return round(total / count, 2) if count > 0 else 0.0

    def compare_outputs(self, output1: str, output2: str, criteria: str) -> Dict:
        """
        Compare two outputs

        Args:
            output1: First output
            output2: Second output
            criteria: Comparison criteria

        Returns:
            Comparison result
        """
        prompt = f"""Compare these two outputs based on: {criteria}

Output 1:
{output1}

Output 2:
{output2}

Provide:
1. Which output is better (1 or 2)
2. Score for each (1-10)
3. Detailed explanation

Format your response as JSON:
{{
    "winner": <1 or 2>,
    "output1_score": <score>,
    "output2_score": <score>,
    "explanation": "<detailed comparison>"
}}"""

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt
            )

            # Parse response
            response_text = response['response']
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)

            return {
                'success': True,
                'comparison': result
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


class ResearchAssistantChain:
    """Pre-configured agent chain for research assistance"""

    def __init__(self):
        """Initialize research assistant chain"""
        self.chain = AgentChain(protocol=AgentProtocol.SEQUENTIAL)

        # Agent 1: Query Understanding
        self.chain.add_agent(Agent(
            name="QueryAnalyzer",
            model="llama3.2",
            role="Query Understanding Specialist",
            instructions="""Analyze the user's question and:
1. Identify the key research topics
2. Determine what type of answer is needed (definition, comparison, explanation, etc.)
3. List relevant keywords for search
4. Suggest the scope of the answer (brief, detailed, comprehensive)

Provide your analysis in a structured format."""
        ))

        # Agent 2: Information Synthesis
        self.chain.add_agent(Agent(
            name="Synthesizer",
            model="llama3.2",
            role="Information Synthesis Expert",
            instructions="""Based on the query analysis:
1. Organize the information logically
2. Identify key concepts and relationships
3. Create a coherent narrative
4. Highlight important findings

Provide a well-structured synthesis."""
        ))

        # Agent 3: Answer Formatting
        self.chain.add_agent(Agent(
            name="Formatter",
            model="llama3.2",
            role="Answer Formatting Specialist",
            instructions="""Format the synthesized information into a clear, concise answer:
1. Use appropriate structure (paragraphs, lists, etc.)
2. Include relevant citations
3. Add clarifications where needed
4. Ensure readability

Provide the final, polished answer."""
        ))

        self.evaluator = OutputEvaluator()

    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Process query through agent chain

        Args:
            query: User query
            context: Additional context

        Returns:
            Processed result with evaluation
        """
        # Execute chain
        result = self.chain.execute(query, context=context)

        if result['success']:
            # Evaluate final output
            evaluation = self.evaluator.evaluate_quality(
                result['final_output'],
                {
                    'Accuracy': 'Is the answer factually correct?',
                    'Completeness': 'Does it fully address the question?',
                    'Clarity': 'Is it clear and easy to understand?',
                    'Relevance': 'Is the information relevant to the question?'
                }
            )

            result['evaluation'] = evaluation

        return result
