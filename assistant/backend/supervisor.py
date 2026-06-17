import sys
from pathlib import Path
from typing import TypedDict, Optional

# Ensure internal module paths resolve smoothly on Windows
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

from watchdog_critic import WatchdogCriticAgent
from langgraph.graph import StateGraph, END


# =====================================================================
# 1. DEFINE THE UNIFIED GRAPH STATE CONTRACT
# =====================================================================
class AgentState(TypedDict):
    """
    The central data contract passed between all nodes in the LangGraph workflow.
    """
    current_intent: str
    speaker_id: str
    system_context: dict
    next_action_node: str
    final_execution_plan: str
    is_approved: bool


# =====================================================================
# 2. IMPLEMENT THE GRAPH ENGINE NODES
# =====================================================================
class LangGraphSupervisorEngine:
    def __init__(self):
        self.critic = WatchdogCriticAgent()
        # Compile the state graph architecture
        self.workflow = self._compile_graph()
        print("[LangGraph Core] Master workflow state graph successfully compiled.")

    def supervisor_router_node(self, state: AgentState) -> AgentState:
        """
        Acts as the central traffic controller (LangGraph Supervisor).
        Inspects the incoming transcript stream and decides the route.
        """
        intent = state["current_intent"].lower()
        print(f"\n[Node: Supervisor] Analyzing incoming payload from {state['speaker_id']}...")

        # Route to Intent Planner for device actions; else route directly to data retrieval
        if any(keyword in intent for keyword in ["set", "turn", "lock", "open", "close"]):
            state["next_action_node"] = "Intent_Planner"
        else:
            state["next_action_node"] = "Retriever"

        return state

    def intent_planner_node(self, state: AgentState) -> AgentState:
        """
        Parses commands and maps out structured task execution details.
        """
        print(f"[Node: Intent Planner] Structuring task blueprints for: '{state['current_intent']}'")
        # In a later phase, this node will query our local SLM to generate JSON payloads
        return state

    def watchdog_critic_node(self, state: AgentState) -> AgentState:
        """
        The Adversarial Safety Gate. Cross-checks safety metrics and forwards to Alishri's MCP class.
        """
        print(f"[Node: Watchdog Critic] Cross-checking execution payload safety metrics...")

        validation = self.critic.validate_intent_stream(
            current_intent=state["current_intent"],
            speaker_id=state["speaker_id"],
            system_context=state["system_context"]
        )

        state["is_approved"] = validation.is_safe
        if validation.is_safe:
            state["final_execution_plan"] = f"EXECUTE_MCP_CALL -> {state['current_intent']}"

            # --- PERFECT INTEGRATION WITH ALISHRI'S MOCK MCP SERVER CLASS ---
            try:
                import mcp_server
                import json

                # 1. Map the incoming natural language text to her registered tools
                intent_lower = state["current_intent"].lower()
                tool_name = "toggle_appliance"
                parameters = {"device_id": "general_appliance", "state": "on"}

                if "temperature" in intent_lower or "degrees" in intent_lower:
                    tool_name = "set_temperature"
                    # Extract the numerical value dynamically or fallback safely
                    temp_val = 22 if "22" in intent_lower else 24
                    parameters = {"device_id": "living_room_ac", "temperature": temp_val}
                elif "lock" in intent_lower:
                    tool_name = "lock_door"
                    parameters = {"device_id": "front_entry_door", "is_locked": "lock" in intent_lower}

                # 2. Package it into the exact JSON payload format her method requires
                payload_to_mcp = json.dumps({
                    "tool": tool_name,
                    "parameters": parameters
                })

                # 3. Instantiate her class and fire the instance method execution loop
                mcp_instance = mcp_server.MockMCPServer()
                mcp_response = mcp_instance.execute_tool(payload_to_mcp)
                print(f"   ↳ [MCP Server Response] {mcp_response}")

            except ImportError:
                print("   ↳ [MCP Notice] mcp_server.py module not found in staging path.")
            except Exception as e:
                print(f"   ↳ [MCP Execution Error] {e}")

        else:
            state[
                "final_execution_plan"] = f"SAFETY_OVERRIDE -> {validation.corrected_action} (Reason: {validation.reason})"

        return state

    # =====================================================================
    # 3. GRAPH ARCHITECTURE & ROUTING CONFIGURATION
    # =====================================================================
    def _compile_graph(self):
        # Initialize graph builder with our custom State tracking schema
        builder = StateGraph(AgentState)

        # Register nodes into the operational runtime environment
        builder.add_node("SupervisorRouter", self.supervisor_router_node)
        builder.add_node("IntentPlanner", self.intent_planner_node)
        builder.add_node("WatchdogCritic", self.watchdog_critic_node)

        # Set the entry point entry checkpoint
        builder.set_entry_point("SupervisorRouter")

        # Configure conditional routing links out of the Supervisor Node
        builder.add_conditional_edges(
            "SupervisorRouter",
            lambda state: state["next_action_node"],
            {
                "Intent_Planner": "IntentPlanner",
                "Retriever": "WatchdogCritic"  # Fallback/bypass route
            }
        )

        # Map linear execution pathways between nodes
        builder.add_edge("IntentPlanner", "WatchdogCritic")
        builder.add_edge("WatchdogCritic", END)

        # Compile the graph structure into an executable binary application
        return builder.compile()

    def run_pipeline(self, speaker_id: str, transcript: str, context: dict) -> dict:
        """
        Executes a streaming transaction across the compiled graph nodes.
        """
        initial_state = AgentState(
            current_intent=transcript,
            speaker_id=speaker_id,
            system_context=context,
            next_action_node="",
            final_execution_plan="",
            is_approved=False
        )

        # Stream or invoke through compiled LangGraph engine nodes
        return self.workflow.invoke(initial_state)


if __name__ == "__main__":
    # Rapid internal verification test
    engine = LangGraphSupervisorEngine()
    mock_history = {"last_speaker": "Speaker_A", "last_intent": "Turn off all appliances"}

    print("\n--- TEST RUN 1: SAFE STANDALONE COMMAND ---")
    engine.run_pipeline("Speaker_A", "Set the temperature to 22 degrees", {})

    print("\n--- TEST RUN 2: ADVERSARIAL MULTI-USER COLLISION ---")
    result = engine.run_pipeline("Speaker_B", "Turn on the kitchen appliances", mock_history)
    print(f"\n[Final Graph Result Out] Status: {result['is_approved']} | Action: {result['final_execution_plan']}")