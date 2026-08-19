import json
import os
from typing import Dict, List, Optional

from openai import OpenAI

from agents.agent import Agent
from agents.deals import Deal, Opportunity
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
from agents.scanner_agent import ScannerAgent


class AutonomousPlanningAgent(Agent):
    """Let an LLM coordinate the scanner, pricer, and messenger as tools."""

    name = "Autonomous Planning Agent"
    color = Agent.GREEN
    MODEL = os.getenv("PLANNING_MODEL", "gpt-5-nano")
    MAX_TURNS = 20

    def __init__(self, collection):
        self.log("Autonomous Planning Agent is initializing")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.openai = OpenAI()
        self.memory = []
        self.opportunity = None
        self.log(f"Autonomous Planning Agent is ready using {self.MODEL}")

    def scan_the_internet_for_bargains(self) -> str:
        self.log("Calling Scanner Agent")
        results = self.scanner.scan(memory=self.memory)
        return results.model_dump_json() if results else '{"deals": []}'

    def estimate_true_value(self, description: str) -> str:
        self.log("Calling Ensemble Agent")
        estimate = self.ensemble.price(description)
        return json.dumps({"description": description, "estimated_true_value": estimate})

    def notify_user_of_deal(
        self,
        description: str,
        deal_price: float,
        estimated_true_value: float,
        url: str,
    ) -> str:
        if self.opportunity is not None:
            self.log("Ignoring a second notification request")
            return "Notification already handled"

        self.log("Calling Messaging Agent")
        self.messenger.notify(description, deal_price, estimated_true_value, url)
        deal = Deal(product_description=description, price=deal_price, url=url)
        self.opportunity = Opportunity(
            deal=deal,
            estimate=estimated_true_value,
            discount=estimated_true_value - deal_price,
        )
        return "Notification handled"

    scan_function = {
        "name": "scan_the_internet_for_bargains",
        "description": "Return promising current bargains and their advertised prices.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }

    estimate_function = {
        "name": "estimate_true_value",
        "description": "Estimate the true value of one product from its description.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "A detailed product description.",
                }
            },
            "required": ["description"],
            "additionalProperties": False,
        },
    }

    notify_function = {
        "name": "notify_user_of_deal",
        "description": "Notify the user about the single best deal. Call at most once.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "deal_price": {"type": "number"},
                "estimated_true_value": {"type": "number"},
                "url": {"type": "string"},
            },
            "required": ["description", "deal_price", "estimated_true_value", "url"],
            "additionalProperties": False,
        },
    }

    def get_tools(self) -> List[Dict]:
        return [
            {"type": "function", "function": self.scan_function},
            {"type": "function", "function": self.estimate_function},
            {"type": "function", "function": self.notify_function},
        ]

    def handle_tool_call(self, message) -> List[Dict[str, str]]:
        mapping = {
            "scan_the_internet_for_bargains": self.scan_the_internet_for_bargains,
            "estimate_true_value": self.estimate_true_value,
            "notify_user_of_deal": self.notify_user_of_deal,
        }
        results = []
        for tool_call in message.tool_calls or []:
            tool_name = tool_call.function.name
            tool = mapping.get(tool_name)
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
                result = tool(**arguments) if tool else f"Unknown tool: {tool_name}"
            except (TypeError, ValueError) as exc:
                result = f"Tool error: {exc}"
            results.append(
                {"role": "tool", "content": str(result), "tool_call_id": tool_call.id}
            )
        return results

    system_message = (
        "You find bargain products with your tools and notify the user about the single "
        "best bargain. Do not invent products, prices, estimates, or URLs."
    )
    user_message = (
        "First scan for bargains. Estimate the true value of every returned deal. "
        "Then choose the single deal with the largest positive discount and notify the user once. "
        "Finally reply OK."
    )

    def plan(self, memory: Optional[List[Opportunity]] = None) -> Optional[Opportunity]:
        self.log("Starting an autonomous planning run")
        self.memory = memory or []
        self.opportunity = None
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.user_message},
        ]

        for _ in range(self.MAX_TURNS):
            response = self.openai.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                tools=self.get_tools(),
            )
            message = response.choices[0].message
            messages.append(message)
            if not message.tool_calls:
                self.log(f"Completed with: {message.content}")
                return self.opportunity
            messages.extend(self.handle_tool_call(message))

        raise RuntimeError(f"Planner exceeded its {self.MAX_TURNS}-turn safety limit")
