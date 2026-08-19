from typing import List, Optional

from agents.agent import Agent
from agents.deals import Deal, Opportunity
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
from agents.scanner_agent import ScannerAgent


class PlanningAgent(Agent):
    """Coordinate the Week 8 agents with an explicit, deterministic workflow."""

    name = "Planning Agent"
    color = Agent.GREEN
    DEAL_THRESHOLD = 50
    MAX_DEALS_TO_PRICE = 5

    def __init__(self, collection):
        self.log("Planning Agent is initializing")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.log("Planning Agent is ready")

    def price_deal(self, deal: Deal) -> Opportunity:
        self.log("Pricing a potential deal")
        estimate = self.ensemble.price(deal.product_description)
        discount = estimate - deal.price
        self.log(f"Processed deal with discount ${discount:.2f}")
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: Optional[List[Opportunity]] = None) -> Optional[Opportunity]:
        """Scan, price up to five deals, and alert when the best discount exceeds $50."""
        self.log("Starting a planning run")
        selection = self.scanner.scan(memory=memory or [])
        if not selection or not selection.deals:
            self.log("No new deals were selected")
            return None

        opportunities = [
            self.price_deal(deal)
            for deal in selection.deals[: self.MAX_DEALS_TO_PRICE]
        ]
        best = max(opportunities, key=lambda opportunity: opportunity.discount)
        self.log(f"Best deal has discount ${best.discount:.2f}")

        if best.discount <= self.DEAL_THRESHOLD:
            self.log(f"No deal exceeded the ${self.DEAL_THRESHOLD:.2f} alert threshold")
            return None

        self.messenger.alert(best)
        self.log("Planning run completed")
        return best
