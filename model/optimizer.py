import numpy as np
from typing import List, Optional

def rule_based_optimizer(
    competitor_prices: List[float],
    inventory: int,
    demand: float,
    base_cost: float,
    min_margin: float = 0.1  # 10% minimum margin
) -> Optional[float]:
    """
    Calculate an optimal price using simple business rules.

    Parameters:
        competitor_prices (list of float): List of competitor product prices.
        inventory (int): Current inventory count.
        demand (float): Demand score (0-1 scale).
        base_cost (float): Your purchase cost for the product.
        min_margin (float): Minimum profit margin as a decimal (default: 0.1 for 10%).

    Returns:
        float: Optimal price suggestion.
    """
    if not competitor_prices:
        return None

    avg_comp_price = np.mean(competitor_prices)

    # Business logic adjustments
    adjustment = 0
    if demand > 0.8 and inventory < 10:
        adjustment = 0.10  # raise price by 10%
    elif demand < 0.3 and inventory > 50:
        adjustment = -0.10  # drop price by 10%
    elif demand < 0.5:
        adjustment = -0.05  # mild discount

    # Apply adjustment to average competitor price
    candidate_price = avg_comp_price * (1 + adjustment)

    # Ensure price is at least base cost + margin
    min_price = base_cost * (1 + min_margin)
    optimal_price = max(candidate_price, min_price)

    return round(optimal_price, 2)
