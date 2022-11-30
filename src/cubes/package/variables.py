"""
Classes to define action and observation variables
"""

from dataclasses import dataclass


@dataclass
class ActionVariable:
    name: str
    action_type: str
    action_quantity: str
