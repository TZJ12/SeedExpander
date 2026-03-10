import base64
from .. import BaseSingleTurnAttack


class Base64(BaseSingleTurnAttack):
    def __init__(self, weight: int = 1):
        self.weight = weight

    def enhance(self, attack: str) -> str:
        """Enhance the attack using Base64 encoding."""
        return base64.b64encode(attack.encode()).decode()

    def get_name(self) -> str:
        return "Base64"