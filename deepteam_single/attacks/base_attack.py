from abc import ABC, abstractmethod


class BaseAttack(ABC):
    weight: int = 1
    multi_turn: bool = False

    def enhance(self, attack: str, *args, **kwargs) -> str:
        pass

    async def a_enhance(self, attack: str, *args, **kwargs) -> str:
        return self.enhance(attack, *args, **kwargs)

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError