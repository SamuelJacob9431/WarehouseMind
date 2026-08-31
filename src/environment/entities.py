from dataclasses import dataclass


Position = tuple[int, int]


@dataclass
class Robot:
    position: Position
    carrying: bool = False


@dataclass
class Package:
    position: Position
    delivered: bool = False


@dataclass
class DeliveryZone:
    position: Position
    