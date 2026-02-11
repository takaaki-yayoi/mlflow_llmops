"""会話スレッド管理。"""
from dataclasses import dataclass, field
from typing import List
import uuid


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Thread:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
