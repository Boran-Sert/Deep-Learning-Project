from typing import Callable, Dict, List, Type
from .events import Event


class EventBus:
    """
    Observer pattern tabanlı, olayların (event) yayınlandığı
    ve dinleyicilerin (listener) abone olduğu merkezi iletişim kanalı.
    """

    # Singleton pattern implementation for global access
    _instance = None
    _subscribers: Dict[Type[Event], List[Callable[[Event], None]]]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            cls._instance._subscribers = {}
        return cls._instance

    def subscribe(
        self, event_type: Type[Event], callback: Callable[[Event], None]
    ) -> None:
        """Belirli bir olay türüne dinleyici kaydeder."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def publish(self, event: Event) -> None:
        """Oluşan bir olayı, abone olan tüm dinleyicilere iletir."""
        event_type = type(event)
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                callback(event)
