import heapq as hq
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field


class HeapMapMember[K, V](ABC):
    @abstractmethod
    def __init__(self, k: K, v: V, /): ...

    @abstractmethod
    def __lt__(self, other) -> bool: ...

    @abstractmethod
    def key(self) -> K: ...

    @abstractmethod
    def val(self) -> V: ...


@dataclass(order=True)
class Item[K, V]:
    data: HeapMapMember
    valid: bool = field(default=True, compare=False)


type Factory[K, V] = Callable[[K, V], HeapMapMember[K, V]]


class HeapMap[K, V]:
    def __init__(self, members: list[tuple[K, V]], factory: Factory[K, V]):
        self.factory = factory
        self.h: list[Item[K, V]] = [Item(self.factory(k, v)) for k, v in members]
        hq.heapify(self.h)
        self.m: dict[K, Item[K, V]] = {item.data.key(): item for item in self.h}

    def __len__(self):
        return len(self.m)

    def __getitem__(self, k: K) -> V:
        return self.m[k].data.val()

    def __contains__(self, k: K) -> bool:
        return k in self.m

    def insert(self, k: K, v: V):
        item = Item(self.factory(k, v))
        hq.heappush(self.h, item)
        self.m[k] = item

    def update(self, key: K, val: V):
        if key not in self.m:
            raise ValueError("key not in map")
        self.m[key].valid = False
        self.insert(key, val)

    def _clean_up(self):
        if len(self.m) / len(self.h) < 0.7:
            self.h = [item for item in self.h if item.valid]
            hq.heapify(self.h)

    def hinvalidate(self, key: K):
        if key not in self.m:
            raise ValueError("key not in map")
        self.m[key].valid = False

    def pop(self) -> tuple[K, V]:
        self._clean_up()
        while len(self.h) > 0:
            item = hq.heappop(self.h)
            if item.valid:
                del self.m[item.data.key()]
                return item.data.key(), item.data.val()
        raise RuntimeError("heap is empty")
