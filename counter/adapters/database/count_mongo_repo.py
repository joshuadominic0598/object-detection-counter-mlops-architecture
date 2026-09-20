from typing import List

from pymongo import MongoClient

from counter.domain.models import ObjectCount
from counter.domain.ports import ObjectCountRepo


class CountInMemoryRepo(ObjectCountRepo):

    def __init__(self):
        self.store = dict()

    def read_values(self, session_id: str, object_classes: List[str] = None) -> List[ObjectCount]:
        session_store = self.store.get(session_id, {})

        if object_classes is None:
            return list(session_store.values())

        return [
            session_store[object_class]
            for object_class in object_classes
            if object_class in session_store
        ]

    def update_values(self, session_id: str, new_values: List[ObjectCount]):
        session_store = self.store.setdefault(session_id, {})

        for new_object_count in new_values:
            key = new_object_count.object_class
            existing = session_store.get(key)
            new_count = existing.count + new_object_count.count if existing else new_object_count.count
            session_store[key] = ObjectCount(key, new_count)


class CountMongoDBRepo(ObjectCountRepo):

    def __init__(self, host, port, database):
        self.__client = MongoClient(host, port)
        self.__database = self.__client[database]
        self.__counter_col = self.__database.counter

    def read_values(self, session_id: str, object_classes: List[str] = None) -> List[ObjectCount]:
        query = {"session_id": session_id}
        if object_classes:
            query["object_class"] = {"$in": object_classes}

        counters = self.__counter_col.find(query)
        return [ObjectCount(c["object_class"], c["count"]) for c in counters]

    def update_values(self, session_id: str, new_values: List[ObjectCount]):
        for value in new_values:
            self.__counter_col.update_one(
                {"session_id": session_id, "object_class": value.object_class},
                {"$inc": {"count": value.count}},
                upsert=True,
            )