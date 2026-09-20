import pytest

from counter.adapters.database.count_mongo_repo import CountMongoDBRepo
from counter.domain.models import ObjectCount


class TestMongoCounter:

    def setup_method(self):
        self.repo = CountMongoDBRepo(
            host="localhost",
            port=27017,
            database="counter",
        )

        # Start each test with a clean counter collection.
        self.repo._CountMongoDBRepo__counter_col.delete_many({})

    def test_update_and_read_counter(self):
        self.repo.update_values(
            "session-1",
            [ObjectCount("cat", 2), ObjectCount("dog", 1)],
        )

        result = self.repo.read_values("session-1")

        assert sorted(result, key=lambda x: x.object_class) == [
            ObjectCount("cat", 2),
            ObjectCount("dog", 1),
        ]

    def test_update_increments_existing_counter(self):
        self.repo.update_values("session-1", [ObjectCount("cat", 2)])
        self.repo.update_values("session-1", [ObjectCount("cat", 3)])

        result = self.repo.read_values("session-1")

        assert result == [ObjectCount("cat", 5)]

    def test_read_specific_classes(self):
        self.repo.update_values(
            "session-1",
            [ObjectCount("cat", 2), ObjectCount("dog", 1), ObjectCount("rabbit", 4)],
        )

        result = self.repo.read_values("session-1", ["cat", "rabbit"])

        assert sorted(result, key=lambda x: x.object_class) == [
            ObjectCount("cat", 2),
            ObjectCount("rabbit", 4),
        ]

    def test_empty_collection_returns_empty_list(self):
        result = self.repo.read_values("session-1")
        assert result == []

    def test_sessions_are_isolated(self):
        self.repo.update_values("session-1", [ObjectCount("cat", 2)])
        self.repo.update_values("session-2", [ObjectCount("cat", 5)])

        session_1_result = self.repo.read_values("session-1")
        session_2_result = self.repo.read_values("session-2")

        assert session_1_result == [ObjectCount("cat", 2)]
        assert session_2_result == [ObjectCount("cat", 5)]