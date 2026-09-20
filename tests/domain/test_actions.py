from unittest.mock import Mock

import pytest

from counter.domain.actions import CountDetectedObjects
from counter.domain.models import ObjectCount
from tests.domain.helpers import generate_prediction


class TestCountDetectedObjects:

    @pytest.fixture
    def object_detector(self) -> Mock:
        object_detector = Mock()
        object_detector.predict.return_value = [
            generate_prediction("cat", 0.9),
            generate_prediction("cat", 0.8),
            generate_prediction("dog", 0.8),
            generate_prediction("dog", 0.1),
            generate_prediction("rabbit", 0.9),
        ]
        return object_detector

    @pytest.fixture
    def count_object_repo(self) -> Mock:
        return Mock()

    def test_count_valid_predictions(self, object_detector, count_object_repo) -> None:
        # execute() returns (CountResponse, result_image_path)
        response, _ = CountDetectedObjects(object_detector, count_object_repo).execute(None, 0.5, "test-session")

        assert sorted(response.current_objects, key=lambda x: x.object_class) == [
            ObjectCount("cat", 2),
            ObjectCount("dog", 1),
            ObjectCount("rabbit", 1),
        ]

    def test_update_count_object_repo(self, object_detector, count_object_repo):
        CountDetectedObjects(object_detector, count_object_repo).execute(None, 0, "test-session")

        count_object_repo.update_values.assert_called_with(
            "test-session",
            [ObjectCount("cat", 2), ObjectCount("dog", 2), ObjectCount("rabbit", 1)],
        )

    def test_empty_predictions(self, count_object_repo):
        object_detector = Mock()
        object_detector.predict.return_value = []

        response, _ = CountDetectedObjects(object_detector, count_object_repo).execute(None, 0.5, "test-session")

        assert response.current_objects == []
        count_object_repo.update_values.assert_called_with("test-session", [])

    def test_read_total_counts_after_update(self, object_detector, count_object_repo):
        CountDetectedObjects(object_detector, count_object_repo).execute(None, 0.5, "test-session")
        count_object_repo.read_values.assert_called_once_with("test-session")

    def test_session_id_is_passed_to_repo(self, object_detector, count_object_repo):
        action = CountDetectedObjects(object_detector, count_object_repo)

        action.execute(None, 0.5, "session-1")
        action.execute(None, 0.5, "session-2")

        assert count_object_repo.update_values.call_count == 2
        assert count_object_repo.update_values.call_args_list[0].args[0] == "session-1"
        assert count_object_repo.update_values.call_args_list[1].args[0] == "session-2"