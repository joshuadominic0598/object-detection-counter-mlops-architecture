"""
Storage port for trained model weights + training run artifacts. Mirrors
counter/domain/ports.py's ObjectDetector/ObjectCountRepo pattern: callers
(registry.py, orchestrator.py, train.py) depend on this interface, not on
Google Drive directly, so a different backend (S3, GCS...) is a new adapter
here rather than a change to any of those callers.
"""

from abc import ABC, abstractmethod


class WeightsStorage(ABC):

    @abstractmethod
    def get_or_create_folder(self, name, parent_id=None):
        """Return the id of a folder named `name` under `parent_id`, creating it if needed."""
        raise NotImplementedError

    @abstractmethod
    def upload_file(self, local_path, filename=None, folder_id=None):
        """Upload a file and return (file_id, share_link)."""
        raise NotImplementedError

    @abstractmethod
    def upload_directory(self, local_dir, filename=None, folder_id=None):
        """Archive `local_dir` and upload it as a single file, return (file_id, share_link)."""
        raise NotImplementedError

    @abstractmethod
    def download(self, file_id, destination):
        """Download `file_id` to local path `destination`."""
        raise NotImplementedError
