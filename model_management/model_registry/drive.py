"""
GoogleDriveStorage: the WeightsStorage (see ports.py) implementation used
to back up trained model weights and training run artifacts (results.csv,
plots, args.yaml - see upload_directory()), and to turn them into "anyone
with the link" share links that get stored in the registry. Performance/
evaluation reports are never uploaded here - they're kept local only, see
model_management/evaluation/evaluator.py.

Requires GOOGLE_DRIVE_CLIENT_ID / GOOGLE_DRIVE_CLIENT_SECRET (an OAuth
"Desktop app" client from Google Cloud Console, APIs & Services ->
Credentials) in .env. GOOGLE_DRIVE_FOLDER_ID is optional - it's the parent
under which the "Model-training-ObjectCounter" folder is created/reused
(defaults to Drive root, i.e. "My Drive"). The first upload opens a browser
for one-time consent (this also works when the notebook runs in Google
Colab - the browser just opens on whatever device you're viewing the
notebook from); the resulting token is cached at TOKEN_PATH so later runs
don't re-auth.

Uses the drive.file scope, so this app can only see/manage files it
created itself - not the rest of the user's Drive.
"""


import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from model_management.model_registry.ports import WeightsStorage

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
REPO_ROOT = Path(__file__).resolve().parents[2]
TOKEN_PATH = REPO_ROOT / "tmp" / "gdrive_token.json"


def _credentials():
    creds = None

    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        client_id = os.getenv("GOOGLE_DRIVE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_DRIVE_CLIENT_SECRET")

        if not client_id or not client_secret:
            raise RuntimeError(
                "Set GOOGLE_DRIVE_CLIENT_ID and GOOGLE_DRIVE_CLIENT_SECRET in .env "
                "(OAuth 'Desktop app' credentials from Google Cloud Console)."
            )

        client_config = {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        }

        flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
        creds = flow.run_local_server(port=0)

    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")

    return creds


def _escape_query_value(value):
    return value.replace("\\", "\\\\").replace("'", "\\'")


class GoogleDriveStorage(WeightsStorage):
    """WeightsStorage backed by Google Drive (drive.file scope)."""

    def __init__(self):
        self._service = None

    def _client(self):
        if self._service is None:
            self._service = build("drive", "v3", credentials=_credentials())

        return self._service

    def get_or_create_folder(self, name, parent_id=None):
        parent_id = parent_id or os.getenv("GOOGLE_DRIVE_FOLDER_ID") or "root"
        service = self._client()

        query = (
            f"name = '{_escape_query_value(name)}' "
            f"and '{parent_id}' in parents "
            "and mimeType = 'application/vnd.google-apps.folder' "
            "and trashed = false"
        )

        results = service.files().list(q=query, fields="files(id)").execute()
        matches = results.get("files", [])

        if matches:
            return matches[0]["id"]

        metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }

        folder = service.files().create(body=metadata, fields="id").execute()

        return folder["id"]

    def upload_file(self, local_path, filename=None, folder_id=None):
        if not folder_id:
            raise RuntimeError("upload_file() requires a folder_id - see get_or_create_folder().")

        local_path = Path(local_path)
        service = self._client()

        metadata = {"name": filename or local_path.name, "parents": [folder_id]}
        media = MediaFileUpload(str(local_path), resumable=True)

        file = service.files().create(body=metadata, media_body=media, fields="id").execute()
        file_id = file["id"]

        service.permissions().create(
            fileId=file_id,
            body={"role": "reader", "type": "anyone"},
        ).execute()

        file = service.files().get(fileId=file_id, fields="webViewLink").execute()

        return file_id, file["webViewLink"]

    def upload_directory(self, local_dir, filename=None, folder_id=None):
        """Zip `local_dir` (e.g. a training run directory - weights, results.csv,
        plots) and upload it as a single archive. Only model weights and this
        kind of training artifact belong here; performance/evaluation reports
        are kept local (see model_management/evaluation/evaluator.py)."""

        local_dir = Path(local_dir)

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_base = Path(tmp_dir) / (filename or local_dir.name)
            archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=local_dir))

            return self.upload_file(archive_path, filename=f"{archive_base.name}.zip", folder_id=folder_id)

    def download(self, file_id, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [sys.executable, "-m", "gdown", file_id, "-O", str(destination)],
            check=True,
        )

        if not destination.exists() or destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"Download failed for '{file_id}': downloaded file is empty.")


def get_storage() -> WeightsStorage:
    """The active WeightsStorage backend - swap this to change where model
    weights live (e.g. an S3Storage implementing the same port)."""

    return GoogleDriveStorage()
