"""
Azure Video Indexer Service.

Wraps Azure AI Video Indexer's ARM-based API: downloading a YouTube video,
uploading it to Azure, waiting for Azure to finish processing it, and
extracting the transcript/OCR/metadata from the result.

Authentication uses DefaultAzureCredential, which picks up whichever
identity is available (Azure CLI login locally, managed identity in Azure).
"""

import logging
import os
import time

import requests
import yt_dlp
from azure.identity import DefaultAzureCredential

logger = logging.getLogger("video-indexer")

ARM_SCOPE = "https://management.azure.com/.default"
POLL_INTERVAL_SECONDS = 30
MAX_WAIT_SECONDS = 1800  # 30 min ceiling - avoids polling forever on a stuck job
REQUEST_TIMEOUT_SECONDS = 30
UPLOAD_TIMEOUT_SECONDS = 180  # file uploads need more time than quick JSON calls


class VideoIndexerService:
    """Thin client around Azure AI Video Indexer's ARM-based REST API."""

    def __init__(self) -> None:
        self.account_id = os.getenv("AZURE_VI_ACCOUNT_ID")
        self.location = os.getenv("AZURE_VI_LOCATION")
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        self.vi_name = os.getenv("AZURE_VI_NAME")

        missing = [
            name
            for name, val in [
                ("AZURE_VI_ACCOUNT_ID", self.account_id),
                ("AZURE_VI_LOCATION", self.location),
                ("AZURE_SUBSCRIPTION_ID", self.subscription_id),
                ("AZURE_RESOURCE_GROUP", self.resource_group),
                ("AZURE_VI_NAME", self.vi_name),
            ]
            if not val
        ]
        if missing:
            raise ValueError(f"Missing required Video Indexer env vars: {missing}")

        # Picks up `az login` locally, or managed identity when running inside Azure
        self.credential = DefaultAzureCredential()

    def download_youtube_video(
        self, url: str, output_path: str = "temp_video.mp4"
    ) -> str:
        """Download a YouTube video to a local file. Returns the local path."""
        logger.info(f"Downloading YouTube video: {url}")
        ydl_opts = {
            "format": "best",
            "outtmpl": output_path,
            "quiet": True,
            "no_warnings": True,
            "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            logger.info(f"Download complete: {output_path}")
            return output_path
        except Exception as e:
            raise RuntimeError(f"YouTube download failed for {url}: {e}") from e

    def upload_video(self, video_path: str, video_name: str) -> str:
        """Upload a local video file to Azure Video Indexer. Returns the Azure video ID."""
        vi_token = self._get_vi_access_token()
        api_url = f"https://api.videoindexer.ai/{self.location}/Accounts/{self.account_id}/Videos"
        params = {
            "accessToken": vi_token,
            "name": video_name,
            "privacy": "Private",
            "indexingPreset": "Default",
        }

        logger.info(f"Uploading {video_path} to Azure Video Indexer...")
        try:
            with open(video_path, "rb") as video_file:
                response = requests.post(
                    api_url,
                    params=params,
                    files={"file": video_file},
                    timeout=UPLOAD_TIMEOUT_SECONDS,
                )
        except requests.RequestException as e:
            raise RuntimeError(f"Network error uploading video to Azure: {e}") from e

        if response.status_code != 200:
            raise RuntimeError(
                f"Azure upload failed ({response.status_code}): {response.text}"
            )

        video_id = response.json().get("id")
        logger.info(f"Upload complete. Azure video ID: {video_id}")
        return video_id

    def wait_for_processing(self, video_id: str) -> dict:
        """Poll Azure until the video finishes processing. Returns the raw insights JSON."""
        logger.info(f"Waiting for video {video_id} to finish processing...")
        elapsed_seconds = (
            0  # Apple Rule: always give a loop counter a known starting value
        )

        while elapsed_seconds < MAX_WAIT_SECONDS:
            vi_token = self._get_vi_access_token()
            url = f"https://api.videoindexer.ai/{self.location}/Accounts/{self.account_id}/Videos/{video_id}/Index"

            try:
                response = requests.get(
                    url,
                    params={"accessToken": vi_token},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                data = response.json()
            except requests.RequestException as e:
                logger.warning(
                    f"Transient error polling video status: {e}. Retrying..."
                )
                time.sleep(POLL_INTERVAL_SECONDS)
                elapsed_seconds += POLL_INTERVAL_SECONDS
                continue

            state = data.get("state")
            if state == "Processed":
                logger.info("Video processing complete.")
                return data
            if state == "Failed":
                raise RuntimeError("Azure reported video indexing failed.")
            if state == "Quarantined":
                raise RuntimeError(
                    "Video quarantined by Azure (copyright/content policy)."
                )

            logger.info(f"Status: {state}. Waiting {POLL_INTERVAL_SECONDS}s...")
            time.sleep(POLL_INTERVAL_SECONDS)
            elapsed_seconds += POLL_INTERVAL_SECONDS

        raise TimeoutError(
            f"Video {video_id} did not finish processing within {MAX_WAIT_SECONDS}s."
        )

    def extract_data(self, vi_json: dict) -> dict:
        """Parse Azure's insights JSON into the fields our graph state needs."""
        transcript_lines = []
        ocr_lines = []
        for video in vi_json.get("videos", []):
            insights = video.get("insights", {})
            transcript_lines.extend(
                item.get("text", "") for item in insights.get("transcript", [])
            )
            ocr_lines.extend(item.get("text", "") for item in insights.get("ocr", []))

        return {
            "transcript": " ".join(transcript_lines),
            "ocr_text": ocr_lines,
            "video_metadata": {
                "duration": vi_json.get("summarizedInsights", {})
                .get("duration", {})
                .get("seconds"),
                "platform": "youtube",
            },
        }

    def _get_vi_access_token(self) -> str:
        """Exchange an ARM token for a Video Indexer account access token."""
        try:
            arm_token = self.credential.get_token(ARM_SCOPE).token
        except Exception as e:
            raise RuntimeError(f"Failed to get Azure ARM token: {e}") from e

        url = (
            f"https://management.azure.com/subscriptions/{self.subscription_id}"
            f"/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.VideoIndexer/accounts/{self.vi_name}"
            f"/generateAccessToken?api-version=2024-01-01"
        )
        headers = {"Authorization": f"Bearer {arm_token}"}
        payload = {"permissionType": "Contributor", "scope": "Account"}

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS
            )
        except requests.RequestException as e:
            raise RuntimeError(f"Network error generating VI access token: {e}") from e

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get VI account token ({response.status_code}): {response.text}"
            )

        return response.json().get("accessToken")
