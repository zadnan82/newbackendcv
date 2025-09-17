# app/cloud/google_drive_service.py
"""
Google Drive focused service - Simplified for debugging and reliability
"""

import io
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from cryptography.fernet import Fernet
from pydantic import ValidationError

from ..config import get_settings
from ..schemas import (
    CloudProvider,
    CloudFileMetadata,
    CompleteCV,
    CVFileMetadata,
    CloudConnectionStatus,
)
from .google_drive import (
    GoogleDriveProvider,
    GoogleDriveError,
    google_drive_oauth,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class GoogleDriveService:
    """Focused service for Google Drive operations"""

    def __init__(self):
        # Use persistent encryption key from settings
        self.encryption_key = settings.get_encryption_key_bytes()
        self.cipher = Fernet(self.encryption_key)
        logger.info("GoogleDriveService initialized")

    def _encrypt_token_data(self, token_data: Dict[str, Any]) -> str:
        """Encrypt Google Drive token data"""
        try:
            token_json = json.dumps(token_data, default=str)
            encrypted = self.cipher.encrypt(token_json.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Token encryption failed: {e}")
            raise GoogleDriveError(f"Failed to encrypt tokens: {str(e)}")

    def _decrypt_token_data(self, encrypted_tokens: str) -> Dict[str, Any]:
        """Decrypt Google Drive token data"""
        try:
            decrypted = self.cipher.decrypt(encrypted_tokens.encode())
            return json.loads(decrypted.decode())
        except Exception as e:
            logger.error(f"Token decryption failed: {e}")
            raise GoogleDriveError(f"Failed to decrypt tokens: {str(e)}")

    def _format_cv_filename(self, title: str, timestamp: datetime = None) -> str:
        """Generate standardized CV filename for Google Drive"""
        timestamp = timestamp or datetime.utcnow()
        safe_title = "".join(
            c for c in title if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        safe_title = safe_title.replace(" ", "_")[:50]
        return f"cv_{safe_title}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"

    def _parse_cv_from_storage(self, content: str) -> CompleteCV:
        """Parse CV data from Google Drive storage"""
        try:
            data = json.loads(content)
            cv_data = data.get("cv_data", data)
            return CompleteCV.parse_obj(cv_data)
        except Exception as e:
            logger.error(f"CV parsing failed: {e}")
            raise ValueError(f"Invalid CV file format: {e}")

    async def get_connection_status(
        self, token_data: Dict[str, Any]
    ) -> CloudConnectionStatus:
        """Get Google Drive connection status"""
        if not token_data or not token_data.get("access_token"):
            return CloudConnectionStatus(
                provider=CloudProvider.GOOGLE_DRIVE,
                connected=False,
                email=None,
                storage_quota=None,
            )

        try:
            access_token = token_data["access_token"]

            async with GoogleDriveProvider(access_token) as provider:
                # Test connection and get user info
                connection_test = await provider.test_connection()

                if not connection_test["success"]:
                    return CloudConnectionStatus(
                        provider=CloudProvider.GOOGLE_DRIVE,
                        connected=False,
                        email=None,
                        storage_quota=None,
                    )

                # Get storage quota
                try:
                    storage_quota = await provider.get_storage_quota()
                except Exception as e:
                    logger.warning(f"Failed to get storage quota: {e}")
                    storage_quota = None

                return CloudConnectionStatus(
                    provider=CloudProvider.GOOGLE_DRIVE,
                    connected=True,
                    email=connection_test["user"]["email"],
                    storage_quota=storage_quota,
                )

        except Exception as e:
            logger.error(f"Google Drive connection status check failed: {e}")
            return CloudConnectionStatus(
                provider=CloudProvider.GOOGLE_DRIVE,
                connected=False,
                email=None,
                storage_quota=None,
                error=str(e),
            )

    # In your google_drive_service.py

    # In google_drive_service.py
    async def save_cv(self, tokens: dict, cv_data: Dict) -> str:
        """Save CV to Google Drive - handle CompleteCV schema"""
        try:
            # Convert to CompleteCV schema for validation
            complete_cv = CompleteCV(**cv_data)

            # Prepare for storage
            file_name = self._format_cv_filename(complete_cv.title)
            content = self._prepare_cv_for_storage(complete_cv)

            # Upload to Google Drive
            access_token = tokens["access_token"]

            async with GoogleDriveProvider(access_token) as provider:
                file_id = await provider.upload_file(file_name, content)

            logger.info(f"✅ CV saved to Google Drive: {file_id}")
            return file_id

        except ValidationError as e:
            logger.error(f"❌ CV validation failed: {e}")
            raise GoogleDriveError(f"Invalid CV data: {e}")
        except Exception as e:
            logger.error(f"❌ Google Drive save failed: {e}")
            raise GoogleDriveError(f"Failed to save CV: {str(e)}")

    def _prepare_cv_for_storage(self, cv_data: CompleteCV) -> str:
        """Prepare CV data for Google Drive storage"""
        storage_data = {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat(),
                "last_modified": datetime.utcnow().isoformat(),
                "created_with": "cv-privacy-platform",
                "provider": "google_drive",
            },
            "cv_data": cv_data.dict(),  # Convert CompleteCV to dict
        }
        return json.dumps(storage_data, indent=2, default=str)

    async def load_cv(self, token_data: Dict[str, Any], file_id: str) -> CompleteCV:
        """Load CV from Google Drive"""
        if not token_data or not token_data.get("access_token"):
            raise GoogleDriveError("No Google Drive access token available")

        access_token = token_data["access_token"]

        try:
            async with GoogleDriveProvider(access_token) as provider:
                content = await provider.download_file(file_id)
                cv_data = self._parse_cv_from_storage(content)
                logger.info(f"CV loaded from Google Drive: {file_id}")
                return cv_data
        except Exception as e:
            logger.error(f"Failed to load CV from Google Drive: {e}")
            raise GoogleDriveError(f"Failed to load CV: {str(e)}")

    async def list_cvs(self, token_data: Dict[str, Any]) -> List[CloudFileMetadata]:
        """List all CVs from Google Drive"""
        if not token_data or not token_data.get("access_token"):
            raise GoogleDriveError("No Google Drive access token available")

        access_token = token_data["access_token"]

        try:
            async with GoogleDriveProvider(access_token) as provider:
                files = await provider.list_files("CVs")
                logger.info(f"Listed {len(files)} CVs from Google Drive")
                return files
        except Exception as e:
            logger.error(f"Failed to list CVs from Google Drive: {e}")
            raise GoogleDriveError(f"Failed to list CVs: {str(e)}")

    async def delete_cv(self, token_data: Dict[str, Any], file_id: str) -> bool:
        """Delete CV from Google Drive"""
        if not token_data or not token_data.get("access_token"):
            raise GoogleDriveError("No Google Drive access token available")

        access_token = token_data["access_token"]

        try:
            async with GoogleDriveProvider(access_token) as provider:
                success = await provider.delete_file(file_id)

                if success:
                    logger.info(f"CV deleted from Google Drive: {file_id}")
                else:
                    logger.warning(f"Failed to delete CV from Google Drive: {file_id}")

                return success
        except Exception as e:
            logger.error(f"Failed to delete CV from Google Drive: {e}")
            raise GoogleDriveError(f"Failed to delete CV: {str(e)}")

    async def test_connection(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test Google Drive connection"""
        if not token_data or not token_data.get("access_token"):
            return {
                "success": False,
                "error": "No access token available",
                "provider": "google_drive",
            }

        access_token = token_data["access_token"]

        try:
            async with GoogleDriveProvider(access_token) as provider:
                result = await provider.test_connection()
                result["provider"] = "google_drive"
                return result
        except Exception as e:
            logger.error(f"Google Drive connection test failed: {e}")
            return {"success": False, "error": str(e), "provider": "google_drive"}

    def get_oauth_url(self, state: str) -> str:
        """Get Google Drive OAuth authorization URL"""
        return google_drive_oauth.get_auth_url(state)

    async def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens"""
        return await google_drive_oauth.exchange_code_for_tokens(code)

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh Google Drive access token"""
        if not refresh_token:
            raise GoogleDriveError("Refresh token is required")
        return await google_drive_oauth.refresh_token(refresh_token)

    async def ensure_valid_token(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure token is valid, refresh if necessary"""
        if not token_data.get("expires_at"):
            return token_data  # No expiration info, assume valid

        try:
            expires_at = datetime.fromisoformat(token_data["expires_at"])
            now = datetime.utcnow()

            # Check if token expires in the next 5 minutes
            if (expires_at - now).total_seconds() < 300:
                logger.info("Google Drive token expired/expiring, refreshing...")

                if not token_data.get("refresh_token"):
                    raise GoogleDriveError(
                        "Token expired and no refresh token available"
                    )

                # Refresh the token
                refreshed_tokens = await self.refresh_access_token(
                    token_data["refresh_token"]
                )

                # Merge old data with new tokens
                updated_token_data = {
                    **token_data,
                    "access_token": refreshed_tokens["access_token"],
                    "expires_at": refreshed_tokens["expires_at"],
                    "refresh_token": refreshed_tokens.get(
                        "refresh_token", token_data.get("refresh_token")
                    ),
                }

                logger.info("Successfully refreshed Google Drive token")
                return updated_token_data

        except Exception as e:
            logger.error(f"Token validation/refresh failed: {e}")
            # Return original token data, let the API call fail if it's actually invalid

        return token_data


# Global Google Drive service instance
google_drive_service = GoogleDriveService()
