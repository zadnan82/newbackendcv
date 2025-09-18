# app/cloud/providers/google_drive.py
"""
Google Drive provider - Separated and simplified for better debugging
"""

import json
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode

from ..config import get_settings
from ..schemas import CloudProvider, CloudFileMetadata, CompleteCV

logger = logging.getLogger(__name__)
settings = get_settings()


class GoogleDriveError(Exception):
    """Google Drive specific errors"""

    pass


class GoogleDriveProvider:
    """Google Drive integration - Simplified and focused"""

    def __init__(self, access_token: str):
        self.access_token = access_token
        self.api_base = "https://www.googleapis.com/drive/v3"
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request to Google Drive API"""
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        # Merge with any existing headers
        if "headers" in kwargs:
            headers.update(kwargs["headers"])
            kwargs["headers"] = headers
        else:
            kwargs["headers"] = headers

        logger.info(f"🔄 Google Drive API Request: {method} {url}")

        try:
            async with self.session.request(method, url, **kwargs) as response:
                logger.info(f"📊 Response Status: {response.status}")

                if response.status == 401:
                    error_text = await response.text()
                    logger.error(f"❌ Google Drive token expired: {error_text}")
                    raise GoogleDriveError("Access token expired or invalid")
                elif response.status >= 400:
                    error_text = await response.text()
                    logger.error(
                        f"❌ Google Drive API error {response.status}: {error_text}"
                    )
                    raise GoogleDriveError(f"API error {response.status}: {error_text}")

                # Handle different content types
                content_type = response.headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    result = await response.json()
                    logger.info(f"✅ JSON Response received")
                    return result
                else:
                    # For file downloads
                    text_result = await response.text()
                    logger.info(f"✅ Text Response received ({len(text_result)} chars)")
                    return {"content": text_result}

        except aiohttp.ClientError as e:
            logger.error(f"❌ Google Drive request failed: {str(e)}")
            raise GoogleDriveError(f"Request failed: {str(e)}")

    async def test_connection(self) -> Dict[str, Any]:
        """Test the connection and get basic user info"""
        try:
            url = f"{self.api_base}/about"
            params = {"fields": "user(displayName,emailAddress)"}

            result = await self._make_request("GET", url, params=params)

            user = result.get("user", {})
            return {
                "success": True,
                "user": {
                    "name": user.get("displayName", ""),
                    "email": user.get("emailAddress", ""),
                },
            }
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return {"success": False, "error": str(e)}

    async def _get_or_create_folder(self, folder_name: str = "CVs") -> str:
        """Get or create CV folder"""
        logger.info(f"🔍 Looking for folder: {folder_name}")

        # Search for existing folder
        search_url = f"{self.api_base}/files"
        params = {
            "q": f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
            "fields": "files(id, name)",
        }

        try:
            result = await self._make_request("GET", search_url, params=params)

            if result.get("files"):
                folder_id = result["files"][0]["id"]
                logger.info(f"✅ Found existing folder: {folder_id}")
                return folder_id

            # Create folder if it doesn't exist
            logger.info(f"📁 Creating new folder: {folder_name}")
            create_data = {
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
            }

            folder = await self._make_request("POST", search_url, json=create_data)
            folder_id = folder["id"]
            logger.info(f"✅ Created folder: {folder_id}")
            return folder_id

        except Exception as e:
            logger.error(f"❌ Folder operation failed: {e}")
            raise GoogleDriveError(f"Failed to get/create folder: {e}")

    async def list_files(self, folder_name: str = "CVs") -> List[CloudFileMetadata]:
        """List CV files in Google Drive folder"""
        try:
            folder_id = await self._get_or_create_folder(folder_name)
            logger.info(f"📋 Listing files in folder: {folder_id}")

            url = f"{self.api_base}/files"
            params = {
                "q": f"parents in '{folder_id}' and name contains '.json' and trashed=false",
                "fields": "files(id, name, createdTime, modifiedTime, size, parents)",
                "orderBy": "modifiedTime desc",
            }

            result = await self._make_request("GET", url, params=params)

            files = []
            for file_data in result.get("files", []):
                try:
                    files.append(
                        CloudFileMetadata(
                            file_id=file_data["id"],
                            name=file_data["name"],
                            provider=CloudProvider.GOOGLE_DRIVE,
                            created_at=datetime.fromisoformat(
                                file_data["createdTime"].replace("Z", "+00:00")
                            ),
                            last_modified=datetime.fromisoformat(
                                file_data["modifiedTime"].replace("Z", "+00:00")
                            ),
                            size_bytes=int(file_data.get("size", 0)),
                        )
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse file data: {e}")
                    continue

            logger.info(f"✅ Found {len(files)} files")
            return files

        except Exception as e:
            logger.error(f"❌ List files failed: {e}")
            raise GoogleDriveError(f"Failed to list files: {e}")

    async def upload_file(
        self, file_name: str, content: str, folder_name: str = "CVs"
    ) -> str:
        """Upload CV file to Google Drive - DEBUG VERSION"""
        try:
            logger.info(f"🐛 PROVIDER STEP 1: Starting upload for file: {file_name}")

            folder_id = await self._get_or_create_folder(folder_name)
            logger.info(f"🐛 PROVIDER STEP 2: Folder ID retrieved: {folder_id}")

            # Create file metadata
            metadata = {
                "name": file_name,
                "parents": [folder_id],
                "mimeType": "application/json",
            }
            logger.info(f"🐛 PROVIDER STEP 3: Metadata created")

            # Use multipart upload
            upload_url = (
                "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
            )
            logger.info(f"🐛 PROVIDER STEP 4: Upload URL set")

            # Prepare multipart data
            boundary = "==boundary=="
            logger.info(f"🐛 PROVIDER STEP 5: About to build multipart body")

            # Build multipart body
            body_parts = []

            # Metadata part
            body_parts.append(f"--{boundary}")
            body_parts.append("Content-Type: application/json")
            body_parts.append("")
            body_parts.append(json.dumps(metadata))

            # File content part
            body_parts.append(f"--{boundary}")
            body_parts.append("Content-Type: application/json")
            body_parts.append("")
            body_parts.append(content)
            body_parts.append(f"--{boundary}--")

            body = "\r\n".join(body_parts)
            logger.info(
                f"🐛 PROVIDER STEP 6: Multipart body created, size: {len(body)} chars"
            )

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": f"multipart/related; boundary={boundary}",
                "Content-Length": str(len(body.encode("utf-8"))),
            }
            logger.info(f"🐛 PROVIDER STEP 7: Headers prepared")

            logger.info(
                f"🐛 PROVIDER STEP 8: About to make HTTP request to Google Drive"
            )
            async with self.session.post(
                upload_url, headers=headers, data=body.encode("utf-8")
            ) as response:
                logger.info(
                    f"🐛 PROVIDER STEP 9: HTTP request completed, status: {response.status}"
                )

                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(
                        f"❌ PROVIDER STEP 9 FAILED: Upload failed: {error_text}"
                    )
                    raise GoogleDriveError(f"Upload failed: {error_text}")

                result = await response.json()
                file_id = result["id"]
                logger.info(
                    f"✅ PROVIDER STEP 10: File uploaded successfully: {file_id}"
                )
                return file_id

        except Exception as e:
            logger.error(f"❌ PROVIDER ERROR: File upload failed: {e}")
            import traceback

            logger.error(f"❌ PROVIDER TRACEBACK: {traceback.format_exc()}")
            raise GoogleDriveError(f"Failed to upload file: {e}")

    async def download_file(self, file_id: str) -> str:
        """Download file content from Google Drive"""
        try:
            logger.info(f"📥 Downloading file: {file_id}")

            url = f"{self.api_base}/files/{file_id}"
            params = {"alt": "media"}

            result = await self._make_request("GET", url, params=params)

            # Handle the response
            if isinstance(result, dict) and "content" in result:
                content = result["content"]
                logger.info(f"✅ File downloaded successfully ({len(content)} chars)")
                return content
            else:
                logger.error("❌ Unexpected response format")
                raise GoogleDriveError("Unexpected response format")

        except Exception as e:
            logger.error(f"❌ File download failed: {e}")
            raise GoogleDriveError(f"Failed to download file: {e}")

    async def delete_file(self, file_id: str) -> bool:
        """Delete file from Google Drive"""
        try:
            logger.info(f"🗑️ Deleting file: {file_id}")

            url = f"{self.api_base}/files/{file_id}"
            await self._make_request("DELETE", url)

            logger.info(f"✅ File deleted successfully")
            return True

        except Exception as e:
            logger.error(f"❌ File deletion failed: {e}")
            return False

    async def get_user_info(self) -> Dict[str, Any]:
        """Get Google Drive user information"""
        try:
            url = f"{self.api_base}/about"
            params = {"fields": "user(displayName,emailAddress)"}

            result = await self._make_request("GET", url, params=params)
            user = result.get("user", {})

            return {
                "name": user.get("displayName", ""),
                "email": user.get("emailAddress", ""),
                "provider": "google_drive",
            }

        except Exception as e:
            logger.error(f"❌ Get user info failed: {e}")
            raise GoogleDriveError(f"Failed to get user info: {e}")

    async def get_storage_quota(self) -> Dict[str, Any]:
        """Get Google Drive storage quota"""
        try:
            url = f"{self.api_base}/about"
            params = {"fields": "storageQuota"}

            result = await self._make_request("GET", url, params=params)
            quota = result.get("storageQuota", {})

            total = int(quota.get("limit", 0))
            used = int(quota.get("usage", 0))

            return {
                "total": total,
                "used": used,
                "available": max(0, total - used),
            }

        except Exception as e:
            logger.error(f"❌ Get storage quota failed: {e}")
            return {"total": 0, "used": 0, "available": 0}


class GoogleDriveOAuth:
    """Handle Google Drive OAuth flow"""

    def __init__(self):
        self.client_id = settings.google_client_id
        self.client_secret = settings.google_client_secret
        self.redirect_uri = settings.google_redirect_uri

    def get_auth_url(self, state: str) -> str:
        """Generate Google OAuth URL"""
        base_url = "https://accounts.google.com/o/oauth2/auth"

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "https://www.googleapis.com/auth/drive.file",
            "response_type": "code",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }

        auth_url = f"{base_url}?{urlencode(params)}"
        logger.info(f"🔗 Generated Google OAuth URL")
        return auth_url

    async def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens"""
        token_url = "https://oauth2.googleapis.com/token"

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
        }

        logger.info(f"🔄 Exchanging code for tokens...")

        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"❌ Token exchange failed: {error_text}")
                    raise GoogleDriveError(f"Token exchange failed: {response.status}")

                token_data = await response.json()

                # Calculate expiry time
                expires_in = token_data.get("expires_in", 3600)
                expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

                result = {
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data.get("refresh_token"),
                    "expires_at": expires_at.isoformat(),
                    "expires_in": expires_in,
                    "token_type": token_data.get("token_type", "Bearer"),
                    "scope": token_data.get("scope", ""),
                }

                logger.info(f"✅ Tokens exchanged successfully")
                return result

    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token"""
        token_url = "https://oauth2.googleapis.com/token"

        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        logger.info(f"🔄 Refreshing token...")

        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"❌ Token refresh failed: {error_text}")
                    raise GoogleDriveError(f"Token refresh failed: {response.status}")

                token_data = await response.json()

                # Calculate expiry time
                expires_in = token_data.get("expires_in", 3600)
                expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

                result = {
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data.get(
                        "refresh_token", refresh_token
                    ),  # Keep old if not provided
                    "expires_at": expires_at.isoformat(),
                    "expires_in": expires_in,
                    "token_type": token_data.get("token_type", "Bearer"),
                    "scope": token_data.get("scope", ""),
                }

                logger.info(f"✅ Token refreshed successfully")
                return result


# Factory function for backwards compatibility
def get_google_drive_provider(access_token: str) -> GoogleDriveProvider:
    """Factory function to create Google Drive provider"""
    return GoogleDriveProvider(access_token)


# OAuth instance
google_drive_oauth = GoogleDriveOAuth()
