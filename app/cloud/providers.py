# app/cloud/providers.py
"""
Cloud provider integrations for Google Drive, OneDrive, Dropbox, Box
"""

import json
import aiohttp
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from ..config import settings, CloudConfig
from ..schemas import CloudProvider, CloudFileMetadata, CompleteCV

logger = logging.getLogger(__name__)


class CloudProviderError(Exception):
    """Base exception for cloud provider errors"""

    pass


class CloudProviderBase(ABC):
    """Base class for all cloud providers"""

    def __init__(self, access_token: str):
        self.access_token = access_token
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @abstractmethod
    async def list_files(self, folder_name: str = "CVs") -> List[CloudFileMetadata]:
        """List CV files in the specified folder"""
        pass

    @abstractmethod
    async def upload_file(
        self, file_name: str, content: str, folder_name: str = "CVs"
    ) -> str:
        """Upload a file and return file ID"""
        pass

    @abstractmethod
    async def download_file(self, file_id: str) -> str:
        """Download file content by ID"""
        pass

    @abstractmethod
    async def delete_file(self, file_id: str) -> bool:
        """Delete file by ID"""
        pass

    @abstractmethod
    async def get_user_info(self) -> Dict[str, Any]:
        """Get user information"""
        pass

    @abstractmethod
    async def get_storage_quota(self) -> Dict[str, Any]:
        """Get storage quota information"""
        pass


class GoogleDriveProvider(CloudProviderBase):
    """Google Drive integration"""

    def __init__(self, access_token: str):
        super().__init__(access_token)
        self.api_base = CloudConfig.PROVIDERS["google_drive"]["api_base"]

    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request to Google Drive API"""
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        try:
            async with self.session.request(
                method, url, headers=headers, **kwargs
            ) as response:
                if response.status == 401:
                    raise CloudProviderError("Google Drive access token expired")
                elif response.status >= 400:
                    error_text = await response.text()
                    raise CloudProviderError(f"Google Drive API error: {error_text}")

                return await response.json()
        except aiohttp.ClientError as e:
            raise CloudProviderError(f"Google Drive request failed: {str(e)}")

    async def _get_or_create_folder(self, folder_name: str) -> str:
        """Get or create CV folder"""
        # Search for existing folder
        search_url = f"{self.api_base}/files"
        params = {
            "q": f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
            "fields": "files(id, name)",
        }

        result = await self._make_request("GET", search_url, params=params)

        if result.get("files"):
            return result["files"][0]["id"]

        # Create folder if it doesn't exist
        create_data = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
        }

        folder = await self._make_request("POST", search_url, json=create_data)
        return folder["id"]

    async def list_files(self, folder_name: str = "CVs") -> List[CloudFileMetadata]:
        """List CV files in Google Drive folder"""
        folder_id = await self._get_or_create_folder(folder_name)

        url = f"{self.api_base}/files"
        params = {
            "q": f"parents in '{folder_id}' and name contains '.json'",
            "fields": "files(id, name, createdTime, modifiedTime, size)",
            "orderBy": "modifiedTime desc",
        }

        result = await self._make_request("GET", url, params=params)

        files = []
        for file_data in result.get("files", []):
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

        return files

    async def upload_file(
        self, file_name: str, content: str, folder_name: str = "CVs"
    ) -> str:
        """Upload CV file to Google Drive"""
        folder_id = await self._get_or_create_folder(folder_name)

        # Create file metadata
        metadata = {"name": file_name, "parents": [folder_id]}

        # Upload file content
        upload_url = "https://www.googleapis.com/upload/drive/v3/files"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
        }

        # Multipart upload
        form_data = aiohttp.FormData()
        form_data.add_field(
            "metadata", json.dumps(metadata), content_type="application/json"
        )
        form_data.add_field(
            "file", content, content_type="application/json", filename=file_name
        )

        try:
            async with self.session.post(
                f"{upload_url}?uploadType=multipart",
                headers={"Authorization": f"Bearer {self.access_token}"},
                data=form_data,
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise CloudProviderError(f"Upload failed: {error_text}")

                result = await response.json()
                return result["id"]

        except aiohttp.ClientError as e:
            raise CloudProviderError(f"Upload request failed: {str(e)}")

    async def download_file(self, file_id: str) -> str:
        """Download file content from Google Drive"""
        url = f"{self.api_base}/files/{file_id}"
        params = {"alt": "media"}

        headers = {"Authorization": f"Bearer {self.access_token}"}

        try:
            async with self.session.get(
                url, headers=headers, params=params
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise CloudProviderError(f"Download failed: {error_text}")

                return await response.text()

        except aiohttp.ClientError as e:
            raise CloudProviderError(f"Download request failed: {str(e)}")

    async def delete_file(self, file_id: str) -> bool:
        """Delete file from Google Drive"""
        url = f"{self.api_base}/files/{file_id}"

        try:
            await self._make_request("DELETE", url)
            return True
        except CloudProviderError:
            return False

    async def get_user_info(self) -> Dict[str, Any]:
        """Get Google Drive user information"""
        url = f"{self.api_base}/about"
        params = {"fields": "user(displayName,emailAddress)"}

        result = await self._make_request("GET", url, params=params)
        user = result.get("user", {})

        return {
            "name": user.get("displayName", ""),
            "email": user.get("emailAddress", ""),
            "provider": "google_drive",
        }

    async def get_storage_quota(self) -> Dict[str, Any]:
        """Get Google Drive storage quota"""
        url = f"{self.api_base}/about"
        params = {"fields": "storageQuota"}

        result = await self._make_request("GET", url, params=params)
        quota = result.get("storageQuota", {})

        return {
            "total": int(quota.get("limit", 0)),
            "used": int(quota.get("usage", 0)),
            "available": int(quota.get("limit", 0)) - int(quota.get("usage", 0)),
        }


class OneDriveProvider(CloudProviderBase):
    """Microsoft OneDrive integration"""

    def __init__(self, access_token: str):
        super().__init__(access_token)
        self.api_base = CloudConfig.PROVIDERS["onedrive"]["api_base"]

    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request to OneDrive API"""
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        try:
            async with self.session.request(
                method, url, headers=headers, **kwargs
            ) as response:
                if response.status == 401:
                    raise CloudProviderError("OneDrive access token expired")
                elif response.status >= 400:
                    error_text = await response.text()
                    raise CloudProviderError(f"OneDrive API error: {error_text}")

                return await response.json()
        except aiohttp.ClientError as e:
            raise CloudProviderError(f"OneDrive request failed: {str(e)}")

    async def _get_or_create_folder(self, folder_name: str) -> str:
        """Get or create CV folder in OneDrive"""
        # Check if folder exists
        search_url = f"{self.api_base}/me/drive/root/children"
        params = {"$filter": f"name eq '{folder_name}' and folder ne null"}

        result = await self._make_request("GET", search_url, params=params)

        if result.get("value"):
            return result["value"][0]["id"]

        # Create folder
        create_data = {
            "name": folder_name,
            "folder": {},
            "@microsoft.graph.conflictBehavior": "rename",
        }

        folder = await self._make_request("POST", search_url, json=create_data)
        return folder["id"]

    async def list_files(self, folder_name: str = "CVs") -> List[CloudFileMetadata]:
        """List CV files in OneDrive folder"""
        folder_id = await self._get_or_create_folder(folder_name)

        url = f"{self.api_base}/me/drive/items/{folder_id}/children"
        params = {"$filter": "endswith(name,'.json')"}

        result = await self._make_request("GET", url, params=params)

        files = []
        for file_data in result.get("value", []):
            files.append(
                CloudFileMetadata(
                    file_id=file_data["id"],
                    name=file_data["name"],
                    provider=CloudProvider.ONEDRIVE,
                    created_at=datetime.fromisoformat(
                        file_data["createdDateTime"].replace("Z", "+00:00")
                    ),
                    last_modified=datetime.fromisoformat(
                        file_data["lastModifiedDateTime"].replace("Z", "+00:00")
                    ),
                    size_bytes=file_data.get("size", 0),
                )
            )

        return files

    async def upload_file(
        self, file_name: str, content: str, folder_name: str = "CVs"
    ) -> str:
        """Upload CV file to OneDrive"""
        folder_id = await self._get_or_create_folder(folder_name)

        url = f"{self.api_base}/me/drive/items/{folder_id}:/{file_name}:/content"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        try:
            async with self.session.put(
                url, headers=headers, data=content.encode()
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise CloudProviderError(f"OneDrive upload failed: {error_text}")

                result = await response.json()
                return result["id"]

        except aiohttp.ClientError as e:
            raise CloudProviderError(f"OneDrive upload request failed: {str(e)}")

    async def download_file(self, file_id: str) -> str:
        """Download file content from OneDrive"""
        url = f"{self.api_base}/me/drive/items/{file_id}/content"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise CloudProviderError(f"OneDrive download failed: {error_text}")

                return await response.text()

        except aiohttp.ClientError as e:
            raise CloudProviderError(f"OneDrive download request failed: {str(e)}")

    async def delete_file(self, file_id: str) -> bool:
        """Delete file from OneDrive"""
        url = f"{self.api_base}/me/drive/items/{file_id}"

        try:
            await self._make_request("DELETE", url)
            return True
        except CloudProviderError:
            return False

    async def get_user_info(self) -> Dict[str, Any]:
        """Get OneDrive user information"""
        url = f"{self.api_base}/me"

        result = await self._make_request("GET", url)

        return {
            "name": result.get("displayName", ""),
            "email": result.get("mail", result.get("userPrincipalName", "")),
            "provider": "onedrive",
        }

    async def get_storage_quota(self) -> Dict[str, Any]:
        """Get OneDrive storage quota"""
        url = f"{self.api_base}/me/drive"

        result = await self._make_request("GET", url)
        quota = result.get("quota", {})

        return {
            "total": quota.get("total", 0),
            "used": quota.get("used", 0),
            "available": quota.get("remaining", 0),
        }


class DropboxProvider(CloudProviderBase):
    """Dropbox integration"""

    def __init__(self, access_token: str):
        super().__init__(access_token)
        self.api_base = CloudConfig.PROVIDERS["dropbox"]["api_base"]

    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request to Dropbox API"""
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        try:
            async with self.session.request(
                method, url, headers=headers, **kwargs
            ) as response:
                if response.status == 401:
                    raise CloudProviderError("Dropbox access token expired")
                elif response.status >= 400:
                    error_text = await response.text()
                    raise CloudProviderError(f"Dropbox API error: {error_text}")

                return await response.json()
        except aiohttp.ClientError as e:
            raise CloudProviderError(f"Dropbox request failed: {str(e)}")

    async def list_files(self, folder_name: str = "CVs") -> List[CloudFileMetadata]:
        """List CV files in Dropbox folder"""
        url = f"{self.api_base}/files/list_folder"
        data = {"path": f"/{folder_name}", "recursive": False}

        try:
            result = await self._make_request("POST", url, json=data)
        except CloudProviderError as e:
            if "not_found" in str(e):
                # Folder doesn't exist, return empty list
                return []
            raise

        files = []
        for entry in result.get("entries", []):
            if entry.get(".tag") == "file" and entry["name"].endswith(".json"):
                files.append(
                    CloudFileMetadata(
                        file_id=entry["id"],
                        name=entry["name"],
                        provider=CloudProvider.DROPBOX,
                        created_at=datetime.fromisoformat(
                            entry["client_modified"].replace("Z", "+00:00")
                        ),
                        last_modified=datetime.fromisoformat(
                            entry["server_modified"].replace("Z", "+00:00")
                        ),
                        size_bytes=entry.get("size", 0),
                    )
                )

        return files

    async def upload_file(
        self, file_name: str, content: str, folder_name: str = "CVs"
    ) -> str:
        """Upload CV file to Dropbox"""
        url = f"{self.api_base}/files/upload"

        # Create folder if it doesn't exist
        await self._create_folder_if_not_exists(folder_name)

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/octet-stream",
            "Dropbox-API-Arg": json.dumps(
                {
                    "path": f"/{folder_name}/{file_name}",
                    "mode": "overwrite",
                    "autorename": True,
                }
            ),
        }

        try:
            async with self.session.post(
                url, headers=headers, data=content.encode()
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise CloudProviderError(f"Dropbox upload failed: {error_text}")

                result = await response.json()
                return result["id"]

        except aiohttp.ClientError as e:
            raise CloudProviderError(f"Dropbox upload request failed: {str(e)}")

    async def _create_folder_if_not_exists(self, folder_name: str):
        """Create folder if it doesn't exist"""
        url = f"{self.api_base}/files/create_folder_v2"
        data = {"path": f"/{folder_name}", "autorename": False}

        try:
            await self._make_request("POST", url, json=data)
        except CloudProviderError:
            # Folder might already exist, ignore error
            pass

    async def download_file(self, file_id: str) -> str:
        """Download file content from Dropbox"""
        # First get file metadata to get the path
        metadata_url = f"{self.api_base}/files/get_metadata"
        metadata_data = {"path": file_id}

        try:
            metadata = await self._make_request(
                "POST", metadata_url, json=metadata_data
            )
            file_path = metadata["path_display"]
        except CloudProviderError:
            # If file_id is already a path, use it directly
            file_path = file_id

        url = f"{self.api_base}/files/download"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Dropbox-API-Arg": json.dumps({"path": file_path}),
        }

        try:
            async with self.session.post(url, headers=headers) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise CloudProviderError(f"Dropbox download failed: {error_text}")

                return await response.text()

        except aiohttp.ClientError as e:
            raise CloudProviderError(f"Dropbox download request failed: {str(e)}")

    async def delete_file(self, file_id: str) -> bool:
        """Delete file from Dropbox"""
        url = f"{self.api_base}/files/delete_v2"
        data = {"path": file_id}

        try:
            await self._make_request("POST", url, json=data)
            return True
        except CloudProviderError:
            return False

    async def get_user_info(self) -> Dict[str, Any]:
        """Get Dropbox user information"""
        url = f"{self.api_base}/users/get_current_account"

        result = await self._make_request("POST", url)

        return {
            "name": result.get("name", {}).get("display_name", ""),
            "email": result.get("email", ""),
            "provider": "dropbox",
        }

    async def get_storage_quota(self) -> Dict[str, Any]:
        """Get Dropbox storage quota"""
        url = f"{self.api_base}/users/get_space_usage"

        result = await self._make_request("POST", url)

        used = result.get("used", 0)
        allocated = result.get("allocation", {})
        total = (
            allocated.get("allocated", 0)
            if allocated.get(".tag") == "individual"
            else 0
        )

        return {"total": total, "used": used, "available": max(0, total - used)}


class BoxProvider(CloudProviderBase):
    """Box integration"""

    def __init__(self, access_token: str):
        super().__init__(access_token)
        self.api_base = CloudConfig.PROVIDERS["box"]["api_base"]

    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request to Box API"""
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        try:
            async with self.session.request(
                method, url, headers=headers, **kwargs
            ) as response:
                if response.status == 401:
                    raise CloudProviderError("Box access token expired")
                elif response.status >= 400:
                    error_text = await response.text()
                    raise CloudProviderError(f"Box API error: {error_text}")

                return await response.json()
        except aiohttp.ClientError as e:
            raise CloudProviderError(f"Box request failed: {str(e)}")

    async def _get_or_create_folder(self, folder_name: str) -> str:
        """Get or create CV folder in Box"""
        # Search for existing folder
        search_url = f"{self.api_base}/search"
        params = {
            "query": folder_name,
            "type": "folder",
            "ancestor_folder_ids": "0",  # Root folder
        }

        result = await self._make_request("GET", search_url, params=params)

        for entry in result.get("entries", []):
            if entry["name"] == folder_name and entry["type"] == "folder":
                return entry["id"]

        # Create folder if not found
        create_url = f"{self.api_base}/folders"
        create_data = {
            "name": folder_name,
            "parent": {"id": "0"},  # Root folder
        }

        folder = await self._make_request("POST", create_url, json=create_data)
        return folder["id"]

    async def list_files(self, folder_name: str = "CVs") -> List[CloudFileMetadata]:
        """List CV files in Box folder"""
        folder_id = await self._get_or_create_folder(folder_name)

        url = f"{self.api_base}/folders/{folder_id}/items"
        params = {"fields": "id,name,created_at,modified_at,size"}

        result = await self._make_request("GET", url, params=params)

        files = []
        for entry in result.get("entries", []):
            if entry["type"] == "file" and entry["name"].endswith(".json"):
                files.append(
                    CloudFileMetadata(
                        file_id=entry["id"],
                        name=entry["name"],
                        provider=CloudProvider.BOX,
                        created_at=datetime.fromisoformat(
                            entry["created_at"].replace("Z", "+00:00")
                        ),
                        last_modified=datetime.fromisoformat(
                            entry["modified_at"].replace("Z", "+00:00")
                        ),
                        size_bytes=entry.get("size", 0),
                    )
                )

        return files

    async def upload_file(
        self, file_name: str, content: str, folder_name: str = "CVs"
    ) -> str:
        """Upload CV file to Box"""
        folder_id = await self._get_or_create_folder(folder_name)

        url = "https://upload.box.com/api/2.0/files/content"

        # Prepare multipart form data
        form_data = aiohttp.FormData()
        form_data.add_field(
            "attributes", json.dumps({"name": file_name, "parent": {"id": folder_id}})
        )
        form_data.add_field(
            "file", content, filename=file_name, content_type="application/json"
        )

        headers = {"Authorization": f"Bearer {self.access_token}"}

        try:
            async with self.session.post(
                url, headers=headers, data=form_data
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise CloudProviderError(f"Box upload failed: {error_text}")

                result = await response.json()
                return result["entries"][0]["id"]

        except aiohttp.ClientError as e:
            raise CloudProviderError(f"Box upload request failed: {str(e)}")

    async def download_file(self, file_id: str) -> str:
        """Download file content from Box"""
        url = f"{self.api_base}/files/{file_id}/content"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise CloudProviderError(f"Box download failed: {error_text}")

                return await response.text()

        except aiohttp.ClientError as e:
            raise CloudProviderError(f"Box download request failed: {str(e)}")

    async def delete_file(self, file_id: str) -> bool:
        """Delete file from Box"""
        url = f"{self.api_base}/files/{file_id}"

        try:
            await self._make_request("DELETE", url)
            return True
        except CloudProviderError:
            return False

    async def get_user_info(self) -> Dict[str, Any]:
        """Get Box user information"""
        url = f"{self.api_base}/users/me"

        result = await self._make_request("GET", url)

        return {
            "name": result.get("name", ""),
            "email": result.get("login", ""),
            "provider": "box",
        }

    async def get_storage_quota(self) -> Dict[str, Any]:
        """Get Box storage quota"""
        url = f"{self.api_base}/users/me"

        result = await self._make_request("GET", url)

        total = result.get("space_amount", 0)
        used = result.get("space_used", 0)

        return {"total": total, "used": used, "available": max(0, total - used)}


# Provider factory
def get_cloud_provider(provider: CloudProvider, access_token: str) -> CloudProviderBase:
    """Factory function to get the appropriate cloud provider"""
    providers = {
        CloudProvider.GOOGLE_DRIVE: GoogleDriveProvider,
        CloudProvider.ONEDRIVE: OneDriveProvider,
        CloudProvider.DROPBOX: DropboxProvider,
        CloudProvider.BOX: BoxProvider,
    }

    if provider not in providers:
        raise ValueError(f"Unsupported cloud provider: {provider}")

    return providers[provider](access_token)
