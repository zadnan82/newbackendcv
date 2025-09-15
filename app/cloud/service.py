# app/cloud/service.py
"""
Unified cloud service that abstracts all cloud providers
FIXED: Proper encryption key management
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from cryptography.fernet import Fernet
import logging

from ..config import get_settings
from ..schemas import (
    CloudProvider,
    CloudFileMetadata,
    CompleteCV,
    CVFileMetadata,
    CloudSession,
    CloudConnectionStatus,
)
from .providers import get_cloud_provider, CloudProviderError

logger = logging.getLogger(__name__)
settings = get_settings()


class CloudService:
    """Unified service for managing CV files across multiple cloud providers"""

    def __init__(self):
        # FIXED: Use persistent encryption key from settings
        self.encryption_key = settings.get_encryption_key_bytes()
        self.cipher = Fernet(self.encryption_key)

        logger.info("CloudService initialized with persistent encryption key")

    def _encrypt_tokens(self, tokens: Dict[str, Any]) -> str:
        """Encrypt cloud provider tokens securely"""
        try:
            tokens_json = json.dumps(tokens, default=str)  # Handle datetime objects
            encrypted = self.cipher.encrypt(tokens_json.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Token encryption failed: {e}")
            raise CloudProviderError(f"Failed to encrypt tokens: {str(e)}")

    def _decrypt_tokens(self, encrypted_tokens: str) -> Dict[str, Any]:
        """Decrypt cloud provider tokens securely"""
        try:
            decrypted = self.cipher.decrypt(encrypted_tokens.encode())
            return json.loads(decrypted.decode())
        except Exception as e:
            logger.error(f"Token decryption failed: {e}")
            raise CloudProviderError(f"Failed to decrypt tokens: {str(e)}")

    def _generate_session_id(self, user_identifier: str = None) -> str:
        """Generate unique session ID"""
        timestamp = str(datetime.utcnow().timestamp())
        data = f"{user_identifier or 'anonymous'}_{timestamp}_{settings.secret_key}"
        return hashlib.sha256(data.encode()).hexdigest()

    def _format_cv_filename(self, title: str, timestamp: datetime = None) -> str:
        """Generate standardized CV filename"""
        timestamp = timestamp or datetime.utcnow()
        # Sanitize title for filename
        safe_title = "".join(
            c for c in title if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        safe_title = safe_title.replace(" ", "_")[:50]  # Limit length
        return f"cv_{safe_title}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"

    def _prepare_cv_for_storage(self, cv_data: CompleteCV) -> str:
        """Prepare CV data for cloud storage with metadata"""
        storage_data = {
            "metadata": CVFileMetadata(
                version="1.0",
                created_at=datetime.utcnow(),
                last_modified=datetime.utcnow(),
                created_with="cv-privacy-platform",
            ).dict(),
            "cv_data": cv_data.dict(),
        }
        return json.dumps(storage_data, indent=2, default=str)

    def _parse_cv_from_storage(self, content: str) -> CompleteCV:
        """Parse CV data from cloud storage"""
        try:
            data = json.loads(content)
            cv_data = data.get("cv_data", data)  # Handle both new and legacy formats
            return CompleteCV.parse_obj(cv_data)
        except Exception as e:
            logger.error(f"CV parsing failed: {e}")
            raise ValueError(f"Invalid CV file format: {e}")

    async def save_cv(
        self,
        session_tokens: Dict[str, Any],
        provider: CloudProvider,
        cv_data: CompleteCV,
        file_name: Optional[str] = None,
    ) -> str:
        """Save CV to specified cloud provider"""

        if provider.value not in session_tokens:
            raise CloudProviderError(f"No access token for {provider.value}")

        access_token = session_tokens[provider.value]["access_token"]

        # Generate filename if not provided
        if not file_name:
            file_name = self._format_cv_filename(cv_data.title)

        # Prepare CV content
        content = self._prepare_cv_for_storage(cv_data)

        # Upload to cloud provider
        try:
            async with get_cloud_provider(provider, access_token) as cloud_provider:
                file_id = await cloud_provider.upload_file(file_name, content)

                logger.info(f"CV saved to {provider.value}: {file_id}")
                return file_id
        except Exception as e:
            logger.error(f"Failed to save CV to {provider.value}: {e}")
            raise CloudProviderError(f"Failed to save CV: {str(e)}")

    async def load_cv(
        self, session_tokens: Dict[str, Any], provider: CloudProvider, file_id: str
    ) -> CompleteCV:
        """Load CV from specified cloud provider"""

        if provider.value not in session_tokens:
            raise CloudProviderError(f"No access token for {provider.value}")

        access_token = session_tokens[provider.value]["access_token"]

        # Download from cloud provider
        try:
            async with get_cloud_provider(provider, access_token) as cloud_provider:
                content = await cloud_provider.download_file(file_id)

                # Parse CV data
                cv_data = self._parse_cv_from_storage(content)

                logger.info(f"CV loaded from {provider.value}: {file_id}")
                return cv_data
        except Exception as e:
            logger.error(f"Failed to load CV from {provider.value}: {e}")
            raise CloudProviderError(f"Failed to load CV: {str(e)}")

    async def list_cvs(
        self, session_tokens: Dict[str, Any], provider: CloudProvider
    ) -> List[CloudFileMetadata]:
        """List all CVs from specified cloud provider"""

        if provider.value not in session_tokens:
            raise CloudProviderError(f"No access token for {provider.value}")

        access_token = session_tokens[provider.value]["access_token"]

        try:
            async with get_cloud_provider(provider, access_token) as cloud_provider:
                files = await cloud_provider.list_files("CVs")

                logger.info(f"Listed {len(files)} CVs from {provider.value}")
                return files
        except Exception as e:
            logger.error(f"Failed to list CVs from {provider.value}: {e}")
            raise CloudProviderError(f"Failed to list CVs: {str(e)}")

    async def delete_cv(
        self, session_tokens: Dict[str, Any], provider: CloudProvider, file_id: str
    ) -> bool:
        """Delete CV from specified cloud provider"""

        if provider.value not in session_tokens:
            raise CloudProviderError(f"No access token for {provider.value}")

        access_token = session_tokens[provider.value]["access_token"]

        try:
            async with get_cloud_provider(provider, access_token) as cloud_provider:
                success = await cloud_provider.delete_file(file_id)

                if success:
                    logger.info(f"CV deleted from {provider.value}: {file_id}")
                else:
                    logger.warning(
                        f"Failed to delete CV from {provider.value}: {file_id}"
                    )

                return success
        except Exception as e:
            logger.error(f"Failed to delete CV from {provider.value}: {e}")
            raise CloudProviderError(f"Failed to delete CV: {str(e)}")

    async def get_connection_status(
        self, session_tokens: Dict[str, Any], provider: CloudProvider
    ) -> CloudConnectionStatus:
        """Get connection status for a cloud provider"""

        if provider.value not in session_tokens:
            return CloudConnectionStatus(provider=provider, connected=False)

        try:
            access_token = session_tokens[provider.value]["access_token"]

            async with get_cloud_provider(provider, access_token) as cloud_provider:
                user_info = await cloud_provider.get_user_info()
                storage_quota = await cloud_provider.get_storage_quota()

                return CloudConnectionStatus(
                    provider=provider,
                    connected=True,
                    email=user_info.get("email"),
                    storage_quota=storage_quota,
                )

        except Exception as e:
            logger.warning(f"Connection check failed for {provider.value}: {e}")
            return CloudConnectionStatus(provider=provider, connected=False)

    async def get_all_connection_statuses(
        self, session_tokens: Dict[str, Any]
    ) -> List[CloudConnectionStatus]:
        """Get connection status for all cloud providers"""

        statuses = []
        for provider in CloudProvider:
            status = await self.get_connection_status(session_tokens, provider)
            statuses.append(status)

        return statuses

    async def search_cvs(
        self,
        session_tokens: Dict[str, Any],
        search_term: str,
        providers: Optional[List[CloudProvider]] = None,
    ) -> List[CloudFileMetadata]:
        """Search for CVs across multiple providers"""

        if providers is None:
            providers = [
                provider
                for provider in CloudProvider
                if provider.value in session_tokens
            ]

        all_files = []

        for provider in providers:
            try:
                files = await self.list_cvs(session_tokens, provider)
                # Filter files by search term
                matching_files = [
                    file for file in files if search_term.lower() in file.name.lower()
                ]
                all_files.extend(matching_files)
            except Exception as e:
                logger.warning(f"Search failed for {provider.value}: {e}")
                continue

        # Sort by last modified date
        all_files.sort(key=lambda x: x.last_modified, reverse=True)

        return all_files

    async def backup_cv(
        self,
        session_tokens: Dict[str, Any],
        source_provider: CloudProvider,
        file_id: str,
        backup_providers: List[CloudProvider],
    ) -> Dict[CloudProvider, str]:
        """Backup CV to multiple cloud providers"""

        # Load CV from source
        cv_data = await self.load_cv(session_tokens, source_provider, file_id)

        # Save to backup providers
        backup_results = {}

        for provider in backup_providers:
            if provider == source_provider:
                continue

            try:
                backup_file_id = await self.save_cv(
                    session_tokens,
                    provider,
                    cv_data,
                    f"backup_{self._format_cv_filename(cv_data.title)}",
                )
                backup_results[provider] = backup_file_id
            except Exception as e:
                logger.error(f"Backup to {provider.value} failed: {e}")
                backup_results[provider] = None

        return backup_results

    async def sync_cv_across_providers(
        self,
        session_tokens: Dict[str, Any],
        cv_data: CompleteCV,
        providers: List[CloudProvider],
    ) -> Dict[CloudProvider, str]:
        """Sync CV across multiple cloud providers"""

        results = {}

        for provider in providers:
            try:
                file_id = await self.save_cv(session_tokens, provider, cv_data)
                results[provider] = file_id
            except Exception as e:
                logger.error(f"Sync to {provider.value} failed: {e}")
                results[provider] = None

        return results

    def validate_cv_data(self, cv_data: Dict[str, Any]) -> bool:
        """Validate CV data structure"""
        try:
            CompleteCV.parse_obj(cv_data)
            return True
        except Exception as e:
            logger.warning(f"CV validation failed: {e}")
            return False

    async def get_provider_health(
        self, provider: CloudProvider, access_token: str
    ) -> Dict[str, Any]:
        """Check health of a specific cloud provider"""
        health_info = {
            "provider": provider.value,
            "status": "unknown",
            "response_time_ms": None,
            "user_info": None,
            "storage_quota": None,
            "error": None,
        }

        try:
            start_time = datetime.utcnow()

            async with get_cloud_provider(provider, access_token) as cloud_provider:
                # Test basic connectivity
                user_info = await cloud_provider.get_user_info()
                storage_quota = await cloud_provider.get_storage_quota()

                end_time = datetime.utcnow()
                response_time = (end_time - start_time).total_seconds() * 1000

                health_info.update(
                    {
                        "status": "healthy",
                        "response_time_ms": round(response_time, 2),
                        "user_info": user_info,
                        "storage_quota": storage_quota,
                    }
                )

        except Exception as e:
            health_info.update({"status": "unhealthy", "error": str(e)})

        return health_info

    async def upload_file(
        self,
        session_tokens: Dict[str, Any],
        provider: CloudProvider,
        file_name: str,
        content: str,
        folder_name: str = "CVs",
    ) -> str:
        """Generic file upload method"""

        if provider.value not in session_tokens:
            raise CloudProviderError(f"No access token for {provider.value}")

        access_token = session_tokens[provider.value]["access_token"]

        try:
            async with get_cloud_provider(provider, access_token) as cloud_provider:
                file_id = await cloud_provider.upload_file(
                    file_name, content, folder_name
                )
                logger.info(f"File uploaded to {provider.value}: {file_id}")
                return file_id
        except Exception as e:
            logger.error(f"Failed to upload file to {provider.value}: {e}")
            raise CloudProviderError(f"Failed to upload file: {str(e)}")

    async def download_file(
        self,
        session_tokens: Dict[str, Any],
        provider: CloudProvider,
        file_id: str,
    ) -> str:
        """Generic file download method"""

        if provider.value not in session_tokens:
            raise CloudProviderError(f"No access token for {provider.value}")

        access_token = session_tokens[provider.value]["access_token"]

        try:
            async with get_cloud_provider(provider, access_token) as cloud_provider:
                content = await cloud_provider.download_file(file_id)
                logger.info(f"File downloaded from {provider.value}: {file_id}")
                return content
        except Exception as e:
            logger.error(f"Failed to download file from {provider.value}: {e}")
            raise CloudProviderError(f"Failed to download file: {str(e)}")

    async def delete_file(
        self,
        session_tokens: Dict[str, Any],
        provider: CloudProvider,
        file_id: str,
    ) -> bool:
        """Generic file deletion method"""

        if provider.value not in session_tokens:
            raise CloudProviderError(f"No access token for {provider.value}")

        access_token = session_tokens[provider.value]["access_token"]

        try:
            async with get_cloud_provider(provider, access_token) as cloud_provider:
                success = await cloud_provider.delete_file(file_id)
                if success:
                    logger.info(f"File deleted from {provider.value}: {file_id}")
                return success
        except Exception as e:
            logger.error(f"Failed to delete file from {provider.value}: {e}")
            raise CloudProviderError(f"Failed to delete file: {str(e)}")

    async def list_files(
        self,
        session_tokens: Dict[str, Any],
        provider: CloudProvider,
        folder_name: str = "CVs",
    ) -> List[CloudFileMetadata]:
        """Generic file listing method"""

        if provider.value not in session_tokens:
            raise CloudProviderError(f"No access token for {provider.value}")

        access_token = session_tokens[provider.value]["access_token"]

        try:
            async with get_cloud_provider(provider, access_token) as cloud_provider:
                files = await cloud_provider.list_files(folder_name)
                logger.info(f"Listed {len(files)} files from {provider.value}")
                return files
        except Exception as e:
            logger.error(f"Failed to list files from {provider.value}: {e}")
            raise CloudProviderError(f"Failed to list files: {str(e)}")


# Global cloud service instance
cloud_service = CloudService()
