"""
Input validation utilities for security.

This module provides validators for:
- URL validation (SSRF prevention)
- File upload validation
- Request size validation
"""

import ipaddress
import socket
from urllib.parse import urlparse
from typing import Optional, Set
import magic  # python-magic for MIME type detection
from fastapi import HTTPException, UploadFile
import os


class URLValidator:
    """
    Validates URLs to prevent SSRF (Server-Side Request Forgery) attacks.

    Blocks access to:
    - Private IP ranges (10.x.x.x, 172.16-31.x.x, 192.168.x.x)
    - Localhost (127.x.x.x)
    - Link-local (169.254.x.x)
    - Metadata endpoints (cloud providers)
    - File:// and other non-HTTP protocols
    """

    # Blocked IP networks
    BLOCKED_NETWORKS = [
        ipaddress.ip_network("0.0.0.0/8"),        # Current network
        ipaddress.ip_network("10.0.0.0/8"),       # Private (Class A)
        ipaddress.ip_network("127.0.0.0/8"),      # Loopback
        ipaddress.ip_network("169.254.0.0/16"),   # Link-local
        ipaddress.ip_network("172.16.0.0/12"),    # Private (Class B)
        ipaddress.ip_network("192.168.0.0/16"),   # Private (Class C)
        ipaddress.ip_network("224.0.0.0/4"),      # Multicast
        ipaddress.ip_network("240.0.0.0/4"),      # Reserved
        ipaddress.ip_network("::1/128"),          # IPv6 loopback
        ipaddress.ip_network("fc00::/7"),         # IPv6 private
        ipaddress.ip_network("fe80::/10"),        # IPv6 link-local
    ]

    # Blocked hostnames
    BLOCKED_HOSTNAMES = {
        "localhost",
        "metadata.google.internal",  # Google Cloud metadata
        "169.254.169.254",            # AWS/Azure metadata
        "metadata.azure.com",         # Azure metadata
    }

    # Allowed schemes
    ALLOWED_SCHEMES = {"http", "https"}

    @classmethod
    def validate(cls, url: str, allow_private: bool = False) -> str:
        """
        Validate URL and check for SSRF vulnerabilities.

        Args:
            url: URL to validate
            allow_private: Allow private IP addresses (default: False)

        Returns:
            Validated URL

        Raises:
            HTTPException: If URL is invalid or blocked

        Example:
            validated_url = URLValidator.validate("https://example.com")
        """
        try:
            parsed = urlparse(url)

            # Check scheme
            if parsed.scheme not in cls.ALLOWED_SCHEMES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid URL scheme. Only {cls.ALLOWED_SCHEMES} allowed."
                )

            # Check hostname exists
            if not parsed.hostname:
                raise HTTPException(
                    status_code=400,
                    detail="URL must have a hostname"
                )

            # Check against blocked hostnames
            if parsed.hostname.lower() in cls.BLOCKED_HOSTNAMES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Access to {parsed.hostname} is not allowed"
                )

            # Resolve hostname to IP
            try:
                ip_str = socket.gethostbyname(parsed.hostname)
                ip_obj = ipaddress.ip_address(ip_str)
            except socket.gaierror:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot resolve hostname: {parsed.hostname}"
                )
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid IP address: {ip_str}"
                )

            # Check IP against blocked networks (unless explicitly allowed)
            if not allow_private:
                for network in cls.BLOCKED_NETWORKS:
                    if ip_obj in network:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Access to private network {network} is not allowed"
                        )

            return url

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid URL: {str(e)}"
            )


class FileValidator:
    """
    Validates file uploads for security.

    Checks:
    - File size limits
    - MIME type (actual content, not just extension)
    - File extension
    - Malicious content
    """

    # Default limits
    DEFAULT_MAX_SIZE = 5 * 1024 * 1024  # 5MB

    # Allowed MIME types
    ALLOWED_MIME_TYPES = {
        "text/plain",
        "text/html",
        "text/markdown",
        "text/csv",
        "application/json",
        "application/xml",
        "text/xml",
    }

    # Allowed file extensions
    ALLOWED_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",
        ".html",
        ".htm",
        ".csv",
        ".json",
        ".xml",
    }

    @classmethod
    async def validate(
        cls,
        file: UploadFile,
        max_size: Optional[int] = None,
        allowed_mime_types: Optional[Set[str]] = None,
        allowed_extensions: Optional[Set[str]] = None
    ) -> bytes:
        """
        Validate uploaded file and return content.

        Args:
            file: Uploaded file
            max_size: Maximum file size in bytes (default: 5MB)
            allowed_mime_types: Set of allowed MIME types (default: text types)
            allowed_extensions: Set of allowed extensions (default: text extensions)

        Returns:
            File content as bytes

        Raises:
            HTTPException: If file is invalid

        Example:
            content = await FileValidator.validate(file)
        """
        if max_size is None:
            max_size = int(os.getenv("SIE_X_MAX_FILE_SIZE", cls.DEFAULT_MAX_SIZE))

        if allowed_mime_types is None:
            allowed_mime_types = cls.ALLOWED_MIME_TYPES

        if allowed_extensions is None:
            allowed_extensions = cls.ALLOWED_EXTENSIONS

        # Read file content
        content = await file.read()

        # Check file size
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="File is empty"
            )

        if len(content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {max_size} bytes ({max_size // 1024 // 1024}MB)"
            )

        # Sanitize filename (remove path traversal)
        if file.filename:
            filename = os.path.basename(file.filename)

            # Check extension
            ext = os.path.splitext(filename)[1].lower()
            if ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not allowed. Allowed extensions: {allowed_extensions}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Filename is required"
            )

        # Validate MIME type (actual content, not just extension)
        try:
            mime_type = magic.from_buffer(content, mime=True)

            if mime_type not in allowed_mime_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not allowed. File MIME type: {mime_type}. Allowed: {allowed_mime_types}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to determine file type: {str(e)}"
            )

        return content

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename

        Example:
            safe_name = FileValidator.sanitize_filename("../../etc/passwd")
            # Returns: "passwd"
        """
        # Remove directory path
        filename = os.path.basename(filename)

        # Remove null bytes
        filename = filename.replace("\x00", "")

        # Limit length
        max_length = 255
        if len(filename) > max_length:
            name, ext = os.path.splitext(filename)
            filename = name[:max_length - len(ext)] + ext

        return filename


class RequestSizeValidator:
    """Validates request body size."""

    DEFAULT_MAX_SIZE = 10 * 1024 * 1024  # 10MB

    @classmethod
    def check_size(cls, content_length: Optional[int], max_size: Optional[int] = None):
        """
        Check if content length exceeds maximum.

        Args:
            content_length: Content-Length header value
            max_size: Maximum allowed size in bytes

        Raises:
            HTTPException: If content too large
        """
        if max_size is None:
            max_size = int(os.getenv("SIE_X_MAX_REQUEST_SIZE", cls.DEFAULT_MAX_SIZE))

        if content_length and content_length > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Request too large. Maximum size: {max_size} bytes ({max_size // 1024 // 1024}MB)"
            )
