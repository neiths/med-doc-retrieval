"""Hugging Face Storage Bucket client supporting both 'hf sync' CLI and S3 / boto3 API."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import boto3
    from botocore.config import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class HFBucketClient:
    """Manages files in Hugging Face Storage Buckets (e.g. hf://buckets/nieths/ViBioMIR)."""

    def __init__(
        self,
        namespace: str = "nieths",
        bucket_name: str = "ViBioMIR",
        endpoint_url: str | None = None,
        region_name: str = "us-east-1",
    ):
        self.namespace = namespace
        self.bucket_name = bucket_name
        self.bucket_uri = f"hf://buckets/{namespace}/{bucket_name}"
        self.endpoint_url = endpoint_url or f"https://s3.hf.co/{namespace}"
        self.region_name = region_name
        self._s3_client = None

    @property
    def s3_client(self):
        """Lazy initialization of S3 client using environment variables."""
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is not installed. Install it via 'uv add boto3' or 'pip install boto3'.")

        if self._s3_client is None:
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                config=Config(
                    region_name=self.region_name,
                    s3={"addressing_style": "path"},
                    request_checksum_calculation="when_required",
                    response_checksum_validation="when_required",
                ),
            )
        return self._s3_client

    def sync_upload_cli(self, local_dir: str | Path, remote_subpath: str = "") -> bool:
        """Uploads local folder to HF bucket using 'hf sync' CLI tool."""
        hf_bin = shutil.which("hf")
        if not hf_bin:
            logger.warning("'hf' CLI is not found in PATH. Falling back to S3 API upload.")
            return self.upload_folder_s3(local_dir, remote_subpath)

        target_uri = f"{self.bucket_uri}/{remote_subpath}".rstrip("/")
        cmd = [hf_bin, "sync", str(local_dir), target_uri]
        logger.info(f"Running HF CLI Sync: {' '.join(cmd)}")
        res = subprocess.run(cmd)
        return res.returncode == 0

    def sync_download_cli(self, local_dir: str | Path, remote_subpath: str = "") -> bool:
        """Downloads from HF bucket to local folder using 'hf sync' CLI tool."""
        hf_bin = shutil.which("hf")
        if not hf_bin:
            logger.warning("'hf' CLI is not found in PATH. Falling back to S3 API download.")
            return self.download_folder_s3(remote_subpath, local_dir)

        source_uri = f"{self.bucket_uri}/{remote_subpath}".rstrip("/")
        cmd = [hf_bin, "sync", source_uri, str(local_dir)]
        logger.info(f"Running HF CLI Sync: {' '.join(cmd)}")
        res = subprocess.run(cmd)
        return res.returncode == 0

    def upload_file_s3(self, local_file: str | Path, remote_key: str):
        """Uploads a single file using boto3 S3 API."""
        local_path = Path(local_file)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_file}")

        logger.info(f"Uploading {local_path} -> s3://{self.bucket_name}/{remote_key}")
        self.s3_client.upload_file(str(local_path), self.bucket_name, remote_key)
        logger.info(f"Uploaded successfully to s3://{self.bucket_name}/{remote_key}")

    def download_file_s3(self, remote_key: str, local_file: str | Path):
        """Downloads a single file using boto3 S3 API."""
        local_path = Path(local_file)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading s3://{self.bucket_name}/{remote_key} -> {local_path}")
        self.s3_client.download_file(self.bucket_name, remote_key, str(local_path))
        logger.info(f"Downloaded successfully: {local_path}")

    def list_files_s3(self, prefix: str = "") -> list[str]:
        """Lists file keys in bucket under given prefix."""
        paginator = self.s3_client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def upload_folder_s3(self, local_dir: str | Path, remote_subpath: str = "") -> bool:
        """Uploads all files in a folder recursively via boto3."""
        base_dir = Path(local_dir)
        if not base_dir.exists():
            raise FileNotFoundError(f"Directory not found: {local_dir}")

        for fpath in base_dir.rglob("*"):
            if fpath.is_file():
                rel_path = fpath.relative_to(base_dir).as_posix()
                key = f"{remote_subpath}/{rel_path}".lstrip("/") if remote_subpath else rel_path
                self.upload_file_s3(fpath, key)
        return True

    def download_folder_s3(self, remote_subpath: str, local_dir: str | Path) -> bool:
        """Downloads all files under a prefix via boto3."""
        keys = self.list_files_s3(prefix=remote_subpath)
        base_out = Path(local_dir)
        for k in keys:
            rel = k[len(remote_subpath) :].lstrip("/") if remote_subpath else k
            dest = base_out / rel
            self.download_file_s3(k, dest)
        return True
