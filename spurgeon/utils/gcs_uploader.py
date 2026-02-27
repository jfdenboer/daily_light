# gcs_uploader.py

import logging
import re
from pathlib import Path
from typing import Union, Optional

from google.cloud import storage
from google.api_core import retry
from google.api_core.exceptions import GoogleAPIError

from spurgeon.config.settings import Settings
from spurgeon.utils.retry_utils import retry_with_backoff

logger = logging.getLogger(__name__)

class GCSUploader:
    """
    Uploads local files to Google Cloud Storage (GCS).
    Features retry on transient failures and blob URL return.
    """

    def __init__(self, settings: Settings, use_global_retry: bool = True) -> None:
        creds = settings.gcs_credentials_path
        bucket_name = settings.gcs_bucket_name

        if not creds or not creds.exists():
            raise ValueError(f"GCS credentials path invalid: {creds}")
        if not bucket_name:
            raise ValueError("GCS bucket name must be provided")

        try:
            self.client = storage.Client.from_service_account_json(str(creds))
            self.bucket = self.client.bucket(bucket_name)
            if not self.bucket.exists():
                raise RuntimeError(f"GCS bucket '{bucket_name}' does not exist or is not accessible")
        except GoogleAPIError as e:
            logger.exception("Failed to initialize GCS client for bucket '%s'", bucket_name)
            raise RuntimeError("Could not initialize GCS client") from e

        self.bucket_name = bucket_name
        self.use_global_retry = use_global_retry
        logger.info("GCSUploader ready for bucket: %s", bucket_name)

    def _sanitize_blob_name(self, name: str) -> str:
        safe = name.lstrip("/")
        safe = re.sub(r"\.\.+", "", safe)
        return re.sub(r"[^A-Za-z0-9\-_/\.]", "_", safe)

    def upload_file(
        self,
        local_path: Union[Path, str],
        destination_blob_name: str,
        make_public: bool = False,
        signed_url_expiration: Optional[int] = None,
    ) -> str:
        lp = Path(local_path)
        if not lp.is_file():
            raise FileNotFoundError(f"Local file not found: {lp}")

        blob_name = self._sanitize_blob_name(destination_blob_name)
        blob = self.bucket.blob(blob_name)

        def do_upload():
            if self.use_global_retry:
                blob.upload_from_filename(str(lp))
            else:
                policy = retry.Retry(
                    predicate=retry.if_transient_error,
                    initial=1.0,
                    maximum=60.0,
                    multiplier=2.0,
                )
                blob.upload_from_filename(str(lp), retry=policy)

            logger.info("Uploaded %s → gs://%s/%s", lp, self.bucket_name, blob_name)

            if make_public:
                blob.make_public()
                return blob.public_url
            if signed_url_expiration:
                return blob.generate_signed_url(signed_url_expiration)
            return blob.public_url

        return retry_with_backoff(
            func=do_upload,
            max_retries=3,
            backoff=2.0,
            error_types=(GoogleAPIError,),
            context=f"GCS upload to {blob_name}",
        )
