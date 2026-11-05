"""Repository-local .env configuration, shared by the cloud commands."""

import os
from pathlib import Path


def cloud_settings():
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    values = {key: os.environ.get(key, "").strip() for key in ("ACCESS_KEY", "SECRET_KEY", "BUCKET_NAME")}
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise ValueError(f"Set {', '.join(missing)} in the repository .env or environment.")
    from helper.snapshot_s3 import create_s3_client

    return create_s3_client(values["ACCESS_KEY"], values["SECRET_KEY"]), values["BUCKET_NAME"]
