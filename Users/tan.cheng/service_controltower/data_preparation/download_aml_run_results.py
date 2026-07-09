"""Download AML snapshot-build results from the configured datastore.

Default behavior:
    Download the most recently submitted job recorded in:

        data_preparation/aml_last_submitted_job.json

Specific historical run:
    Pass a job name at runtime:

        python data_preparation/download_aml_run_results.py --job-name snapshot-build-YYYYMMDD-HHMMSS

    or set this in config.py:

        AML_DOWNLOAD_JOB_NAME = "snapshot-build-YYYYMMDD-HHMMSS"

Results are downloaded to:

    data_preparation/aml_run_results/<job_name>/

Run from service_controltower/ or service_controltower/data_preparation/:

    python data_preparation/download_aml_run_results.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from urllib.parse import unquote

from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential, DefaultAzureCredential, InteractiveBrowserCredential, ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient

import config as run_config


def cfg(name: str, default=None):
    return getattr(run_config, name, default)


def require_value(name: str) -> str:
    value = str(cfg(name, "")).strip()
    if not value:
        raise ValueError(f"Missing required config value: {name}")
    return value




def build_credential():
    """Build the credential used to read AML/datastore results.

    Default is Azure CLI auth to avoid remote-browser timeout. Run:
        az login --use-device-code
    before downloading results, or change AML_AUTH_MODE in config.py.
    """
    auth_mode = str(cfg("AML_AUTH_MODE", "azure_cli")).strip().lower()
    if auth_mode in {"azure_cli", "cli"}:
        print("  auth mode: AzureCliCredential", flush=True)
        return AzureCliCredential()
    if auth_mode in {"managed", "managed_identity"}:
        print("  auth mode: ManagedIdentityCredential", flush=True)
        return ManagedIdentityCredential()
    if auth_mode in {"interactive", "browser"}:
        print("  auth mode: InteractiveBrowserCredential", flush=True)
        return InteractiveBrowserCredential()
    if auth_mode in {"default"}:
        print("  auth mode: DefaultAzureCredential without interactive browser", flush=True)
        return DefaultAzureCredential(exclude_interactive_browser_credential=True)
    raise ValueError("AML_AUTH_MODE must be one of: azure_cli, managed, interactive, default")


def project_root() -> Path:
    return Path(cfg("PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()


def parse_azureml_datastore_uri(uri: str) -> tuple[str, str]:
    pattern = r"azureml://(?:subscriptions/[^/]+/resourcegroups/[^/]+/workspaces/[^/]+/)?datastores/([^/]+)/paths/(.*)"
    match = re.match(pattern, uri.strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError(
            "Expected URI in form azureml://datastores/<datastore_name>/paths/<prefix>/"
        )
    datastore_name = unquote(match.group(1))
    prefix = unquote(match.group(2)).lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return datastore_name, prefix


def normalize_base_uri(uri: str) -> str:
    uri = uri.strip()
    if not uri:
        raise ValueError("Output base URI is empty.")
    if not uri.endswith("/"):
        uri += "/"
    return uri


def output_base_uri() -> str:
    base = str(cfg("AML_OUTPUT_BASE_DATA_URI", "")).strip()
    if base:
        return normalize_base_uri(base)

    old = str(cfg("AML_OUTPUT_DATA_URI", "")).strip()
    if not old:
        raise ValueError("Missing AML_OUTPUT_BASE_DATA_URI or AML_OUTPUT_DATA_URI.")
    old = normalize_base_uri(old)
    if old.rstrip("/").endswith("/latest"):
        return old.rstrip("/").rsplit("/", 1)[0] + "/"
    return old


def join_uri(base_uri: str, *parts: str) -> str:
    out = normalize_base_uri(base_uri)
    clean_parts = [str(p).strip("/") for p in parts if str(p).strip("/")]
    if clean_parts:
        out += "/".join(clean_parts) + "/"
    return out


def last_job_info_path() -> Path:
    root = project_root()
    default_path = root / "data_preparation" / "aml_last_submitted_job.json"
    return Path(cfg("AML_LAST_JOB_INFO_PATH", default_path)).expanduser().resolve()


def load_last_job_info() -> dict:
    path = last_job_info_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No last submitted job info found at {path}. Set AML_DOWNLOAD_JOB_NAME in config.py "
            "or run submit_snapshot_build_aml_job.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download AML snapshot build results from the configured datastore."
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help=(
            "Optional AML job/folder name to download. If omitted, the script uses "
            "AML_DOWNLOAD_JOB_NAME from config.py, then the last submitted job info file."
        ),
    )
    return parser.parse_args()


def resolve_job_name_and_output_uri(cli_job_name: str | None = None) -> tuple[str, str]:
    requested_job_name = (cli_job_name or "").strip()
    if requested_job_name:
        return requested_job_name, join_uri(output_base_uri(), requested_job_name)

    configured_job_name = str(cfg("AML_DOWNLOAD_JOB_NAME", "")).strip()
    if configured_job_name:
        return configured_job_name, join_uri(output_base_uri(), configured_job_name)

    info = load_last_job_info()
    job_name = str(info.get("job_name") or info.get("returned_job_name") or "").strip()
    output_uri = str(info.get("output_uri") or "").strip()
    if not job_name:
        raise ValueError(
            "Last job info does not contain a job_name. Pass --job-name or set "
            "AML_DOWNLOAD_JOB_NAME in config.py."
        )
    if not output_uri:
        output_uri = join_uri(output_base_uri(), job_name)
    return job_name, output_uri


def datastore_account_container(ml_client: MLClient, datastore_name: str) -> tuple[str, str]:
    datastore = ml_client.datastores.get(datastore_name)
    account_name = getattr(datastore, "account_name", None)
    container_name = getattr(datastore, "container_name", None)

    if not account_name and hasattr(datastore, "properties"):
        account_name = getattr(datastore.properties, "account_name", None)
    if not container_name and hasattr(datastore, "properties"):
        container_name = getattr(datastore.properties, "container_name", None)

    if not account_name or not container_name:
        raise ValueError(
            f"Could not resolve account/container for datastore '{datastore_name}'. "
            "Use an Azure Blob datastore for AML_OUTPUT_BASE_DATA_URI."
        )
    return str(account_name), str(container_name)


def main() -> None:
    args = parse_args()
    subscription_id = require_value("AML_SUBSCRIPTION_ID")
    resource_group = require_value("AML_RESOURCE_GROUP")
    workspace_name = require_value("AML_WORKSPACE_NAME")
    job_name, output_uri = resolve_job_name_and_output_uri(args.job_name)

    base_local_dir = Path(
        cfg("AML_RUN_RESULTS_LOCAL_DIR", project_root() / "data_preparation" / "aml_run_results")
    ).expanduser().resolve()
    local_dir = base_local_dir / job_name

    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    datastore_name, prefix = parse_azureml_datastore_uri(output_uri)
    if not prefix.strip("/"):
        raise ValueError("Refusing to download an entire datastore root. Configure a nested output path.")

    credential = build_credential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
    account_name, container_name = datastore_account_container(ml_client, datastore_name)

    service = BlobServiceClient(
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=credential,
    )
    container = service.get_container_client(container_name)

    print("Downloading AML run results from datastore:", flush=True)
    print(f"  job name: {job_name}", flush=True)
    print(f"  output uri: {output_uri}", flush=True)
    print(f"  storage account: {account_name}", flush=True)
    print(f"  container: {container_name}", flush=True)
    print(f"  prefix: {prefix}", flush=True)
    print(f"  local folder: {local_dir}", flush=True)

    count = 0
    total_bytes = 0
    for blob in container.list_blobs(name_starts_with=prefix):
        rel = blob.name[len(prefix):].lstrip("/")
        if not rel:
            continue
        target = local_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            stream = container.download_blob(blob.name)
            f.write(stream.readall())
        count += 1
        total_bytes += int(getattr(blob, "size", 0) or 0)
        if count % 25 == 0:
            print(f"  downloaded {count:,} files...", flush=True)

    print(f"Download complete. Files: {count:,}; bytes: {total_bytes:,}", flush=True)
    print(f"Saved to: {local_dir}", flush=True)

    if count == 0:
        print("WARNING: No result files were found under the configured output prefix.", flush=True)


if __name__ == "__main__":
    main()
