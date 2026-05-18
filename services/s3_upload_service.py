import os
import json
import boto3
import urllib.parse  # Required to safely encode URLs
from botocore.config import Config

# Load config
with open("config.json", "r") as f:
    config = json.load(f)
x = os.getenv("")
ACCOUNT_ID = config["CFR2"]["ACCOUNT_ID"]
ACCESS_KEY = config["CFR2"]["ACCESS_KEY"]
SECRET_KEY = config["CFR2"]["SECRET_KEY"]
BUCKET_NAME = config["CFR2"]["BUCKET_NAME"]
# Ensure no trailing slash on the public domain for clean URL building
PUBLIC_DOMAIN = (
    config["CFR2"]["PUBLIC_DOMAIN"].rstrip("/")
    if "PUBLIC_DOMAIN" in config["CFR2"]
    else None
)

# Initialize S3 Client
s3 = boto3.client(
    service_name="s3",
    endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name="auto",
    config=Config(signature_version="s3v4"),
)


def process_and_upload_analysis(protein_id, accession):
    """
    Navigates through the local 'Output' folder and uploads every file
    directly into the Analysis ID folder in S3, flattening the structure.
    Returns their clean public URLs using the file name as the dictionary key.
    """
    local_folder = "Output"

    if not os.path.exists(local_folder):
        print(f"Error: Local folder '{local_folder}' does not exist.")
        return {
            "success": False,
            "protein_id": protein_id,
            "links": {},
            "error": "Output folder not found",
        }

    if not PUBLIC_DOMAIN:
        print("Error: PUBLIC_DOMAIN is not set in config.json")
        return {"success": False, "error": "Missing PUBLIC_DOMAIN config"}

    result_links = {}
    uploaded_count = 0
    failed_count = 0

    # 1. Navigate through the Output folder (including subdirectories)
    for root, _, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)

            # 2. Upload straight into the ProteinId folder (FLATTENED)
            object_name = f"uploads/{accession}/{file}"

            try:
                # Upload the file
                s3.upload_file(local_path, BUCKET_NAME, object_name)
                print(f"✓ Uploaded: {object_name}")
                uploaded_count += 1

                # 3. Create the clean public URL
                # quote() handles spaces and special characters safely
                safe_object_name = urllib.parse.quote(object_name)
                url = f"{PUBLIC_DOMAIN}/{safe_object_name}"

                # Add the URL to our response dictionary using the filename as the key
                result_links[file] = url

            except Exception as e:
                print(f"✗ Failed to upload {local_path}: {str(e)}")
                failed_count += 1
                result_links[file] = f"Error: {str(e)}"

    # Build the final response dictionary
    return {
        "success": failed_count == 0 and uploaded_count > 0,
        "status": "completed" if failed_count == 0 else "partial_failure",
        "protein_id": protein_id,
        "files_uploaded": uploaded_count,
        "files_failed": failed_count,
        "download_links": result_links,
    }
