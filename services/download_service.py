import os
import logging
import requests
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional

logger = logging.getLogger(__name__)

def alpha_fold_link(pdb_accession: str) -> str:
    """
    Generate the AlphaFold URL for a given PDB accession.

    Args:
        pdb_accession (str): The PDB accession
    """
    return f"https://alphafold.ebi.ac.uk/files/AF-{pdb_accession}-F1-model_v6.pdb"
def download_file(file_url: str, downloads_folder: str = "downloads") -> Optional[str]:
    """
    Download a file from a given URL and save it to the downloads folder.

    Args:
        file_url (str): The URL of the file to download
        downloads_folder (str): The folder path where the file will be saved (default: "downloads")

    Returns:
        Optional[str]: The absolute path to the downloaded file, or None if download failed

    Raises:
        ValueError: If the URL is invalid or empty
        requests.RequestException: If the download request fails
    """
    if not file_url:
        raise ValueError("File URL cannot be empty")

    try:
        # Create downloads folder if it doesn't exist
        downloads_path = Path(downloads_folder)
        downloads_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified downloads folder: {downloads_path.absolute()}")

        # Extract filename from URL
        parsed_url = urlparse(file_url)
        filename = os.path.basename(parsed_url.path)

        # If no filename found in URL, generate one
        if not filename or filename == "":
            filename = f"download_{hash(file_url) % 10000}"

        file_path = downloads_path / filename

        logger.info(f"Starting download from: {file_url}")

        # Download the file with timeout and streaming for large files
        response = requests.get(file_url, timeout=30, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        # Write file in chunks
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        file_path_absolute = file_path.absolute()
        logger.info(f"File downloaded successfully: {file_path_absolute}")

        return str(file_path_absolute)

    except requests.RequestException as e:
        logger.error(f"Failed to download file from {file_url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during file download: {e}")
        raise


def download_file_with_custom_name(
    file_url: str, custom_filename: str, downloads_folder: str = "downloads"
) -> Optional[str]:
    """
    Download a file from a given URL with a custom filename.

    Args:
        file_url (str): The URL of the file to download
        custom_filename (str): Custom name for the downloaded file
        downloads_folder (str): The folder path where the file will be saved (default: "downloads")

    Returns:
        Optional[str]: The absolute path to the downloaded file, or None if download failed

    Raises:
        ValueError: If the URL or custom filename is invalid
        requests.RequestException: If the download request fails
    """
    if not file_url:
        raise ValueError("File URL cannot be empty")

    if not custom_filename:
        raise ValueError("Custom filename cannot be empty")

    try:
        # Create downloads folder if it doesn't exist
        downloads_path = Path(downloads_folder)
        downloads_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified downloads folder: {downloads_path.absolute()}")

        file_path = downloads_path / custom_filename

        logger.info(f"Starting download from: {file_url}")

        # Download the file with timeout and streaming for large files
        response = requests.get(file_url, timeout=30, stream=True)
        response.raise_for_status()

        # Write file in chunks
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        file_path_absolute = file_path.absolute()
        logger.info(f"File downloaded successfully: {file_path_absolute}")

        return str(file_path_absolute)

    except requests.RequestException as e:
        logger.error(f"Failed to download file from {file_url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during file download: {e}")
        raise
