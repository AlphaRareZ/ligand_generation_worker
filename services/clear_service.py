import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def clear_logs_folder(logs_folder: str = "Logs") -> bool:
    """
    Remove the logs folder permanently.

    Args:
        logs_folder (str): Path to the logs folder (default: "Logs")

    Returns:
        bool: True if successfully cleared, False if folder doesn't exist or error occurred
    """
    try:
        logs_path = Path(logs_folder)
        if logs_path.exists():
            shutil.rmtree(logs_path)
            logger.info(f"Logs folder removed: {logs_path.absolute()}")
            return True
        else:
            logger.warning(f"Logs folder does not exist: {logs_path.absolute()}")
            return False
    except Exception as e:
        logger.error(f"Failed to remove logs folder: {e}")
        return False


def clear_downloads_folder(downloads_folder: str = "downloads") -> bool:
    """
    Remove the downloads folder permanently.

    Args:
        downloads_folder (str): Path to the downloads folder (default: "downloads")

    Returns:
        bool: True if successfully cleared, False if folder doesn't exist or error occurred
    """
    try:
        downloads_path = Path(downloads_folder)
        if downloads_path.exists():
            shutil.rmtree(downloads_path)
            logger.info(f"Downloads folder removed: {downloads_path.absolute()}")
            return True
        else:
            logger.warning(
                f"Downloads folder does not exist: {downloads_path.absolute()}"
            )
            return False
    except Exception as e:
        logger.error(f"Failed to remove downloads folder: {e}")
        return False


def clear_output_folder(output_folder: str = "Output") -> bool:
    """
    Remove the output folder permanently.

    Args:
        output_folder (str): Path to the output folder (default: "Output")

    Returns:
        bool: True if successfully cleared, False if folder doesn't exist or error occurred
    """
    try:
        output_path = Path(output_folder)
        if output_path.exists():
            shutil.rmtree(output_path)
            logger.info(f"Output folder removed: {output_path.absolute()}")
            return True
        else:
            logger.warning(f"Output folder does not exist: {output_path.absolute()}")
            return False
    except Exception as e:
        logger.error(f"Failed to remove output folder: {e}")
        return False


def clear_all_folders(
    logs_folder: str = "Logs",
    downloads_folder: str = "downloads",
    output_folder: str = "Output",
) -> dict:
    """
    Remove all folders (logs, downloads, and output) permanently.

    Args:
        logs_folder (str): Path to the logs folder (default: "Logs")
        downloads_folder (str): Path to the downloads folder (default: "downloads")
        output_folder (str): Path to the output folder (default: "Output")

    Returns:
        dict: A dictionary with the status of each folder removal
    """
    results = {
        "logs": clear_logs_folder(logs_folder),
        "downloads": clear_downloads_folder(downloads_folder),
        "output": clear_output_folder(output_folder),
    }

    logger.info(f"Clear operation completed. Results: {results}")
    return results
