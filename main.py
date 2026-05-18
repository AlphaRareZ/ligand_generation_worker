import json
import logging
from random import randint, seed as random_seed
from time import time
from rabbit_mq.rabbit_mq_consumer import create_consumer
from services.delete_excess_ligands_service import cleanup_output_folder
from services.download_service import alpha_fold_link, download_file
from rabbit_mq.rabbit_mq_producer import create_producer
from pipelines import run_pipeline
from services.s3_upload_service import process_and_upload_analysis
from services.clear_service import clear_all_folders
import gc
from os import makedirs

# Configure random seed
random_seed(time())

# Configure logging
makedirs("Logs", exist_ok=True)

# 2. نظبط إعدادات الـ Logging عشان تكتب في الفايل والـ Console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # شكل الرسالة بالوقت
    handlers=[
        logging.FileHandler(
            "Logs/worker_log.txt", mode="a"
        ),  # بيكتب في الفايل (a يعني append)
        logging.StreamHandler(),  # بيكتب في الـ Console
    ],
)
logger = logging.getLogger(__name__)

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)


class MessageModel:

    def __init__(self, data: dict):
        self.PDBAccession = data.get("PDBAccession", "")
        self.ProteinId = data.get("ProteinId", "")
        # self.Username = data.get("Username", "")
        # self.Email = data.get("Email", "")

    def is_valid(self) -> bool:
        return all(
            [
                isinstance(self.PDBAccession, str) and self.PDBAccession.strip() != "",
                isinstance(self.ProteinId, int),
                # isinstance(self.Username, str) and self.Username.strip() != "",
                # isinstance(self.Email, str) and self.Email.strip() != "",
            ]
        )


def process_request(message_data):
    """
    Process the consumed message.

    Args:
        message_data (dict): The jsonified message data
    """
    message_model = MessageModel(message_data)

    # if not message_model.is_valid():
    #     logger.error(f"Invalid message data: {message_data}")
    #     return

    logger.info(f"Processing request: {message_data}")
    logger.info(f"Invoking the Download Service for {message_model.PDBAccession}")

    # Create message producer instance
    producer = create_producer(config)

    try:
        if "PDBAccession" in message_data:
            # Download Files
            pdb_path = download_file(alpha_fold_link(message_model.PDBAccession))
            print(f"✓ Downloaded PDB file to: {pdb_path}")
            # Invoke Pipeline
            run_pipeline.main(
                pdb_path=pdb_path,
                #   pop_size=100,
                # generations=50,
                seed=randint(1, 54673),
            )
            # Delete Excess PDB and SDF
            cleanup_output_folder()
            # Upload Files to Cloudflare R2 and retrieve urls
            response_data = process_and_upload_analysis(
                protein_id=message_model.ProteinId,
                accession=message_model.PDBAccession,
            )
            response_data["message"] = "COMPLETED GENERATING LIGANDS"
            # response_data["username"] = message_model.Username
            # response_data["email"] = message_model.Email
            # Print the formatted JSON output

            print("\nFinal Response Message:")
            print(json.dumps(response_data, indent=4))
            # Publish response message to response_queue
            correlation_id = message_model.ProteinId
            success = producer.publish_message(
                response_data, correlation_id=str(correlation_id)
            )

            if success:
                logger.info(
                    f"Response published successfully for Analysis ID: {correlation_id}"
                )
            else:
                logger.error(
                    f"Failed to publish response for Analysis ID: {correlation_id}"
                )
            clear_all_folders()
        else:
            error_response = {
                "protein_id": message_model.ProteinId,
                "success": False,
                "status": "error",
                "message": "Missing required files: PDBAccession",
            }
            producer.publish_message(
                error_response, correlation_id=message_model.ProteinId
            )
            logger.error(f"Missing required file URLs in request: {message_data}")

    except Exception as e:
        # Send error response to response_queue
        error_response = {
            "protein_id": message_model.ProteinId,
            "success": False,
            "status": "error",
            "message": str(e),
        }
        producer.publish_message(
            error_response, correlation_id=message_data.get("ProteinId")
        )
        logger.error(f"Error processing request: {e}")

    finally:
        producer.close()
        # <-- 2. Add this right here!
        # Force a full garbage collection sweep after every single RabbitMQ message
        # is fully processed or fails. This guarantees a clean slate for the next job.
        gc.collect()
        logger.info("Garbage collection triggered at end of worker cycle.")


if __name__ == "__main__":
    # Create consumer instance
    consumer = create_consumer(config)

    # Start consuming messages from request_queue and jsonify them
    try:
        consumer.consume(callback=process_request)
    except Exception as e:
        logger.error(f"Consumer error: {e}")
    finally:
        consumer.close()
