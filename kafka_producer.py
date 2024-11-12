from kafka import KafkaProducer
import csv
import json
import logging
import sys
import signal

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Kafka producer
producer = KafkaProducer(bootstrap_servers='localhost:9092')

# Define the batch size and maximum batch count
batch_size = 500000  # 500k rows per batch
max_batches = 3  # Stop after 3 batches
batch = []
batch_counter = 0  # Initialize batch counter

def signal_handler(sig, frame):
    logging.info('Gracefully shutting down...')
    # Send any remaining messages in the batch
    for message in batch:
        producer.send('anime_topic_6', message)
    producer.flush()  # Ensure all messages are sent
    producer.close()
    sys.exit(0)

# Register signal handler for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)

# Open the CSV file and read its contents
with open('final_animedataset.csv', 'r') as file:
    reader = csv.DictReader(file, delimiter=',')
    
    # Strip whitespace from fieldnames
    reader.fieldnames = [name.strip() for name in reader.fieldnames]
    logging.info("Detected column names: %s", reader.fieldnames)

    for row in reader:
        try:
            # Convert the entire row into a JSON object
            data = {key: row[key] for key in row}  # This will include all columns
            # Append the JSON-encoded message to the batch
            batch.append(json.dumps(data).encode('utf-8'))
            
            # Check if the batch size has been reached
            if len(batch) >= batch_size:
                # Send all messages in the current batch
                for message in batch:
                    producer.send('anime_topic_6', message)
                logging.info("Sent batch %d with %d messages", batch_counter + 1, batch_size)
                batch.clear()  # Clear the batch after sending
                batch_counter += 1  # Increment the batch counter

                # Stop after sending the maximum number of batches
                if batch_counter >= max_batches:
                    logging.info('Reached the limit of %d batches. Exiting...', max_batches)
                    break

        except KeyError as e:
            logging.error("Column not found: %s", e)
            logging.error("Row causing the error: %s", row)
            continue  # Skip rows that cause errors

    # Send any remaining messages in the batch (if any)
    for message in batch:
        producer.send('anime_topic_6', message)

# Ensure to flush and close the producer
producer.flush()  # Ensure all messages are sent
producer.close()
