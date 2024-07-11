#!/bin/bash

# Function to follow logs of a container
follow_logs() {
    container_id=$1
    echo "Starting to follow logs for container $container_id..."
    docker logs -f $container_id
    echo "Container $container_id has stopped."
}

# Infinite loop to monitor containers on port 9999
while true; do
    echo "Checking for active containers on port 9999..."
    # Get the container ID that is using port 9999
    container_id=$(docker ps --filter "publish=9999" --format "{{.ID}}")

    # Follow logs of the container if it exists
    if [ -z "$container_id" ]; then
        echo "No container found on port 9999."
    else
        follow_logs $container_id
    fi

    # Wait a bit before checking again
    sleep 10
done
