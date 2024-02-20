# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the necessary files or directories into the container
# (Only if you have other files like requirements.txt or other Python scripts)

# Install any needed packages specified in requirements.txt
# (Uncomment the next two lines if you have a requirements.txt)
COPY container_requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
