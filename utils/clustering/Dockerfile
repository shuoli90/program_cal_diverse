# Use an official Anaconda image as a parent image
FROM mcr.microsoft.com/devcontainers/anaconda

# Set the working directory in the container
WORKDIR /usr/src/app



COPY container_requirements.txt ./requirements.txt
# Create a new Python environment named py311 with Python 3.11
RUN conda create -n py311 python=3.11 -y

SHELL ["conda", "run", "-n", "py311", "/bin/bash", "-c"]

RUN pip install --no-cache-dir -r requirements.txt

# Set the default command to execute
CMD ["/bin/bash"]

# Configure the entry point to activate the conda environment
ENTRYPOINT ["conda", "run", "-n", "py311", "--no-capture-output"]