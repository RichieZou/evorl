# Use the official base image matching your CUDA version
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install basic tools and Miniconda
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget git vim && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# Create the working directory
WORKDIR /app

# First, copy only the environment file to leverage Docker's cache
COPY environment.yml .

# Create the Conda environment from the file
RUN conda env create -f environment.yml

# Now, copy all of your project source code into the image
COPY . .

# --- ADD THIS NEW LINE ---
# Install the 'evorl' project itself from the local source code
RUN conda run -n evorl pip install .
# -------------------------

# Set the default command for the container
CMD ["/bin/bash", "-c", "source activate evorl && echo 'Conda env evorl activated. Welcome!' && /bin/bash"]