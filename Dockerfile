# Use Miniconda as base image (with Python 3.10)
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy necessary files
COPY . /app

# Create Conda environment and install dependencies
RUN conda create -n myenv python=3.10 -y && \
    echo "source activate myenv" > ~/.bashrc && \
    conda install -n myenv -c conda-forge faiss-cpu -y && \
    conda install -n myenv -c pytorch -c nvidia \
    pytorch=2.4 torchvision=0.19 torchaudio pytorch-cuda=11.8 -y && \
    /opt/conda/envs/myenv/bin/pip install flask pillow numpy waitress

# Generate embeddings (run after dependencies are installed)
RUN /opt/conda/envs/myenv/bin/python generate.py

# Expose Flask port
EXPOSE 5000

# Command to run the app
CMD ["/opt/conda/envs/myenv/bin/python", "app.py"]