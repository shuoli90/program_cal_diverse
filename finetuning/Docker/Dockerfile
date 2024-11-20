# FROM docker.io/pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y wget git
RUN pip install tqdm wandb lightning fire rotary_embedding_torch pandas numpy matplotlib transformers datasets peft text_generation ninja packaging
RUN MAX_JOBS=48 pip install flash-attn --no-build-isolation
RUN pip install accelerate bitsandbytes
RUN pip install scipy
RUN pip install sentencepiece

# ENV WANDB_API_KEY=yourkeyhere
WORKDIR /workspace