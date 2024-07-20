# model=intfloat/e5-mistral-7b-instruct
model=neulab/codebert-python
volume=$PWD/data

docker run --gpus '"device=0"' -p 8877:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model --pooling=mean