FROM python:3.13-slim

RUN pip install --no-cache-dir fastapi uvicorn torch torchvision pillow gdown

WORKDIR /app
COPY . /app

EXPOSE 8000

ENV MODEL_PATH="faster_best.pth"
ENV MODEL_GDRIVE_ID="19Pi2f3Tfz4kWrM8WCmCKO1eXyhQeX5Zj"
ENV SCORE_THR=0.3

CMD ["uvicorn", "endpoint:app", "--host", "0.0.0.0", "--port", "8000"]
