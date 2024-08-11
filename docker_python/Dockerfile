FROM python:3.11-slim
USER root
WORKDIR /

RUN python -m pip install tqdm opencv-python-headless openvino
RUN python -m pip cache purge
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN mkdir -p /workspace && chown user:user /workspace

USER user
WORKDIR /workspace
COPY --chown=user:user predict.sh predict.sh
COPY --chown=user:user encoder.xml encoder.xml
COPY --chown=user:user encoder.bin encoder.bin
COPY --chown=user:user decoder.xml decoder.xml
COPY --chown=user:user decoder.bin decoder.bin
COPY --chown=user:user validate.py validate.py
