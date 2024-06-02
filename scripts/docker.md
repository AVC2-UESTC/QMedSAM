# Docker

## 1.prepare files
### Dockerfile
```bash
FROM python:3.9-slim
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
# required by cv2
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN mkdir -p /opt/app /inputs /outputs \
    && chown user:user /opt/app /inputs /outputs
USER user
WORKDIR /opt/app
ENV PATH="/home/user/.local/bin:${PATH}"
RUN python -m pip install --user -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple && python -m pip install --user pip-tools -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY --chown=user:user . .
RUN pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```
basicly same as LiteMedSam 
In case it gets stuck,changed to the tsinghua source.

### setup.py
```bash
from setuptools import find_packages, setup

setup(
    name="uestcsd",
    version="0.0.1",
    author="",
    python_requires=">=3.9",
    install_requires=["tqdm","opencv-python","openvino==2023.3"],
    packages=find_packages(exclude="notebooks"),
    extras_require={
    },
)
```
### setup.py
```bash
python3 validate.py -i /workspace/inputs -o /workspace/outputs
```

## 2.Build Docker 

```bash
docker build -f Dockerfile -t uestcsd .
```
> Note: don't forget the `.` in the end

Run the docker on the testing demo images

```bash
docker container run -m 8G --name uestcsd --rm -v {inputpath}:/workspace/inputs/ -v {outputpath}:/workspace/outputs/ uestcsd:latest /bin/bash -c "sh predict.sh"
```
please replace {inputpath} and {outputpath} to your path

> Note: please run `chmod -R 777 ./*` if you run into `Permission denied` error.

Save docker 

```bash
docker save uestcsd | gzip -c > uestcsd.tar.gz
```

## verify
remove images
```bash
docker rmi uestcsd -f
```
Run the commands from the organizer
```bash
docker load -i uestcsd.tar.gz
docker container run -m 8G --name uestcsd --rm -v {inputpath}:/workspace/inputs/ -v {inputpath}:/workspace/outputs/ uestcsd:latest /bin/bash -c "sh predict.sh"
```
please replace {inputpath} and {outputpath} to your path
