# Stage 1. Build the uestcsd_inference exec
FROM ubuntu:20.04 AS build
USER root
WORKDIR /

ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker
RUN echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker

RUN apt update && apt install -y ca-certificates git wget gnupg build-essential ninja-build cmake 

RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
RUN echo "deb https://apt.repos.intel.com/openvino/2024 ubuntu20 main" | tee /etc/apt/sources.list.d/intel-openvino-2024.list
RUN apt update && apt install -y openvino-2024.2.0

RUN mkdir workspace
WORKDIR /workspace

RUN git clone https://github.com/opencv/opencv.git --branch 4.10.0 --single-branch --recursive
RUN git clone https://github.com/xtensor-stack/xtl.git --branch 0.7.7 --single-branch 
RUN git clone https://github.com/xtensor-stack/xsimd.git --branch 13.0.0 --single-branch 
RUN git clone https://github.com/xtensor-stack/xtensor.git --branch 0.25.0 --single-branch 
RUN git clone https://github.com/xtensor-stack/xtensor-io.git --branch 0.13.0 --single-branch 
RUN git clone https://github.com/zlib-ng/zlib-ng.git --branch 2.2.1 --single-branch 
RUN git clone https://github.com/p-ranav/argparse.git --branch v3.1 --single-branch 

WORKDIR /workspace/opencv
RUN cmake -S . -B build -GNinja \
-DCMAKE_BUILD_TYPE=Release \
-DBUILD_TESTS=OFF \
-DBUILD_PERF_TESTS=OFF \
-DBUILD_EXAMPLES=OFF \
-DBUILD_opencv_apps=OFF \
-DBUILD_LIST="core,imgproc" \
-DCPU_BASELINE=AVX2 \
-DCPU_DISPATCH="" \
-DENABLE_LTO=ON \
-DWITH_OPENCL=OFF \
-DWITH_IMGCODEC_HDR=OFF \
-DWITH_IMGCODEC_SUNRASTER=OFF \
-DWITH_IMGCODEC_PXM=OFF \
-DWITH_IMGCODEC_PFM=OFF \
-DWITH_PNG=OFF \
-DWITH_JPEG=OFF \
-DWITH_TIFF=OFF \
-DWITH_WEBP=OFF \
-DWITH_OPENJPEG=OFF \
-DWITH_JASPER=OFF \
-DWITH_OPENEXR=OFF \
-DWITH_V4L=OFF
RUN ninja -C build
RUN ninja -C build install
WORKDIR /workspace

COPY main.cpp main.cpp
COPY CMakeLists.txt CMakeLists.txt
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -GNinja
RUN cmake --build build --parallel
# RUN ldd build/uestcsd_inference

# Stage 2. Create a slim inference docker
FROM openvino/ubuntu20_runtime:2024.2.0 AS inference
USER root
WORKDIR /

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y --no-install-recommends libtbb2

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN mkdir -p /QMedSAM
RUN chown user:user /QMedSAM

USER user
WORKDIR /QMedSAM

COPY --chown=user:user --from=build /workspace/build/uestcsd_inference uestcsd_inference
COPY --chown=user:user --from=build /usr/local/lib/libopencv_imgproc.so.410 /lib/libopencv_imgproc.so.410
COPY --chown=user:user --from=build /usr/local/lib/libopencv_core.so.410 /lib/libopencv_core.so.410
COPY --chown=user:user predict.sh predict.sh
COPY --chown=user:user encoder.xml encoder.xml
COPY --chown=user:user encoder.bin encoder.bin
COPY --chown=user:user decoder.xml decoder.xml
COPY --chown=user:user decoder.bin decoder.bin
# RUN ldd uestcsd_inference
# RUN ldd /lib/libopenvino.so.2420
