FROM nvcr.io/nvidia/l4t-base:r32.5.0

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev python3-setuptools python3-opencv \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev v4l-utils libgtk2.0-dev libcanberra-gtk* \
    wget unzip nano

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python dependencies
RUN pip3 install numpy imutils

# Download YOLOv3 config and weights
WORKDIR /app
RUN wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg && \
    wget https://pjreddie.com/media/files/yolov3.weights && \
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

# Copy the detection script into the container
COPY yolo-csi.py .

# Default command
CMD ["python3", "yolo_csi.py"]
