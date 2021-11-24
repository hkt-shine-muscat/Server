FROM rackspacedot/python38
LABEL author="smoothbear04@gmail.com"

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential

COPY . /app
WORKDIR /app

RUN pip3 install -r requirements.txt
CMD ["python3", "Server/server.py"]