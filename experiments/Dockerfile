FROM python:3.7.0
ADD requirements.txt ./
RUN pip install -r requirements.txt
ADD ping_pong_docker.py ./

CMD ["python", "ping_pong_docker.py", "serve"]
