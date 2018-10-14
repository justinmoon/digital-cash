FROM python:3.7.0
ADD requirements.txt ./
RUN pip install -r requirements.txt
ADD server.py ./

CMD ["python", "server.py"]
