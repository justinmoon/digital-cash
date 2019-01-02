FROM python:3.7.0
ADD requirements.txt ./
RUN pip install -r requirements.txt
ADD mybitcoin.py ./

CMD ["python", "-u", "mybitcoin.py", "serve"]
