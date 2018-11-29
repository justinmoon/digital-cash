FROM python:3.7.0
ADD requirements.txt ./
RUN pip install -r requirements.txt
ADD powp2pcoin.py ./
ADD identities.py ./

CMD ["python", "powp2pcoin.py", "serve"]
