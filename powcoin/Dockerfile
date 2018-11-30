FROM python:3.7.0
ADD requirements.txt ./
RUN pip install -r requirements.txt
ADD utils.py ./
ADD identities.py ./
ADD powp2pcoin.py ./

CMD ["python", "powp2pcoin.py", "serve"]
