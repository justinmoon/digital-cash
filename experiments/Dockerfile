FROM python:3.7.0
ADD requirements.txt ./
RUN pip install -r requirements.txt
ADD alternating.py ./

CMD ["python", "alternating.py", "serve"]
