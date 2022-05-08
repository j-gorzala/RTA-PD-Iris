FROM python:3.8.9
WORKDIR /RTA-PD-IRIS
ADD . /RTA-PD-IRIS/
RUN pip install -r requirements.txt
CMD ["python", "app.py"]