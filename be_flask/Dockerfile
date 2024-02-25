FROM python:3.10
WORKDIR /flask
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
ENV FLASK_APP=iris_backend.py
ENV FLASK_RUN_HOST=0.0.0.0
ENTRYPOINT ["flask", "run"]