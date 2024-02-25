FROM python:3.10
WORKDIR /fe_stream
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "iris_frontend.py"]
