FROM python:3.10-slim
WORKDIR /src
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./app /src/app
COPY ./model /src/model
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]