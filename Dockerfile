FROM python:3.9-slim
WORKDIR /app
ENV FLASK_APP=api.py
ENV FLASK_RUN_HOST=0.0.0.
RUN #apk add --no-cache gcc musl-dev linux-headers
COPY requirements.txt requirements.txt
RUN GRPC_PYTHON_DISABLE_LIBC_COMPATIBILITY=1  pip install -r requirements.txt
EXPOSE 5000
COPY . .
CMD ["python", "api.py"]