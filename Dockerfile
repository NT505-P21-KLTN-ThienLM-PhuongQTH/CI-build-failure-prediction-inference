FROM python:3.9-alpine
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY main.simulation.py main.py
EXPOSE 8000
CMD ["python", "main.py"]