# Core organism (CPU). Language models are reached over the network (e.g. Ollama on the host:
#   docker run -p 8765:8765 -e ORGANISM_LLM__OLLAMA_URL=http://host.docker.internal:11434 -v organism-data:/app/data cortex-ai
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY organism ./organism
COPY config ./config
RUN pip install --no-cache-dir . && pip install --no-cache-dir python-multipart
ENV ORGANISM_API__HOST=0.0.0.0
EXPOSE 8765
VOLUME ["/app/data"]
CMD ["python", "-m", "organism", "serve"]
