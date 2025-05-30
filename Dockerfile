# Imagen base con Python 3.12
FROM python:3.12

# Establece un directorio de trabajo
WORKDIR /workspace

# Instala Git dentro del contenedor
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Comando por defecto: shell interactiva
CMD ["bash"]
