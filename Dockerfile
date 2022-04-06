# Version de Python para la aplicación
FROM python:3.8.5

# Actualizar la version de pip del contenedor y crear la carpeta para los archivos
RUN pip install --upgrade pip \
    && mkdir /app

# Agregar los archivos a la carpeta
ADD . /app

# Establecer como directorio de trabajo
WORKDIR /app

# Instalar los requerimientos
RUN pip install -r requirements.txt

# Ejecutar aplicación
CMD python /app/bot.py