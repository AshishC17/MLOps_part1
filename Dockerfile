FROM python:3.12-slim
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
WORKDIR /usr/app/src
COPY train.py ./
EXPOSE 3030

# Define environment variable
ENV MLOPS basic

# Run predict.py when the container launches
CMD ["python", "train.py"]
