FROM google/cloud-sdk:slim
RUN apt-get update && apt-get upgrade -y && apt-get install -y python3 python3-pip ffmpeg
COPY ./requirements.txt /app/
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt && python3 -m pip install gunicorn && python3 -m nltk.downloader stopwords 
COPY . /app
WORKDIR /app
EXPOSE 80
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["sh", "entrypoint.sh"]
# ENTRYPOINT [ "python3", "-m", "server" ]