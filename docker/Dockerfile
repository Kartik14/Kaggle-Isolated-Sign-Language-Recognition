# Use an official Ubuntu image as a base
FROM tensorflow/tensorflow:latest

COPY pkglist.txt /tmp/pkglist.txt
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -yq --no-install-recommends $(cat /tmp/pkglist.txt)

RUN pip install pandas pyarrow tqdm
