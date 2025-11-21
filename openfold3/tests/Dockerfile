FROM openfoldconsortium/testing-of-dockerhub:0.3.1

COPY openfold3 /opt/openfold3
COPY environments/development.txt /opt/openfold3/development.txt

WORKDIR /opt/openfold3
RUN pip install -r development.txt
RUN pip install --force-reinstall --no-deps .
