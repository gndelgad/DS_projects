# gcloud beta app gen-config --custom
# This docker file was generated from the pre-configured runtime
# runtime: python
# runtime_config:
#     python_version: 3
# and then modified (GDK)

FROM gcr.io/google-appengine/python
# Probably need to 
LABEL python_version=python3.6
# The following apt-get packages are installed to avoid some GPU problems with open-cv (GDK)
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "libsm6", "libxext6", "libxrender-dev"]

RUN virtualenv --no-download /env -p python3.6

# Set virtualenv environment variables. This is equivalent to running
# source /env/bin/activate
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH
ADD requirements.txt /app/
RUN pip install -r requirements.txt
ADD . /app/
CMD exec gunicorn -b:$PORT main:app
