FROM python

RUN mkdir -p /workspace
RUN git config --global credential.helper store

WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY clipai.py .
COPY configurations/ configurations/

ENTRYPOINT ["python", "clipai.py"]