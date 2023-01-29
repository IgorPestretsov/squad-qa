FROM python:3.8.10

WORKDIR /usr/src/bert-qa/

ENV PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VERSION=1.3.1

RUN pip install "poetry==$POETRY_VERSION"

COPY poetry.lock pyproject.toml ./

RUN poetry config virtualenvs.create false \
  && poetry install --only main --no-interaction

COPY ./src/ /usr/src/bert-qa/src/



CMD ["python3", "src/main.py"]

