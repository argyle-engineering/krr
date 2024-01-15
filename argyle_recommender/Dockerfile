FROM python:3.11 as builder

# Set the working directory
WORKDIR /app

# Install system dependencies required for Poetry
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION="1.7.1" \
    POETRY_VENV="/opt/poetry-venv"

RUN apt-get update \
    && dpkg --add-architecture arm64 \
    && apt-get install --no-install-recommends -y \
        build-essential

# https://python-poetry.org/docs/#installing-manually
RUN python -m venv ${POETRY_VENV}
RUN ${POETRY_VENV}/bin/pip install -U pip setuptools
RUN ${POETRY_VENV}/bin/pip install "poetry==${POETRY_VERSION}"
RUN ${POETRY_VENV}/bin/poetry config virtualenvs.create false

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

# Install the project dependencies
RUN ${POETRY_VENV}/bin/poetry install
RUN curl -sLo ksops.tar.gz https://github.com/argyle-engineering/ksops/releases/download/v2.0.8/ksops_Linux_x86_64.tar.gz \
    && tar -xvf ksops.tar.gz ksops \
    && mkdir -p "$HOME/go/bin" \
    && mv ksops "$HOME/go/bin" 
RUN curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash -s - 5.1.0 $HOME/go/bin/ 
ENV PATH="${POETRY_VENV}/bin:/root/go/bin:/google-cloud-sdk/bin::${PATH}"

RUN curl -fsSLO https://dl.google.com/dl/cloudsdk/channels/rapid/google-cloud-sdk.tar.gz \
    && tar xzf google-cloud-sdk.tar.gz -C / \
    && rm google-cloud-sdk.tar.gz \
    && gcloud config set core/disable_usage_reporting true && \
    gcloud config set component_manager/disable_update_check true && \
    gcloud config set metrics/environment github_docker_image && \
    gcloud components install alpha beta gke-gcloud-auth-plugin

RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# Copy the rest of the application code
COPY . .

RUN ${POETRY_VENV}/bin/poetry install
# Run the application using 'poetry run krr simple'
CMD ["$POETRY_VENV/bin/python", "krr.py", "simple"]
