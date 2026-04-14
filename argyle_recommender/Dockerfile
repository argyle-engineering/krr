FROM python:3.11 AS builder

# Set the working directory
WORKDIR /app
ARG KUSTOMIZE_FMT_VERSION=0.0.10


RUN apt-get update \
    && dpkg --add-architecture arm64 \
    && apt-get install --no-install-recommends -y \
        build-essential curl gpg

# Install gh cli to get internal tools
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg;
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null;
RUN apt update && apt install -y gh;


ENV VIRTUAL_ENV=/venv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_FROZEN=1 \
    UV_NO_SYNC=1 \
    UV_NO_DEV=1 \
    UV_NO_EDITABLE=1 \
    UV_PROJECT_ENVIRONMENT=$VIRTUAL_ENV

# Install uv and configure venv
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

ENV PATH="/root/.local/bin:${PATH}"

# get ksops
RUN curl -sLo ksops.tar.gz https://github.com/argyle-engineering/ksops/releases/download/v2.0.8/ksops_Linux_x86_64.tar.gz \
    && tar -xvf ksops.tar.gz ksops \
    && mkdir -p "$HOME/go/bin" \
    && mv ksops "$HOME/go/bin" 
# get kustomize
RUN curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash -s - 5.4.1 $HOME/go/bin/
# get argyle-resource-quota-transformer
RUN --mount=type=secret,id=GITHUB_TOKEN \
    export GH_TOKEN=$(cat /run/secrets/GITHUB_TOKEN) \
    && gh release download --clobber -R "argyle-systems/argyle-resource-quota-transformer" --pattern "argyle-resource-quota-transformer_linux_x86_64.tar.gz" -D "/tmp" \
    && tar -xzf "/tmp/argyle-resource-quota-transformer_linux_x86_64.tar.gz" -C "/tmp" \
    && mv "/tmp/argyle-resource-quota-transformer" "/root/go/bin"
# get argyle-kustomize-fmt
RUN --mount=type=secret,id=GITHUB_TOKEN \
    export GH_TOKEN=$(cat /run/secrets/GITHUB_TOKEN) \
    && gh release download --clobber -R "argyle-systems/argyle-kustomize-fmt" --pattern "argyle-kustomize-fmt_${KUSTOMIZE_FMT_VERSION}_linux_386.tar.gz" -O "/tmp/argyle-kustomize-fmt_Linux_x86_64.tar.gz" \
    && tar -xzf "/tmp/argyle-kustomize-fmt_Linux_x86_64.tar.gz" -C "/tmp" \
    && mv "/tmp/argyle-kustomize-fmt" "/root/go/bin"

# get gke-gcloud-auth-plugin and add to path
ENV PATH="/google-cloud-sdk/bin:${PATH}"

RUN curl -fsSLO https://dl.google.com/dl/cloudsdk/channels/rapid/google-cloud-sdk.tar.gz \
    && tar xzf google-cloud-sdk.tar.gz -C / \
    && rm google-cloud-sdk.tar.gz \
    && gcloud config set core/disable_usage_reporting true && \
    gcloud config set component_manager/disable_update_check true && \
    gcloud config set metrics/environment github_docker_image && \
    gcloud components install alpha beta gke-gcloud-auth-plugin

# get kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# Copy the rest of the application code
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# Run the application using 'poetry run krr simple'
CMD ["/venv/bin/python", "argyle_recommender/recommender.py", "argyle"]
