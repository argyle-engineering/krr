docker buildx build \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --platform linux/arm64,linux/amd64 \
  --tag us-docker.pkg.dev/argyle-artifacts/hub/krr:${TAG} \
  --push \
  .
