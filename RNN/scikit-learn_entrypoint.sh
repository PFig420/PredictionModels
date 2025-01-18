#!/bin/bash

# Conditionally set the environment variable
if [ "$ARCH" = "arm64" ]; then
    export LD_PRELOAD=/usr/local/lib/python3.9/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
fi

# Execute the original entrypoint command
exec "$@"