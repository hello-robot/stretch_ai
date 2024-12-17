#!/usr/bin/env bash
jetson-containers run --volume `pwd`:/stretch_ai --workdir /stretch_ai $(autotag l4t-text-generation)
