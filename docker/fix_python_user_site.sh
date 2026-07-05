#!/usr/bin/env bash
set -e

SITE_PKGS="/home/hello-robot/.local/lib/python3.10/site-packages"

rm -rf "$SITE_PKGS"/setuptools*
rm -rf "$SITE_PKGS"/_distutils_hack
rm -f  "$SITE_PKGS"/distutils-precedence.pth
rm -f  "$SITE_PKGS"/pathlib.py
rm -rf "$SITE_PKGS"/pathlib-*.dist-info
rm -rf "$SITE_PKGS"/pkg_resources
rm -rf "$SITE_PKGS"/packaging*
rm -rf "$SITE_PKGS"/setuptools_scm*

python3 -c "import setuptools, stretch_body; print('setuptools', setuptools.__version__, setuptools.__file__); print('stretch_body', stretch_body.__file__)"
