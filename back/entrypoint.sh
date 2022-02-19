#!/bin/bash
exec gunicorn --config ./gunicorn_config.py --log-level debug server:app