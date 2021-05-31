#!/bin/bash

rm -rf ./lambda_code_uri
cp -r ../src/ ./lambda_code_uri

poetry export --without-hashes -f requirements.txt --output requirements.txt

mv ../requirements.txt ./lambda_code_uri/

sam.cmd build
