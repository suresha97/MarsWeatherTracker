import json

import boto3


def save_json_to_s3(json_obj, bucket, key):
    s3 = boto3.resource("s3")
    s3object = s3.Object(bucket, key)
    s3object.put(Body=(bytes(json.dumps(json_obj).encode("UTF-8"))))
