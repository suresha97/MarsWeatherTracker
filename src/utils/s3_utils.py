from typing import Dict, Any
import json

import boto3


def save_json_to_s3(json_obj: Dict[Any, Any], bucket: str , key: str) -> None:
    s3 = boto3.resource("s3")
    s3object = s3.Object(bucket, key)
    s3object.put(Body=(bytes(json.dumps(json_obj).encode("UTF-8"))))


def load_json_from_s3_as_dict(bucket: str, key: str) -> Dict[Any, Any]:
    s3 = boto3.resource("s3")
    file_object = s3.Object(bucket, key)
    json_obj = file_object.get()["Body"].read().decode("utf-8")
    dict_obj = json.loads(json_obj)

    return dict_obj
