"""Fail closed if a report bucket permits anonymous access."""
import json
from minio.error import S3Error


def require_private_bucket(client, bucket):
    try:
        policy = json.loads(client.get_bucket_policy(bucket))
    except S3Error as error:
        if error.code == 'NoSuchBucketPolicy':
            return
        raise
    for statement in policy.get('Statement', []):
        principal = statement.get('Principal', {})
        identities = principal.get('AWS', []) if isinstance(principal, dict) else principal
        anonymous = identities == '*' or (isinstance(identities, list) and '*' in identities)
        if statement.get('Effect') == 'Allow' and (anonymous or 'NotPrincipal' in statement):
            raise ValueError('Validation reports must be stored in a bucket without anonymous access')
