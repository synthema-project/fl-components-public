"""Freeze complete uploaded collections and configuration versions for a task."""
import hashlib
import json


def select_snapshot(records, use_case, nodes):
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError('Select unique participating nodes before creating a validation task')
    groups = {}
    for row in records:
        if row.get('use_case') == use_case and row.get('node') in nodes and row.get('collection_id'):
            groups.setdefault((row['node'], row['collection_id']), []).append(row)
    selected = {}
    for (node, collection), rows in groups.items():
        roles = {row.get('dataset_role'): row for row in rows}
        if not {'train', 'val', 'test', 'schema'}.issubset(roles):
            continue
        rows = [roles[role] for role in ('train', 'val', 'test', 'schema')]
        versions = {row.get('configuration_version') for row in rows}
        if len(versions) != 1 or None in versions:
            raise ValueError('Collection partitions have inconsistent configuration versions')
        documents = {'metadata': rows[0].get('dataset_meta'), 'evaluation': rows[0].get('evaluation_configuration')}
        if not all(isinstance(document, dict) and document for document in documents.values()):
            raise ValueError('Collection lacks central evaluation documents')
        version = hashlib.sha256(json.dumps(documents, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if versions != {version} or any(row.get('dataset_meta') != documents['metadata'] or
                row.get('evaluation_configuration') != documents['evaluation'] for row in rows):
            raise ValueError('Collection configuration content does not match its version')
        stamp = max(str(row.get('timestamp', '')) for row in rows)
        entry = {'collection_id': collection, 'configuration_version': version,
                 'paths': {role: roles[role]['path'] for role in ('train', 'val', 'test', 'schema')},
                 'documents': documents}
        if node not in selected or (stamp, collection) > selected[node][0]:
            selected[node] = ((stamp, collection), entry)
    if set(selected) != set(nodes):
        raise ValueError('Every selected node needs a complete partitioned upload')
    entries = {node: selected[node][1] for node in nodes}
    if len({entry['configuration_version'] for entry in entries.values()}) != 1:
        raise ValueError('Participating nodes must use the same configuration version')
    return {'use_case': use_case, 'nodes': entries}
