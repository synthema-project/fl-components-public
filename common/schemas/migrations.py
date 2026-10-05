"""Additive task schema upgrades for API and status-listener startup."""
from sqlalchemy import inspect, text


def upgrade_task_node_selection(connection):
    for name in ('selected_nodes', 'validation_snapshot'):
        if connection.dialect.name == 'postgresql':
            connection.execute(text(f'ALTER TABLE taskread ADD COLUMN IF NOT EXISTS {name} JSON'))
        elif name not in {column['name'] for column in inspect(connection).get_columns('taskread')}:
            connection.execute(text(f'ALTER TABLE taskread ADD COLUMN {name} JSON'))
