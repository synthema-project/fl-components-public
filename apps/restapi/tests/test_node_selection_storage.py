"""Persistence and backwards compatibility for selected catalogue nodes."""
import unittest
from sqlalchemy import create_engine, inspect, text
from sqlmodel import SQLModel, Session
from pydantic import ValidationError
from schemas.task import TaskCreate, TaskRead
from schemas.migrations import upgrade_task_node_selection


class NodeSelectionStorageTest(unittest.TestCase):
    def test_selection_survives_database_roundtrip(self):
        engine = create_engine('sqlite://')
        SQLModel.metadata.create_all(engine)
        with Session(engine) as session:
            task = TaskRead.model_validate(TaskCreate(
                user_id='test', use_case='AML', model_name='model', model_version=1,
                num_global_iterations=1, run_name='test', experiment_name='test',
                selected_nodes=['ICH']))
            session.add(task)
            session.commit()
            task_id = task.id
        with Session(engine) as session:
            self.assertEqual(session.get(TaskRead, task_id).selected_nodes, ['ICH'])

    def test_upgrade_preserves_old_tasks_and_can_repeat(self):
        engine = create_engine('sqlite://')
        with engine.begin() as connection:
            connection.execute(text('CREATE TABLE taskread (id INTEGER PRIMARY KEY, run_name TEXT)'))
            connection.execute(text("INSERT INTO taskread VALUES (1, 'existing')"))
            upgrade_task_node_selection(connection)
            upgrade_task_node_selection(connection)
            self.assertIn('selected_nodes', {c['name'] for c in inspect(connection).get_columns('taskread')})
            self.assertEqual(tuple(connection.execute(text('SELECT id, run_name, selected_nodes FROM taskread')).one()), (1, 'existing', None))

    def test_empty_blank_and_duplicate_selections_rejected(self):
        for selection in ([], [' '], ['ICH', 'ICH']):
            with self.subTest(selection=selection), self.assertRaises(ValidationError):
                TaskCreate(user_id='test', use_case='AML', model_name='model', model_version=1,
                           num_global_iterations=1, run_name='test', experiment_name='test',
                           selected_nodes=selection)
