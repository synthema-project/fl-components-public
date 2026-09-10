"""Check the actual Uvicorn entry point without connecting to external services."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class EntrypointTest(unittest.TestCase):
    def test_app_import_and_authentication(self):
        source = Path(__file__).resolve().parents[1] / 'src'
        env = {
            **os.environ,
            'PYTHONPATH': str(source) + os.pathsep + os.environ.get('PYTHONPATH', ''),
            'DB_MODE': 'sqlite',
            'RABBIT_USERNAME': 'test', 'RABBIT_PASSWORD': 'test',
            'RABBIT_HOST': 'localhost', 'RABBIT_PORT': '5672',
            'KEYCLOAK_SERVER_URL': 'http://localhost/',
            'KEYCLOAK_CLIENT_ID': 'test', 'KEYCLOAK_REALM_NAME': 'test',
            'AUTH_DISABLE_TESTS': 'false', 'AUTH_DISABLE_VERIFICATION': 'false',
        }
        result = subprocess.run([sys.executable, '-c', '''
import asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from uvicorn.importer import import_from_string

app = import_from_string('restapi.main:app')
assert isinstance(app, FastAPI)

async def check():
    # ASGITransport does not start lifespan, so RabbitMQ/Postgres stay untouched.
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        assert (await client.get('/')).status_code == 200
        for path in ('/tasks/', '/mlflow/registered-models/list'):
            response = await client.get(path)
            assert response.status_code == 401, (path, response.text)
            assert response.headers['www-authenticate'] == 'Bearer'

    from unittest.mock import AsyncMock, MagicMock, patch
    from fastapi import HTTPException
    from restapi.routers.tasks import create_task
    from restapi.auth import get_mock_user
    from schemas.task import TaskCreate
    import json

    task = TaskCreate(user_id='test-user', use_case='AML', model_name='test',
                      model_version=1, num_global_iterations=1,
                      run_name='test', experiment_name='test')
    session = AsyncMock()
    session.add = MagicMock()
    session.refresh.side_effect = lambda record: setattr(record, "id", 1)
    dispatch = AsyncMock()
    # Creating a task must work without querying the catalogue or forwarding JWTs.
    with patch('requests.sessions.Session.request', side_effect=AssertionError('Unexpected HTTP call')):
        result = await create_task(task, session, {'dispatch': dispatch}, get_mock_user())
    session.commit.assert_awaited_once()
    dispatch.publish_message.assert_awaited_once()
    message = json.loads(dispatch.publish_message.call_args.args[0])
    assert message['use_case'] == 'AML'
    assert message['node_dataset_map_json'] is None
    task.selected_nodes = ['NODE1']
    try:
        await create_task(task, session, {'dispatch': dispatch}, get_mock_user())
    except HTTPException as error:
        assert error.status_code == 422
    else:
        raise AssertionError('Unsupported node selection was accepted')
    session.commit.assert_awaited_once()

asyncio.run(check())
'''], env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
