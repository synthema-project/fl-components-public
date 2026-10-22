"""Historical security settings must not break task-list response serialization."""
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from restapi.routers.tasks import router, get_session, get_rabbitmqs
from restapi.auth import require_authentication, get_mock_user
from schemas.task import TaskRead


@pytest.mark.asyncio
async def test_history_readable_and_invalid_submissions_rejected():
    payload = dict(user_id='test', use_case='AML', model_name='model', model_version=1,
                   num_global_iterations=1, run_name='old', experiment_name='test',
                   SMPC_enabled=True, dp_mode='central_dp', selected_nodes=['ICH'])
    # Match a legacy row loaded by SQLAlchemy, bypassing submission validation.
    historical = TaskRead(**payload, id=1, status='failed')
    session = AsyncMock()
    session.add = Mock()
    session.refresh.side_effect = lambda row: setattr(row, "id", 2)
    result = Mock()
    result.scalars.return_value.all.return_value = [historical]
    session.execute.return_value = result
    session.get.return_value = historical
    dispatch = AsyncMock()
    app = FastAPI()
    app.include_router(router)
    async def session_override():
        return session
    async def user_override():
        return get_mock_user()
    async def rabbitmq_override():
        return {'dispatch': dispatch}
    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[require_authentication] = user_override
    app.dependency_overrides[get_rabbitmqs] = rabbitmq_override
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        for path in ('/tasks/', '/tasks/1', '/tasks/user/test'):
            response = await client.get(path)
            assert response.status_code == 200, response.text
            item = response.json()[0] if isinstance(response.json(), list) else response.json()
            assert item['SMPC_enabled'] is True
            assert item['dp_mode'] == 'central_dp'
            assert item['status'] == 'failed'
        response = await client.post('/tasks/', json=payload)
        assert response.status_code == 422
        assert 'SMPC cannot be enabled' in response.text
        session.commit.assert_not_awaited()
        dispatch.publish_message.assert_not_awaited()
        request = {key: value for key, value in payload.items() if key not in {'dp_mode', 'SMPC_enabled'}}
        request['selected_nodes'] = ['ICH', 'UMCU']
        response = await client.post('/tasks/', json=request)
        assert response.status_code == 200, response.text
        assert response.json()['SMPC_enabled'] is False
        assert response.json()['dp_mode'] is None
    session.commit.assert_awaited_once()
    dispatch.publish_message.assert_awaited_once()
    import json
    message = json.loads(dispatch.publish_message.call_args.args[0])
    assert message['SMPC_enabled'] is False
    assert message['dp_mode'] is None
    assert message['selected_nodes'] == ['ICH', 'UMCU']
