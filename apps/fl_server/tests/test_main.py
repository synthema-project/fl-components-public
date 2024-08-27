from unittest.mock import MagicMock, patch

from fl_server.main import event_handler, get_serverapp

from schemas.task import TaskRead


def test_main():
    task = TaskRead(
        user_id="user_id",
        use_case="use_case",
        model_name="model_name",
        model_version=1,
        num_global_iterations=2,
        id=1,
        status="pending",
        created_at="2021-08-01T00:00:00",
        run_name="run_name",
        experiment_name="experiment_name",
    )

    with patch("fl_server.main.rabbitmq_client") as mock_rabbitmq_client:
        with patch("fl_server.main.run_server_app") as mock_run_server_app:
            mock_listen = mock_rabbitmq_client.listen = MagicMock(return_value=task)
            mock_listen.side_effect = get_serverapp

            event_handler(MagicMock(), MagicMock(), MagicMock(), task.model_dump_json())
            mock_run_server_app.assert_called_once()
