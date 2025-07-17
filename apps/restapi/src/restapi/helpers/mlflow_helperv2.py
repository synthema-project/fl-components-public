"""
MLflow Federated Job Fetcher

This module provides functionality to fetch federated learning job information
from MLflow and return it in a structured format compatible with FastAPI.
"""

import os
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import logging

import mlflow
from mlflow.tracking import MlflowClient
MLFLOW_AVAILABLE = True

logger = logging.getLogger(__name__)


class RunInfo(BaseModel):
    """Information about a run in a federated learning job."""
    
    run_id: str = Field(..., description="Unique identifier for the run")
    run_name: str = Field(..., description="Name of the run")
    status: str = Field(..., description="Status of the run (RUNNING, FINISHED, FAILED, etc.)")
    start_time: Optional[datetime] = Field(None, description="When the run started")
    end_time: Optional[datetime] = Field(None, description="When the run ended")
    duration: Optional[float] = Field(None, description="Duration of the run in seconds")
    client_email: Optional[str] = Field(None, description="Email of the client that ran this")
    node_id: Optional[str] = Field(None, description="Node ID of the client")
    participant_id: Optional[str] = Field(None, description="Participant identifier")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Metrics logged by this client")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters used by this client")

class ModelInfo(BaseModel):
    """Information about a model logged in MLflow."""
    
    name: str = Field(..., description="Name of the model")
    version: str = Field(..., description="Version of the model")
    creation_timestamp: int = Field(..., description="When the model was created (timestamp)")
    last_updated_timestamp: int = Field(..., description="When the model was last updated (timestamp)")
    current_stage: str = Field(..., description="Stage of the model (None, Staging, Production, Archived)")
    description: str = Field(default="", description="Description of the model")
    source: str = Field(..., description="Path to the model artifact")
    run_id: str = Field(..., description="ID of the run that created this model")
    status: str = Field(..., description="Status of the model (READY, etc.)")
    tags: List[Dict[str, str]] = Field(default_factory=list, description="Tags associated with the model")
    run_link: str = Field(default="", description="Link to the run")

class MlflowInfo(BaseModel):
    """Information about Mlflow."""

    experiment_name: str = Field(..., description="Name of the MLflow experiment")
    experiment_id: str = Field(..., description="ID of the MLflow experiment")

    parent_run: RunInfo = Field(..., description="Information about the parent run")
    child_runs: List[RunInfo] = Field(default_factory=list, description="Information about child runs")
    models: List[ModelInfo] = Field(default_factory=list, description="Models logged in this federated job")


class SyftInfo(BaseModel):
    """Information about a Syft job."""
    
    client_email: str = Field(..., description="Email of the client that ran this")
    node_id: str = Field(..., description="Node ID of the client")
    participant_id: str = Field(..., description="Participant identifier")
    

class FederatedJobInfo(BaseModel):
    """Complete information about a federated learning job aligned with task example structure."""

    # Top-level fields
    use_case: str = Field(..., description="Use case identifier")
    user_id: str = Field(..., description="User ID")
    model_version: int = Field(..., description="Model version")
    run_name: str = Field(..., description="Name of the run")
    id: str = Field(..., description="Job ID")  # Changed from int to str
    created_at: datetime = Field(..., description="When the job was created")
    model_name: str = Field(..., description="Name of the model")
    num_global_iterations: int = Field(..., description="Number of global iterations")
    experiment_name: str = Field(..., description="Name of the experiment")
    status: str = Field(..., description="Status of the job")

    # Run information
    run: RunInfo = Field(..., description="Parent run information")
    
    # Children runs
    children_runs: List[RunInfo] = Field(default_factory=list, description="Child runs information")
    
    # Model information
    model: ModelInfo = Field(..., description="Model information")


class FederatedJobFetcher:
    """
    Fetches federated learning job information from MLflow.
    
    This class provides methods to retrieve comprehensive information about
    federated learning jobs, including parent runs, child runs, and models.
    """
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize the federated job fetcher.
        
        Args:
            tracking_uri: MLflow tracking URI (optional)
        """
        if not MLFLOW_AVAILABLE:
            raise RuntimeError("MLflow is required but not available. Please install mlflow: pip install mlflow")
        
        self.tracking_uri = tracking_uri
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.client = MlflowClient()
    
    def get_federated_job_info(self, parent_run_id: str) -> FederatedJobInfo:
        """
        Get comprehensive information about a federated learning job.
        
        Args:
            parent_run_id: ID of the parent run
            
        Returns:
            FederatedJobInfo with complete job information
            
        Raises:
            RuntimeError: If the run is not found or is not a federated job
        """
        try:
            # Get parent run information
            parent_run = self.client.get_run(parent_run_id)
            
            # Extract basic information
            experiment = self.client.get_experiment(parent_run.info.experiment_id)
            
            # Parse timing information
            start_time = datetime.fromtimestamp(parent_run.info.start_time / 1000.0) if parent_run.info.start_time else None
            end_time = datetime.fromtimestamp(parent_run.info.end_time / 1000.0) if parent_run.info.end_time else None
            duration = (parent_run.info.end_time - parent_run.info.start_time) / 1000.0 if parent_run.info.end_time and parent_run.info.start_time else None
            
            # Extract configuration from parameters
            params = parent_run.data.params
            total_rounds = int(params.get("total_rounds", 0))
            num_clients = int(params.get("num_clients", 0))
            log_system_metrics = params.get("log_system_metrics", "true").lower() == "true"
            log_model_per_round = params.get("log_model_per_round", "false").lower() == "true"
            
            # Get child runs (client runs)
            child_runs = self._get_child_runs(parent_run_id)
            
            # Get models
            models = self._get_models(parent_run_id)
            
            # Get aggregated metrics
            # aggregated_metrics = self._get_aggregated_metrics(parent_run)
            
            # Get parent run name safely
            parent_run_name = parent_run.data.tags.get("mlflow.runName")
            if not parent_run_name:
                parent_run_name = parent_run.info.run_name
            if not parent_run_name:
                parent_run_name = f"run_{parent_run_id}"
            
            # Create parent run info
            parent_run_info = RunInfo(
                run_id=parent_run_id,
                run_name=parent_run_name,
                status=parent_run.info.status,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                client_email="federated_coordinator",  # Parent run is the coordinator
                node_id="coordinator_node",
                participant_id="coordinator",
                metrics=dict(parent_run.data.metrics),
                parameters=dict(parent_run.data.params)
            )
            
            # Create MlflowInfo
            mlflow_info = MlflowInfo(
                experiment_name=experiment.name,
                experiment_id=experiment.experiment_id,
                parent_run=parent_run_info,
                child_runs=child_runs,
                models=models
            )
            
            # Create SyftInfo (extract from parent run tags or child runs)
            tags = parent_run.data.tags
            
            # Try to get client info from parent run tags first
            client_email = tags.get("client_email")
            node_id = tags.get("node_id")
            participant_id = tags.get("participant_id")
            
            # Extract email from parent run_name if not in tags
            if not client_email and parent_run_name:
                # Try to extract email from run_name (format: client_<node_id>_<email>)
                parts = parent_run_name.split('_')
                if len(parts) >= 3 and '@' in parts[-1]:
                    client_email = parts[-1]
            
            # If not found in parent run, try to get from child runs
            if not client_email and child_runs:
                # Use the first child run's client info as representative
                first_child = child_runs[0]
                client_email = first_child.client_email or "federated_coordinator"
                node_id = first_child.node_id or "coordinator_node"
                participant_id = first_child.participant_id or "coordinator"
            else:
                # Use defaults if no info found
                client_email = client_email or "federated_coordinator"
                node_id = node_id or "coordinator_node"
                participant_id = participant_id or "coordinator"
            
            syft_info = SyftInfo(
                client_email=client_email,
                node_id=node_id,
                participant_id=participant_id
            )
            
            # Get the first model as the main model
            main_model = models[0] if models else ModelInfo(
                name="default_model",
                version="1",
                creation_timestamp=int(time.time() * 1000),
                last_updated_timestamp=int(time.time() * 1000),
                current_stage="None",
                description="",
                source="",
                run_id=parent_run_id,
                status="READY",
                tags=[],
                run_link=""
            )
            
            return FederatedJobInfo(
                use_case="federated_learning",
                user_id=client_email or "unknown",
                model_version=1,
                run_name=parent_run_name,
                id=parent_run_id,  # Convert hex to int
                created_at=start_time or datetime.now(),
                model_name=main_model.name,
                num_global_iterations=total_rounds,
                experiment_name=experiment.name,
                status=parent_run.info.status,
                run=parent_run_info,
                children_runs=child_runs,
                model=main_model
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch federated job information: {e}")
    
    def _get_child_runs(self, parent_run_id: str) -> List[RunInfo]:
        """Get information about child runs (client runs)."""
        child_runs = []
        
        try:
            # Get the parent run to find its experiment
            parent_run = self.client.get_run(parent_run_id)
            experiment_id = parent_run.info.experiment_id
            
            # Search for child runs using the correct MLflow parent run tag
            search_filter = f"tags.mlflow.parentRunId = '{parent_run_id}'"
            child_run_results = self.client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=search_filter
            )
            
            for run in child_run_results:
                # Parse timing
                start_time = datetime.fromtimestamp(run.info.start_time / 1000.0) if run.info.start_time else None
                end_time = datetime.fromtimestamp(run.info.end_time / 1000.0) if run.info.end_time else None
                duration = (run.info.end_time - run.info.start_time) / 1000.0 if run.info.end_time and run.info.start_time else None
                
                # Extract client information from tags
                tags = run.data.tags
                client_email = tags.get("client_email")
                node_id = tags.get("node_id")
                participant_id = tags.get("participant_id")
                
                # Extract email from run_name if not in tags
                if not client_email and run.info.run_name:
                    # Try to extract email from run_name (format: client_<node_id>_<email>)
                    parts = run.info.run_name.split('_')
                    if len(parts) >= 3 and '@' in parts[-1]:
                        client_email = parts[-1]
                
                # Provide defaults for missing values
                if not client_email:
                    client_email = f"client_{run.info.run_id}"
                if not node_id:
                    node_id = f"node_{run.info.run_id}"
                if not participant_id:
                    participant_id = "participant"
                
                child_run_info = RunInfo(
                    run_id=run.info.run_id,
                    run_name=run.info.run_name or f"run_{run.info.run_id}",
                    status=run.info.status,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    client_email=client_email,
                    node_id=node_id,
                    participant_id=participant_id,
                    metrics=dict(run.data.metrics),
                    parameters=dict(run.data.params)
                )
                
                child_runs.append(child_run_info)
                
        except Exception as e:
            logger.warning(f"Failed to fetch child runs: {e}")
        
        return child_runs
    
    def _get_models(self, parent_run_id: str) -> List[ModelInfo]:
        """Get information about models logged in the parent run (both registered and logged artifacts)."""
        models = []
        
        # Get the parent run for artifact URI
        try:
            parent_run = self.client.get_run(parent_run_id)
        except Exception as e:
            logger.warning(f"Failed to get parent run: {e}")
            return models
        
        try:
            # First, get registered models from the model registry
            model_versions = self.client.search_model_versions(f"run_id = '{parent_run_id}'")
            for mv in model_versions:
                model_info = ModelInfo(
                    name=mv.name,
                    version=str(mv.version),
                    creation_timestamp=int(mv.creation_timestamp),
                    last_updated_timestamp=int(mv.creation_timestamp),
                    current_stage=mv.current_stage or "None",
                    description="",
                    source=mv.source or "",
                    run_id=mv.run_id or "",
                    status="READY",
                    tags=[],
                    run_link=""
                )
                models.append(model_info)
        except Exception as e:
            logger.warning(f"Failed to get registered models from registry: {e}")
        
        # Also try to get logged model artifacts using the MLflow client API
        try:
            # Use the client.search_logged_models API to find models for this run
            logged_models = self.client.search_logged_models(
                experiment_ids=[parent_run.info.experiment_id],
                filter_string=f"source_run_id = '{parent_run_id}'"
            )
            
            for logged_model in logged_models:
                # Get model-specific information from the source run
                model_params = {}
                model_metrics = {}
                model_tags = {}
                
                try:
                    source_run = self.client.get_run(getattr(logged_model, 'source_run_id', ''))
                    # Only include model-specific parameters and metrics
                    # Filter out run-level parameters that are duplicated
                    run_params = dict(source_run.data.params)
                    run_metrics = dict(source_run.data.metrics)
                    run_tags = dict(source_run.data.tags)
                    
                    # Keep only model-specific information
                    # This is a simplified approach - in practice, you might want more sophisticated filtering
                    model_params = {k: v for k, v in run_params.items() if 'model' in k.lower() or 'flavor' in k.lower()}
                    model_metrics = {k: v for k, v in run_metrics.items() if 'model' in k.lower() or 'artifact' in k.lower()}
                    model_tags = {k: v for k, v in run_tags.items() if 'model' in k.lower() or 'artifact' in k.lower()}
                    
                except Exception as e:
                    logger.warning(f"Failed to get detailed model info: {e}")
                
                # Fix timestamp - try different timestamp fields
                creation_time = None
                for timestamp_field in ['creation_timestamp_ms', 'creation_timestamp', 'timestamp']:
                    creation_time = getattr(logged_model, timestamp_field, None)
                    if creation_time:
                        break
                
                logged_at = datetime.fromtimestamp(creation_time / 1000.0) if creation_time else datetime.now()
                
                # Detect model framework from tags/params and artifacts
                model_framework = self._detect_model_framework(model_tags, model_params)
                artifact_framework = self._detect_model_framework_from_artifacts(getattr(logged_model, 'model_id', ''))
                
                # Use artifact detection if it found something, otherwise use tag/param detection
                final_framework = artifact_framework if artifact_framework != 'unknown' else model_framework
                
                model_info = ModelInfo(
                    name=getattr(logged_model, 'name', 'unknown'),
                    version="1",  # Logged models don't have versions
                    creation_timestamp=int(creation_time) if creation_time else int(time.time() * 1000),
                    last_updated_timestamp=int(creation_time) if creation_time else int(time.time() * 1000),
                    current_stage="None",
                    description="",
                    source=getattr(logged_model, 'artifact_uri', ''),
                    run_id=getattr(logged_model, 'source_run_id', ''),
                    status="READY",
                    tags=[],
                    run_link=""
                )
                models.append(model_info)
        except Exception as e:
            logger.warning(f"Failed to get logged models via client.search_logged_models: {e}")
        
        return models
    
    def _get_aggregated_metrics(self, parent_run) -> Dict[str, float]:
        """Extract aggregated metrics from the parent run."""
        aggregated_metrics = {}
        
        try:
            # Look for round-specific metrics and aggregate them
            metrics = parent_run.data.metrics
            
            # Extract final aggregated metrics (usually from the last round)
            for metric_name, value in metrics.items():
                if not metric_name.startswith("round_"):
                    aggregated_metrics[metric_name] = value
            
            # Also look for final round metrics
            for metric_name, value in metrics.items():
                if metric_name.startswith("round_") and "final" in metric_name.lower():
                    clean_name = metric_name.replace("round_", "").replace("_final", "")
                    aggregated_metrics[f"final_{clean_name}"] = value
                    
        except Exception as e:
            logger.warning(f"Failed to extract aggregated metrics: {e}")
        
        return aggregated_metrics
    
    def _infer_model_type(self, model_path: str) -> str:
        """Infer the type of model from its path."""
        if "pytorch" in model_path.lower():
            return "pytorch"
        elif "sklearn" in model_path.lower():
            return "sklearn"
        elif "tensorflow" in model_path.lower():
            return "tensorflow"
        elif "keras" in model_path.lower():
            return "keras"
        else:
            return "unknown"
    
    def _detect_model_framework(self, tags: dict, params: dict) -> str:
        """
        Detect the model framework from tags and parameters.
        
        Args:
            tags: Run tags
            params: Run parameters
            
        Returns:
            Detected framework name
        """
        # Check tags for framework hints
        for key, value in tags.items():
            key_lower = key.lower()
            value_lower = str(value).lower()
            
            if 'pytorch' in key_lower or 'pytorch' in value_lower or 'torch' in value_lower:
                return 'pytorch'
            elif 'tensorflow' in key_lower or 'tensorflow' in value_lower or 'tf' in value_lower:
                return 'tensorflow'
            elif 'sklearn' in key_lower or 'sklearn' in value_lower or 'scikit' in value_lower:
                return 'sklearn'
            elif 'keras' in key_lower or 'keras' in value_lower:
                return 'keras'
            elif 'xgboost' in key_lower or 'xgboost' in value_lower:
                return 'xgboost'
            elif 'lightgbm' in key_lower or 'lightgbm' in value_lower:
                return 'lightgbm'
            elif 'catboost' in key_lower or 'catboost' in value_lower:
                return 'catboost'
        
        # Check parameters for framework hints
        for key, value in params.items():
            key_lower = key.lower()
            value_lower = str(value).lower()
            
            if 'pytorch' in key_lower or 'pytorch' in value_lower or 'torch' in value_lower:
                return 'pytorch'
            elif 'tensorflow' in key_lower or 'tensorflow' in value_lower or 'tf' in value_lower:
                return 'tensorflow'
            elif 'sklearn' in key_lower or 'sklearn' in value_lower or 'scikit' in value_lower:
                return 'sklearn'
            elif 'keras' in key_lower or 'keras' in value_lower:
                return 'keras'
            elif 'xgboost' in key_lower or 'xgboost' in value_lower:
                return 'xgboost'
            elif 'lightgbm' in key_lower or 'lightgbm' in value_lower:
                return 'lightgbm'
            elif 'catboost' in key_lower or 'catboost' in value_lower:
                return 'catboost'
        
        return 'unknown'
    
    def _detect_model_framework_from_artifacts(self, model_id: str) -> str:
        """
        Try to detect model framework by examining model artifacts.
        
        Args:
            model_id: Model ID to examine
            
        Returns:
            Detected framework name
        """
        try:
            # Get the logged model details
            model_details = self.client.get_logged_model(model_id)
            artifact_uri = getattr(model_details, 'artifact_uri', '')
            
            if not artifact_uri:
                return 'unknown'
            
            # Try to list artifacts to see what files are in the model
            try:
                artifacts = self.client.list_artifacts(model_id)
                for artifact in artifacts:
                    path_lower = artifact.path.lower()
                    
                    # Check for framework-specific files
                    if 'pytorch' in path_lower or 'torch' in path_lower or '.pt' in path_lower or '.pth' in path_lower:
                        return 'pytorch'
                    elif 'tensorflow' in path_lower or 'tf' in path_lower or '.pb' in path_lower:
                        return 'tensorflow'
                    elif 'sklearn' in path_lower or 'pickle' in path_lower or '.pkl' in path_lower:
                        return 'sklearn'
                    elif 'keras' in path_lower or '.h5' in path_lower or '.hdf5' in path_lower:
                        return 'keras'
                    elif 'xgboost' in path_lower or '.xgb' in path_lower:
                        return 'xgboost'
                    elif 'lightgbm' in path_lower or '.lgb' in path_lower:
                        return 'lightgbm'
                    elif 'catboost' in path_lower or '.cbm' in path_lower:
                        return 'catboost'
                    
                    # Check for MLmodel file content
                    if artifact.path.endswith('MLmodel'):
                        try:
                            # Try to read the MLmodel file to get flavor information
                            import yaml
                            import requests
                            
                            # Try to get the MLmodel file content
                            response = requests.get(f"{artifact_uri}/{artifact.path}")
                            if response.status_code == 200:
                                mlmodel_content = yaml.safe_load(response.text)
                                flavors = mlmodel_content.get('flavors', {})
                                
                                if 'pytorch' in flavors:
                                    return 'pytorch'
                                elif 'tensorflow' in flavors:
                                    return 'tensorflow'
                                elif 'sklearn' in flavors:
                                    return 'sklearn'
                                elif 'keras' in flavors:
                                    return 'keras'
                                elif 'xgboost' in flavors:
                                    return 'xgboost'
                                elif 'lightgbm' in flavors:
                                    return 'lightgbm'
                                elif 'catboost' in flavors:
                                    return 'catboost'
                        except Exception:
                            pass
                            
            except Exception:
                pass
                
        except Exception:
            pass
        
        return 'unknown'
    
    def _get_model_size(self, model_path: str) -> Optional[int]:
        """Get the size of a model in bytes."""
        try:
            if os.path.exists(model_path):
                return os.path.getsize(model_path)
        except Exception:
            pass
        return None
    
    def fetch_run_metric(self, run_id: str, metric_name: str) -> List[Dict[str, Any]]:
        """
        Fetch a specific metric from a run.
        
        Args:
            run_id: ID of the run to fetch the metric from
            metric_name: Name of the metric to fetch
            
        Returns:
            List of metric objects with the structure:
            {
                'key': str,           # Metric name
                'value': float,        # Metric value
                'timestamp': int,      # Timestamp in milliseconds
                'step': int           # Step number (0 for single values)
            }
            
        Raises:
            RuntimeError: If the run is not found or the metric doesn't exist
        """
        try:
            # Get the run
            run = self.client.get_run(run_id)
            
            # Check if the metric exists
            if metric_name not in run.data.metrics:
                raise RuntimeError(f"Metric '{metric_name}' not found in run {run_id}")
            
            # Get the metric history to get all values
            metric_history = self.client.get_metric_history(run_id, metric_name)
            
            if not metric_history:
                # If no history, try to get the current value
                current_value = run.data.metrics.get(metric_name)
                if current_value is not None:
                    # Return single metric object
                    return [{
                        'key': metric_name,
                        'value': float(current_value),
                        'timestamp': int(run.info.start_time) if run.info.start_time else int(time.time() * 1000),
                        'step': 0
                    }]
                else:
                    raise RuntimeError(f"Metric '{metric_name}' has no values in run {run_id}")
            
            # Extract all metric objects from the metric history
            metric_objects = []
            for metric in metric_history:
                metric_objects.append({
                    'key': metric.key,
                    'value': float(metric.value),
                    'timestamp': int(metric.timestamp),
                    'step': int(metric.step) if hasattr(metric, 'step') else 0
                })
            
            return metric_objects
            
        except Exception as e:
            if "not found" in str(e).lower():
                raise RuntimeError(f"Run {run_id} not found")
            elif "metric" in str(e).lower():
                raise RuntimeError(str(e))
            else:
                raise RuntimeError(f"Failed to fetch metric '{metric_name}' from run {run_id}: {e}")

    def get_models(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch logged models from MLflow runs.
        
        Args:
            model_id: Optional model ID to filter by. Can be:
                - Model name (string)
                - Run ID (string)
                - Model version ID (string, format: m-xxxxxxxxx)
                If None, returns all models.
            
        Returns:
            List of model objects with structure similar to model example.json:
            {
                "name": str,
                "creation_timestamp": int,
                "last_updated_timestamp": int,
                "latest_versions": List[Dict],
                "versions": List[Dict]
            }
        """
        models = []
        
        try:
            # Get all experiments to search for logged models
            experiments = self.client.search_experiments()
            
            for experiment in experiments:
                try:
                    # Search for logged models in this experiment
                    logged_models = self.client.search_logged_models(
                        experiment_ids=[experiment.experiment_id]
                    )
                    
                    for logged_model in logged_models:
                        model_name = getattr(logged_model, 'name', 'unknown')
                        source_run_id = getattr(logged_model, 'source_run_id', '')
                        
                        # Filter by model_id if provided
                        should_include = True  # Default to include if no filtering
                        
                        if model_id:
                            # Try different types of filtering
                            should_include = False
                            
                            # 1. Filter by model name
                            if model_name == model_id:
                                should_include = True
                            
                            # 2. Filter by run ID
                            elif source_run_id == model_id:
                                should_include = True
                            
                            # 3. Filter by logged model ID (format: m-xxxxxxxxx)
                            elif model_id.startswith('m-'):
                                # Get the logged model ID from the logged_model object
                                logged_model_id = getattr(logged_model, 'model_id', None)
                                if logged_model_id == model_id:
                                    should_include = True
                            
                            # 4. Filter by creation timestamp (if model_id is a timestamp)
                            else:
                                try:
                                    # Try to parse model_id as timestamp
                                    timestamp_id = int(model_id)
                                    # We'll check this after building model_info
                                    should_include = None  # Special flag for timestamp checking
                                except (ValueError, TypeError):
                                    should_include = False
                            
                            if should_include is False:
                                continue
                        
                        try:
                            model_info = self._build_logged_model_info(logged_model, experiment)
                            
                            # If we're filtering by timestamp, check now
                            if should_include is None and model_id:
                                try:
                                    timestamp_id = int(model_id)
                                    if model_info.get('creation_timestamp') == timestamp_id:
                                        models.append(model_info)
                                except (ValueError, TypeError):
                                    pass
                            else:
                                models.append(model_info)
                                
                        except Exception as e:
                            logger.warning(f"Failed to build info for logged model {model_name}: {e}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Failed to search logged models in experiment {experiment.name}: {e}")
                    continue
                        
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
        
        return models
    
    def _build_logged_model_info(self, logged_model, experiment) -> Dict[str, Any]:
        """
        Build model information dictionary from MLflow logged model object.
        
        Args:
            logged_model: MLflow logged model object
            experiment: MLflow experiment object
            
        Returns:
            Dictionary with model information
        """
        try:
            model_name = getattr(logged_model, 'name', 'unknown')
            source_run_id = getattr(logged_model, 'source_run_id', '')
            
            # Get source run information if available
            source_run = None
            if source_run_id:
                try:
                    run = self.client.get_run(source_run_id)
                    source_run = {
                        "info": {
                            "run_uuid": run.info.run_id,
                            "experiment_id": str(run.info.experiment_id),
                            "run_name": run.info.run_name,
                            "user_id": run.info.user_id,
                            "status": run.info.status,
                            "start_time": run.info.start_time,
                            "end_time": run.info.end_time,
                            "artifact_uri": run.info.artifact_uri,
                            "lifecycle_stage": run.info.lifecycle_stage,
                            "run_id": run.info.run_id
                        },
                        "data": {
                            "tags": [
                                {"key": tag.key, "value": tag.value}
                                for tag in run.data.tags if hasattr(tag, 'key') and hasattr(tag, 'value')
                            ]
                        },
                        "inputs": {},
                        "outputs": {}
                    }
                except Exception as e:
                    logger.warning(f"Failed to get source run for logged model {model_name}: {e}")
            
            # Get creation timestamp
            creation_time = None
            for timestamp_field in ['creation_timestamp_ms', 'creation_timestamp', 'timestamp']:
                creation_time = getattr(logged_model, timestamp_field, None)
                if creation_time:
                    break
            
            if not creation_time and source_run:
                creation_time = source_run["info"]["start_time"]
            
            if not creation_time:
                creation_time = int(time.time() * 1000)
            
            # Build version info
            version_info = {
                "name": model_name,
                "version": "1",  # Logged models don't have versions
                "creation_timestamp": int(creation_time),
                "last_updated_timestamp": int(creation_time),
                "current_stage": "None",
                "description": "",
                "source": getattr(logged_model, 'artifact_uri', ''),
                "run_id": source_run_id,
                "status": "READY",
                "tags": [
                    {"key": "use_case", "value": "ML"},
                    {"key": "trained", "value": "True"}
                ],
                "run_link": "",
                "source_run": source_run,
                "model_id": getattr(logged_model, 'model_id', '') # Added model_id to version_info
            }
            
            return {
                "name": model_name,
                "creation_timestamp": int(creation_time),
                "last_updated_timestamp": int(creation_time),
                "latest_versions": [version_info],
                "versions": [version_info]
            }
            
        except Exception as e:
            logger.warning(f"Failed to build logged model info: {e}")
            return {
                "name": getattr(logged_model, 'name', 'unknown'),
                "creation_timestamp": int(time.time() * 1000),
                "last_updated_timestamp": int(time.time() * 1000),
                "latest_versions": [],
                "versions": []
            }

    def list_federated_jobs(self, experiment_name: Optional[str] = None) -> List[FederatedJobInfo]:
        """
        List all federated learning jobs.
        
        Args:
            experiment_name: Optional experiment name to filter by
            
        Returns:
            List of FederatedJobInfo objects
        """
        federated_jobs = []
        try:
            # Get experiments to search
            if experiment_name:
                experiments = []
                exp = self.client.get_experiment_by_name(experiment_name)
                if exp:
                    experiments.append(exp)
            else:
                experiments = self.client.search_experiments()

            for experiment in experiments:
                # Search for parent runs that are federated learning jobs in this experiment
                runs = self.client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string="tags.run_type = 'federated_parent'"
                )
                for run in runs:
                    try:
                        job_info = self.get_federated_job_info(run.info.run_id)
                        federated_jobs.append(job_info)
                    except Exception as e:
                        logger.warning(f"Failed to get job info for {run.info.run_id}: {e}")

        except Exception as e:
            logger.error(f"Failed to list federated jobs: {e}")

        return federated_jobs


def create_federated_job_fetcher(tracking_uri: Optional[str] = None) -> FederatedJobFetcher:
    """
    Create a new FederatedJobFetcher instance.
    
    Args:
        tracking_uri: MLflow tracking URI (optional)
        
    Returns:
        Configured FederatedJobFetcher
    """
    return FederatedJobFetcher(tracking_uri) 