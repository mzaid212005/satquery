import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field


class ParameterSpec(BaseModel):
    type: str
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    allowed_values: Optional[List[Any]] = None


class ExpectedInputs(BaseModel):
    min_images: int = 1
    max_images: int = 1
    modalities: Optional[List[str]] = None
    required_modalities: Optional[List[str]] = None
    input_relationship: Optional[str] = None  # "single", "bi_temporal", "cross_modal"
    requires_co_registration: bool = False
    requires_query: bool = False
    supported_formats: List[str] = Field(default_factory=lambda: ["tiff", "geotiff", "png", "jpeg"])


class ModelEntry(BaseModel):
    task_types: List[str]
    name: str
    module: str
    class_name: str
    description: str
    expected_inputs: ExpectedInputs
    permitted_parameters: Dict[str, ParameterSpec] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)


class ToolRegistry(BaseModel):
    version: str
    registry_name: str
    description: str
    models: Dict[str, ModelEntry]


class RegistryLoader:
    """Loads, validates, and manages the specialist model registry."""

    def __init__(self, registry_path: Optional[Union[str, Path]] = None):
        if registry_path is None:
            registry_path = Path(__file__).resolve().parent.parent / "registry.yaml"
        self.registry_path = Path(registry_path)
        self.registry: ToolRegistry = self._load_registry()
        self._instantiated_models: Dict[str, Any] = {}

    def _load_registry(self) -> ToolRegistry:
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry file not found at: {self.registry_path}")

        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return ToolRegistry(**data)

    def reload(self) -> None:
        """Reloads the registry configuration from disk."""
        self.registry = self._load_registry()
        self._instantiated_models.clear()

    def get_model_entry_by_task(self, task_type: str) -> Optional[ModelEntry]:
        """Finds the model entry configured for a given task type."""
        for entry in self.registry.models.values():
            if task_type.lower() in [t.lower() for t in entry.task_types]:
                return entry
        return None

    def get_model_by_name(self, model_key: str) -> Optional[ModelEntry]:
        return self.registry.models.get(model_key)

    def list_available_tasks(self) -> List[str]:
        tasks = []
        for entry in self.registry.models.values():
            tasks.extend(entry.task_types)
        return sorted(list(set(tasks)))

    def validate_and_filter_parameters(
        self, model_entry: ModelEntry, raw_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validates parameters against permitted whitelist in registry schema.
        Applies defaults for omitted parameters and rejects non-whitelisted parameters.
        """
        sanitized: Dict[str, Any] = {}
        raw = raw_params or {}

        # 1. Apply defaults
        for param_name, spec in model_entry.permitted_parameters.items():
            sanitized[param_name] = spec.default

        # 2. Check and validate provided params
        for key, value in raw.items():
            if key not in model_entry.permitted_parameters:
                # Reject parameter not in whitelist to enforce strict auditable contract
                continue

            spec = model_entry.permitted_parameters[key]

            # Type and range checking
            if spec.type == "float" or spec.type == "integer":
                try:
                    num_val = float(value) if spec.type == "float" else int(value)
                    if spec.min is not None and num_val < spec.min:
                        num_val = spec.min
                    if spec.max is not None and num_val > spec.max:
                        num_val = spec.max
                    sanitized[key] = num_val
                except (ValueError, TypeError):
                    pass
            elif spec.type == "string":
                str_val = str(value)
                if spec.allowed_values and str_val not in spec.allowed_values:
                    # Fallback to default
                    str_val = spec.default
                sanitized[key] = str_val
            elif spec.type == "boolean":
                if isinstance(value, bool):
                    sanitized[key] = value
                elif isinstance(value, str):
                    sanitized[key] = value.lower() in ("true", "1", "yes")
            else:
                sanitized[key] = value

        return sanitized

    def get_model_instance(self, model_entry: ModelEntry) -> Any:
        """Dynamically imports and returns a singleton instance of the specialist model."""
        key = f"{model_entry.module}.{model_entry.class_name}"
        if key not in self._instantiated_models:
            mod = importlib.import_module(model_entry.module)
            cls = getattr(mod, model_entry.class_name)
            self._instantiated_models[key] = cls()
        return self._instantiated_models[key]


# Global singleton instance
registry_loader = RegistryLoader()
