from __future__ import annotations

# TODO: use pydantic for other modules in Qlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

try:
    from pydantic_settings import (  # type: ignore[import-not-found]
        BaseSettings,
        EnvSettingsSource,
        PydanticBaseSettingsSource,
        SettingsConfigDict,
    )

    _HAS_PYDANTIC_SETTINGS = True
except Exception:  # pragma: no cover
    BaseSettings = object  # type: ignore[assignment]
    EnvSettingsSource = object  # type: ignore[assignment]
    PydanticBaseSettingsSource = object  # type: ignore[assignment]

    class SettingsConfigDict(dict):  # type: ignore[misc]
        pass

    _HAS_PYDANTIC_SETTINGS = False


if _HAS_PYDANTIC_SETTINGS:

    class ExtendedEnvSettingsSource(EnvSettingsSource):
        def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
            prefixes = [self.config.get("env_prefix", "")]
            if hasattr(self.settings_cls, "__bases__"):
                for base in self.settings_cls.__bases__:
                    if hasattr(base, "model_config"):
                        parent_prefix = base.model_config.get("env_prefix")
                        if parent_prefix and parent_prefix not in prefixes:
                            prefixes.append(parent_prefix)
            for prefix in prefixes:
                self.env_prefix = prefix
                env_val, field_key, value_is_complex = super().get_field_value(field, field_name)
                if env_val is not None:
                    return env_val, field_key, value_is_complex
            return super().get_field_value(field, field_name)

    class ExtendedSettingsConfigDict(SettingsConfigDict, total=False): ...

    class ExtendedBaseSettings(BaseSettings):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,  # noqa
            env_settings: PydanticBaseSettingsSource,  # noqa
            dotenv_settings: PydanticBaseSettingsSource,  # noqa
            file_secret_settings: PydanticBaseSettingsSource,  # noqa
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            return (ExtendedEnvSettingsSource(settings_cls),)

else:

    class ExtendedSettingsConfigDict(dict):
        pass

    def _parse_env_value(raw: str, target_type: type) -> Any:
        if target_type is bool:
            return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
        if target_type is int:
            return int(raw)
        if target_type is float:
            return float(raw)
        if target_type is Path:
            return Path(raw)
        return raw

    class ExtendedBaseSettings:
        def __init__(self, **kwargs: Any) -> None:
            merged: dict[str, Any] = {}
            for cls in reversed(self.__class__.mro()):
                merged.update(getattr(cls, "__dict__", {}))

            annotations: dict[str, Any] = {}
            for cls in reversed(self.__class__.mro()):
                annotations.update(getattr(cls, "__annotations__", {}))

            env_prefix = ""
            config_cls = getattr(self.__class__, "Config", None)
            if config_cls is not None:
                env_prefix = getattr(config_cls, "env_prefix", "") or ""
            if not env_prefix:
                env_prefix = getattr(self.__class__, "model_config", {}).get("env_prefix", "") or ""

            for name, typ in annotations.items():
                if name.startswith("_"):
                    continue
                if name in kwargs:
                    setattr(self, name, kwargs[name])
                    continue
                env_key = f"{env_prefix}{name}".upper()
                if env_key in os.environ:
                    raw = os.environ[env_key]
                    target_type = typ if isinstance(typ, type) else str
                    try:
                        setattr(self, name, _parse_env_value(raw, target_type))
                    except Exception:
                        setattr(self, name, raw)
                    continue
                if name in merged:
                    setattr(self, name, merged[name])

        @classmethod
        def settings_customise_sources(cls, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
            return ()


class RDAgentSettings(ExtendedBaseSettings):
    # TODO: (xiao) I think LLMSetting may be a better name.
    # TODO: (xiao) I think most of the config should be in oai.config
    # Log configs
    # TODO: (xiao) think it can be a separate config.
    log_trace_path: str | None = None

    # azure document intelligence configs
    azure_document_intelligence_key: str = ""
    azure_document_intelligence_endpoint: str = ""
    # factor extraction conf
    max_input_duplicate_factor_group: int = 300
    max_output_duplicate_factor_group: int = 20
    max_kmeans_group_number: int = 40

    # workspace conf
    workspace_path: Path = Path.cwd() / "git_ignore_folder" / "RD-Agent_workspace"

    # multi processing conf
    multi_proc_n: int = 1

    # pickle cache conf
    cache_with_pickle: bool = True  # whether to use pickle cache
    pickle_cache_folder_path_str: str = str(
        Path.cwd() / "pickle_cache/",
    )  # the path of the folder to store the pickle cache
    use_file_lock: bool = (
        True  # when calling the function with same parameters, whether to use file lock to avoid
        # executing the function multiple times
    )


RD_AGENT_SETTINGS = RDAgentSettings()
