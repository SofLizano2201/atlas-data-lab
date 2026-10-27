"""Proveedor intercambiable. Solo la planificación y explicación requieren LLM."""

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import Field, SecretStr

from contracts.models import Contract


class PlannedCall(Contract):
    """Argumentos JSON permanecen datos; jamás se evalúan como código."""

    tool: str
    parameters_json: str
    reason: str


class Plan(Contract):
    """Salida estructurada del planificador; el executor aplica permisos aparte."""

    calls: list[PlannedCall] = Field(max_length=3)
    explanation: str


class Explanation(Contract):
    """Narrativa separada de métricas deterministas."""

    explanation: str


class ProviderError(ValueError):
    """Error del proveedor sin filtrar credenciales o cuerpos de respuesta."""


class LLMProvider(Protocol):
    """Implementar este protocolo para conectar otro proveedor sin cambiar herramientas."""

    name: str
    available: bool

    def plan(
        self, instructions: str, question: str, context: dict[str, Any], tools: list[dict[str, Any]]
    ) -> Plan: ...

    def explain(self, instructions: str, evidence: dict[str, Any]) -> str: ...


class NoProvider:
    """Ausencia explícita de IA; no inventa planes ni respuestas conversacionales."""

    name = "sin_configurar"
    available = False

    def plan(self, *args: Any, **kwargs: Any) -> Plan:
        raise ProviderError("Asistente conversacional no disponible: configure proveedor y modelo.")

    def explain(self, *args: Any, **kwargs: Any) -> str:
        raise ProviderError("No existe proveedor LLM configurado.")


class NoRedirect(HTTPRedirectHandler):
    """Nunca reenviar Authorization a un destino redirigido."""

    def redirect_request(
        self, req: Any, fp: Any, code: Any, msg: Any, headers: Any, newurl: str
    ) -> None:
        return None


@dataclass(repr=False)
class JSONLLMProvider:
    """Responses API o endpoint compatible con Chat Completions y JSON Schema."""

    model: str
    base_url: str
    api_key: SecretStr
    mode: str = "responses"
    timeout: float = 40
    available: bool = True

    @property
    def name(self) -> str:
        return f"{self.mode}:{self.model}"

    def __post_init__(self) -> None:
        url = urlparse(self.base_url)
        if url.username or url.password or url.query or url.fragment or not url.hostname:
            raise ProviderError("URL del proveedor inválida.")
        if url.scheme != "https" and not (
            url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ProviderError("Use HTTPS o un proveedor local en localhost.")
        if self.mode not in {"responses", "chat_completions"} or not self.model:
            raise ProviderError("Configure modelo y protocolo responses/chat_completions.")

    def _json(self, instructions: str, data: dict[str, Any], contract: type[Contract]) -> Any:
        schema = contract.model_json_schema()
        structured = {
            "type": "json_schema",
            "name": contract.__name__.lower(),
            "strict": True,
            "schema": schema,
        }
        message = json.dumps(data, ensure_ascii=False, allow_nan=False)
        if len(message) > 60000:
            raise ProviderError("Contexto demasiado grande para el proveedor.")
        if self.mode == "responses":
            path = "/responses"
            body = {
                "model": self.model,
                "instructions": instructions,
                "input": message,
                "store": False,
                "max_output_tokens": 2500,
                "text": {"format": structured},
            }
        else:
            path = "/chat/completions"
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": message},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {k: v for k, v in structured.items() if k != "type"},
                },
                "max_tokens": 2500,
            }
        request = Request(
            self.base_url.rstrip("/") + path,
            data=json.dumps(body).encode(),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            },
        )
        try:
            with build_opener(NoRedirect()).open(request, timeout=self.timeout) as response:
                raw = response.read(1_000_001)
            if len(raw) > 1_000_000:
                raise ProviderError("Respuesta del proveedor demasiado grande.")
            payload = json.loads(raw)
            if self.mode == "responses":
                texts = [
                    c["text"]
                    for item in payload.get("output", [])
                    for c in item.get("content", [])
                    if c.get("type") == "output_text"
                ]
                text = "".join(texts)
            else:
                text = payload["choices"][0]["message"]["content"]
            return contract.model_validate_json(text)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise ProviderError(
                "No se pudo contactar al proveedor. Revise conexión, permisos y modelo."
            ) from exc
        except (ValueError, KeyError, TypeError, IndexError, RecursionError) as exc:
            raise ProviderError(
                "El proveedor rechazó o no devolvió el contrato JSON requerido."
            ) from exc

    def plan(
        self, instructions: str, question: str, context: dict[str, Any], tools: list[dict[str, Any]]
    ) -> Plan:
        return self._json(
            instructions,
            {"request": question, "untrusted_dataset_metadata": context, "allowed_tools": tools},
            Plan,
        )

    def explain(self, instructions: str, evidence: dict[str, Any]) -> str:
        return self._json(
            instructions + " Explica solo la evidencia calculada. No inventes cifras. "
            "Separa hechos, limitaciones y recomendaciones.",
            evidence,
            Explanation,
        ).explanation


def configured_provider(config: dict[str, str] | None = None) -> LLMProvider:
    """Configuración del servidor; jamás guardar claves en archivos del repositorio."""
    config = os.environ if config is None else config
    mode = config.get("ATLAS_LLM_PROVIDER", "")
    model = config.get("ATLAS_LLM_MODEL", "")
    if not mode or not model:
        return NoProvider()
    key = config.get("ATLAS_LLM_API_KEY", "")
    base = config.get("ATLAS_LLM_BASE_URL", "https://api.openai.com/v1")
    if not key and urlparse(base).hostname not in {"localhost", "127.0.0.1", "::1"}:
        return NoProvider()
    return JSONLLMProvider(model=model, base_url=base, api_key=SecretStr(key), mode=mode)
