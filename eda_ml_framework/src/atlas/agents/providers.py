"""Providers for the modular ATLAS assistant system."""

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import SecretStr

from atlas.core.schemas import (
    Contract,
)

from .contracts import Explanation, Plan, PlannedCall


class ProviderError(ValueError):
    """Define the ProviderError contract."""


class LLMProvider(Protocol):
    """Define the LLMProvider contract."""

    name: str
    available: bool

    def plan(
        self, instructions: str, question: str, context: dict[str, Any], tools: list[dict[str, Any]]
    ) -> Plan: ...

    def explain(self, instructions: str, evidence: dict[str, Any]) -> str: ...


class NoProvider:
    """Define the NoProvider contract."""

    name = "unconfigured"
    available = False

    def plan(self, *args: Any, **kwargs: Any) -> Plan:
        raise ProviderError("Asistente conversacional no disponible: configure proveedor y modelo.")

    def explain(self, *args: Any, **kwargs: Any) -> str:
        raise ProviderError("No existe proveedor LLM configurado.")


class NoRedirect(HTTPRedirectHandler):
    """Define the NoRedirect contract."""

    def redirect_request(
        self, req: Any, fp: Any, code: Any, msg: Any, headers: Any, newurl: str
    ) -> None:
        return None


@dataclass(repr=False)
class JSONLLMProvider:
    """Define the JSONLLMProvider contract."""

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
        if url.username or url.password or url.query or url.fragment or (not url.hostname):
            raise ProviderError("URL del proveedor inválida.")
        if url.scheme != "https" and (
            not (url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"})
        ):
            raise ProviderError("Use HTTPS o un proveedor local en localhost.")
        if self.mode not in {"responses", "responses_tools", "chat_completions"} or not self.model:
            raise ProviderError(
                "Configure modelo y protocolo responses_tools, responses o chat_completions."
            )

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Use a bounded transport that never redirects credentials."""
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
            if not isinstance(payload, dict):
                raise ProviderError("El proveedor no devolvió un objeto JSON.")
            return payload
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            raise ProviderError(
                "No se pudo obtener una respuesta válida del proveedor. Revisa conexión y configuración."
            ) from exc

    def probe(self) -> dict[str, Any]:
        """Check an explicitly requested loopback model catalog without sending dataset context."""
        if urlparse(self.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
            return {
                "available": False,
                "message": "La comprobación de disponibilidad está limitada al modelo local; el proveedor remoto se valida al solicitarlo.",
            }
        try:
            request = Request(self.base_url.rstrip("/") + "/models")
            with build_opener(NoRedirect()).open(request, timeout=2) as response:
                raw = response.read(1_000_001)
            if len(raw) > 1_000_000:
                raise ValueError("Model catalog too large")
            payload = json.loads(raw)
            found = any(item.get("id") == self.model for item in payload.get("data", []))
            return {
                "available": found,
                "message": "Modelo local disponible."
                if found
                else "El servidor responde, pero el modelo configurado no está instalado. El motor local de políticas sigue disponible.",
            }
        except (ValueError, TypeError, AttributeError, URLError, OSError, TimeoutError):
            return {
                "available": False,
                "message": "Modelo local no disponible. Inicia el servidor e instala el modelo; puedes continuar con las políticas offline.",
            }

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
        if self.mode in {"responses", "responses_tools"}:
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
        try:
            payload = self._post(path, body)
            if self.mode in {"responses", "responses_tools"}:
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
        if self.mode == "responses_tools":
            lookup = {tool["name"].replace(".", "__"): tool for tool in tools}
            functions = [
                {
                    "type": "function",
                    "name": name,
                    "description": tool["description"]
                    + " Arguments must match: "
                    + json.dumps(tool["parameters"]),
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "parameters_json": {
                                "type": "string",
                                "description": "JSON object validated against the capability's input schema.",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Concise Spanish decision summary grounded in available evidence.",
                            },
                        },
                        "required": ["parameters_json", "reason"],
                        "additionalProperties": False,
                    },
                }
                for name, tool in lookup.items()
            ]
            message = json.dumps(
                {"request": question, "untrusted_dataset_metadata": context},
                ensure_ascii=False,
                allow_nan=False,
            )
            if len(message) + len(json.dumps(functions)) > 60_000:
                raise ProviderError("Contexto demasiado grande para el proveedor.")
            payload = self._post(
                "/responses",
                {
                    "model": self.model,
                    "instructions": instructions,
                    "input": message,
                    "store": False,
                    "tools": functions,
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                    "max_output_tokens": 2500,
                },
            )
            calls, texts = [], []
            try:
                for item in payload.get("output", []):
                    if item.get("type") == "function_call":
                        if item["name"] not in lookup:
                            raise ProviderError("El proveedor pidió una herramienta no registrada.")
                        arguments = json.loads(item["arguments"])
                        calls.append(PlannedCall(tool=lookup[item["name"]]["name"], **arguments))
                    else:
                        texts.extend(
                            content["text"]
                            for content in item.get("content", [])
                            if content.get("type") == "output_text"
                        )
                if not calls and not texts:
                    raise ProviderError(
                        "El proveedor no devolvió una respuesta ni llamadas válidas."
                    )
                return Plan(calls=calls, explanation=" ".join(texts))
            except (ValueError, TypeError, KeyError) as exc:
                raise ProviderError(
                    "La llamada de herramienta del proveedor no cumple el contrato."
                ) from exc
        return self._json(
            instructions,
            {"request": question, "untrusted_dataset_metadata": context, "allowed_tools": tools},
            Plan,
        )

    def explain(self, instructions: str, evidence: dict[str, Any]) -> str:
        return self._json(
            instructions
            + " Explica solo la evidencia calculada. No inventes cifras. Separa hechos, limitaciones y recomendaciones.",
            evidence,
            Explanation,
        ).explanation


def configured_provider(config: dict[str, str] | None = None) -> LLMProvider:
    """Provide configured provider using validated inputs."""
    config = os.environ if config is None else config
    mode = config.get("ATLAS_LLM_PROVIDER", "")
    model = config.get("ATLAS_LLM_MODEL", "")
    if not mode or not model:
        return NoProvider()
    key = config.get("ATLAS_LLM_API_KEY", "")
    base = config.get(
        "ATLAS_LLM_BASE_URL",
        "http://127.0.0.1:11434/v1" if mode == "ollama" else "https://api.openai.com/v1",
    ) or ("http://127.0.0.1:11434/v1" if mode == "ollama" else "https://api.openai.com/v1")
    if mode == "ollama":
        if urlparse(base).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ProviderError("El modo Ollama requiere una dirección local de loopback.")
        mode = "chat_completions"
    if not key and urlparse(base).hostname not in {"localhost", "127.0.0.1", "::1"}:
        return NoProvider()
    return JSONLLMProvider(model=model, base_url=base, api_key=SecretStr(key), mode=mode)
