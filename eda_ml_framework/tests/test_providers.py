"""Transportes HTTP locales de prueba: nunca se presentan como llamadas reales a un LLM."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from pydantic import SecretStr

from agents.providers import JSONLLMProvider, ProviderError


@pytest.fixture
def endpoint():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            schema = body.get("text", {}).get("format", {}).get("name") or body.get(
                "response_format", {}
            ).get("json_schema", {}).get("name")
            value = (
                {
                    "calls": [{"tool": "eda.profile", "parameters_json": "{}", "reason": "TEST"}],
                    "explanation": "Plan del servidor de prueba",
                }
                if schema == "plan"
                else {"explanation": "Explicación del servidor de prueba"}
            )
            text = json.dumps(value)
            payload = (
                {"output": [{"content": [{"type": "output_text", "text": text}]}]}
                if (self.path == "/responses")
                else {"choices": [{"message": {"content": text}}]}
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", requests
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()


@pytest.mark.parametrize("mode", ["responses", "chat_completions"])
def test_http_adapter(endpoint, mode):
    base, requests = endpoint
    provider = JSONLLMProvider(
        model="test-only-model", base_url=base, api_key=SecretStr("test-only-key"), mode=mode
    )
    result = provider.plan("instructions", "question", {"rows": 10}, [])
    assert result.calls[0].tool == "eda.profile"
    assert provider.explain("instructions", {"metric": 1}) == "Explicación del servidor de prueba"
    assert len(requests) == 2
    assert "test-only-key" not in repr(provider)
    if mode == "responses":
        assert requests[0][1]["store"] is False
    assert "rows" in json.dumps(requests[0][1])


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "https://user:secret@example.com",
        "file:///tmp/model",
        "https://example.com?key=secret",
    ],
)
def test_provider_rejects_unsafe_url(url):
    with pytest.raises(ProviderError):
        JSONLLMProvider(model="test", base_url=url, api_key=SecretStr("test"))


def test_provider_network_failure():
    provider = JSONLLMProvider(
        model="test",
        base_url="http://127.0.0.1:1",
        api_key=SecretStr("do-not-log-test-key"),
        timeout=0.1,
    )
    with pytest.raises(ProviderError) as caught:
        provider.plan("x", "x", {}, [])
    assert "do-not-log-test-key" not in str(caught.value)
