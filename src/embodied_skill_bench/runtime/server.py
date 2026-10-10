from __future__ import annotations

import hmac
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .session import EpisodeSession


class RuntimeHttpServer:
    def __init__(
        self,
        session: EpisodeSession,
        token: str,
        host: str = "127.0.0.1",
        port: int = 0,
        allow_pixel_to_world: bool = True,
    ):
        self.session = session
        self.token = token
        self.allow_pixel_to_world = bool(allow_pixel_to_world)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "EmbodiedSkillBench/1.0"

            def log_message(self, *_args) -> None:
                return None

            def _authorized(self) -> bool:
                supplied = self.headers.get("Authorization", "")
                expected = f"Bearer {owner.token}"
                return hmac.compare_digest(supplied, expected)

            def _send(self, status: int, value: Any) -> None:
                body = json.dumps(value, ensure_ascii=False, sort_keys=True).encode(
                    "utf-8"
                )
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1024 * 1024:
                    raise ValueError("request body exceeds 1 MiB")
                data = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(data, dict):
                    raise ValueError("request must be a JSON object")
                return data

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/health":
                    self._send(HTTPStatus.OK, {"ok": True})
                    return
                if not self._authorized():
                    self._send(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorized"})
                    return
                if path == "/v1/observation/latest":
                    self._send(
                        HTTPStatus.OK,
                        owner.session.latest(
                            acknowledge=True,
                            agent_turn_id=self.headers.get("X-ESB-Agent-Turn-ID"),
                        ),
                    )
                elif path == "/v1/summary":
                    self._send(HTTPStatus.OK, owner.session.summary())
                else:
                    self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                if not self._authorized():
                    self._send(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorized"})
                    return
                try:
                    request = self._read_json()
                    path = urlparse(self.path).path
                    if path == "/v1/execute":
                        self._send(HTTPStatus.OK, owner.session.execute(request))
                    elif path == "/v1/native-tool":
                        execute_native = getattr(owner.session, "execute_native_tool", None)
                        if not callable(execute_native):
                            self._send(
                                HTTPStatus.NOT_FOUND,
                                {"ok": False, "error": "native typed tools are unavailable"},
                            )
                        else:
                            self._send(HTTPStatus.OK, execute_native(request))
                    elif path == "/v1/native-skill":
                        execute_native_skill = getattr(
                            owner.session, "execute_native_skill", None
                        )
                        if not callable(execute_native_skill):
                            self._send(
                                HTTPStatus.NOT_FOUND,
                                {"ok": False, "error": "native selected skills are unavailable"},
                            )
                        else:
                            self._send(HTTPStatus.OK, execute_native_skill(request))
                    elif path == "/v1/pixel-to-world" and owner.allow_pixel_to_world:
                        self._send(HTTPStatus.OK, pixel_to_world(owner.session, request))
                    elif path == "/v1/finish":
                        self._send(
                            HTTPStatus.OK,
                            owner.session.finish(str(request.get("reason", "agent_finished"))),
                        )
                    else:
                        self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                except (ValueError, KeyError, IndexError, FileNotFoundError) as exc:
                    self._send(
                        HTTPStatus.BAD_REQUEST,
                        {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}},
                    )
                except BaseException as exc:
                    self._send(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}},
                    )

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self.host, self.port = self._httpd.server_address[:2]
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> "RuntimeHttpServer":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="embodied-skill-runtime",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None


def pixel_to_world(session: EpisodeSession, request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    camera = str(request.get("camera", "agentview"))
    resolution = str(request.get("resolution", "high"))
    row = int(request["row"])
    col = int(request["col"])
    step = request.get("step")
    if step is None:
        observation = session.store.latest()
    else:
        path = session.store.steps_dir / f"{int(step):04d}" / "observation.json"
        observation = json.loads(path.read_text(encoding="utf-8"))
    camera_data = observation["artifacts"]["cameras"][camera]
    key = "high_world" if resolution == "high" else "world"
    world_path = camera_data[key]["path"]
    world = np.load(world_path, mmap_mode="r")
    if row < 0 or col < 0 or row >= world.shape[0] or col >= world.shape[1]:
        raise IndexError(
            f"pixel ({row}, {col}) is outside {world.shape[1]}x{world.shape[0]}"
        )
    xyz = [float(value) for value in world[row, col, :3]]
    return {
        "ok": True,
        "step_index": observation["step_index"],
        "camera": camera,
        "resolution": resolution,
        "pixel": {"row": row, "col": col},
        "world_xyz_m": xyz,
        "world_map": world_path,
    }
