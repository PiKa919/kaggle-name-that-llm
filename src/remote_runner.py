"""
Remote Kaggle 2xT4 Execution Bridge via Jupyter WebSocket API.
Allows executing python code or bash commands on the remote Kaggle instance,
streaming outputs, and capturing errors and execution results.
"""

import json
import os
import sys
import time
import urllib.request
import uuid
import websocket

KAGGLE_PROXY_URL = (
    "https://kkb-production.jupyter-proxy.kaggle.net/k/360372997/"
    "eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidHlwIjoiSldUIn0.."
    "LS5DJ5oknkgYjBdfpRWKgw.GMpYVlRNHSFhkV55mjlYgJTO7vIg-XJ0N5APzwdrTCSkf-a-KSexHcCh8rKBsgiK"
    "QadU5AFBQT2u8Rl2-bKfaxtK7klyjsp0frBbeidyvbegnUgT7r9Db_X5uyYjvFq4dIAyl2yOHdmFm7p_a8L65N0"
    "mmuWtYrNxG5kvLDhrS2vF1ar_NhuWupiQxAFUqWnsbrcaenGllAkY5ZUKCh9ei7bw5siw5BLFsRoWq3NkFXXxih"
    "DCLGrZdCV3NUxqUOGb.oinDuFOogCtzdkqdpWeSxQ/proxy"
)

def get_active_kernel_id(proxy_url: str = KAGGLE_PROXY_URL) -> str:
    """Fetch the active kernel ID from the Kaggle Jupyter server."""
    req = urllib.request.Request(f"{proxy_url}/api/kernels", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        kernels = json.loads(resp.read().decode("utf-8"))
    if not kernels:
        create_req = urllib.request.Request(
            f"{proxy_url}/api/kernels",
            data=json.dumps({"name": "python3"}).encode("utf-8"),
            headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(create_req, timeout=10) as resp:
            new_kernel = json.loads(resp.read().decode("utf-8"))
            return new_kernel["id"]
    return kernels[0]["id"]


def interrupt_remote_kernel(proxy_url: str = KAGGLE_PROXY_URL) -> bool:
    """Send an interrupt signal to stop running execution on the Kaggle kernel."""
    kid = get_active_kernel_id(proxy_url)
    req = urllib.request.Request(
        f"{proxy_url}/api/kernels/{kid}/interrupt",
        method="POST",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status in (200, 204)


def run_remote_code(code: str, timeout: int = 1200, proxy_url: str = KAGGLE_PROXY_URL, verbose: bool = True) -> dict:
    """
    Execute python or shell code (using ! prefix) on the remote Kaggle kernel.
    Returns dict with stdout, stderr, execution_result, and exit_code/error.
    """
    kernel_id = get_active_kernel_id(proxy_url)
    ws_base = proxy_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_base}/api/kernels/{kernel_id}/channels"

    ws = websocket.create_connection(ws_url, timeout=timeout)
    msg_id = uuid.uuid4().hex
    req_msg = {
        "header": {
            "msg_id": msg_id,
            "username": "user",
            "session": uuid.uuid4().hex,
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
        },
        "channel": "shell",
    }

    ws.send(json.dumps(req_msg))

    stdout_chunks = []
    stderr_chunks = []
    execute_result = None
    error_info = None

    start_time = time.time()
    try:
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Remote execution timed out after {timeout} seconds")
            
            raw_msg = ws.recv()
            if not raw_msg:
                continue
            res = json.loads(raw_msg)
            msg_type = res.get("msg_type")
            content = res.get("content", {})
            parent_id = res.get("parent_header", {}).get("msg_id")

            if parent_id != msg_id:
                continue

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_chunks.append(text)
                    if verbose:
                        sys.stdout.write(text)
                        sys.stdout.flush()
                else:
                    stderr_chunks.append(text)
                    if verbose:
                        sys.stderr.write(text)
                        sys.stderr.flush()
            elif msg_type == "execute_result":
                res_text = content.get("data", {}).get("text/plain", "")
                execute_result = res_text
                if verbose:
                    print(res_text)
            elif msg_type == "error":
                error_info = content
                err_text = "\n".join(content.get("traceback", []))
                if verbose:
                    print(f"Execution Error:\n{err_text}", file=sys.stderr)
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break
    finally:
        ws.close()

    return {
        "stdout": "".join(stdout_chunks),
        "stderr": "".join(stderr_chunks),
        "execute_result": execute_result,
        "error": error_info,
        "success": error_info is None,
    }

def upload_file(local_path: str, remote_path: str, proxy_url: str = KAGGLE_PROXY_URL) -> bool:
    """Upload a local text or binary file to the remote Kaggle filesystem."""
    import base64
    from pathlib import Path

    p = Path(local_path)
    data = p.read_bytes()
    b64_data = base64.b64encode(data).decode("utf-8")

    code = f"""
import base64, os
os.makedirs(os.path.dirname("{remote_path}"), exist_ok=True)
with open("{remote_path}", "wb") as f:
    f.write(base64.b64decode("{b64_data}"))
print("Uploaded {remote_path} ({len(data)} bytes)")
"""
    res = run_remote_code(code, verbose=False)
    return res["success"]


def sync_project_to_kaggle(directories: list = ["src", "experiments"]) -> None:
    """Sync specified local directories to /kaggle/working/ on remote."""
    from pathlib import Path
    for d in directories:
        dp = Path(d)
        if not dp.exists():
            continue
        for file in dp.rglob("*.py"):
            rel_path = file.relative_to(".")
            remote_path = f"/kaggle/working/{rel_path}"
            print(f"Syncing {file} -> {remote_path}...")
            upload_file(str(file), remote_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--sync":
            sync_project_to_kaggle()
        else:
            run_remote_code(sys.argv[1])
    else:
        run_remote_code("!nvidia-smi")
