"""
ComfyUI WebSocket client — talks directly to ComfyUI server.
No external API dependency.
"""
import json
import uuid
import copy
import urllib.request
import urllib.parse
import logging
from typing import Optional

import websocket
import requests

logger = logging.getLogger(__name__)


class ComfyUIClient:
    def __init__(self, server_address: str):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self._ws: Optional[websocket.WebSocket] = None

    def connect(self):
        ws_url = f"ws://{self.server_address}/ws?clientId={self.client_id}"
        logger.info(f"Connecting to ComfyUI: {ws_url}")
        self._ws = websocket.WebSocket()
        self._ws.connect(ws_url)

    def disconnect(self):
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def upload_image(self, filename: str, image_bytes: bytes, content_type: str = "image/png") -> str:
        """Upload image to ComfyUI input folder. Returns stored filename."""
        url = f"http://{self.server_address}/upload/image"
        resp = requests.post(
            url,
            files={"image": (filename, image_bytes, content_type)},
            data={"overwrite": "true", "type": "input"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        name = data.get("name")
        subfolder = data.get("subfolder", "")
        if not name:
            raise RuntimeError(f"Upload failed: {data}")
        return f"{subfolder}/{name}" if subfolder else name

    def execute_workflow(self, workflow: dict) -> dict[str, list[bytes]]:
        """Execute workflow, wait for completion, return output images by node id."""
        if not self._ws:
            raise RuntimeError("Not connected. Call connect() first.")

        # Queue prompt
        payload = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode()
        req = urllib.request.Request(
            f"http://{self.server_address}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        result = json.loads(urllib.request.urlopen(req).read())
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"Failed to queue: {result}")

        logger.info(f"Queued workflow: {prompt_id}")

        # Wait for completion via WebSocket
        while True:
            out = self._ws.recv()
            if isinstance(out, str):
                msg = json.loads(out)
                msg_type = msg.get("type")
                data = msg.get("data", {})

                if msg_type == "executing":
                    if data.get("node") is None and data.get("prompt_id") == prompt_id:
                        break
                elif msg_type == "execution_error":
                    raise RuntimeError(f"ComfyUI error: {data.get('exception_message', 'Unknown')}")

        # Fetch output images from history
        url = f"http://{self.server_address}/history/{prompt_id}"
        history = json.loads(urllib.request.urlopen(url).read())

        if prompt_id not in history:
            raise RuntimeError(f"No history for {prompt_id}")

        outputs = history[prompt_id].get("outputs", {})
        output_images = {}

        for node_id, node_output in outputs.items():
            if "images" not in node_output:
                continue
            images = []
            for img_info in node_output["images"]:
                params = urllib.parse.urlencode({
                    "filename": img_info["filename"],
                    "subfolder": img_info.get("subfolder", ""),
                    "type": img_info.get("type", "output"),
                })
                img_url = f"http://{self.server_address}/view?{params}"
                images.append(urllib.request.urlopen(img_url).read())
            if images:
                output_images[node_id] = images

        logger.info(f"Got {sum(len(v) for v in output_images.values())} images from {len(output_images)} nodes")
        return output_images

    @staticmethod
    def prepare_outfit_workflow(
        workflow_template: dict,
        input_image_name: str,
        prompt: str,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> dict:
        """Inject parameters into the outfit workflow template.

        Node 108: input character image
        Node 16:  positive prompt (outfit change instruction)
        Node 106: KSampler seed
        Node 103: EmptyLatentImage dimensions
        """
        wf = copy.deepcopy(workflow_template)

        # Node 108: input image
        if "108" in wf:
            wf["108"]["inputs"]["image"] = input_image_name

        # Node 16: outfit edit prompt
        if "16" in wf:
            wf["16"]["inputs"]["positive"] = prompt

        # Node 106: seed
        if seed is not None and "106" in wf:
            wf["106"]["inputs"]["seed"] = seed

        # Node 103: dimensions
        if width and height and "103" in wf:
            wf["103"]["inputs"]["width"] = width
            wf["103"]["inputs"]["height"] = height

        return wf

    @staticmethod
    def prepare_pose_workflow(
        workflow_template: dict,
        source_image_name: str,
        pose_image_name: str,
        seed: Optional[int] = None,
    ) -> dict:
        """Inject parameters into the pose workflow template.

        Node 109: source character image (image1 — keep this person's identity)
        Node 170: pose reference image (image2 — copy this pose)
        Node 3:   KSampler seed
        """
        wf = copy.deepcopy(workflow_template)

        # Node 109: character source image
        if "109" in wf:
            wf["109"]["inputs"]["image"] = source_image_name

        # Node 170: pose reference image
        if "170" in wf:
            wf["170"]["inputs"]["image"] = pose_image_name

        # Node 3: seed
        if seed is not None and "3" in wf:
            wf["3"]["inputs"]["seed"] = seed

        return wf

    def check_status(self) -> dict:
        try:
            url = f"http://{self.server_address}/system_stats"
            return json.loads(urllib.request.urlopen(url, timeout=5).read())
        except Exception as e:
            return {"error": str(e)}
