"""
CLI to run an API to serve other functional
"""
import asyncio
import logging
import traceback
from typing import Callable
import janus
from pathlib import Path
from fastapi import APIRouter, FastAPI, Request, WebSocket
from starlette.middleware import Middleware
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.routing import Route
import os
from starlette.middleware.cors import CORSMiddleware
from starlette.types import Scope

import transformers

from axolotl.cli import (
    check_accelerate_default_config,
    check_user_token,
    load_cfg,
    load_datasets,
    prepare_cfg,
    print_axolotl_text_art,
)
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils.dict import DictDefault

LOG = logging.getLogger("axolotl.cli.train")

router = APIRouter()
print_axolotl_text_art()
check_accelerate_default_config()
check_user_token()
app = FastAPI(title="Axolotl API", openapi_url="/api/v1/openapi.json")
# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

@app.websocket("/ws")
async def do_train(websocket: WebSocket):
    await websocket.accept()
    queue = janus.Queue()
    send_json = queue.sync_q.put

    data = await websocket.receive_json()
    fut = asyncio.get_event_loop().run_in_executor(
        None, sync_processing_loop, send_json, data
    )
    print(fut)

    while True:
        val = await queue.async_q.get()
        if val is None:
            break
        await websocket.send_json(val)

    await fut

def sync_processing_loop(send_json: Callable[[dict], None], data: str):
    try:
        request = DictDefault(data)
        print(request)
        parsed_cfg = prepare_cfg(request)
        parsed_cfg["websocket"] = send_json
        parsed_cfg["do_websockets"] = True

        parser = transformers.HfArgumentParser((TrainerCliArgs))
        parsed_cli_args, _ = parser.parse_args_into_dataclasses(
            return_remaining_strings=True
        )
        dataset_meta = load_datasets(cfg=parsed_cfg, cli_args=parsed_cli_args)
        train(cfg=parsed_cfg, cli_args=parsed_cli_args, dataset_meta=dataset_meta)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
    finally:
        send_json(None)

@router.get("/api/v1/health")
def health():
    return Response("OK")


