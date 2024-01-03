"""
CLI to run an API to serve other functional
"""
import logging
from pathlib import Path
from fastapi import APIRouter, FastAPI, Request
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

LOG = logging.getLogger("axolotl.cli.train")

    
router = APIRouter()

@router.post("/api/v1/train")
def do_train(request: Request):
    parsed_cfg = prepare_cfg(request.json())
    parsed_cfg["api_host"] = request.client.host
    parsed_cfg["api_port"] = request.client.port
    parsed_cfg["do_websockets"] = True

    parser = transformers.HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    dataset_meta = load_datasets(cfg=parsed_cfg, cli_args=parsed_cli_args)
    train(cfg=parsed_cfg, cli_args=parsed_cli_args, dataset_meta=dataset_meta)
    return Response("Training started")

@router.get("/api/v1/health")
def health():
    return Response("OK")

def api(config: Path = Path("examples/"), **kwargs):
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
