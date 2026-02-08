# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MultimodalPDWorkerHandler."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    MultimodalEmbeddingCacheManager,
)
from dynamo.vllm.multimodal_handlers import multimodal_pd_worker_handler as mod
from dynamo.vllm.multimodal_utils.protocol import (
    MultiModalGroup,
    MultiModalInput,
    MyRequestOutput,
    PatchedTokensPrompt,
    vLLMMultimodalRequest,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


# ── Helpers ──────────────────────────────────────────────────────────


def _make_config(
    model: str = "test-model",
    is_prefill_worker: bool = False,
    enable_multimodal: bool = True,
    multimodal_embedding_cache_capacity_gb: float = 0,
) -> MagicMock:
    """Create a mock Config with the fields used by MultimodalPDWorkerHandler."""
    config = MagicMock()
    config.model = model
    config.is_prefill_worker = is_prefill_worker
    config.enable_multimodal = enable_multimodal
    config.multimodal_embedding_cache_capacity_gb = (
        multimodal_embedding_cache_capacity_gb
    )
    config.engine_args.create_model_config.return_value.get_diff_sampling_param.return_value = (
        {}
    )
    return config


def _make_handler(
    config: MagicMock | None = None,
    encode_worker_client: MagicMock | None = None,
    decode_worker_client: MagicMock | None = None,
) -> mod.MultimodalPDWorkerHandler:
    """Construct a handler with BaseWorkerHandler.__init__ bypassed."""
    if config is None:
        config = _make_config()
    with patch.object(mod.BaseWorkerHandler, "__init__", return_value=None):
        return mod.MultimodalPDWorkerHandler(
            runtime=MagicMock(),
            component=MagicMock(),
            engine_client=MagicMock(),
            config=config,
            encode_worker_client=encode_worker_client,
            decode_worker_client=decode_worker_client,
        )


def _make_raw_frontend_request(image_urls: list[str] | None = None) -> dict:
    """Build a raw dict that mimics what the Rust frontend sends."""
    mm_data = None
    if image_urls:
        mm_data = {
            "image_url": [{"Url": url} for url in image_urls],
        }
    return {
        "token_ids": [1, 2, 3],
        "multi_modal_data": mm_data,
        "sampling_options": {},
        "stop_conditions": {},
        "output_options": {},
    }


def _make_vllm_request(request_id: str = "req-1") -> vLLMMultimodalRequest:
    """Build a minimal vLLMMultimodalRequest."""
    from vllm.sampling_params import SamplingParams

    return vLLMMultimodalRequest(
        engine_prompt=PatchedTokensPrompt(prompt_token_ids=[1, 2, 3]),
        sampling_params=SamplingParams(),
        request_id=request_id,
        multimodal_inputs=[],
    )


def _make_engine_response(request_id: str = "req-1", finished: bool = True):
    """Create a mock engine response with the fields _serialize_response needs."""
    resp = MagicMock()
    resp.request_id = request_id
    resp.prompt = "test"
    resp.prompt_token_ids = [1, 2, 3]
    resp.prompt_logprobs = None
    resp.outputs = []
    resp.finished = finished
    resp.metrics = None
    resp.kv_transfer_params = {"do_remote_decode": False}
    return resp


# ── Tests ────────────────────────────────────────────────────────────


class TestInit:
    def test_embedding_cache_created_when_capacity_set(self):
        capacity_gb = 0.1
        handler = _make_handler(
            config=_make_config(multimodal_embedding_cache_capacity_gb=capacity_gb)
        )
        assert isinstance(
            handler.embedding_cache_manager, MultimodalEmbeddingCacheManager
        )
        expected_bytes = int(capacity_gb * 1024**3)
        assert handler.embedding_cache_manager._capacity_bytes == expected_bytes


class TestParseFrontendRequest:
    def test_extracts_token_ids_and_sampling_params(self):
        """Parses token_ids and sampling_params from raw frontend dict."""
        handler = _make_handler()
        handler.default_sampling_params = {}

        raw = _make_raw_frontend_request()
        request, image_urls = handler._parse_frontend_request(raw)

        assert request.engine_prompt["prompt_token_ids"] == [1, 2, 3]
        assert image_urls == []

    def test_extracts_image_urls(self):
        """Extracts image URLs from multi_modal_data."""
        handler = _make_handler()
        handler.default_sampling_params = {}

        raw = _make_raw_frontend_request(image_urls=["http://a.png", "http://b.png"])
        request, image_urls = handler._parse_frontend_request(raw)

        assert image_urls == ["http://a.png", "http://b.png"]


class TestLoadMultimodalData:
    @pytest.mark.asyncio
    async def test_no_encode_client_returns_empty(self):
        """Without encode client -> returns empty dict."""
        handler = _make_handler(encode_worker_client=None)
        result = await handler._load_multimodal_data(["http://img.png"], "req-1")
        assert dict(result) == {}

    @pytest.mark.asyncio
    async def test_no_images_returns_empty(self):
        """With encode client but no images -> returns empty dict."""
        handler = _make_handler(encode_worker_client=MagicMock())
        result = await handler._load_multimodal_data([], "req-1")
        assert dict(result) == {}

    @pytest.mark.asyncio
    async def test_with_encode_worker_calls_fetch(self):
        """With encode client -> delegates to fetch_embeddings_via_stream."""
        mock_client = MagicMock()
        handler = _make_handler(encode_worker_client=mock_client)

        fake_group = MultiModalGroup(multimodal_input=MultiModalInput())
        with (
            patch.object(
                mod,
                "fetch_embeddings_via_stream",
                new_callable=AsyncMock,
                return_value=[fake_group],
            ) as mock_fetch,
            patch.object(mod, "load_embedding", new_callable=AsyncMock) as mock_load,
        ):
            mock_load.return_value = torch.randn(1, 10)
            await handler._load_multimodal_data(["http://img.png"], "req-1")

        mock_fetch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_with_cache_calls_fetch_via_local_cache(self):
        """With cache enabled -> delegates to fetch_embeddings_via_local_cache."""
        mock_client = MagicMock()
        config = _make_config(multimodal_embedding_cache_capacity_gb=1.0)
        handler = _make_handler(config=config, encode_worker_client=mock_client)

        fake_group = MultiModalGroup(multimodal_input=MultiModalInput())
        with (
            patch.object(
                mod,
                "fetch_embeddings_via_local_cache",
                new_callable=AsyncMock,
                return_value=[fake_group],
            ) as mock_cache_fetch,
            patch.object(mod, "load_embedding", new_callable=AsyncMock) as mock_load,
            patch.object(mod, "accumulate_embeddings"),
        ):
            mock_load.return_value = torch.randn(1, 10)
            await handler._load_multimodal_data(["http://img.png"], "req-1")

        mock_cache_fetch.assert_awaited_once()


class TestLoadMultimodalDataCachedEmbedding:
    @pytest.mark.asyncio
    async def test_cached_embedding_path(self):
        """_load_multimodal_data uses cached_embedding when set on a group."""
        mock_client = MagicMock()
        handler = _make_handler(encode_worker_client=mock_client)

        tensor = torch.randn(1, 10, dtype=torch.float16)
        group = MultiModalGroup(
            multimodal_input=MultiModalInput(),
            cached_embedding=tensor,
            image_grid_thw=None,
        )

        with (
            patch.object(
                mod,
                "fetch_embeddings_via_stream",
                new_callable=AsyncMock,
                return_value=[group],
            ),
            patch.object(
                mod,
                "accumulate_embeddings",
            ) as mock_accum,
        ):
            await handler._load_multimodal_data(["http://img.png"], "req-1")

        mock_accum.assert_called_once()
        call_args = mock_accum.call_args
        assert torch.equal(call_args[0][3], tensor)
        assert call_args[0][4] is None


class TestGenerateAgg:
    @pytest.mark.asyncio
    async def test_streams_serialized_responses(self):
        """_generate_agg yields dicts formatted by _format_engine_output."""
        handler = _make_handler()
        request = _make_vllm_request()
        engine_resp = _make_engine_response()

        # Add a proper output so we exercise the happy path
        output = MagicMock()
        output.token_ids = [10, 11]
        output.finish_reason = "stop"
        output.stop_reason = None
        engine_resp.outputs = [output]

        async def fake_generate(**kwargs):
            yield engine_resp

        handler.engine_client = MagicMock()
        handler.engine_client.generate = fake_generate

        chunks = []
        async for chunk in handler._generate_agg(request, {"image": []}):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0]["token_ids"] == [10, 11]
        assert chunks[0]["finish_reason"] == "stop"


class TestGenerateDisagg:
    @pytest.mark.asyncio
    async def test_prefills_then_forwards_to_decode(self):
        """_generate_disagg prefills locally, then round-robins to decode worker."""
        config = _make_config(model="test-model", is_prefill_worker=True)
        decode_client = MagicMock()
        handler = _make_handler(config=config, decode_worker_client=decode_client)
        handler.engine_client = MagicMock()

        # Mock prefill engine response
        prefill_resp = _make_engine_response()
        prefill_resp.kv_transfer_params = {"block_ids": [0, 1]}

        async def fake_generate(**kwargs):
            yield prefill_resp

        handler.engine_client.generate = fake_generate

        # Mock decode worker response
        decode_output = MyRequestOutput(
            request_id="req-1",
            prompt="test",
            prompt_token_ids=[1, 2, 3],
            outputs=[],
            finished=True,
            kv_transfer_params={"block_ids": [0, 1]},
        )
        decode_resp = MagicMock()
        decode_resp.data.return_value = decode_output.model_dump_json()

        async def fake_round_robin(payload):
            async def _stream():
                yield decode_resp

            return _stream()

        decode_client.round_robin = fake_round_robin

        request = _make_vllm_request()
        chunks = []
        async for chunk in handler._generate_disagg(request, {"image": []}):
            chunks.append(chunk)

        assert len(chunks) == 1
        parsed = json.loads(chunks[0])
        assert parsed["request_id"] == "req-1"
        assert parsed["finished"] is True
