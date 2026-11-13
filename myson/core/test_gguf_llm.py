"""Glue tests for GGUFLLM — the logic WE wrote around llama-cpp-python: which file gets
loaded, where the weights go, how messages are shaped, locking, and error translation.
Fakes stand in for llama.cpp and the Hub, so no model is downloaded and no GPU is touched.

Run:  python -m unittest myson.core.test_gguf_llm -v
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from myson.core import gguf_llm
from myson.core.gguf_llm import (GGUFLLM, GGUFModelMissing, _pick_quant, _shape_messages,
                                 plan_gpu_layers)

GB = 1024 ** 3
QAT_WEIGHTS = "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
# Real file listings (trimmed) from the two Hugging Face repos config.py names.
QAT_REPO = [
    "MTP/mtp-gemma-4-26B-A4B-it-BF16.gguf",
    "MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf",
    QAT_WEIGHTS,
    "mmproj-BF16.gguf",
    "mtp-gemma-4-26B-A4B-it.gguf",
]
STD_REPO = [
    "BF16/gemma-4-26B-A4B-it-BF16-00001-of-00002.gguf",
    "BF16/gemma-4-26B-A4B-it-BF16-00002-of-00002.gguf",
    "MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf",
    "gemma-4-26B-A4B-it-Q8_0.gguf",
    "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
    "mmproj-F16.gguf",
]


def _touch(folder: str, name: str) -> str:
    path = os.path.join(folder, *name.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"\0" * 8)
    return path


# --------------------------------------------------------------------------------------
# Which file is the model
# --------------------------------------------------------------------------------------
class TestPickQuant(unittest.TestCase):
    def test_default_quant_picks_the_qat_weights(self):
        self.assertEqual(_pick_quant(QAT_REPO, "Q4_K_XL"), [QAT_WEIGHTS])

    def test_companion_files_are_never_the_model(self):
        # the only "Q4_0" in the QAT repo is a multi-token-prediction head, not weights
        self.assertEqual(_pick_quant(QAT_REPO, "Q4_0"), [])
        # and where both exist, the weights win over the MTP head that sorts first
        self.assertEqual(_pick_quant(STD_REPO, "Q8_0"), ["gemma-4-26B-A4B-it-Q8_0.gguf"])

    def test_matching_ignores_case(self):
        self.assertEqual(_pick_quant(STD_REPO, "q4_k_m"), ["gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"])

    def test_split_model_returns_every_shard_in_order(self):
        self.assertEqual(_pick_quant(STD_REPO, "BF16"), STD_REPO[:2])

    def test_unknown_quant_is_empty(self):
        self.assertEqual(_pick_quant(STD_REPO, "Q9_K"), [])


class TestResolvePath(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _llm(self, model: str, quant: str = "Q4_K_XL") -> GGUFLLM:
        llm = GGUFLLM(model=model, quant=quant)
        llm.model_dir = os.path.join(self.tmp, "models")
        return llm

    def test_a_file_path_is_used_as_is(self):
        p = _touch(self.tmp, "anything.bin")
        self.assertEqual(self._llm(p)._resolve_path(), p)

    def test_directory_picks_the_quant_and_skips_companions(self):
        d = os.path.join(self.tmp, "d")
        for name in QAT_REPO:
            _touch(d, name)
        self.assertEqual(os.path.basename(self._llm(d)._resolve_path()), QAT_WEIGHTS)

    def test_directory_without_that_quant_falls_back_to_its_model_file(self):
        d = os.path.join(self.tmp, "d")
        _touch(d, "my-finetune.gguf")
        _touch(d, "mmproj-F16.gguf")
        self.assertEqual(os.path.basename(self._llm(d)._resolve_path()), "my-finetune.gguf")

    def test_directory_with_only_companion_files_is_an_error(self):
        d = os.path.join(self.tmp, "d")
        _touch(d, "mmproj-F16.gguf")
        with self.assertRaises(GGUFModelMissing):
            self._llm(d)._resolve_path()

    def test_half_downloaded_split_model_is_not_loaded(self):
        d = os.path.join(self.tmp, "d")
        _touch(d, "m-BF16-00001-of-00002.gguf")
        with self.assertRaises(GGUFModelMissing):
            self._llm(d, quant="BF16")._resolve_path()

    def test_repo_id_found_in_the_models_folder_needs_no_network(self):
        llm = self._llm("org/repo")
        want = _touch(os.path.join(llm.model_dir, "org--repo"), QAT_WEIGHTS)
        with mock.patch("huggingface_hub.snapshot_download", side_effect=OSError("offline")), \
             mock.patch("huggingface_hub.list_repo_files", side_effect=AssertionError("network used")), \
             mock.patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("network used")):
            self.assertEqual(llm._resolve_path(), want)

    def test_repo_id_found_in_the_hf_cache_needs_no_network(self):
        llm = self._llm("org/repo")
        snap = os.path.join(self.tmp, "snap")
        want = _touch(snap, QAT_WEIGHTS)
        with mock.patch("huggingface_hub.snapshot_download", return_value=snap) as cached, \
             mock.patch("huggingface_hub.list_repo_files", side_effect=AssertionError("network used")):
            self.assertEqual(llm._resolve_path(), want)
        self.assertTrue(cached.call_args.kwargs.get("local_files_only"))

    def test_first_run_downloads_into_the_models_folder_then_works_offline(self):
        llm = self._llm("org/repo")
        dest = os.path.join(llm.model_dir, "org--repo")
        with mock.patch("huggingface_hub.snapshot_download", side_effect=OSError("not cached")), \
             mock.patch("huggingface_hub.list_repo_files", return_value=list(QAT_REPO)), \
             mock.patch("huggingface_hub.hf_hub_download",
                        side_effect=lambda repo, name, local_dir=None: _touch(local_dir, name)) as dl:
            got = llm._resolve_path()
        self.assertEqual(got, os.path.join(dest, QAT_WEIGHTS))
        dl.assert_called_once_with("org/repo", QAT_WEIGHTS, local_dir=dest)
        with mock.patch("huggingface_hub.snapshot_download", side_effect=OSError("offline")), \
             mock.patch("huggingface_hub.list_repo_files", side_effect=AssertionError("network used")):
            self.assertEqual(llm._resolve_path(), got)

    def test_offline_and_not_on_disk_says_so(self):
        llm = self._llm("org/repo")
        with mock.patch("huggingface_hub.snapshot_download", side_effect=OSError("not cached")), \
             mock.patch("huggingface_hub.list_repo_files", side_effect=OSError("offline mode")):
            with self.assertRaises(GGUFModelMissing) as cm:
                llm._resolve_path()
        self.assertIn("not on disk", str(cm.exception))

    def test_repo_without_the_quant_lists_what_it_has(self):
        llm = self._llm("org/repo", quant="Q9_K")
        with mock.patch("huggingface_hub.snapshot_download", side_effect=OSError("not cached")), \
             mock.patch("huggingface_hub.list_repo_files", return_value=list(QAT_REPO)):
            with self.assertRaises(GGUFModelMissing) as cm:
                llm._resolve_path()
        self.assertIn("available", str(cm.exception))


# --------------------------------------------------------------------------------------
# Message shaping
# --------------------------------------------------------------------------------------
class TestShapeMessages(unittest.TestCase):
    U = {"role": "user", "content": "hi"}
    A = {"role": "assistant", "content": "a"}

    def test_system_prompt_stays_a_system_message(self):
        self.assertEqual(_shape_messages([self.U], "S"),
                         [{"role": "system", "content": "S"}, self.U])

    def test_no_system_prompt_no_system_message(self):
        self.assertEqual(_shape_messages([self.U], None), [self.U])

    def test_tool_result_becomes_a_labelled_user_turn(self):
        out = _shape_messages([self.U, self.A, {"role": "tool", "name": "calc", "content": "4"}], None)
        self.assertEqual([m["role"] for m in out], ["user", "assistant", "user"])
        self.assertEqual(out[-1]["content"], "[tool result: calc]\n4")

    def test_same_speaker_turns_are_merged_so_roles_alternate(self):
        convo = [self.U, {"role": "user", "content": "again"}, self.A,
                 {"role": "tool", "name": "t1", "content": "1"},
                 {"role": "tool", "name": "t2", "content": "2"}]
        out = _shape_messages(convo, None)
        self.assertEqual([m["role"] for m in out], ["user", "assistant", "user"])
        self.assertEqual(out[0]["content"], "hi\n\nagain")
        self.assertIn("[tool result: t1]", out[2]["content"])
        self.assertIn("[tool result: t2]", out[2]["content"])

    def test_folded_system_prompt_leads_the_first_user_turn(self):
        out = _shape_messages([self.U, self.A], "S", fold_system=True)
        self.assertEqual(out[0], {"role": "user", "content": "S\n\nhi"})
        self.assertNotIn("system", [m["role"] for m in out])

    def test_folded_system_prompt_with_no_leading_user_turn(self):
        out = _shape_messages([self.A], "S", fold_system=True)
        self.assertEqual(out[0], {"role": "user", "content": "S"})

    def test_stray_system_turns_join_the_system_prompt(self):
        out = _shape_messages([{"role": "system", "content": "extra"}, self.U], "S")
        self.assertEqual(out, [{"role": "system", "content": "S\n\nextra"}, self.U])

    def test_the_callers_messages_are_not_modified(self):
        convo = [dict(self.U), {"role": "user", "content": "again"}]
        _shape_messages(convo, "S", fold_system=True)
        self.assertEqual(convo, [self.U, {"role": "user", "content": "again"}])


# --------------------------------------------------------------------------------------
# Chat / stream around a fake llama.cpp
# --------------------------------------------------------------------------------------
class FakeLlama:
    """Stands in for llama_cpp.Llama: records calls, can refuse the system role or fail."""

    def __init__(self, reject_system=False, fail=None, delay=0.0):
        self.calls, self.reject_system, self.fail, self.delay = [], reject_system, fail, delay
        self.active = self.max_active = 0
        self._guard = threading.Lock()

    def create_chat_completion(self, messages, stream=False, **kw):
        self.calls.append({"messages": messages, "stream": stream, **kw})
        if self.reject_system and messages and messages[0]["role"] == "system":
            raise ValueError("System role not supported")
        if self.fail:
            raise self.fail
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            if stream:
                return iter([{"choices": [{"delta": {"role": "assistant"}}]}] +
                            [{"choices": [{"delta": {"content": p}}]} for p in ("pie", "", "ce")])
            return {"choices": [{"message": {"role": "assistant", "content": "  hello  "}}]}
        finally:
            with self._guard:
                self.active -= 1


def _loaded(fake=None) -> GGUFLLM:
    llm = GGUFLLM(model="fake/model")
    llm._llm = fake or FakeLlama()          # _ensure_loaded() returns at once
    return llm


class TestChat(unittest.TestCase):
    Q = [{"role": "user", "content": "hi"}]

    def test_chat_returns_a_stripped_assistant_message(self):
        llm = _loaded()
        self.assertEqual(llm.chat(self.Q, system="S", temperature=0.3),
                         {"role": "assistant", "content": "hello"})
        call = llm._llm.calls[0]
        self.assertEqual(call["messages"][0], {"role": "system", "content": "S"})
        self.assertEqual(
            (call["temperature"], call["top_p"], call["top_k"], call["max_tokens"], call["stream"]),
            (0.3, llm.top_p, llm.top_k, llm.max_new_tokens, False))

    def test_missing_content_is_an_empty_string(self):
        fake = FakeLlama()
        fake.create_chat_completion = lambda **kw: {"choices": [{"message": {"content": None}}]}
        self.assertEqual(_loaded(fake).chat(self.Q)["content"], "")

    def test_ask_wraps_chat(self):
        self.assertEqual(_loaded().ask("hi", system=None, temperature=0.0), "hello")

    def test_template_without_a_system_role_gets_the_prompt_folded_in(self):
        llm = _loaded(FakeLlama(reject_system=True))
        self.assertEqual(llm.chat(self.Q, system="S")["content"], "hello")
        first, retry = llm._llm.calls
        self.assertEqual(first["messages"][0]["role"], "system")
        self.assertEqual(retry["messages"], [{"role": "user", "content": "S\n\nhi"}])
        llm.chat(self.Q, system="S")                # remembered: no failed attempt this time
        self.assertEqual(len(llm._llm.calls), 3)

    def test_stream_also_folds_the_system_prompt(self):
        llm = _loaded(FakeLlama(reject_system=True))
        self.assertEqual(list(llm.chat_stream(self.Q, system="S")), ["pie", "ce"])

    def test_generation_errors_become_mind_stone_errors(self):
        llm = _loaded(FakeLlama(fail=RuntimeError("boom")))
        with self.assertRaises(GGUFModelMissing) as cm:
            llm.chat(self.Q)
        self.assertIn("generation failed: RuntimeError: boom", str(cm.exception))
        self.assertFalse(llm._gen_lock.locked())    # and the lock is not left held

    def test_context_overflow_says_what_to_do(self):
        err = ValueError("Requested tokens (9000) exceed context window of 8192")
        with self.assertRaises(GGUFModelMissing) as cm:
            _loaded(FakeLlama(fail=err)).chat(self.Q)
        self.assertIn("MYSON_NUM_CTX", str(cm.exception))

    def test_stream_yields_only_non_empty_pieces(self):
        self.assertEqual(list(_loaded().chat_stream(self.Q, system="S")), ["pie", "ce"])

    def test_stream_holds_the_lock_until_finished_or_closed(self):
        llm = _loaded()
        gen = llm.chat_stream(self.Q)
        self.assertEqual(next(gen), "pie")
        self.assertTrue(llm._gen_lock.locked())
        gen.close()                                 # a client that disconnects mid-stream
        self.assertFalse(llm._gen_lock.locked())
        self.assertEqual(list(llm.chat_stream(self.Q)), ["pie", "ce"])
        self.assertFalse(llm._gen_lock.locked())

    def test_stream_errors_become_mind_stone_errors(self):
        with self.assertRaises(GGUFModelMissing):
            list(_loaded(FakeLlama(fail=RuntimeError("boom"))).chat_stream(self.Q))

    def test_two_threads_never_generate_at_once(self):
        llm = _loaded(FakeLlama(delay=0.05))
        threads = [threading.Thread(target=llm.chat, args=(self.Q,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(llm._llm.max_active, 1)
        self.assertEqual(len(llm._llm.calls), 4)

    def test_embed_raises_without_loading_anything(self):
        llm = GGUFLLM(model="fake/model")
        with self.assertRaises(GGUFModelMissing):
            llm.embed("probe")
        self.assertIsNone(llm._llm)


# --------------------------------------------------------------------------------------
# Where the weights go
# --------------------------------------------------------------------------------------
class _LlamaNoMoe:
    """The constructor shape of llama-cpp-python 0.3.36: no n_cpu_moe, swallows **kwargs."""

    def __init__(self, model_path=None, n_gpu_layers=0, n_ctx=0, n_batch=0, flash_attn=False,
                 verbose=True, **kwargs):
        pass


class _LlamaWithMoe:
    def __init__(self, model_path=None, n_gpu_layers=0, n_cpu_moe=0, **kwargs):
        pass


class TestPlacement(unittest.TestCase):
    def test_everything_goes_on_the_gpu_when_it_fits(self):
        self.assertEqual(plan_gpu_layers(4 * GB, 32, GB // 64, 8 * GB, GB), -1)

    def test_a_model_too_big_gets_as_many_layers_as_fit(self):
        # 15 GB over 30 layers is 0.5 GB a layer; 7 GB free minus a 1 GB reserve holds 12
        self.assertEqual(plan_gpu_layers(15 * GB, 30, 0, 7 * GB, GB), 12)

    def test_the_kv_cache_counts_against_the_budget(self):
        self.assertEqual(plan_gpu_layers(15 * GB, 30, GB // 4, 7 * GB, GB), 8)

    def test_no_room_at_all_means_cpu_only(self):
        self.assertEqual(plan_gpu_layers(15 * GB, 30, 0, GB // 2, GB), 0)

    def test_an_explicit_layer_count_is_respected(self):
        llm = GGUFLLM(model="x", n_gpu_layers=7, n_cpu_moe=0)
        with mock.patch.object(gguf_llm, "_free_vram_bytes", side_effect=AssertionError("measured")):
            self.assertEqual(llm._placement(_LlamaNoMoe, "p"), {"n_gpu_layers": 7})

    def test_expert_offload_is_used_when_the_runtime_has_it(self):
        llm = GGUFLLM(model="x", n_gpu_layers=-1, n_cpu_moe=99)
        self.assertEqual(llm._placement(_LlamaWithMoe, "p"), {"n_gpu_layers": -1, "n_cpu_moe": 99})

    def test_expert_offload_request_is_not_silently_dropped(self):
        # llama-cpp-python 0.3.36 has no n_cpu_moe. The request used to vanish and the loader
        # asked for every layer of a 14 GB model on an 8 GB card; now the layer split runs.
        llm = GGUFLLM(model="x", n_gpu_layers=-1, n_cpu_moe=99)
        llm.vram_reserve = GB
        with mock.patch.object(gguf_llm, "_free_vram_bytes", return_value=7 * GB), \
             mock.patch.object(gguf_llm, "_model_shape", return_value=(30, 0)), \
             mock.patch.object(gguf_llm, "_model_bytes", return_value=15 * GB):
            self.assertEqual(llm._placement(_LlamaNoMoe, "p"), {"n_gpu_layers": 12})
        self.assertIn("12 of 30 layers", llm.placement)

    def test_kv_estimate_knows_about_sliding_window_layers(self):
        # Gemma 4 keeps a 1024-token window with smaller heads on most layers. Counting every
        # layer as full attention made the planner put 10 of the 26B's layers on the GPU when
        # 14 fit. Values below are that model's real header.
        md = {"general.architecture": "gemma4", "gemma4.block_count": "30",
              "gemma4.embedding_length": "2816", "gemma4.attention.head_count": "16",
              "gemma4.attention.key_length": "512", "gemma4.attention.value_length": "512",
              "gemma4.attention.sliding_window": "1024",
              "gemma4.attention.key_length_swa": "256", "gemma4.attention.value_length_swa": "256"}

        class Probe:
            def __init__(self, **kw):
                self.metadata = md

            def close(self):
                pass

        full = 4096 * 16 * (512 + 512) * 2
        local = 1024 * 16 * (256 + 256) * 2
        self.assertEqual(gguf_llm._model_shape(Probe, "p", 4096), (30, (full + 4 * local) // 5))
        del md["gemma4.attention.sliding_window"]          # a model without a window: all full
        self.assertEqual(gguf_llm._model_shape(Probe, "p", 4096), (30, full))

    def test_unmeasurable_vram_keeps_the_configured_value(self):
        llm = GGUFLLM(model="x", n_gpu_layers=-1, n_cpu_moe=0)
        with mock.patch.object(gguf_llm, "_free_vram_bytes", return_value=None), \
             mock.patch.object(gguf_llm, "_model_shape", return_value=(30, 0)):
            self.assertEqual(llm._placement(_LlamaNoMoe, "p"), {"n_gpu_layers": -1})
        self.assertIn("could not be read", llm.placement)


class TestLoad(unittest.TestCase):
    def _patched(self, llama_cls):
        fake = types.ModuleType("llama_cpp")
        fake.Llama = llama_cls
        stack = mock.patch.dict(sys.modules, {"llama_cpp": fake})
        return stack

    def test_load_passes_placement_context_and_flash_attention(self):
        made = {}

        class Llama:
            def __init__(self, model_path=None, n_ctx=0, n_batch=0, n_gpu_layers=0,
                         flash_attn=False, verbose=True, **kwargs):
                made.update(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                            flash_attn=flash_attn)

        llm = GGUFLLM(model="x", n_ctx=4096, n_gpu_layers=5, n_cpu_moe=0)
        llm.flash_attn = True
        with self._patched(Llama), mock.patch.object(gguf_llm, "_add_torch_cuda_dlls"), \
             mock.patch.object(GGUFLLM, "_resolve_path", return_value="weights.gguf"):
            llm._ensure_loaded()
        self.assertEqual(made, {"model_path": "weights.gguf", "n_ctx": 4096,
                                "n_gpu_layers": 5, "flash_attn": True})

    def test_a_failed_load_is_a_mind_stone_error_that_names_the_placement(self):
        class Llama:
            def __init__(self, model_path=None, n_gpu_layers=0, **kwargs):
                raise ValueError("Failed to load model from file")

        llm = GGUFLLM(model="x", n_gpu_layers=3, n_cpu_moe=0)
        with self._patched(Llama), mock.patch.object(gguf_llm, "_add_torch_cuda_dlls"), \
             mock.patch.object(GGUFLLM, "_resolve_path", return_value="weights.gguf"):
            with self.assertRaises(GGUFModelMissing) as cm:
                llm._ensure_loaded()
        self.assertIn("3 layers on GPU", str(cm.exception))
        self.assertIsNone(llm._llm)

    def test_available_prepares_the_dll_path_before_importing(self):
        # On Windows a bare `import llama_cpp` fails until torch's CUDA DLL folder is on the
        # search path, so available() used to answer False on a machine that could load.
        order = []
        with mock.patch.object(gguf_llm, "_add_torch_cuda_dlls",
                               side_effect=lambda: order.append("dlls")), \
             mock.patch.dict(sys.modules, {"llama_cpp": types.ModuleType("llama_cpp")}):
            self.assertTrue(GGUFLLM(model="x").available())
        self.assertEqual(order, ["dlls"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
