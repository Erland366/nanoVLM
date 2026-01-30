import torch

from models.language_model import LanguageModel


class _Cfg:
    lm_hidden_dim = 64
    lm_inter_dim = 128
    lm_rms_eps = 1e-5
    lm_re_base = 10000.0
    lm_max_position_embeddings = 1024
    lm_attn_scaling = 1.0
    lm_vocab_size = 100
    lm_n_heads = 4
    lm_n_kv_heads = 2
    lm_dropout = 0.0
    lm_n_blocks = 2
    lm_use_tokens = True
    lm_tie_weights = True


def test_kv_caching_consistency():
    torch.manual_seed(0)
    cfg = _Cfg()
    model = LanguageModel(cfg).eval()

    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, cfg.lm_vocab_size, (batch_size, seq_len))

    # Full forward (no cache).
    output_no_cache, _ = model(input_ids, start_pos=0)

    # Cache path: prefill (all but last token) + decode last token.
    prefill_output, kv_cache_prefill = model(input_ids[:, :-1], start_pos=0)
    assert prefill_output.shape[:2] == (batch_size, seq_len - 1)

    last_token_input = input_ids[:, -1].unsqueeze(-1)
    output_with_cache_last_token, _ = model(
        last_token_input, kv_cache=kv_cache_prefill, start_pos=seq_len - 1
    )

    logits_no_cache_last_token = output_no_cache[:, -1, :]
    logits_with_cache_last_token = output_with_cache_last_token[:, 0, :]
    assert torch.allclose(logits_no_cache_last_token, logits_with_cache_last_token, atol=1e-5)

    # Full token-by-token decode should match full forward.
    kv_cache_step = None
    outputs = []
    for i in range(seq_len):
        token = input_ids[:, i : i + 1]
        out_step, kv_cache_step = model(token, kv_cache=kv_cache_step, start_pos=i)
        outputs.append(out_step)
    output_with_cache_full = torch.cat(outputs, dim=1)
    assert torch.allclose(output_no_cache, output_with_cache_full, atol=1e-5)

