import numpy as np
import torch

from countrycode import labels
from countrycode.train.dataset import COUNTRY_CLASSES, STATE_CLASSES
from countrycode.train.model import CountryModel, combine, soft_ce, target_matrix


def tiny():
    from transformers import ModernBertConfig, ModernBertModel

    cfg = ModernBertConfig(
        vocab_size=300,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
    )
    return CountryModel(ModernBertModel(cfg))


def test_combine_puts_state_mass_under_us():
    pc = np.zeros((1, len(COUNTRY_CLASSES)))
    pc[0, COUNTRY_CLASSES.index("US")] = 0.8
    pc[0, COUNTRY_CLASSES.index("CA")] = 0.2
    ps = np.zeros((1, len(STATE_CLASSES)))
    ps[0, STATE_CLASSES.index("US-CA")] = 0.75
    ps[0, STATE_CLASSES.index("NO_STATE")] = 0.25
    p = combine(pc, ps)
    t = list(labels.targets())
    assert p.shape == (1, 302) and abs(p.sum() - 1) < 1e-9
    assert abs(p[0, t.index("US-CA")] - 0.6) < 1e-9 and abs(p[0, t.index("US")] - 0.2) < 1e-9
    assert abs(p[0, t.index("CA")] - 0.2) < 1e-9


def test_target_matrix_rows_sum_to_one():
    m = target_matrix([{"AT": 1.0}, {"GE": 0.5, "US": 0.5}], COUNTRY_CLASSES)
    assert m.shape == (2, 251) and torch.allclose(m.sum(1), torch.ones(2))


def test_soft_ce_and_training_step_reduce_loss():
    torch.manual_seed(0)
    m = tiny()
    ids = torch.randint(5, 300, (8, 10))
    mask = torch.ones_like(ids)
    tc = target_matrix([{"AT": 1.0}] * 4 + [{"DE": 1.0}] * 4, COUNTRY_CLASSES)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    losses = []
    for _ in range(20):
        c_logits, s_logits = m(ids, mask)
        assert c_logits.shape == (8, 251) and s_logits.shape == (8, 52)
        loss = soft_ce(c_logits, tc)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] * 0.5
