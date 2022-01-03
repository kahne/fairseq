# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.data import encoders
from fairseq.data.audio.audio_utils import (
    get_waveform as get_wav,
    convert_waveform as convert_wav,
    get_fbank,
)
import fairseq.data.audio.feature_transforms.utterance_cmvn as utt_cmvn

logger = logging.getLogger(__name__)


class S2THubInterface(nn.Module):
    def __init__(self, cfg, task, model):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.model = model

    @classmethod
    def get_model_input(cls, task, audio_path: str):
        input_type = task.data_cfg.hub.get("input_type", "fbank80")
        if input_type == "fbank80_w_utt_cmvn":
            feat = get_fbank(audio_path)  # T x D
            feat = utt_cmvn.UtteranceCMVN()(feat)
        elif input_type in {"waveform", "standardized_waveform"}:
            feat, sr = get_wav(audio_path)  # C x T
            feat = convert_wav(feat, sr, to_sample_rate=16_000, to_mono=True)[0]
            feat = feat.squeeze(0)  # 1 x T -> T
        else:
            raise ValueError(f"Unknown value: input_type = {input_type}")

        src_lengths = torch.Tensor([feat.shape[0]]).long()
        src_tokens = torch.from_numpy(feat).unsqueeze(0)  # -> 1 x T (x D)
        if input_type == "standardized_waveform":
            with torch.no_grad():
                src_tokens = F.layer_norm(src_tokens, src_tokens.shape)

        return {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "prev_output_tokens": None,
            },
            "target_lengths": None,
            "speaker": None,
        }

    def detokenize(self, text: str):
        tkn_cfg = self.task.data_cfg.bpe_tokenizer
        tokenizer = encoders.build_bpe(Namespace(**tkn_cfg))
        return text if tokenizer is None else tokenizer.decode(text)

    def predict(self, audio_path: str):
        self.model.eval()

        sample = self.get_model_input(self.task, audio_path)
        generator = self.task.build_generator([self.model], self.task)
        pred_tokens = generator.generate([self.model], sample)[0][0]["tokens"]
        pred = self.detokenize(self.task.tgt_dict.string(pred_tokens))
        return pred
