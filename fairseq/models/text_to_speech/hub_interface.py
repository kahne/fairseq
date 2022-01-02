# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional, Dict
import random

import torch
import torch.nn as nn


class TTSHubInterface(nn.Module):
    def __init__(self, cfg, task, model):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.model = model
        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    @classmethod
    def phonemize(
            cls,
            text: str,
            lang: Optional[str],
            phonemizer: Optional[str] = None,
            preserve_punct: bool = False,
            to_simplified_zh: bool = False
    ):
        if to_simplified_zh:
            import hanziconv
            text = hanziconv.HanziConv.toSimplified(text)

        if phonemizer == "g2p":
            import g2p_en
            g2p = g2p_en.G2p()
            if preserve_punct:
                return " ".join("|" if p == " " else p for p in g2p(text))
            else:
                res = [{",": "sp", ";": "sp"}.get(p, p) for p in g2p(text)]
                return " ".join(p for p in res if p.isalnum())
        if phonemizer == "g2pc":
            import g2pc
            g2p = g2pc.G2pC()
            return " ".join([w[3] for w in g2p(text)])
        elif phonemizer == "ipa":
            assert lang is not None
            import phonemizer
            from phonemizer.separator import Separator
            lang_map = {"en": "en-us", "fr": "fr-fr"}
            return phonemizer.phonemize(
                text,
                backend="espeak",
                language=lang_map.get(lang, lang),
                separator=Separator(word="| ", phone=" "),
            )
        else:
            return text

    @classmethod
    def tokenize(cls, text: str, tkn_cfg: Dict[str, str]):
        sentencepiece_model = tkn_cfg.get("sentencepiece_model", None)
        if sentencepiece_model is not None:
            assert Path(sentencepiece_model).exists()
            import sentencepiece as sp
            spm = sp.SentencePieceProcessor()
            spm.Load(sentencepiece_model)
            return " ".join(spm.Encode(text, out_type=str))
        else:
            return text

    @classmethod
    def _get_speaker(cls, task, speaker: Optional[int] = None):
        speaker = task.data_cfg.config.get("hub", {}).get("speaker", speaker)
        n_speakers = len(task.speaker_to_id or {})
        if speaker is None:
            if n_speakers > 0:
                speaker = random.randint(0, n_speakers - 1)
        else:
            speaker = max(0, min(speaker, n_speakers - 1))
        return speaker

    @classmethod
    def get_model_input(cls, task, text: str, speaker: Optional[int] = None):
        cfg_hub = task.data_cfg.config.get("hub", {})
        phonemized = cls.phonemize(
            text,
            cfg_hub.get("lang", None),
            cfg_hub.get("phonemizer", None),
            cfg_hub.get("preserve_punct", False),
            cfg_hub.get("to_simplified_zh", False),
        )
        tkn_cfg = task.data_cfg.config.get("bpe_tokenizer", {})
        tokenized = cls.tokenize(phonemized, tkn_cfg)
        spk = cls._get_speaker(task, speaker)
        spk = None if spk is None else torch.Tensor([[spk]]).long()

        src_tokens = task.src_dict.encode_line(tokenized).view(1, -1)
        src_lengths = torch.Tensor([len(tokenized.split())]).long()
        return {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "prev_output_tokens": None
            },
            "target_lengths": None,
            "speaker": spk,
        }

    def predict(self, text: str, speaker: Optional[int] = None):
        sample = self.get_model_input(self.task, text, speaker)
        self.model.eval()
        generator = self.task.build_generator([self.model], self.cfg)
        generation = generator.generate(self.model, sample)
        return generation[0]["waveform"], self.task.sr
