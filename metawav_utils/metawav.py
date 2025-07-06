# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu, Zetao Hu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import torchaudio
import logging
import numpy as np
import soundfile as sf
import io
import struct
from typing import NamedTuple
from dataclasses import dataclass
import torch

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

def trim_wav_40ms(wav_data, sample_rate):
    assert 40 * sample_rate % 1000 == 0
    ms40 = 40 * sample_rate // 1000
    return wav_data[:len(wav_data) // ms40 * ms40]

def resample(wav_data: torch.Tensor, sample_rate, target_sr):
    assert 40 * sample_rate % 1000 == 0
    assert 40 * target_sr % 1000 == 0
    ms40 = 40 * sample_rate // 1000
    assert wav_data.shape[-1] % ms40 == 0
    ms40_target = 40 * target_sr // 1000
    resample_wav_data = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr).to(wav_data.device)(
        wav_data
    )
    assert resample_wav_data.shape[-1] % ms40_target == 0
    return resample_wav_data

class MetaWav:
    """
    wav meta五元组，可以作为.wav读写
    text, feat, tokens, embedding可以为None
    如果确保不写入文件，则wav_data和sample_rate也可以为None

    wav_data: np.ndarray    # 1d array of speech waveform, fp32
    sample_rate: int        # sample rate of wav_data
    text: str               # transcript text of speech
    feat: np.ndarray        # 2d array of speech mel features, fp32
    tokens: np.ndarray      # 1d array of speech tokens, int32
    embedding: np.ndarray   # 1d array of speech xvector embedding, fp32
    """
    def __init__(self, wav_data=None, sample_rate=None, text=None, feat=None, tokens=None, embedding=None):
        if wav_data is not None:
            wav_data = trim_wav_40ms(wav_data, sample_rate)
        self.wav_data = wav_data
        self.sample_rate = sample_rate
        self.text = text
        self.feat = feat
        self.tokens = tokens
        self.embedding = embedding
    
    def check_sanity(self):
        if self.wav_data is not None:
            assert self.wav_data.dtype == np.float32
            assert self.wav_data.ndim == 1
        if self.feat is not None:
            assert self.feat.dtype == np.float32
            assert self.feat.ndim == 2
            assert self.feat.shape[1] == 80
        if self.tokens is not None:
            assert self.tokens.dtype == np.int32
            assert self.tokens.ndim == 1
        if self.embedding is not None:
            assert self.embedding.dtype == np.float32
            assert self.embedding.ndim == 1
            assert self.embedding.shape[0] == 192
        
        # 要求所有东西都是40ms整倍数
        wav_len_ms = None if self.wav_data is None else len(self.wav_data) * 1000 / self.sample_rate
        feat_len_ms = None if self.feat is None else len(self.feat) * 20
        tokens_len_ms = None if self.tokens is None else len(self.tokens) * 40
        
        # 断言三个len_ms要么是None，要么相等
        lens = {wav_len_ms, feat_len_ms, tokens_len_ms}
        lens = {l for l in lens if l is not None}
        assert len(lens) <= 1

    def __eq__(self, other):
        if not isinstance(other, MetaWav):
            return False

        if self.sample_rate != other.sample_rate or self.text != other.text:
            return False

        for attr in ['wav_data', 'feat', 'tokens', 'embedding']:
            a = getattr(self, attr)
            b = getattr(other, attr)
            if a is None and b is None:
                continue
            if (a is None) != (b is None):
                return False
            # 对于 numpy 数组，使用 np.array_equal 进行比较
            if not np.array_equal(a, b):
                return False

        return True
    
    @staticmethod
    def infer_from_wav_data(cosyvoice2, wav_data, sample_rate, text=None):
        import torch
        assert wav_data.dtype == np.float32
        assert wav_data.ndim == 1
        wav_data = trim_wav_40ms(wav_data, sample_rate)
        ms = len(wav_data) * 1000 // sample_rate
        assert ms % 40 == 0
        wav_data_ = torch.tensor(wav_data).unsqueeze(0).to(cosyvoice2.model.device)
        wav_data_16k = resample(wav_data_, sample_rate=sample_rate, target_sr=16000)
        wav_data_24k = resample(wav_data_, sample_rate=sample_rate, target_sr=24000)
        tokens, tokens_len = cosyvoice2.frontend._extract_speech_token(wav_data_16k)
        features, features_len = cosyvoice2.frontend._extract_speech_feat(wav_data_24k)
        embedding = cosyvoice2.frontend._extract_spk_embedding(wav_data_16k)
        tokens = tokens.squeeze(0).detach().cpu().numpy().astype(np.int32)
        features = features.squeeze(0).detach().cpu().numpy().astype(np.float32)
        embedding = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)

        # ensure token_mel_ratio is 1:2      
        len1, len2 = len(tokens), len(features)
        assert len2 == len1 * 2
        assert len2 - 1 <= len1 * 2 <= len2 + 1
        if len2 != len1 * 2:
            # if feature is longer, cut feature to match tokens
            if len2 > len1 * 2:
                features = features[:, :len1 * 2]
            elif len2 < len1 * 2:
                # if tokens is longer, cut tokens to match feature
                tokens = tokens[:len2 // 2]
                features = features[:len2 // 2 * 2]

        result = MetaWav(wav_data, sample_rate, text, features, tokens, embedding)
        result.check_sanity()
        return result
    
    @staticmethod
    def infer_from_wav_file(cosyvoice2, wav_file, text=None):
        # read original wav file as mono fp32 wav data using soundfile, don't change sample rate
        wav_data, sample_rate = sf.read(wav_file, dtype='float32')
        if wav_data.ndim == 2:
            wav_data = wav_data.mean(axis=1, keepdims=False)
        return MetaWav.infer_from_wav_data(cosyvoice2, wav_data, sample_rate, text=text)
    
    def __str__(self):
        return f'WavWithMeta(text={self.text}, feat={self.feat}, tokens={self.tokens}, embedding={self.embedding})'

    def __repr__(self):
        return str(self)

    @staticmethod
    def write_to_wav_file(
        wav_file,
        wavmeta: 'MetaWav'
    ):
        """
        将wav_data写入wav_file，并将text、feat、tokens写入RIFF的自定义块中
        """
        wavmeta.check_sanity()
        wav_data, sample_rate, text, feat, tokens, embedding = \
            wavmeta.wav_data, wavmeta.sample_rate, wavmeta.text, wavmeta.feat, wavmeta.tokens, wavmeta.embedding
        def write_chunk(file, chunk_id, chunk_data):
            assert len(chunk_id) == 4
            file.write(struct.pack('<4sI', chunk_id, len(chunk_data)))
            file.write(chunk_data)

        with io.BytesIO() as f:
            f.name = "tmp.wav" # for soundfile to get the file extension
            sf.write(f, wav_data, sample_rate, format='WAV', subtype='FLOAT')
            wav_data = f.getvalue()

        with open(wav_file, 'wb') as f:
            f.write(wav_data)
            if text is not None:
                write_chunk(f, b'text', text.encode('utf-8'))
            if feat is not None:
                assert feat.dtype == np.float32
                write_chunk(f, b'feat', feat.tobytes())
            if tokens is not None:
                assert tokens.dtype == np.int32
                write_chunk(f, b'tokn', tokens.tobytes())
            if embedding is not None:
                assert embedding.dtype == np.float32
                write_chunk(f, b'embd', embedding.tobytes())

    @staticmethod
    def load_from_wav_file(wav_file) -> 'MetaWav':
        """
        从wav文件中读取speech waveform、transcript text、mel features、tokens、xvector embedding
        """
        def read_chunk(file, chunk_id):
            chunk_id = chunk_id.encode('utf-8')
            chunk_cnt = 0
            while True:
                chunk_header = file.read(8)
                chunk_cnt += 1
                if len(chunk_header) < 8:
                    return None
                chunk_id_, chunk_size = struct.unpack('<4sI', chunk_header)
                if chunk_id_ == chunk_id:
                    return file.read(chunk_size)
                file.seek(chunk_size, io.SEEK_CUR)
                if chunk_cnt > 100:
                    return None

        with open(wav_file, 'rb') as f:
            wav_data, sample_rate = sf.read(f, dtype='float32')
            text = read_chunk(f, 'text')
            feat = read_chunk(f, 'feat')
            tokens = read_chunk(f, 'tokn')
            embedding = read_chunk(f, 'embd')
            if text is not None:
                text = text.decode('utf-8')
            if feat is not None:
                feat = np.frombuffer(feat, dtype=np.float32).reshape(-1, 80)
            if tokens is not None:
                tokens = np.frombuffer(tokens, dtype=np.int32)
            if embedding is not None:
                embedding = np.frombuffer(embedding, dtype=np.float32)
        result =  MetaWav(wav_data, sample_rate, text, feat, tokens, embedding)
        result.check_sanity()
        return result


# if __name__ == '__main__':
#     from cosyvoice.cli.cosyvoice import CosyVoice2
#     cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

#     wavmeta = MetaWav.infer_from_wav_file(
#         cosyvoice2=cosyvoice, 
#         wav_file='zhoujie_prompt.wav', 
#         text='印象非常深刻，小杨是说，那首先这件事情是我跟周某某两个人发生的，我们都是成年人，我们两个会自己'
#     )
#     write_wav_with_meta('zhoujie_prompt.meta.wav', wavmeta)

#     wavmeta = MetaWav.infer_from_wav_file(
#         cosyvoice2=cosyvoice, 
#         wav_file='zhoujie_prompt2.wav', 
#         text='不是，您到底想要什么样的呀？能不能一次性说清楚呀？老是这么反反复复的改到底什么时候才能结束啊？'
#     )
#     write_wav_with_meta('zhoujie_prompt2.meta.wav', wavmeta)

#     wavmeta = MetaWav.infer_from_wav_file(
#         cosyvoice2=cosyvoice, 
#         wav_file='zhoujie_prompt3.wav', 
#         text='你好呀，我是小艺，全宇宙最懂你的超级A I助手'
#     )
#     write_wav_with_meta('zhoujie_prompt3.meta.wav', wavmeta)

#     wavmeta = MetaWav.infer_from_wav_file(
#         cosyvoice2=cosyvoice, 
#         wav_file='zhoujie_prompt4.wav', 
#         text='更想跟大家聊的话题，这个话题就是'
#     )
#     write_wav_with_meta('zhoujie_prompt4.meta.wav', wavmeta)

#     wavmeta = MetaWav.infer_from_wav_file(
#         cosyvoice2=cosyvoice, 
#         wav_file='zhoujie_prompt5.wav', 
#         text='诶，真的，诶，我们刚刚没有说啊第二个阶段在你比较开心的时候，你很容易产生赖酒'
#     )
#     write_wav_with_meta('zhoujie_prompt5.meta.wav', wavmeta)
    
#     wavmeta = MetaWav.infer_from_wav_file(
#         cosyvoice2=cosyvoice, 
#         wav_file='F29_prompt.wav', 
#         text='嗯，好的，这样的酒店实在太多了。您对房型有要求吗？看起来不错诶，那他家的房费应该不会很贵的哦。'
#     )
#     write_wav_with_meta('F29_prompt.meta.wav', wavmeta)

#     wavmeta = MetaWav.infer_from_wav_file(
#         cosyvoice2=cosyvoice, 
#         wav_file='F143_prompt.wav', 
#         text='你好，我是小艺，你的智能语音助手。我可以帮你导航、打电话、查天气。你有什么问题都可以随时问我，我一直都在。希望我能给你带来不一样的体验。'
#     )
#     write_wav_with_meta('F143_prompt.meta.wav', wavmeta)


