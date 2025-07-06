import os
import time
from typing import Generator
import torchaudio
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model
from cosyvoice.flow.flow import CausalMaskedDiffWithXvec
from cosyvoice.hifigan.generator import HiFTGenerator
from cosyvoice.llm.llm import Qwen2LM
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.class_utils import get_model_type
from cosyvoice.cli.cosyvoice import CosyVoice2
from metawav_utils.metawav import MetaWav


class Inferrer(CosyVoice2):
    def inference_llm(
        self,
        tts_text_token, tts_text_token_len,
        prompt_text_token, prompt_text_token_len,
        prompt_token, prompt_token_len,
        embedding,
        # cfm_prompt_tokens, cfm_prompt_features, cfm_embedding,
    ):
        get_shape = lambda t: list(t.shape[1:])
        logging.info(f'inference_llm: '
                     f'tts_text_token: {get_shape(tts_text_token)}, '
                     f'prompt_text_token: {get_shape(prompt_text_token)}, '
                     f'prompt_token: {get_shape(prompt_token)}, '
                     f'embedding: {get_shape(embedding)}')
        assert isinstance(self.model.llm, Qwen2LM), 'llm model is not Qwen2LM'
        llm: Qwen2LM = self.model.llm
        tts_tokens = list(llm.inference(
            text=tts_text_token,
            text_len=tts_text_token_len,
            prompt_text=prompt_text_token,
            prompt_text_len=prompt_text_token_len,
            prompt_speech_token=prompt_token,
            prompt_speech_token_len=prompt_token_len,
            embedding=embedding,
            sampling=25,
            max_token_text_ratio=20,
            min_token_text_ratio=2,
        ))
        tts_tokens = torch.tensor(tts_tokens, dtype=torch.int32).unsqueeze(0).to(self.model.device)
        return tts_tokens

    def inference_cfm(
        self, 
        target_tokens,
        prompt_tokens,
        prompt_features,
        cfm_embedding
    ):
        # ensure token_mel_ratio is 1:2
        get_shape = lambda t: list(t.shape[1:])        
        len1, len2 = get_shape(prompt_tokens)[0], get_shape(prompt_features)[0]
        assert len2 - 1 <= len1 * 2 <= len2 + 1
        if len2 != len1 * 2:            
            # if feature is longer, cut feature to match tokens
            if len2 > len1 * 2:
                prompt_features = prompt_features[:, :len1 * 2]
            elif len2 < len1 * 2:
                # if tokens is longer, cut tokens to match feature
                prompt_tokens = prompt_tokens[:, :len2 // 2]
                prompt_features = prompt_features[:, :len2 // 2 * 2]
        logging.info(f'inference_cfm: '
            f'target_tokens: {get_shape(target_tokens)}, '
            f'prompt_tokens: {[len1]} -> {get_shape(prompt_tokens)}, '
            f'prompt_features: {[len2]} -> {get_shape(prompt_features)}, '
            f'cfm_embedding: {get_shape(cfm_embedding)}')

        cfm: CausalMaskedDiffWithXvec = self.model.flow
        tts_mel, _ = cfm.inference(
            token=target_tokens,
            token_len=torch.tensor([target_tokens.shape[1]], dtype=torch.int32).to(self.model.device),
            prompt_token=prompt_tokens.to(self.model.device),
            prompt_token_len=torch.tensor([prompt_tokens.shape[1]], dtype=torch.int32).to(self.model.device),
            prompt_feat=prompt_features.to(self.model.device),
            prompt_feat_len=torch.tensor([prompt_features.shape[1]], dtype=torch.int32).to(self.model.device),
            embedding=cfm_embedding.to(self.model.device),
            streaming=False,
            finalize=True
        )
        return tts_mel
    
    def _inference_cfm_streaming_mode_no_grow(
        self, 
        target_tokens,
        prompt_tokens,
        prompt_features,
        cfm_embedding,
        chunk_size: int=25,
        lookahead_size: int=3,
    ):
        chunk_tts_mels = []
        # 假如chunk_size=25，lookahead_size=3，那么第一次输入的长度是25+3，第二次输入的长度是25+25+3，依此类推
        for chunk_start in range(0, target_tokens.shape[1], chunk_size):
            # 如果不是末次，需要补充lookahead_size帧，但不要超过target_tokens的长度
            chunk_end_with_lookahead = min(chunk_start + chunk_size + lookahead_size, target_tokens.shape[1])
            chunk_end_wo_lookahead = min(chunk_start + chunk_size, target_tokens.shape[1])
            real_lookahead_size = chunk_end_with_lookahead - chunk_end_wo_lookahead
            real_chunk_size = chunk_end_wo_lookahead - chunk_start
            
            growing_target_tokens = target_tokens[:, :chunk_end_with_lookahead]
            chunk_tts_mel = self.inference_cfm(
                target_tokens=growing_target_tokens,
                prompt_tokens=prompt_tokens,
                prompt_features=prompt_features,
                cfm_embedding=cfm_embedding,
            )
            chunk_tts_mel = chunk_tts_mel.transpose(-1, -2)
            # print(chunk_start, chunk_end_with_lookahead, chunk_end_wo_lookahead, real_lookahead_size, real_chunk_size, chunk_tts_mel.shape)
            if real_lookahead_size != 0:
                chunk_tts_mel = chunk_tts_mel[:, -2*(real_chunk_size + real_lookahead_size):-2*real_lookahead_size]
            else:
                chunk_tts_mel = chunk_tts_mel[:, -2*real_chunk_size:]
            chunk_tts_mels.append(chunk_tts_mel)

            

        tts_mel = torch.cat(chunk_tts_mels, dim=1)
        tts_mel = tts_mel.transpose(-1, -2)
        return tts_mel


    def inference_cfm_streaming_mode(
        self, 
        target_tokens,
        prompt_tokens,
        prompt_features,
        cfm_embedding,
        streaming_mode: str='paste_mel',
        chunk_size: int=25,
        lookahead_size: int=3,
    ):
        assert streaming_mode in [
            'paste_mel',
            'no_paste_mel',
            'no_grow'
        ]
        assert lookahead_size < chunk_size

        if streaming_mode == 'no_grow':
            return self._inference_cfm_streaming_mode_no_grow(
                target_tokens=target_tokens,
                prompt_tokens=prompt_tokens,
                prompt_features=prompt_features,
                cfm_embedding=cfm_embedding,
                chunk_size=chunk_size,
                lookahead_size=lookahead_size,
            )

        chunk_tts_mels = []
        # 假如chunk_size=25，lookahead_size=3，那么第一次输入的长度是25+3，第二次输入的长度是25+25+3，依此类推
        for chunk_start in range(0, target_tokens.shape[1], chunk_size):
            # 如果不是末次，需要补充lookahead_size帧，但不要超过target_tokens的长度
            chunk_end_with_lookahead = min(chunk_start + chunk_size + lookahead_size, target_tokens.shape[1])
            chunk_end_wo_lookahead = min(chunk_start + chunk_size, target_tokens.shape[1])
            real_lookahead_size = chunk_end_with_lookahead - chunk_end_wo_lookahead
            
            if chunk_start == 0:
                # 第一次输入，不需要贴任何前面的mel
                growing_prompt_features = prompt_features
                growing_prompt_tokens = prompt_tokens
            else:
                # 将chunk_tts_mels贴到prompt_features上得到growing_prompt_features
                growing_prompt_features = torch.cat([prompt_features] + chunk_tts_mels, dim=1)
                if streaming_mode == 'no_paste_mel':
                    growing_prompt_features[:, prompt_features.shape[1]:] = 0
                growing_prompt_tokens = torch.cat([prompt_tokens, target_tokens[:, :chunk_start]], dim=1)

            cur_target_tokens = target_tokens[:, chunk_start:chunk_end_with_lookahead]
            chunk_tts_mel = self.inference_cfm(
                target_tokens=cur_target_tokens,
                prompt_tokens=growing_prompt_tokens,
                prompt_features=growing_prompt_features,
                cfm_embedding=cfm_embedding,
            )
            chunk_tts_mel = chunk_tts_mel.transpose(-1, -2)
            if real_lookahead_size != 0:
                chunk_tts_mel = chunk_tts_mel[:, :-2*real_lookahead_size]
            chunk_tts_mels.append(chunk_tts_mel)
        
        tts_mel = torch.cat(chunk_tts_mels, dim=1)
        tts_mel = tts_mel.transpose(-1, -2)
        return tts_mel

    
    def inference_hift(
        self,
        tts_mel,
    ):
        get_shape = lambda t: list(t.shape[1:])
        logging.info(f'inference_hift: tts_mel: {get_shape(tts_mel)}')
        hift: HiFTGenerator = self.model.hift
        tts_speech, _ = hift.inference(speech_feat=tts_mel)
        return tts_speech
    
    def inference_target_tokens_using_metawav(
        self,
        target_metawav: MetaWav,
        llm_prompt_metawav: MetaWav,
        target_token_mode: str='infer',
    ):
        if target_token_mode == 'provide':
            assert target_metawav.tokens is not None
            target_tokens = torch.tensor(target_metawav.tokens).unsqueeze(0).to(self.model.device)
            return target_tokens
        
        if target_token_mode == 'extract':
            assert target_metawav.wav_data is not None
            target_wav_data = torch.tensor(target_metawav.wav_data).unsqueeze(0).to(self.model.device)
            target_wav_data_16k = torchaudio.transforms.Resample(orig_freq=target_metawav.sample_rate, new_freq=16000).to(self.model.device)(target_wav_data)
            target_tokens, _ = self.frontend._extract_speech_token(target_wav_data_16k)
            return target_tokens
        
        # 1.1. prepare llm target text
        text_tokens, text_token_len = self.frontend._extract_text_token(target_metawav.text)

        # 1.2. prepare llm prompt text
        if llm_prompt_metawav.text is None or len(llm_prompt_metawav.text) == 0:
            # no llm prompt text
            llm_prompt_text_token = torch.zeros(1, 0, dtype=torch.int32).to(self.model.device)
            llm_prompt_text_token_len = torch.zeros(1, dtype=torch.int32).to(self.model.device)
        else:
            llm_prompt_text_token, llm_prompt_text_token_len = self.frontend._extract_text_token(llm_prompt_metawav.text)
        
        # 1.3. prepare llm prompt tokens and embedding
        if llm_prompt_metawav.wav_data is not None:
            # 1.3.1 run llm's prompt_token/embedding extraction if not provided
            llm_wav_data = torch.tensor(llm_prompt_metawav.wav_data).unsqueeze(0).to(self.model.device)
            llm_wav_data_16k = torchaudio.transforms.Resample(orig_freq=llm_prompt_metawav.sample_rate, new_freq=16000).to(self.model.device)(llm_wav_data)

            if llm_prompt_metawav.tokens is None:
                llm_prompt_tokens, llm_prompt_token_len = self.frontend._extract_speech_token(llm_wav_data_16k)
            else:
                llm_prompt_tokens = torch.tensor(llm_prompt_metawav.tokens).unsqueeze(0).to(self.model.device)
                llm_prompt_token_len = torch.tensor([len(llm_prompt_tokens)], dtype=torch.int32).to(self.model.device)

            if llm_prompt_metawav.embedding is None:
                llm_prompt_embedding = self.frontend._extract_spk_embedding(llm_wav_data_16k)
            else:
                llm_prompt_embedding = torch.tensor(llm_prompt_metawav.embedding).unsqueeze(0).to(self.model.device)
        else:
            # 1.3.2 use provided llm prompt tokens and embedding
            if llm_prompt_tokens is None or len(llm_prompt_tokens) == 0:
                # 1.3.2.1 no llm prompt tokens. 
                # - if LLM is not sft-ed on single speaker，it performs random speaker sampling.
                # - otherwise, it performs sft behaviour.
                llm_prompt_tokens = torch.zeros(1, 0, dtype=torch.int32)
                llm_prompt_text_token_len = torch.zeros(1, dtype=torch.int32)
            else:
                # 1.3.2.2 have llm prompt tokens. 
                # - in this mode, the LLM performs zero-shot behaviour.
                llm_prompt_tokens = torch.tensor(llm_prompt_tokens, dtype=torch.int32).unsqueeze(0).to(self.model.device)
                llm_prompt_text_token_len = torch.tensor([len(llm_prompt_tokens)], dtype=torch.int32).to(self.model.device)

        # 1.4. run llm
        target_tokens = self.inference_llm(
            tts_text_token=text_tokens,
            tts_text_token_len=text_token_len,
            prompt_text_token=llm_prompt_text_token,
            prompt_text_token_len=llm_prompt_text_token_len,
            prompt_token=llm_prompt_tokens,
            prompt_token_len=llm_prompt_token_len,
            embedding=llm_prompt_embedding,
        )
        return target_tokens

    def _prepare_cfm_prompt(
        self,
        cfm_prompt_metawav: MetaWav,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if cfm_prompt_metawav.wav_data is not None:
            # 2.1 run cfm's prompt_token/promt_feature/embedding extraction if not provided
            cfm_wav_data = torch.tensor(cfm_prompt_metawav.wav_data).unsqueeze(0).to(self.model.device)
            cfm_wav_data_16k = torchaudio.transforms.Resample(orig_freq=cfm_prompt_metawav.sample_rate, new_freq=16000).to(self.model.device)(cfm_wav_data)
            cfm_wav_data_24k = torchaudio.transforms.Resample(orig_freq=cfm_prompt_metawav.sample_rate, new_freq=24000).to(self.model.device)(cfm_wav_data)
            
        if cfm_prompt_metawav.tokens is None:
            cfm_prompt_tokens, cfm_prompt_tokens_len = self.frontend._extract_speech_token(cfm_wav_data_16k)
        else:
            cfm_prompt_tokens = torch.tensor(cfm_prompt_metawav.tokens).unsqueeze(0).to(self.model.device)
            cfm_prompt_tokens_len = torch.tensor([len(cfm_prompt_tokens)], dtype=torch.int32).to(self.model.device)
        
        if cfm_prompt_metawav.feat is None:
            cfm_prompt_features, cfm_prompt_features_len = self.frontend._extract_speech_feat(cfm_wav_data_24k)
        else:
            cfm_prompt_features = torch.tensor(cfm_prompt_metawav.feat).unsqueeze(0).to(self.model.device)
            cfm_prompt_features_len = torch.tensor([len(cfm_prompt_features)], dtype=torch.int32).to(self.model.device)
        
        if cfm_prompt_metawav.embedding is None:
            cfm_prompt_embedding = self.frontend._extract_spk_embedding(cfm_wav_data_16k)
        else:
            cfm_prompt_embedding = torch.tensor(cfm_prompt_metawav.embedding).unsqueeze(0).to(self.model.device)

        return cfm_prompt_tokens, cfm_prompt_features, cfm_prompt_embedding

    def inference_mel_using_metawav(
        self,
        target_metawav: MetaWav,
        llm_prompt_metawav: MetaWav,
        cfm_prompt_metawav: MetaWav,
        target_token_mode: str='infer',
        target_mel_mode: str='infer',
        cfm_streaming_mode: str='none',
        cfm_chunk_size: int=25,
        cfm_lookahead_size: int=3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert cfm_streaming_mode in ['none', 'paste_mel', 'no_paste_mel', 'no_grow']
        assert cfm_chunk_size > cfm_lookahead_size
        if target_mel_mode == 'extract':
            tts_mel, _ = self.frontend._extract_speech_feat(
                torch.tensor(target_metawav.wav_data).unsqueeze(0).to(self.model.device))
            target_tokens = None
            tts_mel = tts_mel.transpose(-1, -2)
            return tts_mel, target_tokens
        
        if target_mel_mode == 'provide':
            tts_mel = torch.tensor(target_metawav.feat).unsqueeze(0).to(self.model.device)
            tts_mel = tts_mel.transpose(-1, -2)
            target_tokens = None
            return tts_mel, target_tokens

        target_tokens = self.inference_target_tokens_using_metawav(
            target_metawav=target_metawav,
            llm_prompt_metawav=llm_prompt_metawav,
            target_token_mode=target_token_mode,
        )

        cfm_prompt_tokens, cfm_prompt_features, cfm_prompt_embedding = self._prepare_cfm_prompt(cfm_prompt_metawav)
        if cfm_streaming_mode == 'none':
            tts_mel = self.inference_cfm(
                target_tokens=target_tokens,
                prompt_tokens=cfm_prompt_tokens,
                prompt_features=cfm_prompt_features,
                cfm_embedding=cfm_prompt_embedding,
            )
        else:
            tts_mel = self.inference_cfm_streaming_mode(
                target_tokens=target_tokens,
                prompt_tokens=cfm_prompt_tokens,
                prompt_features=cfm_prompt_features,
                cfm_embedding=cfm_prompt_embedding,
                streaming_mode=cfm_streaming_mode,
                chunk_size=cfm_chunk_size,
                lookahead_size=cfm_lookahead_size,
            )
        return tts_mel, target_tokens


    def inference_using_metawav(
        self,
        target_metawav: MetaWav,
        llm_prompt_metawav: MetaWav,
        cfm_prompt_metawav: MetaWav,
        target_token_mode: str='infer',
        target_mel_mode: str='infer',
        cfm_streaming_mode: str='none',
        cfm_chunk_size: int=25,
        cfm_lookahead_size: int=3,
    ) -> MetaWav:
        """
        Perform tts using provided metawav. Performs different behaviours based on the provided metawav.
        """
        assert target_token_mode in ['infer', 'provide', 'extract']
        assert target_mel_mode in ['infer', 'provide', 'extract']

        tts_mel, tts_tokens = self.inference_mel_using_metawav(
            target_metawav,
            llm_prompt_metawav,
            cfm_prompt_metawav,
            target_token_mode,
            target_mel_mode,
            cfm_streaming_mode=cfm_streaming_mode,
            cfm_chunk_size=cfm_chunk_size,
            cfm_lookahead_size=cfm_lookahead_size,
        )
        tts_speech = self.inference_hift(tts_mel=tts_mel)

        tts_speech = tts_speech.squeeze(0).detach().cpu().numpy()
        tts_mel = tts_mel.squeeze(0).T.detach().cpu().numpy()
        if tts_tokens is not None:
            tts_tokens = tts_tokens.squeeze(0).detach().cpu().numpy()
            assert len(tts_tokens) * 40 * 24000 // 1000 == len(tts_speech)
        assert len(tts_mel) * 20 * 24000 // 1000 == len(tts_speech)
        tts_metawav = MetaWav(
            wav_data=tts_speech,
            sample_rate=self.sample_rate,
            text=target_metawav.text,
            feat=tts_mel,
            tokens=tts_tokens,
            embedding=None
        )
        return tts_metawav
    

    def inference_with_metawav_long(
        self,
        target_metawav: MetaWav,
        llm_prompt_metawav: MetaWav,
        cfm_prompt_metawav: MetaWav
    ):
        assert target_metawav.tokens is None, 'target tokens is not None'
        # for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
        # TODO: finish this
        pass

