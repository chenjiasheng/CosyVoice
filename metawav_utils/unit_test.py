import sys
sys.path.append('third_party/Matcha-TTS')
from metawav_utils.metawav import MetaWav
from metawav_utils.inferrer import Inferrer
import torch
from cosyvoice.cli.cosyvoice import CosyVoice2
import sys
import os


def make_output_dir():
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


def vocoder_reconstruct(inferrer, example_audios_dir):
    """
    vocoder分析合成
    """
    target_metawav = MetaWav.load_from_wav_file(os.path.join(example_audios_dir, 'F143_prompt.meta.wav'))
    tts_wav = inferrer.inference_using_metawav(
        target_metawav,
        llm_prompt_metawav=None,
        cfm_prompt_metawav=None,
        target_mel_mode='provide',
    )
    MetaWav.write_to_wav_file(os.path.join(make_output_dir(), 'vocoder_reconstruct.meta.wav'), tts_wav)


def cfm_reconstruct(inferrer, example_audios_dir):
    """
    cfm vocoder分析合成
    """
    target_metawav = MetaWav.load_from_wav_file(os.path.join(example_audios_dir, 'F143_prompt.meta.wav'))
    tts_wav = inferrer.inference_using_metawav(
        target_metawav,
        llm_prompt_metawav=None,
        cfm_prompt_metawav=target_metawav,
        target_token_mode='provide',
        target_mel_mode='infer',
    )
    MetaWav.write_to_wav_file(os.path.join(make_output_dir(), 'cfm_reconstruct.meta.wav'), tts_wav)


def streaming_cfm_reconstruct(inferrer, example_audios_dir):
    """
    cfm vocoder分析合成
    """
    target_metawav = MetaWav.load_from_wav_file(os.path.join(example_audios_dir, 'F143_prompt.meta.wav'))
    tts_wav = inferrer.inference_using_metawav(
        target_metawav,
        llm_prompt_metawav=None,
        cfm_prompt_metawav=target_metawav,
        target_token_mode='provide',
        target_mel_mode='infer',
        cfm_streaming_mode='paste_mel',
    )
    MetaWav.write_to_wav_file(os.path.join(make_output_dir(), 'streaming_cfm_reconstruct.meta.wav'), tts_wav)


def vc(inferrer, example_audios_dir):
    """
    vc分析合成
    """
    target_metawav = MetaWav.load_from_wav_file(os.path.join(example_audios_dir, 'F29_prompt.meta.wav'))
    cfm_prompt_metawav = MetaWav.load_from_wav_file(os.path.join(example_audios_dir, 'F143_prompt.meta.wav'))
    tts_wav = inferrer.inference_using_metawav(
        target_metawav,
        llm_prompt_metawav=None,
        cfm_prompt_metawav=cfm_prompt_metawav,
        target_token_mode='provide',
        target_mel_mode='infer',
    )
    MetaWav.write_to_wav_file(os.path.join(make_output_dir(), 'vc.meta.wav'), tts_wav)


def tts(inferrer, example_audios_dir):
    """
    tts常规合成
    """
    target_metawav = MetaWav(text='今天天气怎么样呀？')
    llm_prompt_metawav = MetaWav.load_from_wav_file(os.path.join(example_audios_dir, 'F143_prompt.meta.wav'))
    cfm_prompt_metawav = MetaWav.load_from_wav_file(os.path.join(example_audios_dir, 'F143_prompt.meta.wav'))
    tts_wav = inferrer.inference_using_metawav(
        target_metawav,
        llm_prompt_metawav=llm_prompt_metawav,
        cfm_prompt_metawav=cfm_prompt_metawav,
    )
    MetaWav.write_to_wav_file(os.path.join(make_output_dir(), 'tts.meta.wav'), tts_wav)


def vc_tts(inferrer, example_audios_dir):
    """
    vc+tts合成
    """
    target_metawav = MetaWav(text='今天天气怎么样呀？')
    llm_prompt_metawav = MetaWav.load_from_wav_file(os.path.join(example_audios_dir, 'F29_prompt.meta.wav'))
    cfm_prompt_metawav = MetaWav.load_from_wav_file(os.path.join(example_audios_dir, 'F143_prompt.meta.wav'))
    tts_wav = inferrer.inference_using_metawav(
        target_metawav,
        llm_prompt_metawav=llm_prompt_metawav,
        cfm_prompt_metawav=cfm_prompt_metawav,
        target_token_mode='infer',
        target_mel_mode='infer',
    )
    MetaWav.write_to_wav_file(os.path.join(make_output_dir(), 'vc_tts.meta.wav'), tts_wav)


def jdslj_tts(inferrer, example_audios_dir):
    """
    jdslj tts合成
    """
    jdslj_llm_model_dir = '/mnt/d/exp/jdslj/cosyvoice2/CosyVoice2-0.5B/llm/torch_ddp/epoch_1_whole.pt'
    inferrer.model.llm.load_state_dict(torch.load(jdslj_llm_model_dir, map_location='cuda'), strict=False)
    target_metawav = MetaWav(text='今天天气怎么样呀？')
    llm_prompt_metawav = MetaWav.load_from_wav_file(os.path.join(example_audios_dir, 'zhoujie_prompt.meta.wav'))
    cfm_prompt_metawav = MetaWav.load_from_wav_file(os.path.join(example_audios_dir, 'zhoujie_prompt.meta.wav'))
    tts_wav = inferrer.inference_using_metawav(
        target_metawav,
        llm_prompt_metawav=llm_prompt_metawav,
        cfm_prompt_metawav=cfm_prompt_metawav,
        target_token_mode='infer',
        target_mel_mode='infer',
    )
    MetaWav.write_to_wav_file(os.path.join(make_output_dir(), 'jdslj_tts.meta.wav'), tts_wav)


if __name__ == '__main__':
    example_audios_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example_audios')
    official_model_dir = '/home/chenjiasheng/model/CosyVoice/pretrained_models/CosyVoice2-0.5B'
    inferrer = Inferrer(model_dir=official_model_dir, load_jit=False, load_trt=False, fp16=False)
    vocoder_reconstruct(inferrer, example_audios_dir)
    cfm_reconstruct(inferrer, example_audios_dir)
    streaming_cfm_reconstruct(inferrer, example_audios_dir)
    vc(inferrer, example_audios_dir)
    tts(inferrer, example_audios_dir)
    vc_tts(inferrer, example_audios_dir)
    jdslj_tts(inferrer, example_audios_dir)
    print('Done')
