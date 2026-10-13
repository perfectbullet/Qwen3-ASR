"""CPU regression checks for bounded streaming sessions (no model loading).

Run with the demo dependencies installed:
    python tests/test_streaming_sessions.py [path/to/demo_streaming.py]
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from dataclasses import dataclass, field
from unittest.mock import patch

import numpy as np


demo_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'qwen_asr/cli/demo_streaming.py'
stub = types.ModuleType('qwen_asr')
stub.Qwen3ASRModel = object
with patch.dict(sys.modules, {'qwen_asr': stub}):
    spec = importlib.util.spec_from_file_location('streaming_demo_test', demo_path)
    demo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = demo
    spec.loader.exec_module(demo)


@dataclass
class State:
    chunk_size_samples: int = 4
    audio_accum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    text: str = ''
    language: str = ''


class FakeASR:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.successful_sizes = []

    def init_streaming_state(self, **kwargs):
        return State()

    def decode(self, state):
        if state.audio_accum.size > self.capacity:
            raise ValueError('The decoder prompt (length 8198) is longer than the maximum model length of 8192.')
        self.successful_sizes.append(state.audio_accum.size)
        state.text = ''.join(chr(int(value)) for value in state.audio_accum)
        state.language = 'Chinese'

    def streaming_transcribe(self, wav, state):
        state.buffer = np.concatenate([state.buffer, wav])
        while state.buffer.size >= state.chunk_size_samples:
            part = state.buffer[:state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples:]
            state.audio_accum = np.concatenate([state.audio_accum, part])
            self.decode(state)

    def finish_streaming_transcribe(self, state):
        if state.buffer.size:
            state.audio_accum = np.concatenate([state.audio_accum, state.buffer])
            state.buffer = np.zeros(0, dtype=np.float32)
            self.decode(state)


class StreamingTests(unittest.TestCase):
    def setUp(self):
        demo.SESSIONS.clear()
        demo.asr = FakeASR()
        demo.SEGMENT_MAX_SEC = 8 / demo.SAMPLE_RATE
        self.client = demo.app.test_client()
        self.sid = self.client.post('/api/start').get_json()['session_id']

    def push(self, text):
        audio = np.array([ord(c) for c in text], dtype=np.float32)
        response = self.client.post('/api/chunk', query_string={'session_id': self.sid},
                                    data=audio.tobytes(), content_type='application/octet-stream')
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()

    def finish(self):
        response = self.client.post('/api/finish', query_string={'session_id': self.sid})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()

    def test_large_upload_rolls_over_without_losing_or_duplicating_audio(self):
        original = 'abcdefghijklmnopqrstuvwxyz'
        result = self.push(original)
        self.assertEqual(result['segments_completed'], 3)
        self.assertLessEqual(demo._segment_samples(demo.SESSIONS[self.sid]), 8)
        self.assertEqual(self.finish()['text'], original)
        self.assertLessEqual(max(demo.asr.successful_sizes), 8)
        self.assertNotIn(self.sid, demo.SESSIONS)

    def test_small_uploads_preserve_completed_text_and_final_tail(self):
        for part in ['ab', 'cde', 'fghi', 'jkl', 'm']:
            self.push(part)
        self.assertEqual(self.finish()['text'], 'abcdefghijklm')

    def test_overflow_restores_state_and_replays_pending_samples(self):
        demo.SEGMENT_MAX_SEC = 30
        demo.asr.capacity = 6
        self.push('abcdef')
        self.push('ghijklmnopq')
        self.assertEqual(self.finish()['text'], 'abcdefghijklmnopq')
        self.assertLessEqual(max(demo.asr.successful_sizes), 6)

    def test_overflow_on_finish_preserves_partial_tail(self):
        demo.SEGMENT_MAX_SEC = 30
        demo.asr.capacity = 4
        self.push('abcdef')
        self.assertEqual(self.finish()['text'], 'abcdef')

    def test_too_small_context_returns_actionable_error(self):
        demo.asr.capacity = 2
        response = self.client.post('/api/chunk', query_string={'session_id': self.sid},
                                    data=np.ones(4, dtype=np.float32).tobytes(),
                                    content_type='application/octet-stream')
        self.assertEqual(response.status_code, 422)
        self.assertIn('chunk-size-sec', response.get_json()['error'])
        self.assertEqual(demo._segment_samples(demo.SESSIONS[self.sid]), 0)

    def test_quiet_boundary_finalizes_before_maximum(self):
        demo.SEGMENT_MAX_SEC = 2
        demo.asr.init_streaming_state = lambda **kwargs: State(chunk_size_samples=16000)
        demo.SESSIONS[self.sid].state = demo.asr.init_streaming_state()
        demo.asr.capacity = 32000
        demo.asr.decode = lambda state: None
        audio = np.concatenate([np.full(8000, 0.1, dtype=np.float32), np.zeros(8000, dtype=np.float32)])
        demo._feed_session(demo.SESSIONS[self.sid], audio)
        self.assertEqual(demo.SESSIONS[self.sid].segments_completed, 1)
        self.assertEqual(demo._segment_samples(demo.SESSIONS[self.sid]), 0)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]], verbosity=2)
