from io import BytesIO
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chat import DEFAULT_MODEL, Model, ModelError


class ModelConfigTests(unittest.TestCase):
    def request_body(self, model, content='question'):
        response = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"action":"reply"}'}}]}
        with patch('chat.urlopen', return_value=BytesIO(json.dumps(response).encode())) as call:
            self.assertEqual(model.json('Return JSON.', content, timeout=10, tokens=1800), {'action': 'reply'})
            self.assertEqual(call.call_args.kwargs['timeout'], 10)
            return json.loads(call.call_args.args[0].data)

    def test_default_model_uses_low_reasoning_with_token_headroom(self):
        body = self.request_body(Model('test-only'))
        self.assertEqual(body['model'], DEFAULT_MODEL)
        self.assertEqual(body['reasoning_effort'], 'low')
        self.assertEqual(body['max_completion_tokens'], 4096)
        self.assertEqual(body['response_format'], {'type': 'json_object'})

    def test_image_content_is_preserved(self):
        content = [{'type': 'text', 'text': 'Return JSON.'},
                   {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]
        body = self.request_body(Model('test-only'), content)
        self.assertEqual(body['messages'][1]['content'], content)

    def test_legacy_model_does_not_get_unsupported_reasoning_parameter(self):
        body = self.request_body(Model('test-only', 'gpt-4.1-mini'))
        self.assertNotIn('reasoning_effort', body)
        self.assertEqual(body['max_completion_tokens'], 1800)

    def test_invalid_reasoning_is_rejected_before_network(self):
        with patch('chat.urlopen') as call:
            with self.assertRaises(ModelError):
                Model('test-only', reasoning_effort='none').json('JSON', 'test')
            call.assert_not_called()

    def test_truncated_output_is_not_accepted_as_success(self):
        response = {'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]}
        with patch('chat.urlopen', return_value=BytesIO(json.dumps(response).encode())):
            with self.assertRaises(ModelError):
                Model('test-only').json('JSON', 'test')


if __name__ == '__main__':
    unittest.main()
