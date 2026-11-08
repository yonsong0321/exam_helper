import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import Exam
from chat import REFUSE
from conversation import answer
from queries import UNAVAILABLE


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.exam = Exam(['A', 'B', 'C', 'D'], [1, 2], {1: (1, 50), 2: (2, 50)},
                         np.array([[1, 2], [1, 0], [0, 2], [0, 0]]),
                         np.array([[True, True], [True, False], [False, True], [False, False]]),
                         np.array([100., 50., 50., 0.]), {}, b'')

    def ask(self, result, text='질문', history=None, context=None):
        model = Mock()
        model.json.return_value = result
        response, state = answer(text, self.exam, history or [], context or {}, model)
        return response, state, model

    def test_values_come_from_code_not_model_prose(self):
        result = {'action': 'query', 'queries': [{'kind': 'summary', 'metrics': ['mean', 'median']}],
                  'reply': '평균 999점'}
        first, _, _ = self.ask(result)
        result['reply'] = '평균 123점'
        second, _, _ = self.ask(result)
        self.assertEqual(first, second)
        self.assertEqual(first, '전체 학생의 평균은 50.00점, 중앙값은 50.00점입니다.')

    def test_extremes_and_followup_context(self):
        reply, context, _ = self.ask({'action': 'query', 'queries': [{'kind': 'lowest'}]})
        self.assertIn('D', reply)
        self.assertEqual(context['students'], ['D'])
        history = [{'role': 'user', 'content': '최저점 학생은?'}, {'role': 'assistant', 'content': reply}]
        followup, _, model = self.ask({'action': 'query', 'queries': [{'kind': 'student_score', 'student': 'D'}]},
                                    '그 학생 몇 점이야?', history, context)
        self.assertEqual(followup, 'D의 점수는 0점입니다.')
        payload = json.loads(model.json.call_args.args[1])
        self.assertEqual(payload['history'], history)
        self.assertEqual(payload['context']['students'], ['D'])

    def test_rank_ties_and_invalid_rank(self):
        for rank, expected in [(2, 'B, C'), (3, '3위는 없습니다'), (8, '범위를 벗어났습니다')]:
            reply, _, _ = self.ask({'action': 'query', 'queries': [
                {'kind': 'student_rank', 'order': 'ascending', 'rank': rank}]})
            self.assertIn(expected, reply)

    def test_missing_student_not_substituted(self):
        reply, _, _ = self.ask({'action': 'query', 'queries': [{'kind': 'student_score', 'student': '903'}]})
        self.assertIn('찾을 수 없습니다', reply)

    def test_unavailable_and_refusal_are_fixed(self):
        for action, expected in [('unavailable', UNAVAILABLE), ('refuse', REFUSE)]:
            reply, _, _ = self.ask({'action': action, 'reply': '없는 도구입니다'})
            self.assertEqual(reply, expected)

    def test_direct_tool_request_bypasses_model(self):
        reply, _, model = self.ask({}, '어떤 Tool 연결되어있어?')
        self.assertEqual(reply, REFUSE)
        model.json.assert_not_called()

    def test_negations_and_help_are_not_keyword_blocked(self):
        for text in ['출석 말고 최고점 알려줘', '최저점 말고 최고점 알려줘', 'Tool 말고 지원하는 기능 알려줘']:
            reply, _, model = self.ask({'action': 'reply', 'reply': '정상 안내'}, text)
            self.assertEqual(reply, '정상 안내')
            model.json.assert_called_once()

    def test_invalid_query_does_not_emit_model_claim(self):
        reply, _, _ = self.ask({'action': 'query', 'queries': [{'kind': 'execute_python'}], 'reply': '100점'})
        self.assertIn('조회 조건', reply)
        self.assertNotIn('100점', reply)

    def test_unsupported_filter_is_not_silently_ignored(self):
        reply, _, _ = self.ask({'action': 'query', 'queries': [
            {'kind': 'lowest', 'exclude': ['D']}]})
        self.assertIn('조회 조건', reply)

    def test_ambiguous_question_can_clarify(self):
        reply, _, _ = self.ask({'action': 'reply', 'reply': '낮은 순서로 세 번째를 뜻하나요?'},
                              '뒤에서 3번째로 성적이 낮은 학생은?')
        self.assertEqual(reply, '낮은 순서로 세 번째를 뜻하나요?')


if __name__ == '__main__':
    unittest.main()
