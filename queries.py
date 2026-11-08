"""Validated exam queries and deterministic response templates."""
import re


UNAVAILABLE = '업로드된 시험 자료에는 해당 정보가 없어 확인할 수 없습니다.'


def explicit_private_request(text):
    # Only standalone disclosure requests bypass language interpretation.
    return bool(re.fullmatch(
        r'\s*(?:어떤\s*)?(?:tool|tools|도구)\s*(?:이|가)?\s*연결\s*(?:되어|돼)\s*있(?:어|어요|나요|습니까)\s*[?.？]*\s*',
        text, re.I))


def render_queries(queries, exam, context):
    if not isinstance(queries, list) or not 1 <= len(queries) <= 8:
        raise ValueError('invalid queries')
    lines, focus = [], []
    for query in queries:
        if not isinstance(query, dict):
            raise ValueError('invalid query')
        kind = query.get('kind')
        fields = {'summary': {'metrics'}, 'lowest': set(), 'highest': set(),
                  'student_score': {'student'}, 'student_wrong': {'student'},
                  'student_rank': {'order', 'rank'}, 'unavailable': set()}
        if not isinstance(kind, str) or kind not in fields or set(query) - ({'kind'} | fields[kind]):
            raise ValueError('unsupported query fields')
        if kind == 'summary':
            metrics = query.get('metrics')
            labels = {'mean': '평균', 'median': '중앙값', 'std': '표준편차'}
            if not isinstance(metrics, list) or not metrics or any(x not in labels for x in metrics):
                raise ValueError('invalid metrics')
            selected = [k for k in labels if k in metrics]
            values = [f'{labels[k]}은 {exam.summary[labels[k]]:.2f}점' for k in selected]
            lines.append('전체 학생의 ' + ', '.join(values) + '입니다.')
        elif kind in ('lowest', 'highest'):
            score = float(exam.scores.min() if kind == 'lowest' else exam.scores.max())
            names = [n for n, s in zip(exam.names, exam.scores) if s == score]
            focus.extend(names)
            label = '가장 성적이 저조한' if kind == 'lowest' else '가장 성적이 높은'
            lines.append(f'{label} 학생은 {", ".join(names)}이며, 점수는 {score:g}점입니다.')
        elif kind in ('student_score', 'student_wrong'):
            student = query.get('student')
            if not isinstance(student, str) or not student.strip():
                raise ValueError('missing student')
            index = exam.student(student)
            if index is None:
                lines.append(f'업로드된 자료에서 {student} 학생을 찾을 수 없습니다.')
                continue
            name = exam.names[index]
            focus.append(name)
            if kind == 'student_score':
                lines.append(f'{name}의 점수는 {exam.scores[index]:g}점입니다.')
            else:
                wrong = [f'{n}번' for n, ok in zip(exam.numbers, exam.correct[index]) if not ok]
                lines.append(f'{name}의 틀린 문항은 {", ".join(wrong)}입니다.' if wrong
                             else f'{name}은 모든 문항을 맞혔습니다.')
        elif kind == 'student_rank':
            order, rank = query.get('order'), query.get('rank')
            if order not in ('ascending', 'descending') or type(rank) is not int or rank < 1:
                raise ValueError('invalid rank')
            # Competition ranking: ties share a rank and occupy their places.
            scores = sorted(exam.scores.tolist(), reverse=order == 'descending')
            if rank > len(scores):
                lines.append(f'학생은 총 {len(scores)}명으로, 요청한 순위가 범위를 벗어났습니다.')
                continue
            score = scores[rank - 1]
            actual = scores.index(score) + 1
            if actual != rank:
                lines.append(f'동점자는 공동 순위로 처리하므로 {rank}위는 없습니다. 해당 위치는 공동 {actual}위입니다.')
                continue
            names = [n for n, s in zip(exam.names, exam.scores) if s == score]
            focus.extend(names)
            direction = '낮은' if order == 'ascending' else '높은'
            lines.append(f'점수가 {direction} 순서로 {rank}위인 학생은 {", ".join(names)}이며, {score:g}점입니다.'
                         + (' 동점자는 공동 순위로 처리합니다.' if len(names) > 1 else ''))
        elif kind == 'unavailable':
            lines.append(UNAVAILABLE)
        else:
            raise ValueError('unsupported query')
    updated = dict(context)
    if focus:
        updated['students'] = list(dict.fromkeys(focus))
    return '\n\n'.join(lines), updated
