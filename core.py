from dataclasses import dataclass
from io import BytesIO
import re

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
import pdfplumber


class InputError(ValueError):
    pass


def choice(value):
    text = str(value).strip()
    if text in '①②③④⑤' and len(text) == 1:
        return '①②③④⑤'.index(text) + 1
    if re.fullmatch(r'[1-5](?:\.0)?', text):
        return int(float(text))
    raise InputError(f'선택지는 1~5여야 합니다: {text[:30]}')


def read_key(data):
    result = {}
    try:
        with pdfplumber.open(BytesIO(data)) as pdf:
            if len(pdf.pages) > 20:
                raise InputError('정답 PDF는 20쪽 이내로 업로드해 주세요.')
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        for offset in range(0, len(row) - 2, 3):
                            n, a, points = row[offset:offset + 3]
                            n = re.sub(r'\s+', '', str(n))
                            if not n.isdigit():
                                continue
                            try:
                                number, answer, score = int(n), choice(a), float(str(points).strip())
                            except (ValueError, TypeError):
                                raise InputError(f'{n}번 정답 또는 배점을 읽지 못했습니다.')
                            if number < 1 or not np.isfinite(score) or score <= 0:
                                raise InputError('문항번호 또는 배점이 올바르지 않습니다.')
                            if number in result:
                                raise InputError(f'{number}번 정답이 중복되었습니다. 한 시험의 정답만 업로드해 주세요.')
                            result[number] = (answer, score)
    except InputError:
        raise
    except Exception as exc:
        raise InputError('정답 PDF를 읽지 못했습니다. 텍스트 표가 포함된 정답지를 사용해 주세요.') from exc
    if not result:
        raise InputError('정답 표를 찾지 못했습니다. 스캔 이미지 정답지는 지원하지 않습니다.')
    return dict(sorted(result.items()))


def read_responses(data, numbers):
    try:
        book = openpyxl.load_workbook(BytesIO(data), read_only=True, data_only=False)
        sheet = book['OMR_응답'] if 'OMR_응답' in book.sheetnames else book.active
        rows = list(sheet.values)
        book.close()
    except Exception as exc:
        raise InputError('OMR XLSX를 읽지 못했습니다.') from exc
    if len(rows) < 2 or len(rows) > 10001:
        raise InputError('학생 수는 1~10,000명이어야 합니다.')
    header = [str(v).strip() if v is not None else '' for v in rows[0]]
    if header.count('학생명') != 1:
        raise InputError('학생명 열이 정확히 하나 있어야 합니다.')
    mapping = {}
    for i, label in enumerate(header):
        match = re.fullmatch(r'Q0*(\d+)', label, re.I)
        if match:
            n = int(match[1])
            if n in mapping:
                raise InputError(f'Q{n} 응답 열이 중복되었습니다.')
            mapping[n] = i
    if set(mapping) != set(numbers):
        missing, extra = sorted(set(numbers) - set(mapping)), sorted(set(mapping) - set(numbers))
        raise InputError(f'정답지와 OMR 문항이 다릅니다. 누락: {missing}, 추가: {extra}')
    names, responses = [], []
    for row in rows[1:]:
        if all(v is None for v in row):
            continue
        raw_name = row[header.index('학생명')]
        name = str(raw_name).strip() if raw_name is not None else ''
        if not name or name in names or name.startswith('='):
            raise InputError('학생명이 비어 있거나 중복되었거나 수식입니다.')
        values = []
        for n in numbers:
            value = row[mapping[n]]
            if value is None or str(value).strip() in ('', '0', '0.0'):
                values.append(0)
            else:
                try:
                    values.append(choice(value))
                except InputError as exc:
                    raise InputError(f'{name}, Q{n}: {exc}') from exc
        names.append(name)
        responses.append(values)
    if not names:
        raise InputError('학생 응답이 없습니다.')
    return names, np.array(responses, dtype=int)


def read_questions(data, numbers):
    chunks = []
    try:
        with pdfplumber.open(BytesIO(data)) as pdf:
            if len(pdf.pages) > 50:
                raise InputError('문제 PDF는 50쪽 이내로 업로드해 주세요.')
            for page_index, page in enumerate(pdf.pages):
                # Provided exam uses two columns. Preserve column order and all page text.
                for box in ((0, 0, page.width / 2, page.height),
                            (page.width / 2, 0, page.width, page.height)):
                    text = page.crop(box).extract_text(x_tolerance=2) or ''
                    chunks.append((page_index, text))
    except InputError:
        raise
    except Exception as exc:
        raise InputError('문제 PDF를 읽지 못했습니다.') from exc
    questions, current, shared = {}, None, ''
    shared_members = set()
    for page_index, text in chunks:
        for line in text.splitlines():
            group = re.match(r'^\s*\[(\d+)\s*[～~〜–-]\s*(\d+)\]', line)
            start = re.match(r'^\s*(\d{1,3})\.\s*(.*)', line)
            if group:
                shared_members = set(range(int(group[1]), int(group[2]) + 1))
                shared = line + '\n'
                current = None
            elif start and int(start[1]) in numbers:
                current = int(start[1])
                questions[current] = {'text': (shared if current in shared_members else '') + line + '\n',
                                      'pages': [page_index]}
            elif current is not None:
                questions[current]['text'] += line + '\n'
                if page_index not in questions[current]['pages']:
                    questions[current]['pages'].append(page_index)
            elif shared_members:
                shared += line + '\n'
    return questions


@dataclass
class Exam:
    names: list
    numbers: list
    key: dict
    responses: np.ndarray
    correct: np.ndarray
    scores: np.ndarray
    questions: dict
    problem_pdf: bytes

    @property
    def summary(self):
        return {'학생 수': len(self.names), '만점': sum(v[1] for v in self.key.values()),
                '평균': float(self.scores.mean()), '중앙값': float(np.median(self.scores)),
                '표준편차': float(self.scores.std(ddof=0))}

    def ranking(self):
        counts = (~self.correct).sum(axis=0)
        return sorted(zip(self.numbers, counts.tolist()), key=lambda item: (-item[1], item[0]))

    def student(self, query):
        query = str(query).strip()
        if query in self.names:
            return self.names.index(query)
        canonical = lambda s: re.sub(r'[\s번학생]', '', s)
        matches = [i for i, name in enumerate(self.names) if canonical(name) == canonical(query)]
        return matches[0] if len(matches) == 1 else None


def analyze(problem, answers, omr):
    key = read_key(answers)
    numbers = list(key)
    names, responses = read_responses(omr, numbers)
    expected = np.array([key[n][0] for n in numbers])
    correct = responses == expected
    scores = correct @ np.array([key[n][1] for n in numbers])
    questions = read_questions(problem, numbers)
    if not questions:
        raise InputError('문제지에서 문항을 찾지 못했습니다. 텍스트가 포함된 문제지를 업로드해 주세요.')
    return Exam(names, numbers, key, responses, correct, scores, questions, problem)


def safe_text(text):
    value = str(text)
    return "'" + value if value.startswith(('=', '+', '-', '@')) else value


def export_excel(exam):
    wb = openpyxl.Workbook()
    overview = wb.active
    overview.title = '전체 성적 개요'
    overview.append(['항목', '값'])
    for label, value in exam.summary.items():
        overview.append([label, value])
    overview.append(['표준편차 기준', '모집단 (ddof=0)'])
    overview.append(['동률 정렬', '오답 수 내림차순, 문항번호 오름차순'])
    overview.append([])
    overview.append(['최다 오답 문항', '오답 수', '오답률'])
    for n, count in exam.ranking()[:3]:
        overview.append([n, count, count / len(exam.names)])
        overview.cell(overview.max_row, 3).number_format = '0.0%'
    students = wb.create_sheet('학생별 성적')
    students.append(['학생명', '총점'] + [f'Q{n:02}' for n in exam.numbers])
    for name, score, row in zip(exam.names, exam.scores, exam.correct):
        students.append([safe_text(name), float(score)] + ['O' if v else 'X' for v in row])
    items = wb.create_sheet('문항별 분석')
    items.append(['문항번호', '정답', '배점', '오답 수', '오답률'])
    counts = dict(exam.ranking())
    for n, (a, points) in exam.key.items():
        items.append([n, a, points, counts[n], counts[n] / len(exam.names)])
        items.cell(items.max_row, 5).number_format = '0.0%'
    for sheet in wb:
        sheet.freeze_panes = 'C2' if sheet == students else 'A2'
        if sheet != overview:
            sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(name='맑은 고딕', color='FFFFFF', bold=True)
            cell.fill = PatternFill('solid', fgColor='2563EB')
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.font = Font(name='맑은 고딕', size=11)
                cell.alignment = Alignment(vertical='center')
                if isinstance(cell.value, float) and cell.number_format == 'General':
                    cell.number_format = '0.00'
        for column in sheet.columns:
            letter = column[0].column_letter
            sheet.column_dimensions[letter].width = 23 if column[0].column == 1 else 14
        if sheet == overview:
            sheet.column_dimensions['B'].width = 46
    out = BytesIO()
    wb.save(out)
    return out.getvalue()
