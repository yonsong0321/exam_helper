import hashlib
import os
import time

import pandas as pd
import streamlit as st

from core import InputError, analyze, export_excel
from chat import Model
from conversation import answer

st.set_page_config(page_title='영어학원 평가 도우미', page_icon='📘', layout='wide')


def setting(name, default=''):
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except FileNotFoundError:
        return os.environ.get(name, default)


def clear_analysis():
    for key in ('exam', 'excel', 'messages', 'context', 'pending_analysis', 'analysis_error'):
        st.session_state.pop(key, None)


def landing():
    st.image("https://www.gachon.ac.kr/sites/kor/images/sub/slogan_2.png", width=340)
    st.caption('ENGLISH ASSESSMENT ASSISTANT')
    st.title('채점부터 다음 수업 준비까지')
    st.subheader('영어학원 평가 도우미')
    st.write('시험 자료를 올리면 학생별 성적과 문항별 오답을 분석합니다. '
             '궁금한 결과를 대화로 확인하고, 취약한 유형의 연습문제를 만들어 보세요.')
    cols = st.columns(3)
    for col, title, body in zip(cols, ['01 · 자동 채점', '02 · 성적 분석', '03 · 맞춤 연습'],
            ['정답 PDF의 배점으로 정확하게 채점합니다.',
             '전체 통계와 학생별 결과를 엑셀로 내려받습니다.',
             '오답이 많은 유형을 바탕으로 새로운 영어 문제를 만듭니다.']):
        with col:
            with st.container(border=True):
                st.subheader(title)
                st.write(body)
    st.divider()
    st.write('**준비할 파일:** 문제지 PDF, 정답 PDF, OMR 응답 XLSX')
    st.caption('OMR은 사진이 아닌 인식 완료된 표 데이터입니다. 학생명과 Q01, Q02… 열을 포함해 주세요.')
    st.page_link(analysis_page, label='자료를 올리고 분석 시작하기', icon='➡️')


def analysis_view():
    st.title('시험 분석')
    pending = st.session_state.pop('pending_analysis', None)
    if pending is not None:
        try:
            if any(len(data) > 20 * 1024 * 1024 for data in pending):
                raise InputError('각 파일은 20MB 이내여야 합니다.')
            with st.spinner('정답 추출 및 채점 중…'):
                exam = analyze(*pending)
                excel = export_excel(exam)
            st.session_state.update(exam=exam, excel=excel, messages=[], context={})
        except InputError as exc:
            st.session_state['analysis_error'] = str(exc)
        except Exception:
            st.session_state['analysis_error'] = '자료 처리에 실패했습니다. 파일 형식과 내용을 확인해 주세요.'
        st.rerun()
    if st.session_state.get('analysis_error'):
        st.error(st.session_state['analysis_error'])
    exam = st.session_state.get('exam')
    if exam is None:
        st.info('왼쪽에서 파일 3개를 업로드하고 분석 시작을 눌러 주세요.')
        st.chat_input('채점 완료 후 질문할 수 있어요.', disabled=True)
        return
    summary = exam.summary
    cols = st.columns(4)
    for col, label, value in zip(cols, ['학생 수', '평균', '중앙값', '만점'],
                [f'{len(exam.names)}명', f'{summary["평균"]:.2f}점',
                 f'{summary["중앙값"]:.2f}점', f'{summary["만점"]:g}점']):
        col.metric(label, value)
    with st.expander('학생별 성적과 오답 현황 보기'):
        left, right = st.columns(2)
        left.dataframe(pd.DataFrame({'학생명': exam.names, '점수': exam.scores}), hide_index=True)
        right.dataframe(pd.DataFrame(exam.ranking(), columns=['문항번호', '오답 수']), hide_index=True)
    st.caption('예: 평균과 중앙값 알려줘 / 가장 성적이 저조한 학생은? / 가장 많이 틀린 문제로 하나 만들어줘')
    messages = st.session_state.setdefault('messages', [])
    for message in messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    prompt = st.chat_input('성적이나 연습문제에 대해 질문해 주세요.', max_chars=3000)
    if prompt:
        with st.chat_message('user'):
            st.write(prompt)
        model = Model(setting('OPENAI_API_KEY'), setting('OPENAI_MODEL', 'gpt-4.1-mini'))
        start = time.monotonic()
        with st.chat_message('assistant'):
            with st.spinner('확인하고 있어요…'):
                try:
                    response, context = answer(prompt, exam, messages,
                                               st.session_state.get('context', {}), model)
                except Exception:
                    response, context = '요청을 처리하지 못했습니다. 질문이나 자료를 확인하고 다시 시도해 주세요.', st.session_state.get('context', {})
            st.markdown(response)
            st.caption(f'응답 완료 · {time.monotonic() - start:.2f}초')
        messages.extend([{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}])
        st.session_state['context'] = context


home_page = st.Page(landing, title='서비스 소개', icon='🏠', default=True)
analysis_page = st.Page(analysis_view, title='시험 분석', icon='📊')
page = st.navigation([home_page, analysis_page], position="top")

with st.sidebar:
    st.subheader('시험 자료')
    problem = st.file_uploader('문제지 PDF', type='pdf', key='problem_upload')
    key = st.file_uploader('정답 PDF', type='pdf', key='key_upload')
    omr = st.file_uploader('OMR 응답 XLSX', type='xlsx', key='omr_upload')
    files = [problem, key, omr]
    fingerprint = tuple(
        hashlib.sha256(upload.getvalue()).hexdigest() if upload else None
        for upload in files
    )

    if st.session_state.get('fingerprint') != fingerprint:
        clear_analysis()
        st.session_state['fingerprint'] = fingerprint

    ready = all(upload is not None for upload in files)
    if st.button(
        '분석 시작',
        type='primary',
        disabled=page.url_path == home_page.url_path or not ready,
        use_container_width=True,
    ):
        clear_analysis()
        st.session_state['pending_analysis'] = tuple(
            upload.getvalue() for upload in files
        )
        st.switch_page(analysis_page)

    st.download_button(
        '엑셀 추출',
        data=st.session_state.get('excel', b''),
        file_name='exam_results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        disabled='exam' not in st.session_state,
        use_container_width=True,
    )
    if not setting('OPENAI_API_KEY'):
        st.warning('LLM 설정이 필요합니다. 채점·엑셀·기본 성적 조회는 사용 가능합니다.')
    st.caption('자료와 대화는 현재 세션에서만 사용합니다. 새 파일로 바꾸면 분석과 대화가 초기화됩니다.')

page.run()
