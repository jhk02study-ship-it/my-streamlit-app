
import os
import io
import base64
import smtplib
from datetime import date
from email.message import EmailMessage

import streamlit as st
from openai import OpenAI


# ==================================================
# 1. 기본 환경 설정
# ==================================================

st.set_page_config(
    page_title="AIVLE AI 개인비서",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_secret(key, default=""):
    value = os.getenv(key)
    if value:
        return value

    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


api_key = get_secret("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY가 등록되지 않았습니다.")
    st.stop()

client = OpenAI(api_key=api_key)


# ==================================================
# 2. 상태 관리
# ==================================================

DEFAULTS = {
    "page": "대시보드",
    "dashboard_view": "home",
    "user_name": "사용자",
    "demo_login": False,
    "voice": "alloy",
    "auto_tts": True,
    "history": [],
    "meetings": [],
    "schedules": [],
    "stt_result": "",
    "last_answer": "",
    "last_audio": None,
    "last_mode": "",
    "last_question": ""
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# 3. 화면 디자인
# ==================================================

st.markdown("""
<style>
.block-container {
    max-width: 1250px;
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(120, 120, 120, 0.25);
}

div[data-testid="stButton"] button {
    border-radius: 12px;
    min-height: 42px;
}

div[data-testid="stPopover"] button {
    border-radius: 12px;
}

div[data-testid="stTextInput"] input {
    border-radius: 14px;
}

div[data-testid="stAudioInput"] {
    border-radius: 16px;
}

.main-title {
    font-size: 1.7rem;
    font-weight: 750;
    margin-bottom: .2rem;
}

.stt-label {
    font-weight: 650;
    margin-bottom: .5rem;
}

.mic-label {
    text-align: center;
    font-size: 1.1rem;
    font-weight: 650;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 4. 공통 함수
# ==================================================

def navigate(page, view=None):
    st.session_state.page = page

    if page == "대시보드":
        st.session_state.dashboard_view = view or "home"

    st.rerun()


def save_history(mode, question, answer):
    st.session_state.history.append({
        "mode": mode,
        "question": question,
        "answer": answer
    })

    # 질문과 답변을 하나의 대화로 계산
    st.session_state.history = st.session_state.history[-10:]


def read_text_file(uploaded):
    if uploaded is None:
        return ""

    data = uploaded.getvalue()

    if len(data) > 5 * 1024 * 1024:
        raise ValueError("TXT 파일은 5MB 이하로 첨부하세요.")

    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp949")


def speech_to_text(audio):
    audio_bytes = io.BytesIO(audio.getvalue())
    audio_bytes.name = "recording.wav"

    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_bytes
    )

    return result.text


def ask_assistant(mode, question, reference="", image=None):
    system_prompt = """
당신은 AI 개인비서 '에이블'입니다.

역할:
- 김 대리의 업무를 지원하는 전문 AI 개인비서
- 업무 관련 질문에 친절하고 정확하게 답변
- 문서 및 회의 요약
- 이메일 작성 및 일정 정리 지원

규칙:
- 한국어로 답변합니다.
- 이전 대화의 맥락을 고려합니다.
- 확인되지 않은 사실은 임의로 만들지 않습니다.
- 중요한 사항은 보기 쉽게 정리합니다.
"""

    if mode == "회의 요약":
        system_prompt += """
회의 내용을 아래 형식으로 정리하세요.

1. 회의 주제
2. 핵심 논의사항
3. 결정사항
4. 담당자별 후속 업무
5. 미결정 사항

원문에서 확인되지 않는 담당자, 날짜,
결정사항은 만들어내지 마세요.
"""

    elif mode == "일정 관리":
        system_prompt += """
사용자의 일정을 정리하고 업무 우선순위를
추천하는 개인비서 역할을 수행하세요.

일정과 마감일이 불명확하다면
확인되지 않았음을 표시하세요.
"""

    if reference:
        system_prompt += (
            "\n\n첨부한 참고자료:\n"
            + reference[:16000]
        )

    messages = []

    # 최근 5개 대화를 현재 요청에 활용
    for item in st.session_state.history[-5:]:
        messages.append({
            "role": "user",
            "content": item["question"]
        })
        messages.append({
            "role": "assistant",
            "content": item["answer"]
        })

    user_content = [
        {"type": "input_text", "text": question}
    ]

    # 사진 첨부 시 이미지 분석
    if image is not None:
        raw = image.getvalue()

        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("사진은 8MB 이하로 첨부하세요.")

        mime = image.type or "image/png"
        encoded = base64.b64encode(raw).decode("utf-8")

        user_content.append({
            "type": "input_image",
            "image_url": f"data:{mime};base64,{encoded}"
        })

    messages.append({
        "role": "user",
        "content": user_content
    })

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=system_prompt,
        input=messages
    )

    return response.output_text


def text_to_speech(text):
    result = client.audio.speech.create(
        model="tts-1",
        voice=st.session_state.voice,
        input=text[:4000],
        response_format="mp3"
    )

    return result.read()


def send_email(receiver, subject, body):
    host = get_secret("SMTP_HOST", "smtp.gmail.com")
    port = int(get_secret("SMTP_PORT", "465"))
    user = get_secret("SMTP_USER")
    password = get_secret("SMTP_PASSWORD")

    if not user or not password:
        raise ValueError("이메일 인증 정보가 없습니다.")

    if not receiver.strip() or "@" not in receiver:
        raise ValueError("받는 사람 이메일을 확인하세요.")

    if not subject.strip() or not body.strip():
        raise ValueError("제목과 내용을 입력하세요.")

    message = EmailMessage()
    message["From"] = user
    message["To"] = receiver.strip()
    message["Subject"] = subject
    message.set_content(body)

    with smtplib.SMTP_SSL(
        host, port, timeout=20
    ) as server:
        server.login(user, password)
        server.send_message(message)


# ==================================================
# 5. 왼쪽 메뉴 - 실제 페이지 이동
# ==================================================

with st.sidebar:

    st.title("🤖 AIVLE 비서")
    st.caption("나만의 AI 업무 개인비서")
    st.divider()

    menus = [
        ("🏠", "대시보드"),
        ("📝", "회의 요약"),
        ("📅", "일정 관리")
    ]

    for emoji, page in menus:

        active = st.session_state.page == page

        if st.button(
            f"🔴 {emoji} {page}",
            key=f"nav_{page}",
            use_container_width=True,
            type="primary" if active else "secondary"
        ):
            navigate(page)

    st.divider()
    st.subheader("최근 대화")

    if not st.session_state.history:
        st.caption("아직 저장된 대화가 없습니다.")

    for i, item in enumerate(
        reversed(st.session_state.history), 1
    ):
        if st.button(
            f"💬 {item['question'][:20]}",
            key=f"recent_{i}",
            use_container_width=True
        ):
            st.session_state.last_answer = item["answer"]
            st.session_state.last_question = item["question"]
            st.session_state.last_mode = "대시보드"
            st.session_state.last_audio = None
            navigate("대시보드", "assistant")

    if st.button(
        "🗑️ 대화 기록 삭제",
        use_container_width=True
    ):
        st.session_state.history = []
        st.session_state.last_answer = ""
        st.session_state.last_audio = None
        st.rerun()


# ==================================================
# 6. 상단 프로필 메뉴
# ==================================================

header_col, profile_col = st.columns([8, 2])

with header_col:
    st.markdown(
        '<div class="main-title">AIVLE 비서</div>',
        unsafe_allow_html=True
    )

with profile_col:
    with st.popover(
        f"👤 {st.session_state.user_name}님",
        use_container_width=True
    ):
        st.markdown("#### 프로필 설정")

        new_name = st.text_input(
            "사용자 이름",
            value=st.session_state.user_name
        )

        if st.button("이름 저장"):
            st.session_state.user_name = (
                new_name.strip() or "사용자"
            )
            st.rerun()

        st.divider()

        st.markdown("#### 비서 음성 설정")

        st.selectbox(
            "음성 선택",
            ["alloy", "echo", "fable",
             "onyx", "nova", "shimmer"],
            key="voice"
        )

        st.checkbox(
            "답변 자동 음성 재생",
            key="auto_tts"
        )

        st.divider()

        # 실제 사용자 인증이 아닌 시연용 기능
        st.caption("데모 로그인 (실제 계정 인증 아님)")

        if not st.session_state.demo_login:
            if st.button("데모 로그인"):
                st.session_state.demo_login = True
                st.rerun()
        else:
            st.success("데모 로그인 상태")

            if st.button("로그아웃"):
                st.session_state.demo_login = False
                st.rerun()

st.divider()


# ==================================================
# 7. AI 답변, TTS 및 이메일 발송
# ==================================================

def show_answer(mode):

    if not st.session_state.last_answer:
        return

    if st.session_state.last_mode != mode:
        return

    st.divider()
    st.subheader("🤖 에이블의 답변")

    st.markdown(st.session_state.last_answer)

    # LLM 응답을 음성으로 출력
    if st.session_state.last_audio is not None:
        st.audio(
            st.session_state.last_audio,
            format="audio/mpeg",
            autoplay=st.session_state.auto_tts
        )

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "🔊 답변 다시 듣기",
            key=f"tts_{mode}",
            use_container_width=True
        ):
            try:
                with st.spinner("음성을 생성하고 있습니다..."):
                    st.session_state.last_audio = (
                        text_to_speech(
                            st.session_state.last_answer
                        )
                    )
                st.rerun()
            except Exception as e:
                st.error(f"음성 생성 오류: {e}")

    with col2:
        with st.popover(
            "📧 이메일 보내기",
            use_container_width=True
        ):
            receiver = st.text_input(
                "받는 사람",
                key=f"receiver_{mode}"
            )

            subject = st.text_input(
                "제목",
                value="[에이블] 업무 지원 결과",
                key=f"subject_{mode}"
            )

            body = st.text_area(
                "이메일 내용",
                value=st.session_state.last_answer,
                height=200,
                key=f"body_{mode}"
            )

            if st.button(
                "발송",
                key=f"email_send_{mode}"
            ):
                try:
                    send_email(receiver, subject, body)
                    st.success("이메일 발송 완료!")
                except Exception as e:
                    st.error(f"이메일 발송 오류: {e}")


# ==================================================
# 8. 손그림의 중앙 AI 입력 화면
# ==================================================

def assistant_input(mode):

    # STT 변환 결과 출력 영역
    st.markdown(
        '<div class="stt-label">음성을 텍스트로 변환한 값</div>',
        unsafe_allow_html=True
    )

    stt_placeholder = st.empty()

    if st.session_state.stt_result:
        stt_placeholder.info(st.session_state.stt_result)
    else:
        stt_placeholder.info(
            "마이크로 녹음한 음성이 이곳에 텍스트로 표시됩니다."
        )

    st.write("")

    # 가운데 마이크
    left, center, right = st.columns([1, 2, 1])

    with center:
        st.markdown(
            '<div class="mic-label">🎤 음성 입력</div>',
            unsafe_allow_html=True
        )

        audio = st.audio_input(
            "마이크를 눌러 질문하세요",
            key=f"audio_{mode}"
        )

    st.write("")

    # 아래 입력창과 첨부 버튼
    attach_col, input_col, send_col = st.columns(
        [1.5, 7, 1.3]
    )

    with attach_col:
        with st.popover(
            "➕ 첨부",
            use_container_width=True
        ):
            uploaded_file = st.file_uploader(
                "파일 첨부",
                type=["txt"],
                key=f"file_{mode}"
            )

            uploaded_image = st.file_uploader(
                "사진 첨부",
                type=["jpg", "jpeg", "png", "webp"],
                key=f"image_{mode}"
            )

            st.caption(
                "녹음은 위의 마이크 버튼을 이용하세요."
            )

    with input_col:
        question = st.text_input(
            "질문 입력",
            placeholder="무엇을 도와드릴까요?",
            label_visibility="collapsed",
            key=f"text_{mode}"
        )

    with send_col:
        send = st.button(
            "➤",
            type="primary",
            use_container_width=True,
            key=f"send_{mode}"
        )

    # 전송 버튼 클릭 시 AI 실행
    if send:

        final_question = question.strip()

        try:
            reference = read_text_file(uploaded_file)

            # STT
            if audio is not None:

                with st.spinner("음성을 인식하고 있습니다..."):
                    transcript = speech_to_text(audio)

                    st.session_state.stt_result = transcript
                    stt_placeholder.success(transcript)

                    if not final_question:
                        final_question = transcript

            # 첨부 자료만 보낸 경우
            if not final_question:

                if uploaded_file is not None or uploaded_image is not None:
                    final_question = (
                        "첨부한 자료를 분석하고 요약해주세요."
                    )
                else:
                    st.warning(
                        "텍스트를 입력하거나 음성을 녹음해주세요."
                    )
                    return

            # LLM
            with st.spinner("답변을 준비하고 있습니다..."):

                answer = ask_assistant(
                    mode=mode,
                    question=final_question,
                    reference=reference,
                    image=uploaded_image
                )

            st.session_state.last_answer = answer
            st.session_state.last_question = final_question
            st.session_state.last_mode = mode
            st.session_state.last_audio = None

            save_history(mode, final_question, answer)

            # 회의 요약 결과 별도 보관
            if mode == "회의 요약":

                st.session_state.meetings.append({
                    "title": final_question[:35],
                    "summary": answer
                })

                st.session_state.meetings = (
                    st.session_state.meetings[-10:]
                )

            # PPT 요구사항: 최종 답변 즉시 음성 출력
            if st.session_state.auto_tts:

                try:
                    with st.spinner("음성 답변 생성 중..."):
                        st.session_state.last_audio = (
                            text_to_speech(answer)
                        )
                except Exception:
                    st.warning(
                        "텍스트 답변은 완료됐지만 "
                        "음성 생성에 실패했습니다."
                    )

            st.rerun()

        except Exception as e:
            st.error(f"AI 처리 중 오류: {e}")

    # 답변 표시
    show_answer(mode)


# ==================================================
# 9. 대시보드 - 그림판 왼쪽 화면
# ==================================================

if st.session_state.page == "대시보드":

    if st.session_state.dashboard_view == "home":

        st.title("🏠 대시보드")

        st.write(
            f"안녕하세요, "
            f"{st.session_state.user_name}님!"
        )

        st.subheader("최근 회의 요약")

        with st.container(border=True):

            if st.session_state.meetings:

                latest = st.session_state.meetings[-1]

                st.markdown(latest["summary"][:1200])

                if st.button("▶ 자세히 보기"):
                    navigate("회의 요약")

            else:
                st.info("아직 저장된 회의 요약이 없습니다.")

                if st.button("▶ 회의 요약 시작하기"):
                    navigate("회의 요약")

        st.write("")

        st.subheader("AI 개인비서")

        with st.container(border=True):

            st.write(
                "음성 또는 텍스트로 업무를 요청하세요."
            )

            if st.button(
                "🎤 에이블과 대화 시작",
                type="primary",
                use_container_width=True
            ):
                navigate("대시보드", "assistant")

        st.divider()

        st.subheader("최근 대화")

        if not st.session_state.history:
            st.caption("대화 기록이 없습니다.")

        for i, item in enumerate(
            reversed(st.session_state.history[-5:]), 1
        ):
            with st.expander(
                f"{i}. {item['question'][:40]}"
            ):
                st.write(item["answer"])

    else:

        st.title("🤖 AI 개인비서 에이블")

        if st.button("← 대시보드로 돌아가기"):
            navigate("대시보드", "home")

        st.write(
            "음성이나 텍스트로 질문하면 "
            "에이블이 업무를 도와드립니다."
        )

        assistant_input("대시보드")


# ==================================================
# 10. 회의 요약 페이지
# ==================================================

elif st.session_state.page == "회의 요약":

    st.title("📝 회의 요약")

    st.write(
        "회의 내용을 입력하거나 녹음하여 "
        "핵심 내용과 결정사항을 요약하세요."
    )

    assistant_input("회의 요약")

    st.divider()
    st.subheader("저장된 회의 요약")

    if not st.session_state.meetings:
        st.info("저장된 회의 요약이 없습니다.")

    for i, item in enumerate(
        reversed(st.session_state.meetings), 1
    ):
        with st.expander(
            f"회의 {i}: {item['title']}"
        ):
            st.markdown(item["summary"])


# ==================================================
# 11. 일정 관리 페이지
# ==================================================

elif st.session_state.page == "일정 관리":

    st.title("📅 일정 관리")

    st.write("일정을 등록하고 관리할 수 있습니다.")

    with st.container(border=True):

        schedule_title = st.text_input(
            "일정 제목",
            key="schedule_title"
        )

        schedule_date = st.date_input(
            "날짜",
            value=date.today(),
            key="schedule_date"
        )

        schedule_memo = st.text_area(
            "메모",
            key="schedule_memo"
        )

        if st.button(
            "➕ 일정 등록",
            type="primary"
        ):

            if not schedule_title.strip():
                st.warning("일정 제목을 입력해주세요.")

            else:
                st.session_state.schedules.append({
                    "title": schedule_title.strip(),
                    "date": str(schedule_date),
                    "memo": schedule_memo
                })

                st.rerun()

    st.subheader("등록된 일정")

    schedules = sorted(
        enumerate(st.session_state.schedules),
        key=lambda x: x[1]["date"]
    )

    if not schedules:
        st.info("등록된 일정이 없습니다.")

    for index, item in schedules:

        col1, col2 = st.columns([8, 1])

        with col1:
            st.write(
                f"📌 {item['date']} | "
                f"{item['title']}"
            )

            if item["memo"]:
                st.caption(item["memo"])

        with col2:
            if st.button(
                "삭제",
                key=f"delete_schedule_{index}"
            ):
                st.session_state.schedules.pop(index)
                st.rerun()

    st.divider()
    st.subheader("AI 일정 비서")

    assistant_input("일정 관리")


# ==================================================
# 12. 최근 대화 히스토리
# ==================================================

st.divider()

with st.expander("📜 최근 10개 대화 기록"):

    if not st.session_state.history:
        st.info("아직 저장된 대화가 없습니다.")

    for i, item in enumerate(
        reversed(st.session_state.history), 1
    ):
        with st.expander(
            f"{i}. [{item['mode']}] "
            f"{item['question'][:35]}"
        ):
            st.markdown("**사용자 질문**")
            st.write(item["question"])

            st.markdown("**에이블 답변**")
            st.markdown(item["answer"])


st.caption(
    "AIVLE 비서 | KT AIVLE School AI 개인비서 미니 프로젝트"
)
