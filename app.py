
import os
import io
import json
import smtplib
import streamlit as st
from email.message import EmailMessage
from openai import OpenAI

# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="AI 개인비서 에이블",
    page_icon="🤖",
    layout="wide"
)

# Streamlit Secrets 및 환경변수 지원
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
    st.error("OPENAI_API_KEY를 등록해주세요.")
    st.stop()

client = OpenAI(api_key=api_key)

st.title("🤖 AI 개인비서 에이블")
st.write("음성 또는 텍스트로 질문하시면 답변해 드립니다.")

# --------------------------------------------------
# 2. AI 개인비서 역할 정의 [추가]
# --------------------------------------------------

sys_role = """
당신은 AI 개인비서 '에이블'입니다.

주요 역할:
1. 김 대리의 업무를 돕는 개인비서
2. 질문에 친절하고 정확하게 답변
3. 업무 내용 및 문서 요약
4. 이메일 작성 지원
5. 일정 및 업무 정리 지원

답변은 한국어로 작성합니다.
모르는 내용은 임의로 만들어내지 않습니다.
"""

# --------------------------------------------------
# 3. 상태 관리 [추가]
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "answer_audio" not in st.session_state:
    st.session_state.answer_audio = None

# --------------------------------------------------
# 4. 사이드바 및 에이블 이미지 [추가]
# --------------------------------------------------

with st.sidebar:
    st.header("🤖 에이블")

    st.image(
        "https://raw.githubusercontent.com/"
        "Jangrae/img/master/ai_robot.png",
        width=180
    )

    st.write("AI 개인비서 서비스")

    st.divider()

    st.subheader("주요 기능")
    st.write("🎤 음성 인식")
    st.write("💬 AI 답변")
    st.write("🔊 음성 답변")
    st.write("📧 이메일 발송")
    st.write("📜 대화 기록")

    if st.button("대화 기록 초기화"):
        st.session_state.messages = []
        st.session_state.last_answer = ""
        st.session_state.last_question = ""
        st.session_state.answer_audio = None
        st.rerun()

# --------------------------------------------------
# 5. 사용자 입력
# --------------------------------------------------

st.subheader("💬 AI에게 질문하기")

uploaded_file = st.file_uploader(
    "텍스트 파일 업로드",
    type=["txt"]
)

# [추가] 텍스트 입력
text_question = st.text_area(
    "질문을 입력하세요",
    placeholder="예: 오늘 업무를 정리해줘."
)

# 음성 녹음
audio_value = st.audio_input(
    "🎤 음성으로 질문하기"
)

# 기존 st.form 대신 일반 버튼 사용
submit = st.button(
    "🚀 실행",
    type="primary"
)

# --------------------------------------------------
# 6. 음성 인식 및 AI 답변
# --------------------------------------------------

if submit:

    # 참고자료 읽기
    reference_text = ""

    if uploaded_file is not None:
        try:
            reference_text = uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.error("UTF-8 형식의 TXT 파일을 업로드해주세요.")
            st.stop()

    question = text_question.strip()

    try:
        # STT: 음성 → 텍스트
        if audio_value is not None:

            with st.spinner("음성을 인식하고 있습니다..."):

                audio_file = io.BytesIO(
                    audio_value.getvalue()
                )
                audio_file.name = "audio.wav"

                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

                question = transcript.text

        if not question:
            st.warning("텍스트를 입력하거나 음성을 녹음해주세요.")
            st.stop()

        # [추가] STT 결과 화면 출력
        st.subheader("📝 인식된 질문")
        st.info(question)

        # 시스템 메시지
        system_content = sys_role

        if reference_text:
            system_content += (
                "\n\n다음 참고자료를 활용하세요:\n"
                + reference_text[:12000]
            )

        input_messages = [
            {
                "role": "system",
                "content": system_content
            }
        ]

        # [추가] 이전 대화 내용 활용
        for item in st.session_state.messages[-5:]:
            input_messages.append({
                "role": "user",
                "content": item["question"]
            })
            input_messages.append({
                "role": "assistant",
                "content": item["answer"]
            })

        input_messages.append({
            "role": "user",
            "content": question
        })

        # LLM 호출
        with st.spinner("답변을 준비하고 있습니다..."):

            response = client.responses.create(
                model="gpt-4o-mini",
                input=input_messages
            )

            answer = response.output_text

        st.session_state.last_question = question
        st.session_state.last_answer = answer
        st.session_state.answer_audio = None

        # [추가] 대화 기록 저장
        st.session_state.messages.append({
            "question": question,
            "answer": answer
        })

        # 최근 10개 대화만 유지
        st.session_state.messages = (
            st.session_state.messages[-10:]
        )

    except Exception as e:
        st.error(f"AI 처리 중 오류가 발생했습니다: {e}")

# --------------------------------------------------
# 7. AI 답변 출력
# --------------------------------------------------

if st.session_state.last_answer:

    st.subheader("🤖 AI 답변")
    st.write(st.session_state.last_answer)

    # --------------------------------------------------
    # 8. 답변 음성 재생(TTS) [추가]
    # --------------------------------------------------

    st.subheader("🔊 AI 답변 듣기")

    if st.button("음성 답변 생성"):

        try:
            with st.spinner("음성을 생성하고 있습니다..."):

                speech = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=st.session_state.last_answer[:4000]
                )

                st.session_state.answer_audio = speech.read()

        except Exception as e:
            st.error(f"음성 생성 오류: {e}")

    if st.session_state.answer_audio:
        st.audio(
            st.session_state.answer_audio,
            format="audio/mpeg"
        )

# --------------------------------------------------
# 9. 이메일 발송 기능 [추가]
# --------------------------------------------------

st.divider()
st.subheader("📧 이메일 발송")

with st.expander("이메일 작성 및 발송"):

    receiver = st.text_input("받는 사람 이메일")

    subject = st.text_input(
        "이메일 제목",
        value="AI 개인비서 에이블 업무 보고"
    )

    email_body = st.text_area(
        "이메일 내용",
        value=st.session_state.last_answer,
        height=200
    )

    if st.button("이메일 보내기"):

        if not receiver or not email_body.strip():
            st.warning("받는 사람과 이메일 내용을 입력해주세요.")

        elif not (
            get_secret("SMTP_USER")
            and get_secret("SMTP_PASSWORD")
        ):
            st.error("이메일 발송 정보가 등록되지 않았습니다.")

        else:
            try:
                smtp_host = get_secret(
                    "SMTP_HOST", "smtp.gmail.com"
                )
                smtp_port = int(
                    get_secret("SMTP_PORT", "465")
                )
                smtp_user = get_secret("SMTP_USER")
                smtp_password = get_secret("SMTP_PASSWORD")

                msg = EmailMessage()
                msg["From"] = smtp_user
                msg["To"] = receiver
                msg["Subject"] = subject
                msg.set_content(email_body)

                with smtplib.SMTP_SSL(
                    smtp_host,
                    smtp_port,
                    timeout=20
                ) as server:
                    server.login(
                        smtp_user,
                        smtp_password
                    )
                    server.send_message(msg)

                st.success("이메일 발송이 완료되었습니다.")

            except Exception as e:
                st.error(f"이메일 발송 오류: {e}")

# --------------------------------------------------
# 10. 대화 히스토리 [추가]
# --------------------------------------------------

st.divider()
st.subheader("📜 최근 대화 기록")

if not st.session_state.messages:
    st.info("아직 저장된 대화가 없습니다.")

else:
    for i, item in enumerate(
        reversed(st.session_state.messages), 1
    ):

        with st.expander(f"대화 {i}: {item['question'][:35]}"):

            st.markdown("**사용자 질문**")
            st.write(item["question"])

            st.markdown("**에이블 답변**")
            st.write(item["answer"])

st.caption(
    "AI 개인비서 에이블 | KT AIVLE School 미니 프로젝트"
)
