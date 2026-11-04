
import io
import os
import smtplib
import streamlit as st
from email.message import EmailMessage
from openai import OpenAI

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="AIVLE 비서",
    page_icon="🤖",
    layout="wide"
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
    st.error("OPENAI_API_KEY를 Streamlit Secrets 또는 환경변수에 등록해주세요.")
    st.stop()

client = OpenAI(api_key=api_key)

# --------------------------------------------------
# 상태 초기화
# --------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "stt_text" not in st.session_state:
    st.session_state.stt_text = ""

if "tts_audio" not in st.session_state:
    st.session_state.tts_audio = None

if "voice_name" not in st.session_state:
    st.session_state.voice_name = "alloy"

if "login_name" not in st.session_state:
    st.session_state.login_name = "ooo님"

# --------------------------------------------------
# CSS
# --------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}
.menu-box {
    border: 1px solid #d9d9d9;
    border-radius: 12px;
    padding: 0.5rem;
    background-color: #ffffff08;
}
.section-card {
    border: 1px solid #d9d9d9;
    border-radius: 20px;
    padding: 1rem 1.2rem;
    background-color: #ffffff08;
    margin-bottom: 1rem;
}
.stt-box {
    border: 2px solid #d9d9d9;
    border-radius: 24px;
    padding: 1rem;
    text-align: center;
    min-height: 70px;
    background: #ffffff05;
    font-size: 1.05rem;
}
.center-box {
    display: flex;
    justify-content: center;
    align-items: center;
}
.small-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #666;
    margin-bottom: 0.4rem;
}
.answer-card {
    border: 1px solid #cfcfcf;
    border-radius: 16px;
    padding: 1rem;
    background-color: #ffffff05;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 공통 함수
# --------------------------------------------------
def build_system_prompt(mode, extra_text=""):
    base = """
당신은 AI 개인비서 'AIVLE 비서'입니다.
사용자의 업무를 도와주는 친절하고 정확한 비서 역할을 수행합니다.
답변은 한국어로 작성하세요.
필요하면 항목별로 정리하세요.
불확실한 내용은 추측하지 말고 그 점을 명확히 밝히세요.
"""
    if mode == "회의 요약":
        base += """
현재 역할은 회의 요약 비서입니다.
사용자가 준 회의 내용에서 핵심 안건, 결정사항, 담당자, 후속 액션을 정리하세요.
"""
    elif mode == "일정 관리":
        base += """
현재 역할은 일정 관리 비서입니다.
해야 할 일, 일정, 우선순위, 마감일을 깔끔하게 정리하세요.
"""
    elif mode == "웹 리서치":
        base += """
현재 역할은 웹 리서치 비서입니다.
사용자의 질문에 대해 조사 방향, 핵심 포인트, 정리된 요약 형식으로 답변하세요.
"""
    else:
        base += """
현재 역할은 일반 업무 비서입니다.
업무 질문, 자료 요약, 정리 요청에 자연스럽게 답변하세요.
"""

    if extra_text:
        base += f"\n\n참고자료:\n{extra_text[:12000]}"

    return base


def ask_ai(mode, question, extra_text=""):
    system_prompt = build_system_prompt(mode, extra_text=extra_text)

    messages = [{"role": "system", "content": system_prompt}]

    for item in st.session_state.history[-5:]:
        messages.append({"role": "user", "content": item["question"]})
        messages.append({"role": "assistant", "content": item["answer"]})

    messages.append({"role": "user", "content": question})

    response = client.responses.create(
        model="gpt-4o-mini",
        input=messages
    )
    return response.output_text


def make_tts(text):
    speech = client.audio.speech.create(
        model="tts-1",
        voice=st.session_state.voice_name,
        input=text[:4000]
    )
    return speech.read()


def send_email(receiver, subject, body):
    smtp_host = get_secret("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(get_secret("SMTP_PORT", "465"))
    smtp_user = get_secret("SMTP_USER")
    smtp_password = get_secret("SMTP_PASSWORD")

    if not smtp_user or not smtp_password:
        raise ValueError("SMTP_USER / SMTP_PASSWORD가 설정되지 않았습니다.")

    msg = EmailMessage()
    msg["From"] = smtp_user
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20) as server:
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


# --------------------------------------------------
# 좌측 메뉴
# --------------------------------------------------
with st.sidebar:
    st.markdown("### AIVLE 비서")

    menu = st.radio(
        "메뉴",
        ["대시보드", "회의 요약", "일정 관리", "웹 리서치"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("최근 대화")
    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history[-5:]), 1):
            st.write(f"{idx}. {item['question'][:20]}")
    else:
        st.write("아직 대화가 없습니다.")

    st.markdown("---")
    if st.button("대화 기록 초기화"):
        st.session_state.history = []
        st.session_state.last_question = ""
        st.session_state.last_answer = ""
        st.session_state.stt_text = ""
        st.session_state.tts_audio = None
        st.rerun()

# --------------------------------------------------
# 상단 헤더
# --------------------------------------------------
top1, top2 = st.columns([8, 2])

with top1:
    st.markdown(f"## {menu}")

with top2:
    with st.popover(f"🔴 {st.session_state.login_name}"):
        st.write("로그인/로그아웃")
        selected_voice = st.selectbox(
            "비서 음성 설정",
            ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            index=["alloy", "echo", "fable", "onyx", "nova", "shimmer"].index(st.session_state.voice_name)
        )
        st.session_state.voice_name = selected_voice

        new_name = st.text_input("프로필 설정", value=st.session_state.login_name)
        if st.button("프로필 저장"):
            st.session_state.login_name = new_name
            st.success("프로필이 저장되었습니다.")

# --------------------------------------------------
# 공통 입력 영역 함수
# --------------------------------------------------
def render_main_input(mode):
    col_left, col_main = st.columns([1.2, 4.8])

    with col_left:
        st.markdown("### 메뉴")
        st.markdown("""
- 🔴 대시보드  
- 🔴 회의 요약  
- 🔴 일정 관리  
- 🔴 웹 리서치
""")

    with col_main:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.markdown("#### 음성을 텍스트로 변환한 값")
        display_text = st.session_state.stt_text if st.session_state.stt_text else "아직 변환된 내용이 없습니다."
        st.markdown(f'<div class="stt-box">{display_text}</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown("#### 🎤 음성 입력")
        audio_value = st.audio_input("음성 녹음", label_visibility="collapsed")

        st.write("")
        attach_col1, attach_col2 = st.columns([1, 1])

        with attach_col1:
            uploaded_file = st.file_uploader("파일", type=["txt", "pdf", "csv", "docx"])
        with attach_col2:
            uploaded_image = st.file_uploader("사진", type=["png", "jpg", "jpeg"])

        text_question = st.text_input("무엇을 도와드릴까요?", placeholder="무엇을 도와드릴까요?")
        send_clicked = st.button("➤ 보내기", type="primary", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # 참고 텍스트
        extra_text = ""
        if uploaded_file is not None:
            try:
                extra_text = uploaded_file.read().decode("utf-8")
            except:
                extra_text = "업로드된 파일이 있으나 현재 텍스트로 직접 읽지 못했습니다."

        if uploaded_image is not None:
            extra_text += "\n\n[참고] 이미지 파일이 첨부되었습니다."

        if send_clicked:
            final_question = text_question.strip()

            # 음성 -> 텍스트
            if audio_value is not None:
                try:
                    with st.spinner("음성을 텍스트로 변환 중입니다..."):
                        audio_file = io.BytesIO(audio_value.getvalue())
                        audio_file.name = "audio.wav"

                        transcript = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file
                        )
                        st.session_state.stt_text = transcript.text
                        if not final_question:
                            final_question = transcript.text
                except Exception as e:
                    st.error(f"음성 인식 오류: {e}")

            if not final_question:
                st.warning("텍스트를 입력하거나 음성을 녹음해주세요.")
                return

            st.session_state.last_question = final_question

            try:
                with st.spinner("답변을 생성하고 있습니다..."):
                    answer = ask_ai(mode, final_question, extra_text=extra_text)
                    st.session_state.last_answer = answer
                    st.session_state.tts_audio = None

                    st.session_state.history.append({
                        "mode": mode,
                        "question": final_question,
                        "answer": answer
                    })
                    st.session_state.history = st.session_state.history[-10:]

            except Exception as e:
                st.error(f"AI 응답 생성 오류: {e}")

        # 결과 출력
        if st.session_state.last_answer:
            st.markdown("### 답변")
            st.markdown(f'<div class="answer-card">{st.session_state.last_answer}</div>', unsafe_allow_html=True)

            btn1, btn2 = st.columns([1, 1])

            with btn1:
                if st.button("🔊 답변 음성 듣기"):
                    try:
                        with st.spinner("음성을 생성하고 있습니다..."):
                            st.session_state.tts_audio = make_tts(st.session_state.last_answer)
                    except Exception as e:
                        st.error(f"TTS 오류: {e}")

            with btn2:
                with st.popover("📧 이메일 보내기"):
                    receiver = st.text_input("받는 사람 이메일")
                    subject = st.text_input("제목", value=f"[AIVLE 비서] {mode} 결과")
                    body = st.text_area("내용", value=st.session_state.last_answer, height=200)

                    if st.button("이메일 발송"):
                        try:
                            send_email(receiver, subject, body)
                            st.success("이메일 발송 완료")
                        except Exception as e:
                            st.error(f"이메일 발송 오류: {e}")

            if st.session_state.tts_audio:
                st.audio(st.session_state.tts_audio, format="audio/mpeg")


# --------------------------------------------------
# 페이지별 UI
# --------------------------------------------------
if menu == "대시보드":
    render_main_input("대시보드")

elif menu == "회의 요약":
    st.info("최근 회의 요약을 빠르게 정리하는 화면입니다.")
    render_main_input("회의 요약")

elif menu == "일정 관리":
    st.info("할 일, 일정, 우선순위를 정리하는 화면입니다.")
    render_main_input("일정 관리")

elif menu == "웹 리서치":
    st.info("주제 조사, 정보 정리, 조사 포인트 정리에 사용하는 화면입니다.")
    render_main_input("웹 리서치")

st.markdown("---")
st.caption("AIVLE 비서 | 인터페이스 스케치 기반 Streamlit 버전")
