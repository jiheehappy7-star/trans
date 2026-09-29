"""AI 번역기 - Streamlit UI."""

from datetime import datetime

import streamlit as st
from openai import OpenAI

from translator import LANGUAGES, MAX_CHARS, TONES, TranslationError, get_api_key, get_model, translate

HISTORY_LIMIT = 10
ACCENTS = {"en": "#2563EB", "ja": "#DC2626", "vi": "#059669"}

CSS = """
<style>
.block-container { max-width: 1100px; padding-top: 2.5rem; }

.hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #DB2777 100%);
    color: #fff; padding: 28px 32px; border-radius: 20px; margin-bottom: 1.5rem;
    box-shadow: 0 12px 32px rgba(79, 70, 229, .22);
}
.hero h1 { color: #fff; margin: 0; padding: 0; font-size: 2rem; font-weight: 800; }
.hero p { margin: .4rem 0 .9rem; opacity: .9; }
.hero .chip {
    display: inline-block; margin-right: 6px; padding: 3px 12px; border-radius: 999px;
    background: rgba(255, 255, 255, .18); font-size: .8rem; font-weight: 600;
}

.badge {
    display: inline-block; min-width: 34px; padding: 2px 8px; border-radius: 8px;
    color: #fff; font-size: .75rem; font-weight: 700; text-align: center; letter-spacing: .5px;
}
.card-title { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.05rem; }
.card-title small { color: #6B7280; font-weight: 500; font-size: .8rem; }

[class*="st-key-card_"] {
    background: #fff; border-radius: 16px !important; border-top: 4px solid var(--accent) !important;
    box-shadow: 0 4px 16px rgba(17, 24, 39, .06);
}
[data-testid="stColumn"]:has([class*="st-key-card_"]) > div { height: 100%; }
[class*="st-key-card_"] { height: 100%; }
[class*="st-key-card_"] pre, [class*="st-key-card_"] code {
    font-family: inherit !important; font-size: 1rem !important; line-height: 1.65 !important;
}
.st-key-card_en { --accent: #2563EB; }
.st-key-card_ja { --accent: #DC2626; }
.st-key-card_vi { --accent: #059669; }

[data-testid="stForm"] {
    background: #fff; border: 1px solid #E5E7EB; border-radius: 16px; padding: 1.25rem;
    box-shadow: 0 4px 16px rgba(17, 24, 39, .04);
}
.empty {
    text-align: center; color: #6B7280; padding: 48px 16px;
    border: 2px dashed #D1D5DB; border-radius: 16px; background: #fff;
}
.empty b { display: block; color: #374151; font-size: 1.05rem; margin-bottom: 4px; }
.footer { text-align: center; color: #9CA3AF; font-size: .8rem; margin-top: 2rem; }
</style>
"""


@st.cache_resource
def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, timeout=30)


def init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("current", None)


def select_entry(entry: dict) -> None:
    st.session_state.current = entry


def clear_history() -> None:
    st.session_state.history = []
    st.session_state.current = None


def format_as_text(entry: dict) -> str:
    lines = [f"[원문]\n{entry['source']}\n"]
    for code, translated in entry["result"].items():
        lines.append(f"[{LANGUAGES[code]['label']}]\n{translated}\n")
    return "\n".join(lines)


def badge(code: str) -> str:
    return f'<span class="badge" style="background:{ACCENTS[code]}">{code.upper()}</span>'


def render_header(targets: list[str]) -> None:
    chips = "".join(f'<span class="chip">{LANGUAGES[code]["label"]}</span>' for code in targets)
    st.markdown(
        f"""<div class="hero">
            <h1>🌐 AI 번역기</h1>
            <p>텍스트를 입력하면 여러 언어로 한 번에 번역해 드립니다.</p>
            {chips or '<span class="chip">번역할 언어를 선택하세요</span>'}
        </div>""",
        unsafe_allow_html=True,
    )


def render_settings() -> tuple[list[str], str]:
    with st.sidebar:
        st.subheader("⚙️ 번역 설정")
        st.caption("번역할 언어")
        targets = [
            code
            for code, lang in LANGUAGES.items()
            if st.checkbox(f"{lang['label']} ({lang['name']})", value=True, key=f"lang_{code}")
        ]
        tone = st.radio("말투", list(TONES), horizontal=True)
    return targets, tone


def render_history() -> None:
    with st.sidebar:
        st.divider()
        st.subheader("🕘 번역 기록")
        history = st.session_state.history
        if not history:
            st.caption("아직 번역 기록이 없습니다.")
        for i, entry in enumerate(history):
            preview = entry["source"][:18] + ("…" if len(entry["source"]) > 18 else "")
            st.button(
                f"{entry['time']} · {preview}",
                key=f"history_{i}",
                use_container_width=True,
                on_click=select_entry,
                args=(entry,),
            )
        if history:
            st.button("🗑️ 기록 지우기", use_container_width=True, on_click=clear_history)


def render_results(entry: dict) -> None:
    head, action = st.columns([3, 1], vertical_alignment="bottom")
    head.markdown(
        f"#### 번역 결과 <small style='color:#9CA3AF'>· {entry['tone']} · {entry['time']}</small>",
        unsafe_allow_html=True,
    )
    action.download_button(
        "📄 .txt 다운로드",
        data=format_as_text(entry),
        file_name=f"translation_{entry['time'].replace(':', '')}.txt",
        mime="text/plain",
        on_click="ignore",
        use_container_width=True,
    )

    columns = st.columns(len(entry["result"]))
    for column, (code, translated) in zip(columns, entry["result"].items()):
        lang = LANGUAGES[code]
        with column.container(border=True, key=f"card_{code}"):
            st.markdown(
                f'<div class="card-title">{badge(code)} {lang["name"]} <small>{lang["label"]}</small></div>',
                unsafe_allow_html=True,
            )
            if translated:
                st.code(translated, language=None, wrap_lines=True)
            else:
                st.caption("번역 결과를 받지 못했습니다. 다시 시도해 주세요.")


def render_empty() -> None:
    st.markdown(
        '<div class="empty"><b>번역 결과가 여기에 표시됩니다</b>텍스트를 입력하고 번역하기 버튼을 눌러 보세요.</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="AI 번역기", page_icon="🌐", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    api_key = get_api_key()
    targets, tone = render_settings()
    render_header(targets)

    if not api_key:
        st.error("`.env` 파일에 OPENAI_API_KEY를 설정하세요.")
        st.stop()

    with st.form("translate_form", border=False):
        text = st.text_area(
            "번역할 텍스트",
            height=140,
            max_chars=MAX_CHARS,
            placeholder="예: 내일 오후 3시에 회의 가능할까요?",
        )
        submitted = st.form_submit_button("✨ 번역하기", type="primary", use_container_width=True)

    if submitted:
        try:
            with st.spinner("번역 중..."):
                result = translate(get_client(api_key), get_model(), text.strip(), targets, tone)
        except TranslationError as e:
            st.error(str(e))
        else:
            entry = {"time": datetime.now().strftime("%H:%M:%S"), "source": text.strip(), "tone": tone, "result": result}
            st.session_state.current = entry
            st.session_state.history = [entry, *st.session_state.history][:HISTORY_LIMIT]

    render_history()
    if st.session_state.current:
        render_results(st.session_state.current)
    else:
        render_empty()

    st.markdown(f'<div class="footer">Powered by OpenAI · {get_model()}</div>', unsafe_allow_html=True)


main()
