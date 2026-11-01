
import hashlib
import streamlit as st

from document_processor import extract_text
from text_splitter import split_text
from vector_store import (
    add_chunks,
    list_documents,
    delete_document,
    document_exists
)
from rag_pipeline import answer_question


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# DARK APPLE STYLE
# --------------------------------------------------

st.markdown("""
<style>

:root {
    --bg: #090a0c;
    --surface: #14161a;
    --border: #292d35;
    --text: #f5f5f7;
    --muted: #9297a2;
    --accent: #a5b4fc;
}

.stApp,
[data-testid="stAppViewContainer"] {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}

[data-testid="stSidebar"] {
    background: #101216;
    border-right: 1px solid var(--border);
}

h1 {
    color: #f5f5f7 !important;
    font-size: 2.4rem !important;
    font-weight: 650 !important;
    letter-spacing: -1.5px;
}

h2, h3 {
    color: #f5f5f7 !important;
}

p, label {
    color: #c3c6ce;
}

hr {
    border-color: var(--border) !important;
}

.brand {
    font-size: 20px;
    font-weight: 700;
    color: #f5f5f7;
    letter-spacing: -0.7px;
}

.brand-subtitle {
    color: #858b98;
    font-size: 11px;
    letter-spacing: 1.5px;
    margin-top: 5px;
}

.hero-label {
    color: var(--accent);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 12px;
}

.hero-description {
    color: var(--muted);
    font-size: 15px;
    line-height: 1.7;
    max-width: 680px;
}

.section-label {
    color: #858b98;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.7px;
    margin-bottom: 8px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
}

.stTextInput input {
    background: #171a20 !important;
    color: var(--text) !important;
    border: 1px solid #303540 !important;
    border-radius: 11px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #171a20 !important;
    border: 1px dashed #414754 !important;
    border-radius: 14px !important;
}

div.stButton > button {
    background: #252933;
    color: #f5f5f7;
    border: 1px solid #373d49;
    border-radius: 10px;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background: #343a47;
    color: white;
    border-color: #626b7d;
}

div.stButton > button[kind="primary"] {
    background: #e8eaf0;
    color: #111318;
    border: 1px solid #e8eaf0;
    font-weight: 650;
}

div.stButton > button[kind="primary"]:hover {
    background: #cbd0dc;
    color: #111318;
}

[data-testid="stExpander"] {
    background: #171a20;
    border: 1px solid var(--border);
    border-radius: 12px;
}

[data-testid="stMetric"] {
    background: #1b1e25;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stMetricValue"] {
    color: #f5f5f7 !important;
}

[data-testid="stChatMessage"] {
    background: #14161a;
    border: 1px solid #292d35;
    border-radius: 15px;
    padding: 12px;
    margin-bottom: 12px;
}

[data-testid="stChatInput"] {
    border-color: #343a47;
}

[data-testid="stChatInput"] textarea {
    color: #f5f5f7 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #858b98 !important;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

.stCaption,
[data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
}

a {
    color: var(--accent) !important;
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    h1 {
        font-size: 2rem !important;
    }
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        '<div class="brand">◈ AI Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'PERSONAL KNOWLEDGE SYSTEM'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("**ВОЗМОЖНОСТИ**")

    st.caption("Загрузка документов")
    st.caption("Поиск по содержимому")
    st.caption("AI-ответы с источниками")
    st.caption("История переписки")

    st.divider()

    st.markdown("**ТЕХНОЛОГИИ**")

    st.caption("Gemini API")
    st.caption("ChromaDB")
    st.caption("Python / Streamlit")
    st.caption("RAG Architecture")

    st.divider()

    st.markdown("**ЧАТ**")

    st.caption(
        f"Сообщений: {len(st.session_state.messages)}"
    )

    if st.button(
        "Очистить историю",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption("AI Knowledge Assistant")
    st.caption("Portfolio Project · 2026")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="hero-label">PERSONAL AI WORKSPACE</div>',
    unsafe_allow_html=True
)

st.title("AI Knowledge Assistant")

st.markdown(
    '<div class="hero-description">'
    'Интеллектуальный помощник для работы с документами. '
    'Загружайте файлы, находите нужную информацию '
    'и получайте ответы с указанием источников.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# --------------------------------------------------
# 1. DOCUMENT UPLOAD
# --------------------------------------------------

st.markdown(
    '<div class="section-label">01 / KNOWLEDGE BASE</div>',
    unsafe_allow_html=True
)

st.header("Загрузка документов")

with st.container(border=True):

    st.write(
        "Добавьте документы, по которым AI будет искать информацию."
    )

    uploaded_file = st.file_uploader(
        "Выберите PDF, DOCX или TXT",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.sha256(file_bytes).hexdigest()

        file_id = uploaded_file.name + "_" + file_hash

        if document_exists(file_id):

            st.session_state.processed_files.add(file_id)

            st.info(
                "Этот документ уже находится в базе данных."
            )

        else:

            try:

                with st.spinner("Обрабатываем документ..."):

                    document_text = extract_text(
                        uploaded_file.name,
                        file_bytes
                    )

                    if not document_text.strip():

                        st.warning(
                            "В документе не найден текст."
                        )

                    else:

                        chunks = split_text(document_text)

                        count = add_chunks(
                            chunks=chunks,
                            source=uploaded_file.name,
                            document_id=file_id
                        )

                        st.session_state.processed_files.add(
                            file_id
                        )

                        st.success(
                            f"Документ обработан! "
                            f"Сохранено фрагментов: {count}"
                        )

            except Exception as error:

                st.error(
                    f"Ошибка обработки документа: {error}"
                )


st.divider()


# --------------------------------------------------
# 2. DOCUMENT LIBRARY
# --------------------------------------------------

st.markdown(
    '<div class="section-label">02 / LIBRARY</div>',
    unsafe_allow_html=True
)

st.header("Мои документы")

documents = list_documents()

if not documents:

    st.info(
        "Пока нет загруженных документов."
    )

else:

    st.caption(
        f"Всего документов: {len(documents)}"
    )

    for document in documents:

        document_id = document["document_id"]

        with st.container(border=True):

            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:

                st.write(
                    f"**{document['source']}**"
                )

            with col2:

                st.write(
                    f"Фрагментов: {document['chunks_count']}"
                )

            with col3:

                if document_id is not None:

                    if st.button(
                        "Удалить",
                        key=f"delete_{document_id}"
                    ):

                        try:

                            delete_document(document_id)

                            st.session_state.processed_files.discard(
                                document_id
                            )

                            st.success("Документ удалён.")

                            st.rerun()

                        except Exception as error:

                            st.error(
                                f"Ошибка удаления: {error}"
                            )

                else:

                    st.caption("Старая запись")


st.divider()


# --------------------------------------------------
# 3. CHAT
# --------------------------------------------------

st.markdown(
    '<div class="section-label">03 / AI ASSISTANT</div>',
    unsafe_allow_html=True
)

st.header("Диалог с AI")

st.caption(
    "Задавайте вопросы по содержимому загруженных документов."
)


# Показываем предыдущие сообщения

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        # Источники конкретного ответа AI

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            for i, source in enumerate(sources, start=1):

                chunk_index = source.get("chunk_index")
                distance = source.get("distance")

                if chunk_index is not None:
                    chunk_number = chunk_index + 1
                else:
                    chunk_number = "Неизвестно"

                if distance is not None:
                    distance_text = f"{distance:.4f}"
                else:
                    distance_text = "Неизвестно"

                with st.expander(
                    f"Источник {i}: {source['source']}"
                ):

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Номер фрагмента",
                            chunk_number
                        )

                    with col2:

                        st.metric(
                            "Расстояние",
                            distance_text
                        )

                    st.caption(
                        "Чем меньше расстояние, "
                        "тем ближе найденный фрагмент к вопросу."
                    )

                    st.markdown(
                        "**Текст найденного фрагмента**"
                    )

                    st.write(source["text"])


# Поле ввода в нижней части чата

user_question = st.chat_input(
    "Напишите свой вопрос..."
)


if user_question:

    # Добавляем вопрос пользователя

    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    # Показываем вопрос

    with st.chat_message("user"):
        st.write(user_question)

    # Получаем ответ AI

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "AI анализирует документы..."
            ):

                answer, sources = answer_question(user_question)

            st.write(answer)

            for i, source in enumerate(sources, start=1):

                chunk_index = source.get("chunk_index")
                distance = source.get("distance")

                chunk_number = (
                    chunk_index + 1
                    if chunk_index is not None
                    else "Неизвестно"
                )

                distance_text = (
                    f"{distance:.4f}"
                    if distance is not None
                    else "Неизвестно"
                )

                with st.expander(
                    f"Источник {i}: {source['source']}"
                ):

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Номер фрагмента",
                            chunk_number
                        )

                    with col2:

                        st.metric(
                            "Расстояние",
                            distance_text
                        )

                    st.caption(
                        "Чем меньше расстояние, "
                        "тем ближе найденный фрагмент к вопросу."
                    )

                    st.markdown(
                        "**Текст найденного фрагмента**"
                    )

                    st.write(source["text"])

            # Сохраняем ответ и источники в истории

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as error:

            error_message = (
                f"Не удалось получить ответ: {error}"
            )

            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": []
            })
