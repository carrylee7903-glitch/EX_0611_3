import os
import tempfile

import streamlit as st
# 추가: Streamlit 화면 글자 깨짐(NoSessionContext) 방지용 모듈
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# 수정: langchain_classic은 서버 배포 시 에러가 나므로 표준 langchain으로 변경
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

st.title("📄 PDF File Reader")
st.write("----------------")

openai_key = st.text_input("OPENAI_API_KEY", type="password")

uploaded_file = st.file_uploader("PDF 파일을 올려주세요", type=["pdf"])
st.write("----------------")

def pdf_to_document(uploaded_file):
    """   
    Streamlit 업로드 PDF를
    LangChain Document 형태로 변환
    """
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)

    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader(temp_filepath)
    pages = loader.load()
    return pages

class StreamHandler(BaseCallbackHandler):
    """
    GPT가 토큰을 생성할 때마다 Streamlit 화면에 출력하는 Handler
    """
    def __init__(self, container):
        self.container = container
        self.text = ""
        # 백그라운드 쓰레드 실행을 위해 정상 상태일 때의 화면 주소(Context) 기억
        self.ctx = get_script_run_ctx()

    def on_llm_new_token(self, token, **kwargs):
        # 글자를 쓸 때 주소(Context) 강제 주입
        add_script_run_ctx(ctx=self.ctx)
        
        self.text += token
        self.container.markdown(self.text)

if uploaded_file is not None:
    # 수정: API 키가 없으면 코드가 실행되지 않고 여기서 멈춤 (빨간 에러 방지)
    if not openai_key:
        st.info("👈 진행하려면 먼저 상단에 OPENAI_API_KEY를 입력해 주세요.")
        st.stop()

    pages = pdf_to_document(uploaded_file)
