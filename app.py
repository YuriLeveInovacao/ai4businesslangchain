--- START OF FILE app (1).py ---

import os
import tempfile
import pandas as pd
import pytesseract
from PIL import Image
import docx
import streamlit as st

# workaround para sqlite3 em ambientes como Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# Fim do workaround sqlite3


from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma # Este import agora usará a versão de pysqlite3
from pypdf.errors import PdfReadError
# Importação corrigida para os erros da biblioteca OpenAI (v1+)
from openai import AuthenticationError, BadRequestError, APIError

# Adicionar a imagem no cabeçalho
image_url = "https://cienciadosdados.com/images/CINCIA_DOS_DADOS_4.png"
# CORREÇÃO: Substituindo use_column_width por use_container_width
st.image(image_url, use_container_width=True)

# Adicionar o nome do aplicativo
st.subheader("Q&A com IA - PLN usando LangChain")

# Componentes interativos
file_input = st.file_uploader("Upload a file", type=['pdf', 'txt', 'csv', 'docx', 'jpeg', 'png'])
# Nota: Não é recomendado incluir a chave de API diretamente no código ou no input padrão.
# Use variáveis de ambiente ou st.secrets (recomendado no Streamlit Cloud).
# Mantendo a estrutura original conforme solicitado, mas esteja ciente dessa prática.
# O texto padrão do input foi removido, pois não deve conter a chave de API.
openaikey = st.text_input("Enter your OpenAI API Key", type='password')
prompt = st.text_area("Enter your questions", height=160)
run_button = st.button("Run!")

select_k = st.slider("Number of relevant chunks", min_value=1, max_value=5, value=2)
select_chain_type = st.radio("Chain type", ['stuff', 'map_reduce', "refine", "map_rerank"])

# Função para carregar documentos
def load_document(file_path, file_type):
    if file_type == 'application/pdf':
        try:
            loader = PyPDFLoader(file_path)
            return loader.load()
        except PdfReadError as e:
             st.error(f"Error reading PDF file: {e}")
             return None
    elif file_type == 'text/plain':
        loader = TextLoader(file_path)
        return loader.load()
    elif file_type == 'text/csv':
        try:
            df = pd.read_csv(file_path)
            # Convertendo o DataFrame para string. Pode precisar de ajustes dependendo do CSV
            return [{"page_content": df.to_string(index=False)}]
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        try:
            doc = docx.Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return [{"page_content": "\n".join(full_text)}]
        except Exception as e:
            st.error(f"Error reading DOCX file: {e}")
            return None
    elif file_type in ['image/jpeg', 'image/png']:
        try:
            # Garante que o Tesseract está instalado no ambiente de deploy
            text = pytesseract.image_to_string(Image.open(file_path))
            return [{"page_content": text}]
        except pytesseract.TesseractNotFoundError:
            st.error("Tesseract is not installed or not in PATH. OCR requires Tesseract.")
            return None
        except Exception as e:
            st.error(f"Error processing image file with Tesseract: {e}")
            return None
    else:
        st.error("Unsupported file type.")
        return None

# Função de perguntas e respostas
def qa(file_path, file_type, query, chain_type, k):
    try:
        documents = load_document(file_path, file_type)
        if not documents:
            return None

        # split the documents into chunks
        # Ajustei chunk_overlap para 200 para evitar cortar palavras no meio e manter contexto
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # select which embeddings we want to use
        # OpenAIEmbeddings agora usa OPENAI_API_KEY do ambiente ou passada diretamente
        embeddings = OpenAIEmbeddings()

        # create the vectorestore to use as the index
        db = Chroma.from_documents(texts, embeddings)

        # expose this index in a retriever interface
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

        # create a chain to answer questions
        # ChatOpenAI agora usa OPENAI_API_KEY do ambiente ou passada diretamente
        # Você pode precisar especificar 'model="gpt-3.5-turbo"' se 'gpt-4' for caro ou não estiver disponível.
        # Mantendo "gpt-4" conforme o original.
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4"),
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True
        )
        result = qa({"query": query})
        return result
    except AuthenticationError as e: # Usa a classe importada de openai
        st.error(f"Authentication error: {e}")
        return None
    except BadRequestError as e: # Usa a classe importada de openai (substitui InvalidRequestError)
        st.error(f"Invalid request error: {e}")
        return None
    except APIError as e: # Adicionado para capturar outros erros gerais da API OpenAI (v1+)
        st.error(f"General OpenAI API error: {e}")
        return None
    except Exception as e: # Captura outros erros não específicos da API
        st.error(f"An unexpected error occurred during QA processing: {e}")
        # Opcional: imprimir o traceback completo para debug nos logs do Streamlit Cloud
        # import traceback
        # st.error(traceback.format_exc())
        return None


# Função para exibir o resultado no Streamlit
def display_result(result):
    if result:
        st.markdown("### Result:")
        st.write(result.get("result", "No answer found.")) # Use .get para evitar KeyError
        if "source_documents" in result and result["source_documents"]:
            st.markdown("### Relevant source text:")
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Source Document {i+1}:**")
                st.markdown(doc.page_content)
                st.markdown("---")
        elif "source_documents" not in result:
             st.info("No source documents returned by the QA chain.") # Informa se não há fontes (alguns chain_types não retornam)


# Execução do app
if run_button and file_input and openaikey and prompt:
    # Configurar a chave de API do OpenAI ANTES de instanciar qualquer objeto OpenAI/Langchain que a use
    os.environ["OPENAI_API_KEY"] = openaikey

    with st.spinner("Running QA..."):
        # Salvar o arquivo em um local temporário
        # tempfile.gettempdir() já cria o diretório se necessário.
        temp_file_path = os.path.join(tempfile.gettempdir(), file_input.name)
        try:
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_input.read())

            # Verificar se a chave de API é válida (opcional, mas boa prática)
            try:
                # Testar a chave de API com uma chamada simples
                # Instanciar embeddings para testar a chave. Isso pode lançar AuthenticationError.
                embeddings_test = OpenAIEmbeddings()
                embeddings_test.embed_documents(["test"])

                # Se o teste passou, continue com o QA
                result = qa(temp_file_path, file_input.type, prompt, select_chain_type, select_k)
                # Exibir o resultado
                display_result(result)

            except AuthenticationError as e: # Usa a classe importada de openai
                st.error(f"Invalid OpenAI API Key: {e}")
            except BadRequestError as e: # Captura erros de requisição durante o teste
                 st.error(f"Bad request during API key test: {e}")
            except APIError as e: # Captura outros erros da API durante o teste (ex: limite de rate)
                 st.error(f"Error testing OpenAI API Key: {e}")
            except Exception as e: # Captura outros erros inesperados durante o teste
                 st.error(f"An unexpected error occurred during API key test: {e}")

        finally:
            # Opcional: Tentar remover o arquivo temporário após o uso
            # Pode não ser estritamente necessário no Streamlit Cloud, mas é boa prática.
            if os.path.exists(temp_file_path):
                 try:
                     os.remove(temp_file_path)
                 except Exception as e:
                     st.warning(f"Could not remove temporary file {temp_file_path}: {e}")
