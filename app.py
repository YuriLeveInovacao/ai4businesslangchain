import os
import tempfile
import pandas as pd
import pytesseract
from PIL import Image
import docx
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # Já estava correto do passo anterior
# Importação corrigida para OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from pypdf.errors import PdfReadError
# Importação corrigida para os erros da biblioteca OpenAI (v1+)
from openai import AuthenticationError, BadRequestError, APIError

# Adicionar o nome do aplicativo
st.subheader("Q&A com IA - PLN usando LangChain")

# Componentes interativos
file_input = st.file_uploader("Upload a file", type=['pdf', 'txt', 'csv', 'docx', 'jpeg', 'png'])
# Nota: Não é recomendado incluir a chave de API diretamente no código.
# Use variáveis de ambiente ou st.secrets (recomendado no Streamlit Cloud).
# Mantendo a estrutura original conforme solicitado, mas esteja ciente dessa prática.
openaikey = st.text_input("Enter your OpenAI API Key", type='password')
prompt = st.text_area("Enter your questions", height=160)
run_button = st.button("Run!")

select_k = st.slider("Number of relevant chunks", min_value=1, max_value=5, value=2)
select_chain_type = st.radio("Chain type", ['stuff', 'map_reduce', "refine", "map_rerank"])

# Função para carregar documentos
def load_document(file_path, file_type):
    if file_type == 'application/pdf':
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_type == 'text/plain':
        loader = TextLoader(file_path)
        return loader.load()
    elif file_type == 'text/csv':
        df = pd.read_csv(file_path)
        # Convertendo o DataFrame para string. Pode precisar de ajustes dependendo do CSV
        return [{"page_content": df.to_string(index=False)}]
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return [{"page_content": "\n".join(full_text)}]
    elif file_type in ['image/jpeg', 'image/png']:
        try:
            # Garante que o Tesseract está instalado no ambiente de deploy
            text = pytesseract.image_to_string(Image.open(file_path))
            return [{"page_content": text}]
        except pytesseract.TesseractNotFoundError:
            st.error("Tesseract is not installed or not in PATH. OCR requires Tesseract.")
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
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
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
    except PdfReadError as e:
        st.error(f"Error reading PDF file: {e}")
        return None
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
        st.error(f"An unexpected error occurred: {e}")
        # Opcional: imprimir o traceback completo para debug nos logs do Streamlit Cloud
        # import traceback
        # st.error(traceback.format_exc())
        return None


# Função para exibir o resultado no Streamlit
def display_result(result):
    if result:
        st.markdown("### Result:")
        st.write(result["result"])
        if "source_documents" in result and result["source_documents"]:
            st.markdown("### Relevant source text:")
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Source Document {i+1}:**")
                st.markdown(doc.page_content)
                st.markdown("---")


# Execução do app
if run_button and file_input and openaikey and prompt:
    # Configurar a chave de API do OpenAI ANTES de instanciar qualquer objeto OpenAI/Langchain que a use
    os.environ["OPENAI_API_KEY"] = openaikey

    with st.spinner("Running QA..."):
        # Salvar o arquivo em um local temporário
        # Crie o diretório temporário se não existir (tempfile.gettempdir() já faz isso)
        # Assegure-se de que o arquivo temporário seja fechado/removido se necessário,
        # mas para um app Streamlit simples, o ambiente temporário geralmente lida com isso.
        temp_file_path = os.path.join(tempfile.gettempdir(), file_input.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_input.read())

        # Verificar se a chave de API é válida
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
        except APIError as e: # Captura outros erros da API durante o teste (ex: limite de rate)
             st.error(f"Error testing OpenAI API Key: {e}")
        except Exception as e: # Captura outros erros inesperados durante o teste
             st.error(f"An unexpected error occurred during API key test: {e}")
