import os
import sys
import argparse
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Função para obter a chave da API (similar à do app.py, mas adaptada para script)
def get_api_key():
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        logging.error("Chave da API do Google Gemini não encontrada na variável de ambiente GEMINI_API_KEY.")
        logging.error("Por favor, defina a variável de ambiente antes de executar o script.")
        sys.exit(1) # Termina o script se a chave não for encontrada
    return api_key

def process_documents(docs_path, index_path):
    """Carrega, processa documentos e cria um índice FAISS."""
    
    api_key = get_api_key()
    genai.configure(api_key=api_key)
    
    logging.info(f"Iniciando carregamento de documentos do diretório: {docs_path}")
    
    # Usar DirectoryLoader com UnstructuredFileLoader para lidar com vários tipos
    # Glob para incluir todos os tipos de arquivos suportados pelo unstructured
    # Usar silent_errors=True para pular arquivos que causam erros
    loader = DirectoryLoader(
        docs_path, 
        glob="**/*.*", 
        use_multithreading=True, 
        show_progress=True,
        loader_cls=UnstructuredFileLoader, # Tenta carregar qualquer tipo
        silent_errors=True # Ignora arquivos que falham ao carregar
    )
    
    try:
        documents = loader.load()
        if not documents:
            logging.warning("Nenhum documento foi carregado. Verifique o diretório e os arquivos.")
            return
        logging.info(f"{len(documents)} documentos carregados com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao carregar documentos: {e}")
        return

    # Dividir documentos em chunks
    logging.info("Dividindo documentos em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    logging.info(f"{len(chunks)} chunks criados.")

    # Gerar embeddings e criar índice FAISS
    logging.info("Gerando embeddings e criando índice FAISS...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Modelo de embedding recomendado
        vector_store = FAISS.from_documents(chunks, embeddings)
        logging.info("Índice FAISS criado com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao gerar embeddings ou criar índice FAISS: {e}")
        return

    # Salvar índice FAISS localmente
    try:
        vector_store.save_local(index_path)
        logging.info(f"Índice FAISS salvo em: {index_path}")
    except Exception as e:
        logging.error(f"Erro ao salvar o índice FAISS: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa documentos e cria um índice FAISS.")
    parser.add_argument("--docs_path", default="./materiais_curso", help="Caminho para o diretório com os documentos.")
    parser.add_argument("--index_path", default="./faiss_index", help="Caminho para salvar o índice FAISS.")
    
    args = parser.parse_args()
    
    process_documents(args.docs_path, args.index_path)
