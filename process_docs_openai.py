import os
import sys
import argparse
import logging
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_api_key():
    """Obtém a chave da API da OpenAI."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        logging.error("Chave da API da OpenAI não encontrada na variável de ambiente OPENAI_API_KEY.")
        logging.error("Por favor, defina a variável de ambiente antes de executar o script.")
        sys.exit(1)
    return api_key

def process_documents(docs_path, index_path, incremental=False):
    """Carrega, processa documentos e cria ou atualiza um índice FAISS."""
    
    api_key = get_api_key()
    
    # Verificar se o diretório de documentos existe
    docs_dir = Path(docs_path)
    if not docs_dir.exists() or not docs_dir.is_dir():
        logging.error(f"O diretório {docs_path} não existe ou não é um diretório válido.")
        return
    
    logging.info(f"Iniciando carregamento de documentos do diretório: {docs_path}")
    
    # Usar DirectoryLoader com UnstructuredFileLoader para lidar com vários tipos
    loader = DirectoryLoader(
        docs_path, 
        glob="**/*.*", 
        use_multithreading=True, 
        show_progress=True,
        loader_cls=UnstructuredFileLoader,
        silent_errors=True
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

    # Configurar embeddings da OpenAI
    logging.info("Gerando embeddings...")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Verificar se é uma atualização incremental ou criação de novo índice
    index_dir = Path(index_path)
    if incremental and index_dir.exists():
        try:
            logging.info(f"Carregando índice FAISS existente de: {index_path}")
            vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            
            logging.info("Adicionando novos documentos ao índice existente...")
            vector_store.add_documents(chunks)
            logging.info(f"Índice FAISS atualizado com {len(chunks)} novos chunks.")
        except Exception as e:
            logging.error(f"Erro ao atualizar o índice FAISS: {e}")
            logging.warning("Criando um novo índice em vez de atualizar...")
            vector_store = FAISS.from_documents(chunks, embeddings)
    else:
        try:
            logging.info("Criando novo índice FAISS...")
            vector_store = FAISS.from_documents(chunks, embeddings)
            logging.info("Índice FAISS criado com sucesso.")
        except Exception as e:
            logging.error(f"Erro ao criar o índice FAISS: {e}")
            return

    # Salvar índice FAISS localmente
    try:
        vector_store.save_local(index_path)
        logging.info(f"Índice FAISS salvo em: {index_path}")
    except Exception as e:
        logging.error(f"Erro ao salvar o índice FAISS: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa documentos e cria ou atualiza um índice FAISS.")
    parser.add_argument("--docs_path", default="./materiais_curso", help="Caminho para o diretório com os documentos.")
    parser.add_argument("--index_path", default="./faiss_index", help="Caminho para salvar o índice FAISS.")
    parser.add_argument("--incremental", action="store_true", help="Atualizar índice existente em vez de criar um novo.")
    
    args = parser.parse_args()
    
    process_documents(args.docs_path, args.index_path, args.incremental)
