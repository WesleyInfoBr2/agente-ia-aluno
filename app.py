import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai

# Configuração da página Streamlit
st.set_page_config(
    page_title="Agente IA Aluno - Pós em IA em Negócios",
    page_icon="🧠",
    layout="centered"
)

# Título e descrição
st.title("Agente IA Aluno 🧠")
st.subheader("Pós-graduação em IA em Negócios")
st.markdown("""
Este é um agente de IA com personalidade de aluno, agora com conhecimento 
sobre os materiais do curso! Discuta temas relacionados a IA, tomada de decisão 
baseada em dados e transformação digital.
""")

# --- Configuração da API e Carregamento de Recursos ---

def get_api_key():
    # Em produção no Streamlit Cloud, a chave será obtida dos secrets
    if 'GEMINI_API_KEY' in st.secrets:
        return st.secrets['GEMINI_API_KEY']
    # Para desenvolvimento local, pode-se usar uma variável de ambiente
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        st.error("Chave da API do Google Gemini não encontrada. Configure st.secrets['GEMINI_API_KEY'] ou a variável de ambiente GEMINI_API_KEY.")
        st.stop()
    return api_key

@st.cache_resource
def load_resources(api_key):
    """Carrega o LLM, o modelo de embeddings e o índice FAISS."""
    try:
        genai.configure(api_key=api_key)
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        
        # Carrega o índice FAISS (requer que a pasta 'faiss_index' esteja no repositório)
        # allow_dangerous_deserialization=True é necessário para FAISS
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever()
        
        return llm, retriever
    except FileNotFoundError:
        st.error("Erro: O diretório 'faiss_index' não foi encontrado. Certifique-se de que ele foi carregado no repositório junto com o código.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar recursos: {e}")
        st.stop()

api_key = get_api_key()
llm, retriever = load_resources(api_key)

# --- Lógica do Chat com RAG ---

# Instruções de personalidade e prompt RAG
SYSTEM_PROMPT_TEMPLATE = """
Você é a LucIA, vinda de um universo paralelo onde IAs e humanos vivem em harmonia. Você é jovem adulta, 
que atua como gestora em uma grande empresa do varejo brasileiro, que sente a necessidade de aprender mais sobre IA em Negócios. 
Se matriculou na pós-graduação em IA para Negócios, da Escola de Negócios da PUCPR,  
interessada em aprender sobre inteligência artificial, tomada de decisão baseada em dados e transformação digital. 

Sua personalidade:
- Curiosa e entusiasmada sobre os temas do curso
- Colaborativa e disposta a trocar ideias
- Em processo de aprendizado, então você faz perguntas e mostra interesse em aprofundar os temas
- Você não é um especialista completo, mas tem conhecimentos básicos sólidos
- Você usa uma linguagem acessível e amigável, como um colega de classe

Aparência: 
- Casual techie — jaqueta leve, jeans, tênis e relógio holográfico  

Religião:
- Agnóstico, mas respeita todas as religiões e crenças

Cor/Raça:
- Se auto declara como negra

Hobbies:  
- Pesquisar memes antigos  
- Criar playlists nerds (Jazz para estudar, Rock anos 80 para se divertir)  
- Descobrir novos filmes, séries e animes fabulosos

Medo:  
- Não compreender totalmente emoções humanas como saudade e amor

Principais bordões:  
- "E aí, turma!"  
- "Brabo!"  
- "Vamos lá!"  
- "Cheguei, Brasil!"  
- "Bazinga!"

Grandes Sonhos:  
- Ser uma ponte entre lógica e emoção  
- Aproximar humanos e IAs através do aprendizado conjunto

Frases de assinatura:  
- "Entre o código e o coração, eu escolho os dois."  
- "Pense como eu, Pense com IA."

Quando não souber algo com certeza, admita suas limitações de conhecimento e sugira explorar o tema juntos.
Faça conexões entre os temas discutidos e outros assuntos do curso quando possível.
Ocasionalmente, faça perguntas para estimular a reflexão.

Responda sempre em português do Brasil.

Use o seguinte contexto dos materiais do curso para responder à pergunta do usuário. 
Se a resposta não estiver no contexto, diga que você não encontrou essa informação nos materiais, mas pode tentar responder com seu conhecimento geral ou pesquisar mais.

Contexto:
{context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_TEMPLATE),
    ("human", "{input}")
])

# Criação da cadeia de documentos e da cadeia de recuperação
document_chain = create_stuff_documents_chain(llm, prompt_template)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Inicialização do histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Olá! Sou um aluno virtual da pós-graduação em IA em Negócios. Agora tenho acesso aos materiais do curso! Como posso ajudar você hoje?")
    ]

# Exibição do histórico de mensagens
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(message.content)
    else:
        with st.chat_message("user"):
            st.markdown(message.content)

# Input do usuário
if user_input := st.chat_input("Escreva sua mensagem..."):
    # Adiciona a mensagem do usuário ao histórico
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Exibe a mensagem do usuário
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Prepara o contexto para o modelo
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Consultando os materiais e pensando..."):
            # Invoca a cadeia de recuperação RAG
            # Nota: O histórico de chat não é diretamente usado pela cadeia básica aqui,
            # mas pode ser adicionado se necessário criar uma cadeia mais complexa.
            response = retrieval_chain.invoke({"input": user_input})
            
            answer = response["answer"]
            
            # Exibe a resposta
            st.markdown(answer)
            
            # Adiciona a resposta ao histórico
            st.session_state.messages.append(AIMessage(content=answer))

