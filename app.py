import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

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
Este é um agente de IA com personalidade de aluno, 
que pode discutir temas relacionados a IA, tomada de decisão baseada em dados 
e transformação digital.
""")

# Configuração da API do Google Gemini
def get_api_key():
    # Em produção no Streamlit Cloud, a chave será obtida dos secrets
    if 'GEMINI_API_KEY' in st.secrets:
        return st.secrets['GEMINI_API_KEY']
    # Para desenvolvimento local, pode-se usar uma variável de ambiente ou um input
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        # Apenas para desenvolvimento - em produção, use secrets
        if 'api_key_input' not in st.session_state:
            st.session_state.api_key_input = None
        
        if st.session_state.api_key_input is None:
            st.session_state.api_key_input = st.text_input(
                "Insira sua chave de API do Google Gemini (apenas para desenvolvimento):",
                type="password"
            )
        api_key = st.session_state.api_key_input
    
    return api_key

# Inicialização do modelo
@st.cache_resource
def get_llm():
    api_key = get_api_key()
    if not api_key:
        return None
    
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=api_key,
        temperature=0.7,
        convert_system_message_to_human=True
    )

# Inicialização do histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Olá! Sou um aluno virtual da pós-graduação em IA em Negócios. Estou aprendendo sobre IA, tomada de decisão baseada em dados e transformação digital. Como posso ajudar você hoje?")
    ]

# Exibição do histórico de mensagens
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(message.content)
    else:
        with st.chat_message("user"):
            st.markdown(message.content)

# Instruções de personalidade para o modelo
SYSTEM_PROMPT = """
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
"""

# Input do usuário
if prompt := st.chat_input("Escreva sua mensagem..."):
    # Adiciona a mensagem do usuário ao histórico
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Exibe a mensagem do usuário
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepara o contexto para o modelo
    with st.chat_message("assistant", avatar="🧠"):
        # Verifica se a API key está disponível
        llm = get_llm()
        if not llm:
            st.error("Por favor, forneça uma chave de API válida para o Google Gemini para continuar.")
        else:
            with st.spinner("Pensando..."):
                # Prepara as mensagens para o modelo
                messages_for_model = [
                    {"role": "system", "content": SYSTEM_PROMPT}
                ]
                
                # Adiciona o histórico de mensagens
                for msg in st.session_state.messages:
                    if isinstance(msg, HumanMessage):
                        messages_for_model.append({"role": "human", "content": msg.content})
                    else:
                        messages_for_model.append({"role": "ai", "content": msg.content})
                
                # Obtém a resposta do modelo
                response = llm.invoke(messages_for_model)
                
                # Exibe a resposta
                st.markdown(response.content)
                
                # Adiciona a resposta ao histórico
                st.session_state.messages.append(AIMessage(content=response.content))
