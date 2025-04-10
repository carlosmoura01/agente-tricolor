import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração da página do Streamlit
st.set_page_config(page_title="Agente Interativo", layout="wide")

# Título da aplicação
st.title("Agente Interativo com Memória")

# Inicialização do modelo de linguagem
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# --- Modificações para Memória ---

# 1. Inicializar memória e histórico de mensagens no session_state
if "memory" not in st.session_state:
    # Cria a memória (guarda o histórico para o LLM)
    # return_messages=True é importante para ChatModels
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
if "messages" not in st.session_state:
    # Cria o histórico de mensagens (guarda para exibição no Streamlit)
    st.session_state.messages = []

# 2. Modificar o prompt para incluir o histórico
# Adicionamos MessagesPlaceholder para onde o histórico da memória será inserido
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente útil. Responda às perguntas do usuário da melhor forma possível."),
    MessagesPlaceholder(variable_name="history"), # Onde o histórico da memória será injetado
    ("human", "{input}")
])

# 3. Criar a LLMChain com o LLM, prompt e memória
# A LLMChain gerenciará automaticamente o carregamento e salvamento da memória
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=st.session_state.memory,
    verbose=True # Opcional: mostra o que a chain está fazendo no console
)

# --- Fim das Modificações para Memória ---

# Exibir mensagens do histórico ao recarregar a página
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada do usuário
if user_query := st.chat_input("Qual sua dúvida?"):
    # Adicionar mensagem do usuário ao histórico de exibição e exibir
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Exibir mensagem de "pensando..."
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Pensando...")

        # 4. Invocar a LLMChain (que usa a memória)
        # A resposta já virá formatada como string por padrão na LLMChain simples
        # O input deve ser um dicionário com a chave correspondente no prompt ("input")
        response = chain.invoke({"input": user_query})
        ai_response = response['text'] # A LLMChain retorna um dicionário

        # Atualizar a mensagem de "pensando..." com a resposta real
        message_placeholder.markdown(ai_response)

    # Adicionar resposta da IA ao histórico de exibição
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

    # Debug: Opcional - Ver o que está na memória após a interação
    # st.sidebar.write("Conteúdo da Memória:")
    # st.sidebar.write(st.session_state.memory.load_memory_variables({})) 