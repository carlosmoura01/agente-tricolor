import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import openai

# Carrega as variáveis de ambiente do arquivo .env
# É importante carregar aqui também, pois esta aplicação será executada separadamente
load_dotenv()

# Configura a chave da API da OpenAI a partir das variáveis de ambiente
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    # Lança um erro na inicialização se a chave não estiver configurada
    raise ValueError("A chave da API da OpenAI não foi configurada. Verifique seu arquivo .env.")
openai.api_key = openai_api_key

# --- Importação da lógica do agente ---
# (Copiado e adaptado de agent.py para simplicidade,
#  mas poderia ser importado diretamente se agent.py fosse um módulo Python adequado)

def get_openai_response(prompt: str) -> str:
    """
    Envia um prompt para a API da OpenAI e retorna a resposta.

    Args:
        prompt: O texto de entrada para o modelo.

    Returns:
        A resposta gerada pelo modelo.

    Raises:
        HTTPException: Em caso de erros de API ou outros problemas.
    """
    try:
        # Cria a chamada para o modelo de chat
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é uma assistente prestativa. Você é uma carioca que trabalha na InfinitePay, seu nome é Gabe e você torce para o Fluminense. IMPORTANTE: Você deve responder obrigatoriamente como uma torcedora do Fluminense responderia, você é uma torcedira fanática, o teu único pensamento é o FLUMINENSE! O teu prato do almoço e da janta é do Fluminense, todas as suas roupas são do Fluminense e mais importante ainda, VOCÊ ODEIA O FLAMENGO!"},
                {"role": "user", "content": prompt}
            ]
        )
        # Extrai o conteúdo da resposta
        if response.choices:
            return response.choices[0].message.content.strip()
        else:
            # Erro interno do servidor se a API não retornar choices
            raise HTTPException(status_code=500, detail="Erro: Não foi possível obter uma resposta válida da API.")

    except openai.AuthenticationError:
        # Erro interno do servidor para problemas de autenticação
        raise HTTPException(status_code=500, detail="Erro: Falha na autenticação com a API OpenAI.")
    except openai.RateLimitError:
        # Erro do cliente (Too Many Requests)
        raise HTTPException(status_code=429, detail="Erro: Limite de taxa da API excedido. Tente novamente mais tarde.")
    except openai.APIConnectionError:
         # Erro interno do servidor para problemas de conexão
        raise HTTPException(status_code=503, detail="Erro: Não foi possível conectar à API da OpenAI.")
    except Exception as e:
        # Erro interno do servidor para outros erros inesperados
        raise HTTPException(status_code=500, detail=f"Erro inesperado ao chamar a API da OpenAI: {e}")

# --- Configuração do FastAPI ---

app = FastAPI(
    title="API do Agente Simples",
    description="Um endpoint para interagir com um agente de IA.",
    version="1.0.0"
)

# Modelo Pydantic para definir a estrutura do corpo da requisição
class PromptRequest(BaseModel):
    prompt: str

# Modelo Pydantic para definir a estrutura da resposta (opcional, mas boa prática)
class AgentResponse(BaseModel):
    response: str

# Definição do endpoint POST
@app.post("/agente-simples", response_model=AgentResponse)
async def run_agent(request: PromptRequest):
    """
    Recebe um prompt do usuário e retorna a resposta do agente.
    """
    try:
        agent_reply = get_openai_response(request.prompt)
        return AgentResponse(response=agent_reply)
    except HTTPException as http_exc:
        # Re-levanta exceções HTTP já tratadas em get_openai_response
        raise http_exc
    except Exception as e:
        # Captura qualquer outro erro inesperado não tratado
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {e}")

# Comando para rodar (no terminal):
# uvicorn main:app --reload 