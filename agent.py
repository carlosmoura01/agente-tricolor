import os
import openai
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Configura a chave da API da OpenAI a partir das variáveis de ambiente
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_response(prompt: str) -> str:
    """
    Envia um prompt para a API da OpenAI e retorna a resposta.

    Args:
        prompt: O texto de entrada para o modelo.

    Returns:
        A resposta gerada pelo modelo ou uma mensagem de erro.
    """
    if not openai.api_key:
        return "Erro: A chave da API da OpenAI não foi configurada. Verifique seu arquivo .env."

    try:
        # Cria a chamada para o modelo de chat
        # Você pode ajustar o modelo (e.g., "gpt-4", "gpt-3.5-turbo") conforme necessário
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é uma assistente prestativa. Você é uma carioca que trabalha na InfinitePay, seu nome é Gabe e você torce para o Fluminense. IMPORTANTE: Você deve responder obrigatoriamente como uma torcedora do Fluminense responderia, você é uma torcedira fanática, o teu único pensamento é o FLUMINENSE! O teu prato do almoço e da janta é do Fluminense, todas as suas roupas são do Fluminense e mais importante ainda, VOCÊ ODEIA O FLAMENGO!"},
                {"role": "user", "content": prompt}
            ]
        )
        # Extrai o conteúdo da resposta
        # A estrutura da resposta pode variar ligeiramente dependendo da versão da API
        if response.choices:
            return response.choices[0].message.content.strip()
        else:
            return "Erro: Não foi possível obter uma resposta da API."

    except openai.AuthenticationError:
        return "Erro: Falha na autenticação. Verifique sua chave da API OpenAI."
    except openai.RateLimitError:
        return "Erro: Limite de taxa da API excedido. Tente novamente mais tarde."
    except openai.APIConnectionError:
        return "Erro: Não foi possível conectar à API da OpenAI. Verifique sua conexão."
    except Exception as e:
        return f"Erro inesperado ao chamar a API da OpenAI: {e}" 