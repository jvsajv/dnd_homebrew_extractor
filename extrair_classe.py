import openai
from pdfminer.high_level import extract_text
import json
import sys
import os
from dotenv import load_dotenv

# Carrega a chave da API
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extrair_texto_pdf(path_pdf):
    try:
        return extract_text(path_pdf)
    except Exception as e:
        raise RuntimeError(f"Erro ao extrair texto do PDF: {e}")

def montar_prompt(texto):
    return f"""
Você é um especialista em Dungeons & Dragons 5e. Com base no conteúdo abaixo, gere um JSON estruturado contendo todas as informações da classe em **português**.

📌 Instruções:
- Use apenas nomes legíveis, não inclua campos como "indice".
- Todos os nomes de atributos e valores devem estar em português.
- O campo `equipamento_inicial` deve conter apenas o que é fixo.
- O campo `opcoes_equipamento` deve conter uma lista de escolhas possíveis por linha no estilo (a) ou (b):
  - Cada item da lista tem "descricao" e "opcoes" (sublistas com as opções A e B).
  - Exemplo:
    {{
      "descricao": "(a) uma besta OU (b) um arco",
      "opcoes": [["uma besta"], ["um arco"]]
    }}

🔤 Traduza todos os termos (mesmo que o texto esteja em inglês).

❌ NÃO inclua:
- Markdown (```)
- Campos com "índice"
- URLs ou explicações

✅ JSON esperado (exemplo):
{{
  "nome": "Guerreiro",
  "dado_vida": 10,
  "testes_resistencia": ["Força", "Constituição"],
  "proficiencias": {{
    "armaduras": ["Todas as Armaduras", "Escudos"],
    "armas": ["Armas Simples", "Armas Marciais"],
    "ferramentas": []
  }},
  "pericias_opcionais": {{
    "descricao": "Escolha duas entre: Atletismo, Intimidação...",
    "escolher": 2,
    "opcoes": ["Atletismo", "Intimidação", "Natureza"]
  }},
  "equipamento_inicial": [],
  "opcoes_equipamento": [...],
  "niveis_classe": "classes/guerreiro/niveis",
  "multi_classe": {{
    "requisitos": ["Força 13"],
    "proficiencias": ["Armas Simples", "Armas Marciais"]
  }},
  "subclasses": ["Campeão", "Mestre de Batalha"],
  "descricao": "Guerreiros são especialistas em combate..."
}}

🔽 Abaixo está o conteúdo extraído do PDF:

{texto}

Gere apenas o JSON acima. Nenhuma explicação.
"""


def gerar_json_com_llm(texto):
    prompt = montar_prompt(texto)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    raw_content = response.choices[0].message.content.strip()

    # 🔍 Remove blocos de markdown se existirem
    if raw_content.startswith("```json") or raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1].strip()

    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Erro ao decodificar JSON da LLM: {e}\n\nResposta recebida:\n{raw_content[:300]}...")

def salvar_json(dados):
    nome = dados.get("nome", "classe_desconhecida").lower().replace(" ", "-")
    nome_arquivo = f"extracted_classes/classe_{nome}.json"
    with open(nome_arquivo, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    print(f"💾 Classe salva em: {nome_arquivo}")


def main():
    if len(sys.argv) < 2:
        print("❗ Uso: python extrair_classe.py caminho_para_pdf.pdf")
        return

    caminho_pdf = sys.argv[1]
    print(f"📖 Lendo PDF: {caminho_pdf}")

    try:
        texto = extrair_texto_pdf(caminho_pdf)
        print("🤖 Enviando para LLM...")
        dados = gerar_json_com_llm(texto)
        salvar_json(dados)
        print("✅ JSON gerado com sucesso!")
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()
