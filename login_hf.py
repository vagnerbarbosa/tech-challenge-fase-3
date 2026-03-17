#!/usr/bin/env python3
"""
🤗 Script Auxiliar para Login no Hugging Face
=============================================

Este script resolve problemas de login no Hugging Face, especialmente
em ambientes Windows/Git Bash onde o comando `hf` pode não estar no PATH.

Uso:
    python login_hf.py

O que este script faz:
    1. Tenta carregar o token da variável de ambiente HUGGINGFACE_TOKEN (via .env)
    2. Valida se o token existe e não está vazio
    3. Tenta fazer login com o token
    4. Se houver qualquer erro, solicita ao usuário um token válido
    5. Após receber o token do usuário, valida-o fazendo login

Onde obter seu token:
    https://huggingface.co/settings/tokens
    
Certifique-se de criar um token com permissão de 'read' (leitura)
para acessar modelos, ou 'write' (escrita) se precisar fazer uploads.
"""

import sys
import os
from getpass import getpass


def print_header():
    """Exibe o cabeçalho do script."""
    print("=" * 60)
    print("🤗 Login no Hugging Face")
    print("=" * 60)
    print()


def print_token_instructions():
    """Exibe instruções para obter o token."""
    print("📝 Para obter seu token de acesso:")
    print("   1. Acesse: https://huggingface.co/settings/tokens")
    print("   2. Crie um novo token (ou copie um existente)")
    print("   3. Cole o token quando solicitado")
    print()


def load_token_from_env():
    """
    Tenta carregar o token do arquivo .env ou variáveis de ambiente.
    
    Returns:
        str ou None: O token se encontrado, None caso contrário.
    """
    # Tentar carregar do arquivo .env usando python-dotenv
    try:
        from dotenv import load_dotenv
        # Carregar .env do diretório atual
        load_dotenv()
        print("📂 Arquivo .env carregado com sucesso.")
    except ImportError:
        print("⚠️  python-dotenv não instalado. Tentando variáveis de ambiente do sistema...")
    except Exception as e:
        print(f"⚠️  Erro ao carregar .env: {e}")
    
    # Verificar variáveis de ambiente (tanto HUGGINGFACE_TOKEN quanto HF_TOKEN)
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    
    if token:
        # Remover espaços em branco
        token = token.strip()
        if token:
            print("🔑 Token encontrado nas variáveis de ambiente.")
            return token
    
    return None


def validate_token(token):
    """
    Valida o token tentando fazer login no Hugging Face.
    
    Args:
        token: Token do Hugging Face a ser validado.
        
    Returns:
        bool: True se o login foi bem-sucedido, False caso contrário.
    """
    if not token or not token.strip():
        print("❌ Token vazio ou inválido.")
        return False
    
    try:
        from huggingface_hub import login, HfApi
        
        # Fazer login com o token
        login(token=token, add_to_git_credential=False)
        
        # Verificar se o login foi bem-sucedido tentando acessar a API
        api = HfApi()
        user_info = api.whoami()
        
        print(f"✅ Login realizado com sucesso!")
        print(f"   👤 Usuário: {user_info.get('name', 'N/A')}")
        print(f"   📧 Email: {user_info.get('email', 'N/A')}")
        return True
        
    except ImportError:
        print("❌ Erro: huggingface_hub não está instalado!")
        print()
        print("Execute primeiro:")
        print("    pip install huggingface_hub")
        print()
        sys.exit(1)
        
    except Exception as e:
        error_msg = str(e).lower()
        if "invalid" in error_msg or "401" in error_msg:
            print("❌ Token inválido ou expirado.")
        elif "unauthorized" in error_msg:
            print("❌ Token não autorizado.")
        elif "network" in error_msg or "connection" in error_msg:
            print("❌ Erro de conexão. Verifique sua internet.")
        else:
            print(f"❌ Erro durante o login: {e}")
        return False


def request_token_from_user():
    """
    Solicita ao usuário que insira um token manualmente.
    
    Returns:
        str: O token inserido pelo usuário.
    """
    print()
    print("-" * 60)
    print_token_instructions()
    print("-" * 60)
    
    try:
        # Usar getpass para não exibir o token na tela
        token = getpass("🔐 Cole seu token do Hugging Face: ")
        return token.strip() if token else None
    except (KeyboardInterrupt, EOFError):
        print("\n\n⚠️  Operação cancelada pelo usuário.")
        return None


def main():
    """Função principal do script."""
    print_header()
    
    # Verificar se huggingface_hub está instalado
    try:
        from huggingface_hub import login
    except ImportError:
        print("❌ Erro: huggingface_hub não está instalado!")
        print()
        print("Execute primeiro:")
        print("    pip install huggingface_hub")
        print()
        sys.exit(1)
    
    # Passo 1: Tentar carregar token do .env ou variáveis de ambiente
    print("🔍 Procurando token nas variáveis de ambiente...")
    token = load_token_from_env()
    
    if token:
        print()
        print("🔄 Tentando fazer login com o token encontrado...")
        if validate_token(token):
            print()
            print("🎉 Tudo pronto! Você já pode usar os modelos do Hugging Face.")
            return
        else:
            print()
            print("⚠️  O token encontrado no .env não é válido.")
    else:
        print("⚠️  Nenhum token encontrado no .env ou variáveis de ambiente.")
    
    # Passo 2: Solicitar token do usuário
    print()
    print("📋 Por favor, insira um token válido manualmente.")
    
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        user_token = request_token_from_user()
        
        if not user_token:
            if attempt < max_attempts:
                print(f"\n⚠️  Token não fornecido. Tentativa {attempt}/{max_attempts}.")
                continue
            else:
                print("\n❌ Número máximo de tentativas excedido.")
                print()
                print("💡 Dicas:")
                print("   1. Configure a variável HUGGINGFACE_TOKEN no arquivo .env")
                print("   2. Ou exporte: export HUGGINGFACE_TOKEN=hf_seu_token")
                print("   3. Obtenha um token em: https://huggingface.co/settings/tokens")
                sys.exit(1)
        
        print()
        print("🔄 Validando o token inserido...")
        
        if validate_token(user_token):
            print()
            print("🎉 Tudo pronto! Você já pode usar os modelos do Hugging Face.")
            print()
            print("💡 Dica: Para não precisar inserir o token novamente,")
            print("   adicione ao seu arquivo .env:")
            print(f"   HUGGINGFACE_TOKEN={user_token[:10]}...")
            return
        else:
            if attempt < max_attempts:
                print(f"\n⚠️  Token inválido. Tentativa {attempt}/{max_attempts}.")
            else:
                print("\n❌ Número máximo de tentativas excedido.")
                print()
                print("💡 Verifique se:")
                print("   1. O token foi copiado corretamente")
                print("   2. O token não está expirado")
                print("   3. O token tem as permissões necessárias (read)")
                print()
                print("Obtenha um novo token em: https://huggingface.co/settings/tokens")
                sys.exit(1)

