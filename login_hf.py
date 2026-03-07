#!/usr/bin/env python3
"""
🤗 Script Auxiliar para Login no Hugging Face
=============================================

Este script resolve problemas de login no Hugging Face, especialmente
em ambientes Windows/Git Bash onde o comando `hf` pode não estar no PATH.

Uso:
    python login_hf.py

O que este script faz:
    1. Importa a função login do huggingface_hub
    2. Abre um prompt interativo para você inserir seu token
    3. Salva o token para uso futuro

Onde obter seu token:
    https://huggingface.co/settings/tokens
    
Certifique-se de criar um token com permissão de 'read' (leitura)
para acessar modelos, ou 'write' (escrita) se precisar fazer uploads.
"""

import sys

def main():
    print("="*60)
    print("🤗 Login no Hugging Face")
    print("="*60)
    print()
    
    try:
        from huggingface_hub import login
    except ImportError:
        print("❌ Erro: huggingface_hub não está instalado!")
        print()
        print("Execute primeiro:")
        print("    pip install huggingface_hub")
        print()
        sys.exit(1)
    
    print("Este script irá fazer login na sua conta do Hugging Face.")
    print()
    print("📝 Para obter seu token de acesso:")
    print("   1. Acesse: https://huggingface.co/settings/tokens")
    print("   2. Crie um novo token (ou copie um existente)")
    print("   3. Cole o token quando solicitado abaixo")
    print()
    print("-"*60)
    
    try:
        # A função login() abre um prompt interativo
        login()
        print()
        print("✅ Login realizado com sucesso!")
        print()
    except Exception as e:
        print(f"\n❌ Erro durante o login: {e}")
        print()
        print("💡 Alternativa: Configure a variável de ambiente HF_TOKEN")
        print("   No arquivo .env, adicione:")
        print("   HF_TOKEN=hf_seu_token_aqui")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
