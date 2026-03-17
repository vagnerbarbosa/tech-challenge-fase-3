"""
Módulo de Treinamento (Fine-tuning) do LLM
==========================================

Responsável por:
- Carregar modelo base (LLaMA/Falcon/TinyLlama)
- Configurar LoRA para fine-tuning eficiente
- Treinar o modelo com os dados preparados
- Validar existência de modelo treinado antes do treinamento
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig
from trl import SFTTrainer
from datasets import Dataset

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Classe para fine-tuning de modelos LLM usando LoRA.
    
    Inclui validação para verificar modelo existente antes do treinamento,
    permitindo ao usuário escolher entre sobrescrever ou reutilizar.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Inicializa o treinador.
        
        Args:
            model_name: Nome do modelo base no Hugging Face
            output_dir: Diretório para salvar o modelo treinado
        """
        self.model_name = model_name or os.getenv(
            "BASE_MODEL_NAME", 
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        self.output_dir = Path(output_dir or os.getenv("MODEL_PATH", "./models"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurações de treinamento
        self.max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", 512))
        self.batch_size = int(os.getenv("BATCH_SIZE", 4))
        self.learning_rate = float(os.getenv("LEARNING_RATE", 2e-4))
        self.num_epochs = int(os.getenv("NUM_EPOCHS", 3))
        
        # Detecta dispositivo
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ModelTrainer inicializado. Device: {self.device}")
        logger.info(f"Modelo base: {self.model_name}")
    
    def _check_existing_model(self) -> bool:
        """
        Verifica se já existe um modelo treinado no diretório de saída.
        
        Returns:
            bool: True se existe um modelo treinado, False caso contrário
        """
        final_model_path = self.output_dir / "final_model"
        
        # Verifica se o diretório existe e contém arquivos de modelo
        if final_model_path.exists() and final_model_path.is_dir():
            # Verifica se contém arquivos essenciais do modelo
            model_files = [
                "config.json",
                "pytorch_model.bin",
                "model.safetensors",
                "adapter_config.json",
                "adapter_model.bin",
                "adapter_model.safetensors",
            ]
            
            has_model_file = any(
                (final_model_path / f).exists() for f in model_files
            )
            
            if has_model_file:
                logger.info(f"Modelo existente encontrado em: {final_model_path}")
                return True
        
        return False
    
    def _prompt_user_for_overwrite(self) -> str:
        """
        Pergunta ao usuário se deseja sobrescrever o modelo existente.
        
        Returns:
            str: 'overwrite' para sobrescrever, 'use_existing' para usar o existente
        """
        final_model_path = self.output_dir / "final_model"
        
        print("\n" + "=" * 60)
        print("⚠️  MODELO EXISTENTE DETECTADO")
        print("=" * 60)
        print(f"\nJá existe um modelo treinado em:\n  {final_model_path}")
        print("\nO que você deseja fazer?")
        print("  [1] Sobrescrever o modelo existente e treinar novamente")
        print("  [2] Usar o modelo existente e pular o treinamento")
        print("=" * 60)
        
        while True:
            choice = input("\nDigite sua escolha (1 ou 2): ").strip()
            
            if choice == "1":
                logger.info("Usuário escolheu sobrescrever o modelo existente")
                return "overwrite"
            elif choice == "2":
                logger.info("Usuário escolheu usar o modelo existente")
                return "use_existing"
            else:
                print("❌ Entrada inválida. Por favor, digite 1 ou 2.")
    
    def _load_existing_model(self) -> Tuple[Any, Any]:
        """
        Carrega o modelo existente com LoRA do diretório de saída.
        
        Returns:
            Tuple contendo (modelo, tokenizer)
        """
        final_model_path = self.output_dir / "final_model"
        logger.info(f"Carregando modelo existente de: {final_model_path}")

        # Carrega tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(final_model_path), trust_remote_code=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Carrega configuração PEFT para saber o modelo base
        peft_config = PeftConfig.from_pretrained(str(final_model_path))

        # Carrega modelo base
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=self._get_quantization_config(),
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=False,
        )

        # Aplica LoRA
        model = PeftModel.from_pretrained(base_model, str(final_model_path))

        logger.info("Modelo existente com LoRA carregado com sucesso!")

        return model, tokenizer

    def _get_quantization_config(self) -> BitsAndBytesConfig:
        """
        Retorna configuração de quantização 4-bit.
        
        Returns:
            Configuração BitsAndBytes
        """
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    
    def _get_lora_config(self) -> LoraConfig:
        """
        Retorna configuração LoRA para fine-tuning eficiente.
        
        Returns:
            Configuração LoRA
        """
        return LoraConfig(
            r=16,  # Rank da decomposição
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    
    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """
        Carrega o modelo e tokenizer.
        
        Returns:
            Tupla (modelo, tokenizer)
        """
        logger.info(f"Carregando modelo: {self.model_name}")
        
        # Carrega tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=False,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Configuração de quantização
        bnb_config = self._get_quantization_config()
        
        # Carrega modelo
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=False,
        )
        
        # Prepara para treinamento com quantização
        model = prepare_model_for_kbit_training(model)
        
        # Aplica LoRA
        lora_config = self._get_lora_config()
        model = get_peft_model(model, lora_config)
        
        # Log de parâmetros treináveis
        trainable, total = model.get_nb_trainable_parameters()
        logger.info(f"Parâmetros treináveis: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
        return model, tokenizer
    
    def get_training_arguments(self) -> TrainingArguments:
        """
        Retorna argumentos de treinamento.
        
        Returns:
            TrainingArguments configurado
        """

        return TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            warmup_steps=10,  
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="no",
            fp16=True if self.device == "cuda" else False,
            optim="adamw_torch",  
            report_to="none",
        )
    
    def train(
        self,
        dataset: Dataset,
        force_retrain: bool = False,
    ) -> Tuple[Any, Any]:
        """
        Executa o fine-tuning do modelo.
        
        Verifica se já existe um modelo treinado e pergunta ao usuário
        se deseja sobrescrever ou usar o existente.
        
        Args:
            dataset: Dataset preparado para treinamento
            force_retrain: Se True, força o retreinamento sem perguntar ao
                          usuário. Útil para automação e scripts não-interativos.
            
        Returns:
            Tupla (modelo treinado, tokenizer)
        """
        logger.info("Iniciando processo de treinamento...")
        
        # Verifica se já existe um modelo treinado
        if self._check_existing_model():
            if force_retrain:
                logger.info("force_retrain=True: Sobrescrevendo modelo existente sem perguntar")
            else:
                user_choice = self._prompt_user_for_overwrite()
                
                if user_choice == "use_existing":
                    logger.info("Pulando treinamento - usando modelo existente")
                    model, tokenizer = self._load_existing_model()
                    return model, tokenizer
                
                # Se chegou aqui, user_choice == "overwrite"
                logger.info("Usuário optou por sobrescrever - iniciando novo treinamento")
        
        # Carrega modelo e tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Argumentos de treinamento
        training_args = self.get_training_arguments()
        
        # Configura trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            processing_class=tokenizer
        )
        
        # Treina
        logger.info("Executando treinamento...")
        trainer.train()
        
        # Salva modelo final
        final_path = self.output_dir / "final_model"
        model.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        logger.info(f"Modelo salvo em: {final_path}")
        
        return model, tokenizer
    
    def get_model_path(self) -> Path:
        """
        Retorna o caminho do modelo treinado.
        
        Returns:
            Path do modelo final
        """
        return self.output_dir / "final_model"




if __name__ == "__main__":
    """
    Execução isolada do módulo de treinamento.
    
    Uso:
        python -m src.fine_tuning.training
    
    Exibe informações sobre a configuração de treinamento
    e verifica se existe modelo treinado previamente.
    """
    from src.utils.logging_config import setup_logging
    setup_logging()
    
    print("=" * 60)
    print("🏋️ MÓDULO DE TREINAMENTO (Fine-tuning)")
    print("=" * 60)
    
    trainer = ModelTrainer()
    
    print(f"\n📋 Configuração atual:")
    print(f"   Modelo base: {trainer.model_name}")
    print(f"   Diretório de saída: {trainer.output_dir}")
    print(f"   Device: {trainer.device}")
    print(f"   Max seq length: {trainer.max_seq_length}")
    print(f"   Batch size: {trainer.batch_size}")
    print(f"   Learning rate: {trainer.learning_rate}")
    print(f"   Num epochs: {trainer.num_epochs}")
    
    has_model = trainer._check_existing_model()
    print(f"\n🔍 Modelo existente: {'✅ Sim' if has_model else '❌ Não'}")
    print(f"   Caminho: {trainer.get_model_path()}")
    
    print(f"\n💡 Para treinar, use no pipeline completo:")
    print(f"   python main.py")
    print(f"\n   Ou integre via código:")
    print(f"   trainer = ModelTrainer()")
    print(f"   model, tokenizer = trainer.train(dataset)")
