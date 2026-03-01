"""
Módulo de Treinamento (Fine-tuning) do LLM
==========================================

Responsável por:
- Carregar modelo base (LLaMA/Falcon)
- Configurar LoRA para fine-tuning eficiente
- Treinar o modelo com os dados preparados
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Classe para fine-tuning de modelos LLM usando LoRA.
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
            "meta-llama/Llama-2-7b-chat-hf"
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
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Configuração de quantização
        bnb_config = self._get_quantization_config()
        
        # Carrega modelo
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
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
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
            fp16=True if self.device == "cuda" else False,
            optim="paged_adamw_8bit",
            report_to="none",
        )
    
    def train(self, dataset: Dataset) -> Tuple[Any, Any]:
        """
        Executa o fine-tuning do modelo.
        
        Args:
            dataset: Dataset preparado para treinamento
            
        Returns:
            Tupla (modelo treinado, tokenizer)
        """
        logger.info("Iniciando treinamento...")
        
        # Carrega modelo e tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Argumentos de treinamento
        training_args = self.get_training_arguments()
        
        # Configura trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            packing=False,
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


if __name__ == "__main__":
    # Teste do módulo
    from src.fine_tuning.data_preparation import DataPreparation
    
    prep = DataPreparation()
    dataset = prep.prepare_dataset()
    
    trainer = ModelTrainer()
    model, tokenizer = trainer.train(dataset)
