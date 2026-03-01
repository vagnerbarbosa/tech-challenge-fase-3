"""
Módulo de Treinamento (Fine-tuning) do LLM

Implementa fine-tuning com:
- LoRA (Low-Rank Adaptation)
- Quantização 4-bit
- Gradient checkpointing
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Classe para fine-tuning de modelos LLM."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        output_dir: str = "./models",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        max_seq_length: int = 512,
        use_4bit: bool = True,
        use_lora: bool = True
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.use_4bit = use_4bit
        self.use_lora = use_lora
        
        self.model = None
        self.tokenizer = None
        
        # Cria diretório de saída
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Retorna configuração de quantização 4-bit."""
        if not self.use_4bit:
            return None
            
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    
    def _get_lora_config(self) -> LoraConfig:
        """Retorna configuração do LoRA."""
        return LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
    
    def load_model(self):
        """Carrega modelo e tokenizer."""
        logger.info(f"Carregando modelo: {self.model_name}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Modelo com quantização
        quantization_config = self._get_quantization_config()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepara para treinamento k-bit
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Aplica LoRA
        if self.use_lora:
            lora_config = self._get_lora_config()
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        logger.info("Modelo carregado com sucesso!")
    
    def load_dataset(self, data_path: str = "./data/processed/medical_dataset.json") -> Dataset:
        """Carrega dataset de treinamento."""
        logger.info(f"Carregando dataset: {data_path}")
        
        dataset = load_dataset("json", data_files=data_path, split="train")
        logger.info(f"Dataset carregado: {len(dataset)} exemplos")
        return dataset
    
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokeniza exemplos para treinamento."""
        # Formata prompt no estilo instrução
        prompts = []
        for instruction, inp, output in zip(
            examples["instruction"],
            examples["input"],
            examples["output"]
        ):
            if inp:
                prompt = f"### Instrução:\n{instruction}\n\n### Contexto:\n{inp}\n\n### Resposta:\n{output}"
            else:
                prompt = f"### Instrução:\n{instruction}\n\n### Resposta:\n{output}"
            prompts.append(prompt)
        
        # Tokeniza
        return self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length"
        )
    
    def train(self, data_path: Optional[str] = None):
        """Executa o fine-tuning."""
        logger.info("=" * 50)
        logger.info("Iniciando Fine-tuning")
        logger.info("=" * 50)
        
        # Carrega modelo
        self.load_model()
        
        # Carrega dados
        data_path = data_path or "./data/processed/medical_dataset.json"
        dataset = self.load_dataset(data_path)
        
        # Tokeniza
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Argumentos de treinamento
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            fp16=True,
            optim="paged_adamw_32bit",
            gradient_checkpointing=True,
            report_to="wandb" if os.getenv("WANDB_API_KEY") else "none"
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        
        # Treina
        logger.info("Iniciando treinamento...")
        trainer.train()
        
        # Salva modelo
        model_save_path = self.output_dir / "assistente-medico-final"
        self.model.save_pretrained(str(model_save_path))
        self.tokenizer.save_pretrained(str(model_save_path))
        
        logger.info(f"Modelo salvo em: {model_save_path}")
        logger.info("Fine-tuning concluído!")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
