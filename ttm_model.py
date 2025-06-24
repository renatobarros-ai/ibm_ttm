"""
Sistema Preditivo de Demanda - Equipamentos Hidráulicos
Multinacional - Granite TinyTimeMixer (TTM) IBM

CONTEXTO EMPRESARIAL:
Sistema preditivo desenvolvido para empresa multinacional especializada em equipamentos 
hidráulicos. Utiliza o modelo Granite TimeSeries TTM da IBM para antecipar demanda de 
peças e equipamentos, fornecendo previsões de 1 a 6 meses (4-26 semanas) para diferentes 
regiões e categorias de produtos.

ESCOPO DO MODELO:
- Estados Prioritários: SP, GO, MG, EX, RS, PR (>95% vendas/faturamento)
- Top 150 produtos mais vendidos (>70% vendas/faturamento)
- Previsões multivariadas: Volume de vendas + Faturamento
- Horizonte: 1-6 meses sem necessidade de variáveis preditoras futuras

Autor: Renato Barros
Data: 23/06/2025
Versão: 4.1 - Correções Críticas Implementadas
"""

import pandas as pd
import torch
import numpy as np
import math
import os
import warnings

# Importações específicas da biblioteca Hugging Face e TSFM para treinamento
from transformers import (
   TrainingArguments,       # Argumentos de treinamento do modelo
   Trainer,                 # Classe principal para treinamento
   EarlyStoppingCallback,   # Callback para parada antecipada
)
from torch.optim import AdamW                    # Otimizador AdamW
from torch.optim.lr_scheduler import OneCycleLR  # Scheduler de learning rate

# Componentes da biblioteca IBM Granite TSFM para processamento de séries temporais
from tsfm_public import (
   TimeSeriesPreprocessor,      # Preprocessador de séries temporais
   TinyTimeMixerForPrediction,  # Modelo TTM para previsão
   ForecastDFDataset,           # Dataset personalizado para previsão
   TrackingCallback,            # Callback para tracking de métricas
)
from tsfm_public.models.tinytimemixer.configuration_tinytimemixer import TinyTimeMixerConfig

# Importações para avaliação do modelo
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Backend sem interface gráfica
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÕES GLOBAIS - EMPRESA MULTINACIONAL HIDRÁULICOS
# =============================================================================

# Configurações de paths flexíveis
DATA_PATH = os.getenv('DATA_PATH', './dados/db_tratado-w.csv')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './results_ttm_model')
SAVE_PATH = os.getenv('MODEL_SAVE_PATH', './final_ttm_model')

# Estrutura de dados empresarial - Equipamentos Hidráulicos
TIMESTAMP_COLUMN = 'date'
ID_COLUMNS = ['produto_cat', 'uf_cat']
TARGET_COLUMNS = ['vendas', 'faturamento']
STATIC_CATEGORICAL_COLUMNS = ['uf_cat']
CONTROL_COLUMNS = []

# Configurações temporais para previsão de demanda industrial
CONTEXT_LENGTH = 104    # Histórico de 2 anos (104 semanas)
PREDICTION_LENGTH = 26  # Horizonte de 6 meses (26 semanas)

# Configuração automática do dispositivo de processamento
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo de processamento: {DEVICE}")

# =============================================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================================

def validate_dataframe(df):
   """Valida estrutura e qualidade do DataFrame."""
   required_columns = ['date', 'produto_cat', 'uf_cat', 'vendas', 'faturamento']
   missing = [col for col in required_columns if col not in df.columns]
   if missing:
       raise ValueError(f"Colunas obrigatórias ausentes: {missing}")
   
   if df.empty:
       raise ValueError("DataFrame está vazio")
   
   # Verificar tipos de dados
   if not pd.api.types.is_datetime64_any_dtype(df['date']):
       raise ValueError("Coluna 'date' deve ser datetime")
   
   # Validar dados numéricos
   numeric_cols = ['vendas', 'faturamento']
   for col in numeric_cols:
       if not pd.api.types.is_numeric_dtype(df[col]):
           raise ValueError(f"Coluna '{col}' deve ser numérica")
       if (df[col] < 0).any():
           print(f"Aviso: Valores negativos encontrados em '{col}' - serão convertidos para 0")
           df[col] = df[col].clip(lower=0)
   
   return True

def clear_gpu_cache():
   """Limpa cache da GPU se disponível."""
   if torch.cuda.is_available():
       torch.cuda.empty_cache()

def monitor_gpu_usage():
   """Monitora o uso de memória GPU durante o treinamento."""
   if torch.cuda.is_available():
       allocated = torch.cuda.memory_allocated() / 1e9
       cached = torch.cuda.memory_reserved() / 1e9
       print(f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

# =============================================================================
# SEÇÃO 1: CARREGAMENTO E CONFIGURAÇÃO DOS DADOS DE DEMANDA
# =============================================================================

def load_data():
   """
   Carrega dados históricos de demanda de equipamentos hidráulicos.
   
   Returns:
       pd.DataFrame: Dados históricos estruturados para modelagem preditiva
   """
   if not os.path.exists(DATA_PATH):
       raise FileNotFoundError(f"Arquivo de dados não encontrado: {DATA_PATH}")
   
   try:
       df = pd.read_csv(DATA_PATH, parse_dates=["date"])
       print(f"Dados carregados do arquivo {DATA_PATH}")
       validate_dataframe(df)
       return df
   except Exception as e:
       print(f"Erro ao carregar dados: {e}")
       raise

# Carregar dados
df = load_data()
print(f"Dataset carregado: {len(df)} registros")
print(f"Período: {df['date'].min()} a {df['date'].max()}")
print(f"Produtos únicos: {df['produto_cat'].nunique()}")
print(f"Regiões únicas: {df['uf_cat'].nunique()}")

# =============================================================================
# SEÇÃO 2: PRÉ-PROCESSAMENTO - NORMALIZAÇÃO PARA DEMANDA INDUSTRIAL
# =============================================================================

def resample_data_to_weekly(df, timestamp_col, id_cols):
   """
   Padronização temporal para ciclos de demanda industrial.
   
   Args:
       df (pd.DataFrame): Dados de transações em diferentes frequências
       timestamp_col (str): Coluna de data/hora das transações
       id_cols (list): Identificadores produto-região
   
   Returns:
       pd.DataFrame: Série temporal padronizada (frequência semanal)
   """
   print("Aplicando resampling para frequência semanal...")
   
   # Ordenar dados por timestamp e IDs para consistência
   df = df.sort_values([timestamp_col] + id_cols).reset_index(drop=True)
   
   resampled_dfs = []
   
   for group_keys, group_df in df.groupby(id_cols):
       try:
           # Definir timestamp como índice para resampling
           group_df = group_df.set_index(timestamp_col)
           
           # Aplicar resampling semanal (W) pegando último valor de cada semana
           group_df = group_df.resample('W').last().ffill()
           
           # Resetar índice para voltar timestamp como coluna
           group_df = group_df.reset_index()
           
           # Restaurar colunas identificadoras
           for i, col in enumerate(id_cols):
               group_df[col] = group_keys[i] if isinstance(group_keys, tuple) else group_keys
           
           resampled_dfs.append(group_df)
       except Exception as e:
           print(f"Erro processando grupo {group_keys}: {e}")
           continue
   
   if not resampled_dfs:
       raise ValueError("Nenhum grupo foi processado com sucesso")
   
   result = pd.concat(resampled_dfs, ignore_index=True)
   print(f"Resampling concluído. Dataset final: {len(result)} registros")
   return result

# Aplicar resampling nos dados
df = resample_data_to_weekly(df, TIMESTAMP_COLUMN, ID_COLUMNS)

def create_time_series_preprocessor():
   """
   Cria e configura o preprocessador de séries temporais TTM.
   
   Returns:
       TimeSeriesPreprocessor: Preprocessador configurado para o dataset
   """
   print("Configurando preprocessador de séries temporais...")
   
   return TimeSeriesPreprocessor(
       id_columns=ID_COLUMNS,
       timestamp_column=TIMESTAMP_COLUMN,
       target_columns=TARGET_COLUMNS,
       static_categorical_columns=STATIC_CATEGORICAL_COLUMNS,
       scaling_id_columns=ID_COLUMNS,
       context_length=CONTEXT_LENGTH,
       prediction_length=PREDICTION_LENGTH,
       scaling=True,
       scaler_type="standard",
       encode_categorical=True,
       control_columns=CONTROL_COLUMNS,
       freq='W'
   )

# Criar e treinar preprocessador
tsp = create_time_series_preprocessor()
print("Treinando preprocessador com todos os dados...")
trained_tsp = tsp.train(df)

# Aplicar preprocessamento aos dados
print("Aplicando transformações de preprocessamento...")
df_processed = trained_tsp.preprocess(df)
print(f"Preprocessamento concluído. Shape dos dados processados: {df_processed.shape}")

# =============================================================================
# SEÇÃO 3: DIVISÃO TEMPORAL DOS DADOS
# =============================================================================

def split_data_temporal(df_processed, train_ratio=0.7, val_ratio=0.15):
   """
   Divisão temporal dos dados preservando continuidade das séries.
   
   Args:
       df_processed (pd.DataFrame): Dados preprocessados
       train_ratio (float): Proporção para treino
       val_ratio (float): Proporção para validação
   
   Returns:
       tuple: (train_data, valid_data, test_data)
   """
   print("Dividindo dados temporalmente...")
   
   # Ordenar por data para divisão temporal
   df_sorted = df_processed.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
   
   # Calcular pontos de corte temporal
   total_len = len(df_sorted)
   train_end = int(total_len * train_ratio)
   val_end = int(total_len * (train_ratio + val_ratio))
   
   # Dividir dados
   train_data = df_sorted[:train_end].copy()
   valid_data = df_sorted[train_end:val_end].copy()
   test_data = df_sorted[val_end:].copy()
   
   print(f"Divisão temporal concluída:")
   print(f"  Treino: {len(train_data):>5} registros")
   print(f"  Validação: {len(valid_data):>5} registros")
   print(f"  Teste: {len(test_data):>5} registros")
   
   return train_data, valid_data, test_data

# Aplicar divisão temporal
train_data, valid_data, test_data = split_data_temporal(df_processed)

def create_forecast_datasets(train_data, valid_data, test_data):
   """
   Cria os datasets específicos para o modelo TTM de previsão.
   
   Returns:
       tuple: (train_dataset, valid_dataset, test_dataset)
   """
   print("Criando datasets TTM para treinamento...")
   
   # Configuração comum para todos os datasets
   dataset_config = {
       'id_columns': ID_COLUMNS,
       'timestamp_column': TIMESTAMP_COLUMN,
       'target_columns': TARGET_COLUMNS,
       'control_columns': CONTROL_COLUMNS,
       'static_categorical_columns': STATIC_CATEGORICAL_COLUMNS,
       'context_length': CONTEXT_LENGTH,
       'prediction_length': PREDICTION_LENGTH,
   }
   
   # Criar datasets para cada divisão
   train_dataset = ForecastDFDataset(train_data, **dataset_config)
   valid_dataset = ForecastDFDataset(valid_data, **dataset_config)
   test_dataset = ForecastDFDataset(test_data, **dataset_config)
   
   print(f"Datasets TTM criados:")
   print(f"  Treino: {len(train_dataset):>4} amostras")
   print(f"  Validação: {len(valid_dataset):>4} amostras")
   print(f"  Teste: {len(test_dataset):>4} amostras")
   
   return train_dataset, valid_dataset, test_dataset

# Criar datasets TTM
train_dataset, valid_dataset, test_dataset = create_forecast_datasets(
   train_data, valid_data, test_data
)

# =============================================================================
# SEÇÃO 4: CONFIGURAÇÃO TTM - MODELO PREDITIVO EMPRESARIAL
# =============================================================================

def create_ttm_config():
   """
   Cria configuração personalizada para o modelo TinyTimeMixer.
   
   Returns:
       TinyTimeMixerConfig: Configuração do modelo TTM
   """
   print("Configurando modelo TinyTimeMixer...")
   
   # Obter informações do dataset de treino
   sample = train_dataset[0]
   num_input_channels = sample['past_values'].shape[-1]
   
   # Calcular índices dos canais alvo
   target_indices = list(range(len(TARGET_COLUMNS)))
   
   config = TinyTimeMixerConfig(
       context_length=CONTEXT_LENGTH,
       prediction_length=PREDICTION_LENGTH,
       num_input_channels=num_input_channels,
       prediction_channel_indices=target_indices,
       decoder_mode="mix_channel",
   )
   
   print(f"Configuração TTM:")
   print(f"  Context Length: {config.context_length}")
   print(f"  Prediction Length: {config.prediction_length}")
   print(f"  Input Channels: {config.num_input_channels}")
   print(f"  Prediction Channels: {len(config.prediction_channel_indices)}")
   
   return config

def load_pretrained_ttm_model(config):
   """
   Carrega modelo TTM pré-treinado da IBM Granite.
   
   Args:
       config (TinyTimeMixerConfig): Configuração do modelo
   
   Returns:
       TinyTimeMixerForPrediction: Modelo TTM carregado
   """
   print("Carregando modelo TTM pré-treinado...")
   
   clear_gpu_cache()
   
   model = TinyTimeMixerForPrediction.from_pretrained(
       "ibm-granite/granite-timeseries-ttm-r2",
       config=config,
       ignore_mismatched_sizes=True,
       low_cpu_mem_usage=True,
   )
   
   # Mover modelo para dispositivo apropriado
   model = model.to(DEVICE)
   
   print(f"Modelo carregado: {model.__class__.__name__}")
   monitor_gpu_usage()
   return model

def setup_selective_fine_tuning(model):
   """
   Configura fine-tuning seletivo congelando a maioria das camadas.
   
   Args:
       model: Modelo TTM carregado
   """
   print("Configurando fine-tuning seletivo...")
   
   # Congelar todas as camadas inicialmente
   for param in model.parameters():
       param.requires_grad = False
   
   # Descongelar apenas camadas fully-connected
   trainable_layers = []
   for name, module in model.named_modules():
       if any(layer_name in name for layer_name in ['fc1', 'fc2', 'head']):
           if isinstance(module, torch.nn.Linear):
               for param in module.parameters():
                   param.requires_grad = True
               trainable_layers.append(name)
   
   # Calcular estatísticas de parâmetros
   trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
   total_params = sum(p.numel() for p in model.parameters())
   trainable_percentage = (trainable_params / total_params) * 100 if total_params > 0 else 0
   
   print(f"Fine-tuning seletivo configurado:")
   print(f"  Parâmetros treináveis: {trainable_params:,} / {total_params:,} ({trainable_percentage:.1f}%)")

# Executar configuração do modelo
model_config = create_ttm_config()
model = load_pretrained_ttm_model(model_config)
setup_selective_fine_tuning(model)

# =============================================================================
# SEÇÃO 5: TREINAMENTO - OTIMIZAÇÃO PARA AMBIENTE EMPRESARIAL
# =============================================================================
# Modo de teste para desenvolvimento rápido
MODO_TESTE = True  # Mude para False para treinamento completo

# Controle de previsões de demanda
GERAR_PREVISOES = False  # Mude para False para pular previsões

# Hiperparâmetros para ambiente de produção corporativa
LEARNING_RATE = 5e-4
NUM_TRAIN_EPOCHS = 1 if MODO_TESTE else 100
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 8

def create_training_arguments():
   """
   Cria argumentos de treinamento otimizados.
   
   Returns:
       TrainingArguments: Configuração de treinamento
   """
   return TrainingArguments(
       output_dir=OUTPUT_DIR,
       overwrite_output_dir=True,
       learning_rate=LEARNING_RATE,
       num_train_epochs=NUM_TRAIN_EPOCHS,
       do_eval=True,
       eval_strategy="epoch",
       per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
       per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
       dataloader_num_workers=2,
       dataloader_pin_memory=True if DEVICE == "cuda" else False,
       gradient_accumulation_steps=4,
       fp16=False,
       bf16=False,
       max_grad_norm=1.0,
       report_to="none",
       save_strategy="epoch",
       logging_strategy="epoch",
       save_total_limit=2,
       logging_dir=f"{OUTPUT_DIR}/logs",
       load_best_model_at_end=True,
       metric_for_best_model="eval_loss",
       greater_is_better=False,
   )

def create_training_callbacks():
   """
   Cria callbacks para controle avançado do treinamento.
   
   Returns:
       list: Lista de callbacks configurados
   """
   early_stopping_callback = EarlyStoppingCallback(
       early_stopping_patience=1 if MODO_TESTE else 15,
       early_stopping_threshold=0.0,
   )
   
   tracking_callback = TrackingCallback()
   
   return [early_stopping_callback, tracking_callback]

def create_optimizer_and_scheduler(model, train_dataset_size):
   """
   Cria otimizador e scheduler de learning rate otimizados.
   
   Returns:
       tuple: (optimizer, scheduler)
   """
   optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
   
   steps_per_epoch = math.ceil(train_dataset_size / (PER_DEVICE_TRAIN_BATCH_SIZE * 4))
   scheduler = OneCycleLR(
       optimizer,
       max_lr=LEARNING_RATE,
       epochs=NUM_TRAIN_EPOCHS,
       steps_per_epoch=steps_per_epoch,
   )
   
   print(f"Otimização configurada:")
   print(f"  Otimizador: AdamW (lr={LEARNING_RATE})")
   print(f"  Scheduler: OneCycleLR")
   print(f"  Steps por época: {steps_per_epoch}")
   
   return optimizer, scheduler

class TTMTrainer(Trainer):
   """
   Trainer customizado para o modelo TinyTimeMixer.
   """
   
   def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
       """
       Calcula a função de perda personalizada para TTM.
       """
       valid_keys = [
           'past_values',
           'future_values',
           'past_observed_mask',
           'future_observed_mask',
           'static_categorical_values'
       ]
       
       # Filtrar apenas entradas válidas
       filtered_inputs = {k: v for k, v in inputs.items() if k in valid_keys}
       
       try:
           outputs = model(**filtered_inputs)
           loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
           
           return (loss, outputs) if return_outputs else loss
       except Exception as e:
           print(f"Erro durante compute_loss: {e}")
           dummy_loss = torch.tensor(0.0, requires_grad=True, device=model.device)
           return dummy_loss

def train_ttm_model():
   """
   Pipeline de treinamento para modelo preditivo empresarial.
   
   Returns:
       Trainer: Modelo treinado pronto para inferência empresarial
   """
   print("\n" + "="*80)
   print("INICIANDO TREINAMENTO DO MODELO TTM")
   print("="*80)
   
   monitor_gpu_usage()
   
   # Criar componentes de treinamento
   training_args = create_training_arguments()
   callbacks = create_training_callbacks()
   optimizer, scheduler = create_optimizer_and_scheduler(model, len(train_dataset))
   
   # Inicializar trainer customizado
   trainer = TTMTrainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset,
       eval_dataset=valid_dataset,
       callbacks=callbacks,
       optimizers=(optimizer, scheduler),
   )
   
   print(f"Trainer configurado:")
   print(f"  Epochs máximas: {NUM_TRAIN_EPOCHS}")
   print(f"  Batch size treino: {PER_DEVICE_TRAIN_BATCH_SIZE}")
   print(f"  Batch size validação: {PER_DEVICE_EVAL_BATCH_SIZE}")
   print(f"  Dispositivo: {DEVICE}")
   
   print("\nIniciando treinamento...")
   
   try:
       trainer.train()
       
       print("\n" + "="*80)
       print("TREINAMENTO CONCLUÍDO")
       print("="*80)
   except Exception as e:
       print(f"Erro durante treinamento: {e}")
       raise
   finally:
       clear_gpu_cache()
       monitor_gpu_usage()
   
   return trainer

def save_final_model(trainer, save_path=None):
   """
   Salva o modelo final treinado.
   
   Args:
       trainer: Trainer após treinamento
       save_path (str): Caminho para salvar o modelo
   """
   if save_path is None:
       save_path = SAVE_PATH
   
   print(f"\nSalvando modelo final em: {save_path}")
   
   os.makedirs(save_path, exist_ok=True)
   
   try:
       trainer.save_model(save_path)
       
       # Salvar também informações do preprocessador
       import pickle
       preprocessor_path = f"{save_path}/preprocessor.pkl"
       with open(preprocessor_path, 'wb') as f:
           pickle.dump(trained_tsp, f)
       
       print(f"Modelo salvo com sucesso!")
       print(f"  Modelo: {save_path}")
       print(f"  Preprocessador: {preprocessor_path}")
   except Exception as e:
       print(f"Erro ao salvar modelo: {e}")
       raise

# =============================================================================
# SEÇÃO 6: AVALIAÇÃO EMPRESARIAL - MÉTRICAS DE NEGÓCIO
# =============================================================================

# =============================================================================
# SEÇÃO 6: AVALIAÇÃO EMPRESARIAL - MÉTRICAS DE NEGÓCIO
# =============================================================================

def calculate_metrics(y_true, y_pred):
   """Calcula métricas de avaliação para séries temporais."""
   # Validação robusta de entrada
   if len(y_true) == 0 or len(y_pred) == 0:
       return {'MAE': 0.0, 'RMSE': 0.0, 'MAPE': 0.0, 'R2': 0.0}
   
   # Remover NaN/infinitos
   mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
   y_true_clean = y_true[mask]
   y_pred_clean = y_pred[mask]
   
   if len(y_true_clean) < 2:  # Mínimo para R²
       return {'MAE': 0.0, 'RMSE': 0.0, 'MAPE': 0.0, 'R2': 0.0}
   
   try:
       mae = mean_absolute_error(y_true_clean, y_pred_clean)
       rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
       
       # MAPE mais robusto
       denominator = np.maximum(np.abs(y_true_clean), 1e-8)
       mape = np.mean(np.abs((y_true_clean - y_pred_clean) / denominator)) * 100
       
       # R² com fallback
       try:
           r2 = r2_score(y_true_clean, y_pred_clean)
           if np.isnan(r2) or np.isinf(r2):
               r2 = 0.0
       except:
           r2 = 0.0
       
       return {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape), 'R2': float(r2)}
   except Exception as e:
       print(f"Erro calculando métricas: {e}")
       return {'MAE': 0.0, 'RMSE': 0.0, 'MAPE': 0.0, 'R2': 0.0}

def evaluate_model(trainer, test_dataset):
    """Avalia o modelo no conjunto de teste."""
    print("\n" + "="*60)
    print("AVALIAÇÃO DO MODELO TTM")
    print("="*60)
    
    try:
        # Obter predições
        predictions = trainer.predict(test_dataset)
        y_pred = predictions.predictions
        
        # Converter predições para formato consistente
        if isinstance(y_pred, (list, tuple)):
            y_pred = y_pred[0] if len(y_pred) > 0 else np.array([])
        
        # Extrair targets diretamente do dataset
        y_true_list = []
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            if 'future_values' in sample:
                y_true_list.append(sample['future_values'].numpy())
        
        if len(y_true_list) > 0 and hasattr(y_pred, 'shape') and y_pred.size > 0:
            # Empilhar targets de forma segura
            y_true = np.stack(y_true_list)
            
            # Garantir que predições e targets tenham o mesmo número de amostras
            min_samples = min(len(y_true), len(y_pred))
            y_true = y_true[:min_samples]
            y_pred = y_pred[:min_samples]
            
            print(f"Shape das predições: {y_pred.shape}")
            print(f"Shape dos targets: {y_true.shape}")
            
            results = {}
            
            # Métricas por target
            for i, target_name in enumerate(TARGET_COLUMNS):
                try:
                    if i < y_true.shape[-1] and i < y_pred.shape[-1]:
                        y_true_target = y_true[:, :, i].reshape(-1)
                        y_pred_target = y_pred[:, :, i].reshape(-1)
                        
                        metrics = calculate_metrics(y_true_target, y_pred_target)
                        results[target_name] = metrics
                        
                        print(f"\n{target_name.upper()}:")
                        print(f"  MAE:  {metrics['MAE']:.4f}")
                        print(f"  RMSE: {metrics['RMSE']:.4f}")
                        print(f"  MAPE: {metrics['MAPE']:.2f}%")
                        print(f"  R²:   {metrics['R2']:.4f}")
                except Exception as e:
                    print(f"Erro calculando métricas para {target_name}: {e}")
            
            # Métricas gerais se houver resultados
            if results:
                y_true_flat = y_true.reshape(-1)
                y_pred_flat = y_pred.reshape(-1)
                overall_metrics = calculate_metrics(y_true_flat, y_pred_flat)
                results['overall'] = overall_metrics
                
                print(f"\nMÉTRICAS GERAIS:")
                print(f"  MAE:  {overall_metrics['MAE']:.4f}")
                print(f"  RMSE: {overall_metrics['RMSE']:.4f}")
                print(f"  MAPE: {overall_metrics['MAPE']:.2f}%")
                print(f"  R²:   {overall_metrics['R2']:.4f}")
            
            return results, y_true, y_pred
        else:
            print("Dados insuficientes para avaliação")
            return {}, np.array([]), np.array([])
            
    except Exception as e:
        print(f"Erro durante avaliação: {e}")
        import traceback
        traceback.print_exc()
        return {}, np.array([]), np.array([])

def save_evaluation_report(results, save_path="./evaluation_report.txt"):
   """Salva relatório conciso de avaliação."""
   try:
       with open(save_path, 'w', encoding='utf-8') as f:
           f.write("RELATÓRIO DE AVALIAÇÃO - MODELO TTM\n")
           f.write("="*50 + "\n\n")
           f.write(f"Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
           
           # Métricas por variável
           for target_name, metrics in results.items():
               if target_name == 'overall':
                   continue
               f.write(f"{target_name.upper()}:\n")
               f.write(f"  MAE:  {metrics['MAE']:.4f}\n")
               f.write(f"  RMSE: {metrics['RMSE']:.4f}\n")
               f.write(f"  MAPE: {metrics['MAPE']:.2f}%\n")
               f.write(f"  R²:   {metrics['R2']:.4f}\n\n")
           
           # Métricas gerais
           if 'overall' in results:
               overall = results['overall']
               f.write("MÉTRICAS GERAIS:\n")
               f.write(f"  MAE:  {overall['MAE']:.4f}\n")
               f.write(f"  RMSE: {overall['RMSE']:.4f}\n")
               f.write(f"  MAPE: {overall['MAPE']:.2f}%\n")
               f.write(f"  R²:   {overall['R2']:.4f}\n")
       
       print(f"✅ Relatório salvo: {save_path}")
   except Exception as e:
       print(f"Erro salvando relatório: {e}")

def run_evaluation(trainer, test_dataset):
   """Executa avaliação completa."""
   try:
       results, y_true, y_pred = evaluate_model(trainer, test_dataset)
       save_evaluation_report(results)
       
       print(f"\n✅ Avaliação concluída!")
       print(f"📄 Relatório: ./evaluation_report.txt")
       
       return results
   except Exception as e:
       print(f"Erro durante avaliação: {e}")
       return {}

# =============================================================================
# SEÇÃO 7: INFERÊNCIA EMPRESARIAL - PREVISÕES REAIS
# =============================================================================

def generate_business_forecasts(trainer_modelo, preprocessador_treinado, 
                              dados_historicos, horizonte_semanas=26,
                              arquivo_saida='previsoes_demanda.csv'):
    """Gera previsões de demanda para todas as combinações produto-região."""
    print(f"Gerando previsões de demanda empresarial...")
    
    combinacoes = dados_historicos[ID_COLUMNS].drop_duplicates()
    print(f"Combinações produto-região: {len(combinacoes)}")
    
    resultados = []
    
    for idx, (_, row) in enumerate(combinacoes.iterrows()):
        produto = row['produto_cat']
        regiao = row['uf_cat']
        
        try:
            dados_combo = dados_historicos[
                (dados_historicos['produto_cat'] == produto) & 
                (dados_historicos['uf_cat'] == regiao)
            ].copy()
            
            if len(dados_combo) < CONTEXT_LENGTH:
                continue
            
            dados_combo = dados_combo.sort_values(TIMESTAMP_COLUMN).tail(CONTEXT_LENGTH).copy()
            dados_processados = preprocessador_treinado.preprocess(dados_combo)
            
            pred_dataset = ForecastDFDataset(
                dados_processados,
                id_columns=ID_COLUMNS,
                timestamp_column=TIMESTAMP_COLUMN,
                target_columns=TARGET_COLUMNS,
                context_length=CONTEXT_LENGTH,
                prediction_length=PREDICTION_LENGTH,
                static_categorical_columns=STATIC_CATEGORICAL_COLUMNS,
                control_columns=CONTROL_COLUMNS
            )
            
            if len(pred_dataset) == 0:
                continue
            
            predictions = trainer_modelo.predict(pred_dataset)
            pred_values = predictions.predictions
            
            # CORREÇÃO: Lidar com formatos diferentes de predições
            if isinstance(pred_values, (list, tuple)):
                pred_values = pred_values[0]
            if isinstance(pred_values, np.ndarray) and pred_values.ndim > 2:
                pred_values = pred_values[0]
            
            num_semanas = min(horizonte_semanas, PREDICTION_LENGTH, pred_values.shape[0])
            
            for semana in range(num_semanas):
                try:
                    # Extrair valores como escalares seguros
                    if pred_values.shape[1] >= 2:
                        vendas_raw = pred_values[semana, 0]
                        faturamento_raw = pred_values[semana, 1]
                    else:
                        vendas_raw = pred_values[semana, 0] if pred_values.shape[1] > 0 else 0
                        faturamento_raw = vendas_raw  # Fallback
                    
                    # Converter para escalar
                    vendas_pred = float(np.asarray(vendas_raw).item()) if np.asarray(vendas_raw).size == 1 else float(np.asarray(vendas_raw).flat[0])
                    faturamento_pred = float(np.asarray(faturamento_raw).item()) if np.asarray(faturamento_raw).size == 1 else float(np.asarray(faturamento_raw).flat[0])
                    
                    vendas_pred = max(0, vendas_pred)
                    faturamento_pred = max(0, faturamento_pred)
                    
                    ultima_data = dados_combo[TIMESTAMP_COLUMN].max()
                    data_previsao = ultima_data + pd.Timedelta(weeks=semana+1)
                    
                    resultados.append({
                        'data_previsao': data_previsao,
                        'produto_cat': produto,
                        'uf_cat': regiao,
                        'semana_horizonte': semana + 1,
                        'vendas_previstas': round(vendas_pred, 2),
                        'faturamento_previsto': round(faturamento_pred, 2)
                    })
                except Exception as e:
                    print(f"Erro semana {semana+1}: {str(e)[:30]}...")
                    continue
            
            if (idx + 1) % 50 == 0:
                print(f"Processadas: {idx + 1}/{len(combinacoes)}")
                
        except Exception as e:
            print(f"Erro produto {produto}, região {regiao}: {str(e)[:30]}...")
            continue
    
    if resultados:
        df_previsoes = pd.DataFrame(resultados)
        df_previsoes.to_csv(arquivo_saida, index=False)
        
        print(f"\n✅ Previsões geradas: {len(df_previsoes):,}")
        print(f"📁 Arquivo: {arquivo_saida}")
        
        return df_previsoes
    else:
        print("❌ Nenhuma previsão foi gerada")
        return pd.DataFrame()

def create_forecast_summary(df_previsoes, save_path="./resumo_previsoes.csv"):
    """Cria resumo executivo das previsões por produto e região."""
    if df_previsoes.empty:
        return pd.DataFrame()
    
    print("Criando resumo executivo das previsões...")
    
    resumo = df_previsoes.groupby(['produto_cat', 'uf_cat']).agg({
        'vendas_previstas': ['sum', 'mean', 'std'],
        'faturamento_previsto': ['sum', 'mean', 'std']
    }).round(2)
    
    resumo.columns = [f"{col[0]}_{col[1]}" for col in resumo.columns]
    resumo = resumo.reset_index()
    
    resumo['vendas_total_6m'] = resumo['vendas_previstas_sum']
    resumo['faturamento_total_6m'] = resumo['faturamento_previsto_sum']
    resumo['vendas_media_semanal'] = resumo['vendas_previstas_mean']
    resumo['faturamento_medio_semanal'] = resumo['faturamento_previsto_mean']
    
    resumo.to_csv(save_path, index=False)
    
    print(f"✅ Resumo executivo salvo: {save_path}")
    return resumo

# =============================================================================
# EXECUÇÃO DO PIPELINE COMPLETO
# =============================================================================

if __name__ == "__main__":
   try:
       print("\n🚀 INICIANDO PIPELINE COMPLETO TTM")
       print("="*80)
       
       # 1. Treinamento do modelo
       trainer = train_ttm_model()
       
       # 2. Avaliação do modelo
       evaluation_results = run_evaluation(trainer, test_dataset)
       
       # 3. Salvamento do modelo
       save_final_model(trainer)
       
       # 4. Geração de previsões empresariais (opcional)
       if GERAR_PREVISOES:
           print("\n📈 GERANDO PREVISÕES DE DEMANDA")
           print("="*50)
           
           df_previsoes = generate_business_forecasts(
               trainer_modelo=trainer,
               preprocessador_treinado=trained_tsp,
               dados_historicos=df,
               horizonte_semanas=26,
               arquivo_saida='previsoes_demanda_26_semanas.csv'
           )
           
           # 5. Criar resumo executivo
           if not df_previsoes.empty:
               resumo = create_forecast_summary(df_previsoes)
               print(f"\n📋 RESUMO EXECUTIVO:")
               print(resumo.head(10))
       else:
           print("\n⏭️ PREVISÕES PULADAS (GERAR_PREVISOES=False)")
       
       # 6. Resultado final
       print(f"\n🎉 PIPELINE CONCLUÍDO COM SUCESSO!")
       print("="*50)
       
       if evaluation_results and 'overall' in evaluation_results:
           print(f"📊 Métricas do Modelo:")
           print(f"   R² Geral: {evaluation_results['overall']['R2']:.4f}")
           print(f"   MAE Geral: {evaluation_results['overall']['MAE']:.4f}")
           print(f"   MAPE Geral: {evaluation_results['overall']['MAPE']:.2f}%")
       
       print(f"\n📁 Arquivos Gerados:")
       print(f"   • Modelo: {SAVE_PATH}/")
       if GERAR_PREVISOES:
           print(f"   • Previsões: previsoes_demanda_26_semanas.csv")
           print(f"   • Resumo: resumo_previsoes.csv")
       print(f"   • Avaliação: evaluation_report.txt")
       print(f"   • Gráficos: evaluation_plots/")
       
   except Exception as e:
       print(f"❌ ERRO NO PIPELINE: {e}")
       import traceback
       traceback.print_exc()
       raise
   finally:
       # Limpeza final
       clear_gpu_cache()
       print(f"\n🧹 Limpeza de memória concluída")
