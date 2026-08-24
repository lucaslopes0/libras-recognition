# Scripts legados

Ferramentas pontuais que não fazem parte do pipeline numerado principal
(`01_download.py` … `10_prepare_merged.py`). Mantidas por documentarem
soluções reais para problemas que já aconteceram e podem voltar a
acontecer — mas não são parte do fluxo normal de uso.

- **`find_classes.py`** — busca fuzzy por nomes de classe no V-LIBRASIL
  quando o nome não bate exatamente com o esperado.

- **`load_weights_compat.py`** / **`load_weights_npz.py`** — importam um
  modelo treinado no Google Colab (arquitetura ligeiramente diferente da
  atual em `src/libras/models/lstm.py`: usa `recurrent_dropout` e
  `l2` regularization) para o formato local `.keras`, contornando
  incompatibilidade de versão do Keras entre os dois ambientes. Use se
  precisar reimportar um modelo treinado no Colab.

- **`prepare_for_colab.py`** — empacota um subconjunto fixo de 10 classes
  de `data/processed/` em `.tar.gz` para upload manual ao Colab. Específico
  do vocabulário de 10 palavras escolhido em uma iteração anterior do
  projeto; ajuste `SELECTED_CLASSES` antes de reusar.
