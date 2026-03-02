## Migrações (SQLite com Alembic via Flask-Migrate)

> Comandos:
`python -m venv .venv`

- Linux/Mac
    - `source .venv/bin/activate`
- Windows
    - `.venv\Scripts\activate`


```bash 
pip install -r requirements.txt

export FLASK_APP=wsgi.py  # Windows (PowerShell): $env:FLASK_APP="wsgi.py"

flask db init
flask db migrate -m "create users and params tables"
flask db upgrade

flask run
```


## Rodar testes

`pytest -q`


## Rotas de acesso

> HTML: /, /params, /pipeline
> Swagger UI: /api/swagger-ui





## Checklist objetivo para rodar e testar tudo:

- Banco
- Seed (só rodar se necessário)
- Pipeline
- SSE
- Pré-processamento via browser

### ✅ 1️⃣ Criar e ativar ambiente virtual

Na raiz do projeto:
```bash
python -m venv .venv
# Linux / Mac:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\activate
```
### ✅ 2️⃣ Instalar dependências
```bash
pip install -r requirements.txt
```

Se você adicionou as libs do pipeline manualmente, confirme que tem:
`pip install librosa soundfile pandas scipy numpy`


### ✅ 3️⃣ Configurar variáveis de ambiente
```bash 
# Linux / Mac:
export FLASK_APP=wsgi.py
export FLASK_ENV=development
# Windows PowerShell:
$env:FLASK_APP="wsgi.py"
$env:FLASK_ENV="development"
```

### ✅ 4️⃣ Criar banco e tabelas (migrações)

Se ainda não rodou:
```bash 
flask db init
flask db migrate -m "initial"
flask db upgrade
```
Se já tinha migrations:
```bash
flask db upgrade
```

### ✅ 5️⃣ Rodar seed inicial
```bash
python seeds/seed.py
```
Deve aparecer algo como:

```
Iniciando seed...
Usuário admin criado com sucesso.
Params iniciais criados.
Seed finalizado.
```

### ✅ 6️⃣ Estrutura de dados necessária
Confirme que você tem:

```
data/
├── audio_raw/
│   ├── HC_AH/
│   │   ├── arquivo1.wav
│   │   └── ...
│   └── PD_AH/
│       ├── arquivo2.wav
│       └── ...
├── metadata/
│   └── manifest.csv
```

> ⚠️ O manifest.csv precisa existir em:
> `data/metadata/manifest.csv`


### ✅ 7️⃣ Rodar aplicação
```bash
flask run
```

Saída esperada:

`Running on http://127.0.0.1:5000`

### ✅ 8️⃣ Testar no navegador
- 🔹 Home
    - [](http://127.0.0.1:5000/)
- 🔹 Lista de áudios
    - [](http://127.0.0.1:5000/audio-raw)
- 🔹 Parâmetros (tabela técnica)
    - [](http://127.0.0.1:5000/params/view)
- 🔹 Swagger
    - [](http://127.0.0.1:5000/api/swagger-ui)


### ✅ 9️⃣ Testar pré-processamento (SSE)
Vá para:

- [](http://127.0.0.1:5000/audio-raw)

- Clique em Pré-processar

- Deve abrir o modal

- Os print() do pipeline devem aparecer em tempo real

- Se quiser testar via terminal:

```bash
- [](curl -X POST http://127.0.0.1:5000/audio-raw/preprocess/HC_AH/arquivo.wav)
```

### ✅ 🔎 Se der erro
- 🔹 Erro de import do pacote pre_proccess

Verifique se:

```bash
pre_proccess/
├── __init__.py
├── pre_proccess_pipeline.py
├── filters.py
...
```

O `__init__.py` é obrigatório.

- 🔹 Erro: soundfile / libsndfile

- No Windows pode precisar:

```pip install soundfile```

- No Linux:
```sudo apt install libsndfile1 ```


- 🔹 Erro SSE não atualiza

- Certifique-se que está rodando com:

```FLASK_ENV=development```

- E não usando algum proxy (nginx) sem configuração para streaming.

- 🧪 Teste rápido de sanity check

- Abra um terminal Python:

```python```

E teste:
```python
from pre_proccess import pre_proccess_pipeline
print("Import OK")
```

Se isso funcionar, o pipeline está importável.

### 🎯 Fluxo completo esperado

- App sobe

- Singleton carrega params

- Clique em pré-processar

- Thread inicia

- Prints aparecem no modal

- Arquivo processado é salvo em: `data/audio_processed/<grupo>/<arquivo>.wav`

