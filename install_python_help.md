# Linux Debian
```bash
sudo apt update

# Instalar dependências 
sudo apt install -y build-essential libssl-dev zlib1g-dev \ 
libbz2-dev libreadline-dev libsqlite3-dev curl \ 
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev 

# Instalar pyenv 
curl https://pyenv.run | bash

# Depois, adicione ao seu ~/.bashrc:
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Recarregue o shell:
source ~/.bashrc

# E instale Python 3.11:
pyenv install 3.11.9
pyenv local 3.11.9

# Apague o ambiente virtual atual:
rm -rf .venv

# Crie um novo venv:
python3.11 -m venv .venv 
source .venv/bin/activate

python3 -m venv .venv
source .venv/bin/activate

# Instale os pacotes:
pip install -r requirements.txt
```


# iOS Mac
### Atualizar Homebrew
```bash
brew update

# Instalar dependências necessárias para compilar Python
brew install openssl readline sqlite3 xz zlib tcl-tk

# Instalar pyenv
brew install pyenv
brew install pyenv-virtualenv

# Adicione ao seu ~/.zshrc (macOS usa zsh por padrão):
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Recarregue o shell:
source ~/.zshrc

# Instale Python 3.11
pyenv install 3.11.9
pyenv local 3.11.9

# Apague o ambiente virtual atual (se existir)
rm -rf .venv

# Crie um novo ambiente virtual
python3.11 -m venv .venv
source .venv/bin/activate

# Instale os pacotes necessários
pip install -r requirements.txt
```


### O que funciona
- Homebrew substitui o apt do Debian para instalar dependências.
- pyenv e pyenv-virtualenv funcionam da mesma forma no macOS.
- O fluxo de criar e ativar ambientes virtuais (python3.11 -m venv .venv) é idêntico.
- A configuração no shell é feita em ~/.zshrc (já que o macOS usa zsh por padrão).

### ⚠️ Pontos de atenção
Dependências: no macOS, algumas bibliotecas podem estar em caminhos diferentes. O pyenv geralmente detecta isso, mas às vezes é preciso exportar variáveis como:

```bash
export LDFLAGS="-L$(brew --prefix zlib)/lib -L$(brew --prefix bzip2)/lib"
export CPPFLAGS="-I$(brew --prefix zlib)/include -I$(brew --prefix bzip2)/include"
```
- Isso ajuda na compilação de versões específicas do Python.

 -Shell: se você ainda usa bash, edite `~/.bashrc` em vez de `~/.zshrc`.

- Versões múltiplas: para alternar entre várias versões (ex.: 3.9, 3.10, 3.11), basta instalar cada uma com pyenv install e depois usar:

```bash
pyenv global 3.11.9
pyenv local 3.10.14
```
Isso define a versão padrão ou a versão específica por projeto.

### 🚀 Teste rápido
Depois de rodar o script, verifique se está funcionando com:

```bash
python --version
which python
```

Isso deve mostrar a versão gerenciada pelo pyenv e confirmar que o ambiente virtual está ativo.

## Tutorial para configurar duas versões de Python lado a lado no macOS usando pyenv. Permite alternar facilmente entre elas em diferentes projetos.

## 📌 Script para múltiplas versões de Python no macOS
```bash
# Atualizar Homebrew
brew update

# Instalar dependências necessárias
brew install openssl readline sqlite3 xz zlib tcl-tk

# Instalar pyenv e pyenv-virtualenv
brew install pyenv
brew install pyenv-virtualenv

# Configurar no ~/.zshrc (ou ~/.bashrc se usar bash)
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Recarregar o shell
source ~/.zshrc

# Instalar múltiplas versões de Python
pyenv install 3.9.18
pyenv install 3.11.9

# Definir versão global (padrão para todo o sistema)
pyenv global 3.11.9

# Definir versão local (apenas para um projeto específico)
cd ~/meu_projeto
pyenv local 3.9.18

# Criar ambiente virtual com a versão escolhida
python -m venv .venv
source .venv/bin/activate

# Instalar dependências do projeto
pip install -r requirements.txt
```


## 🔄 Alternando entre versões
- Versão global: `pyenv global 3.11.9` → usada em todos os diretórios.

- Versão local: `pyenv local 3.9.18` → usada apenas no diretório atual (cria um arquivo .python-version).

- Ambiente virtual: cada projeto pode ter seu próprio .venv com a versão de Python que você escolher.

## 🧪 Teste rápido
Depois de configurar, rode:
```bash
bash
python --version
which python
```
Isso confirma qual versão está ativa e se o ambiente virtual está funcionando.


## correçao para mac i386
- Caminho A (recomendado se você estiver em Mac Intel): 
    -  pin de numba/llvmlite compatíveis
    - Você não instalou numba diretamente, mas o librosa puxa. Então você precisa fixar as versões para evitar llvmlite 0.46.x.
    - Apague e recrie o venv (para não ficar lixo de tentativas anteriores):
```bash
deactivate 2>/dev/null
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

No seu requirements.txt, adicione estas duas linhas (coloque perto do topo, antes do librosa):
```python
numba==0.62.1
llvmlite==0.45.1
```

Instale de novo:
```bash
pip install -r requirements.txt
```

- ✅ Isso força o pip a usar versões que ainda oferecem wheel para macOS Intel e evita a compilação do llvmlite.

> Observação: no seu print aparece numba 0.63.1 sendo instalado e o wheel dele é macosx_15_0_x86_64, mas o llvmlite está vindo como tar.gz (compilação). Esse combo é a raiz do problema.



## Execucao do pipeline de otimizacao do valor de q
``` bash
# 1) criar manifest
python -m src.pipeline.build_manifest

# 2) cache de p_amp e p_f0
python -m src.pipeline.compute_p_cache

# 3) nested CV para estimar q global supervisionado
python -m src.pipeline.nested_cv_q_global

# 4) extrair features finais com q_global_final
python -m src.pipeline.extract_features_with_q
```
