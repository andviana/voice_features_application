import pandas as pd

# Caminho para o arquivo de metadados
MANIFEST_PATH = 'manifest.csv'

def get_sex_from_manifest(recording_id):
    """Busca o sexo do paciente no arquivo manifest.csv."""
    try:
        df = pd.read_csv(MANIFEST_PATH)
        row = df[df['recording_id'] == recording_id]
        if not row.empty:
            return row.iloc[0]['sex']
    except Exception as e:
        print(f"Erro ao ler manifest: {e}")
    return 'Unknown'
