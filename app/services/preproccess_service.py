from __future__ import annotations

from pathlib import Path
from typing import Iterable

from app.globals import params_singleton
from app.services.pipeline_service import pipeline_manager  


def _run_core(
        tasks: list[tuple[Path, Path]], 
        is_batch: bool = False
        ):
    """    
    Realiza o processamento, logs e tratamento de erros.
    """
    from pre_proccess import pre_proccess_pipeline
    p = params_singleton.get()
    
    label = "LOTE" if is_batch else "UNITÁRIO"
    
    try:
        print(f"== Pré-processamento {label} iniciado ==")
        print(f"Total de arquivos: {len(tasks)}")
        print(f"Config: SR={int(p.SR)}Hz | Duração={p.duration}s | Alvo={p.target_db}dB")
        print("Filtro: Butterworth 4ª Ordem (80-8000 Hz) [Item 4.1.3]")

        for idx, (input_wav, output_wav) in enumerate(tasks, start=1):
            if is_batch:
                print(f"\n--- Processando [{idx}/{len(tasks)}] ---")
            
            print(f"Arquivo: {input_wav.name}")
            output_wav.parent.mkdir(parents=True, exist_ok=True)

            # Chamada centralizada ao pipeline
            info = pre_proccess_pipeline.executar_pipeline(
                input_path=str(input_wav),
                output_path=str(output_wav),
                target_sr=int(p.SR),
                duration=p.duration,
            )

            print("Métricas de Validação:")
            for k, v in info.items():
                print(f"  > {k}: {v}")

        print(f"\n== Pré-processamento {label} finalizado com sucesso ==")

    except Exception as e:
        print(f"\n[ERRO] Falha no processamento {label}: {str(e)}")
        raise e


def start_preprocess_batch_run(
    tasks: Iterable[tuple[Path, Path]],    
) -> str:
    """Inicia processamento em lote para múltiplos arquivos/pastas."""
    
    def job():        
        _run_core(list(tasks), is_batch=True)

    return pipeline_manager.start(job)



def start_preprocess_run(
    input_wav_path: Path,
    output_wav_path: Path,
) -> str:
    """Inicia processamento unitário para um único arquivo."""

    def job():
        _run_core([(input_wav_path, output_wav_path)], is_batch=False)

    return pipeline_manager.start(job)