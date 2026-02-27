from __future__ import annotations

from pathlib import Path
from typing import Callable

from app.globals import params_singleton
from app.services.pipeline_service import pipeline_manager  # o manager SSE que você já tem


def start_preprocess_run(
    input_wav_path: Path,
    output_wav_path: Path,
    manifest_path: Path,
) -> str:
    """
    Inicia uma execução do pré-processamento em background e retorna run_id.
    Captura tudo que for print() via redirect_stdout (já implementado no PipelineManager).
    """

    def job():
        # Importa aqui dentro pra garantir que ocorre dentro da thread (prints capturados)
        from pre_proccess import pre_proccess_pipeline, filters

        p = params_singleton.get()

        # 1) Atualiza constante do manifest (sem mexer no processamento)
        pre_proccess_pipeline.MANIFEST_PATH = str(manifest_path)

        # 2) Atualiza limites do filtro por sexo em runtime
        #    Mantém a mesma lógica: M usa faixa masculina, F usa faixa feminina, fallback banda padrão.
        original_apply_shen = filters.apply_shen_filter

        def patched_apply_shen_filter(y, sr, sex):
            if str(sex).upper() == "M":
                low, high = p.f_low_man, p.f_high_man
            elif str(sex).upper() == "F":
                low, high = p.f_low_woman, p.f_high_woman
            else:
                low, high = 80, 10000  # mantém fallback como no código original:contentReference[oaicite:4]{index=4}
            return filters.apply_bandpass(y, sr, low, high)

        filters.apply_shen_filter = patched_apply_shen_filter

        try:
            print("== Pré-processamento iniciado ==")
            print(f"Input : {input_wav_path}")
            print(f"Output: {output_wav_path}")
            print(f"Manifest: {manifest_path}")
            print(f"Params: SR={p.SR} duration={p.duration} target_db={p.target_db}")
            print(
                f"F0 woman=[{p.f_low_woman},{p.f_high_woman}] Hz | "
                f"man=[{p.f_low_man},{p.f_high_man}] Hz"
            )

            output_wav_path.parent.mkdir(parents=True, exist_ok=True)

            info = pre_proccess_pipeline.executar_pipeline(
                input_path=str(input_wav_path),
                output_path=str(output_wav_path),
                target_sr=p.SR,
                duration=p.duration,
            )

            print("== Info retornada pelo pipeline ==")
            for k, v in info.items():
                print(f"{k}: {v}")

            print("== Pré-processamento finalizado ==")

        finally:
            # restaura para não “vazar” patch para outras execuções
            filters.apply_shen_filter = original_apply_shen

    return pipeline_manager.start(job)