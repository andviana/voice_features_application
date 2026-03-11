/**
 * AudioVisualizer: Utilitários unificados para visualização de sinais.
 * Utiliza as chaves 'x', 'y' e 'z' retornadas pelo AudioSignalsService.
 */

const AudioVisualizer = {
    defaultLayout: {
        paper_bgcolor: "white",
        plot_bgcolor: "white",
        margin: { l: 60, r: 20, t: 30, b: 45 },
        font: { size: 11 }
    },

    purge: function(divId) {
        const el = document.getElementById(divId);
        if (el) Plotly.purge(el);
    },

    /**
     * Gráfico de Linha Genérico (Waveform, Spectrum, PSD, Autocorr)
     */
    plotLine: function(divId, data, titleX, titleY, options = {}) {
        const el = document.getElementById(divId);
        if (!el || !data || !data.x || !data.y) return;
        this.purge(divId);

        const trace = {
            x: data.x,
            y: data.y,
            type: "scatter",
            mode: "lines",
            name: options.name || "",
            line: { 
                color: options.color || "#2563eb", 
                width: 1.5, 
                dash: options.dash || "solid" 
            }
        };

        const layout = {
            ...this.defaultLayout,
            height: options.height || 300,
            xaxis: { 
                title: titleX, 
                range: options.xRange,
                autorange: options.xRange ? false : true
            },
            yaxis: { 
                title: titleY, 
                type: options.logY ? "log" : "linear",
                range: options.yRange,
                autorange: options.yRange ? false : true 
            },
            showlegend: !!options.showLegend
        };

        Plotly.newPlot(divId, [trace], layout, { responsive: true, displaylogo: false });
    },

    /**
     * Gráfico específico para Resposta Teórica do Filtro (Butterworth)
     */
    plotButterResponse: function(divId, data) {
        const el = document.getElementById(divId);
        if (!el || !data) return;
        this.purge(divId);

        const traces = [
            {
                x: data.x,
                y: data.gain_db,
                name: "Ganho do filtro",
                line: { color: "#1d4ed8", width: 2 }
            },
            {
                x: [data.cutoff_hz, data.cutoff_hz],
                y: [-80, 10],
                name: "Frequência de corte",
                line: { color: "#dc2626", dash: "dash", width: 2 }
            }
        ].map(t => ({ ...t, type: "scatter", mode: "lines" }));

        const layout = {
            ...this.defaultLayout,
            height: 420,
            xaxis: { title: "Frequência (Hz)", range: [0, 1000] },
            yaxis: { title: "Ganho (dB)", range: [-80, 10] },
            shapes: [{
                type: "rect", x0: 0, x1: data.cutoff_hz, y0: -80, y1: 10,
                fillcolor: "rgba(239,68,68,0.12)", line: { width: 0 }
            }]
        };

        Plotly.newPlot(divId, traces, layout, { responsive: true, displaylogo: false });
    },

    /**
     * Espectrograma (Heatmap)
     */
    plotHeatmap: function(divId, data, titleX, titleY, options = {}) {
        const el = document.getElementById(divId);
        if (!el || !data || !data.z) return;
        this.purge(divId);
  
        const layout = {
            ...this.defaultLayout,
            height: options.height || 400,
            xaxis: { title: titleX, range: options.xRange },
            yaxis: { title: titleY, range: options.yRange }
        };
        
        const trace = {
            x: data.x, y: data.y, z: data.z,
            type: "heatmap",
            colorscale: "Viridis",
            zmin: options.zRange ? options.zRange[0] : undefined,
            zmax: options.zRange ? options.zRange[1] : undefined,
            colorbar: { title: "dB" }
        };

        Plotly.newPlot(divId, [trace], layout, { responsive: true, displaylogo: false });
    },

    /**
     * Histograma de Amplitude
     */
    plotHistogram: function(divId, data, titleX, titleY, options = {}) {
        const el = document.getElementById(divId);
        if (!el || !data) return;
        this.purge(divId);

        const trace = {
            x: data.x, y: data.y,
            type: "bar",
            marker: { color: options.color || "#2563eb" }
        };

        const layout = {
            ...this.defaultLayout,
            height: options.height || 300,
            xaxis: { title: titleX, range: options.xRange },
            yaxis: { title: titleY, range: options.yRange }
        };

        Plotly.newPlot(divId, [trace], layout, { responsive: true, displaylogo: false });
    },

    /**
     * Gráfico Comparativo de Efeito de Filtro (Específico Scientific)
     */
    plotFilterEffect: function(divId, rawPsd, procPsd, cutoffHz) {
        const el = document.getElementById(divId);
        if (!el) return;
        this.purge(divId);

        const ymax = Math.max(...(rawPsd?.y || []), ...(procPsd?.y || []));

        const traces = [
            { x: rawPsd.x, y: rawPsd.y, name: "RAW", line: { color: "#2563eb" } },
            { x: procPsd.x, y: procPsd.y, name: "PROCESSED", line: { color: "#16a34a" } },
            { x: [cutoffHz, cutoffHz], y: [1e-12, ymax], name: `Corte ${cutoffHz}Hz`, line: { color: "#dc2626", dash: "dash" } }
        ].map(t => ({ ...t, type: "scatter", mode: "lines" }));

        const layout = {
            ...this.defaultLayout,
            height: 420,
            xaxis: { title: "Frequência (Hz)", range: [0, 500] },
            yaxis: { title: "PSD", type: "log" },
            shapes: [{
                type: "rect", x0: 0, x1: cutoffHz, y0: 0, y1: 1, xref: "x", yref: "paper",
                fillcolor: "rgba(239,68,68,0.1)", line: { width: 0 }
            }]
        };
        Plotly.newPlot(divId, traces, layout, { responsive: true, displaylogo: false });
    }






};