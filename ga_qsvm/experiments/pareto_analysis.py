import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_pareto_front(points):
    """
    Calcula el frente de Pareto exacto para una lista de puntos (accuracy, depth).
    Maximiza Accuracy, Minimiza Depth.
    """
    # Ordenar por Accuracy (descendente) y luego Depth (ascendente)
    sorted_points = sorted(points, key=lambda x: (x[0], -x[1]), reverse=True)
    pareto_front = []
    min_depth = float('inf')
    
    for p in sorted_points:
        if p[1] < min_depth:
            pareto_front.append(p)
            min_depth = p[1]
            
    return pareto_front

def plot_pareto_evolution(history, output_dir):
    """
    Genera un panel 2x2 mostrando la evolución del frente de Pareto
    en 4 momentos clave (inicio, medio-temprano, medio-tardío, final).
    """
    total_gens = len(history)
    # Seleccionar 4 generaciones representativas
    target_gens = [
        min(10, total_gens) - 1, 
        total_gens // 3 - 1, 
        (total_gens * 2) // 3 - 1, 
        total_gens - 1
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Evolución del Frente de Pareto (Accuracy vs Depth)', fontsize=16)
    
    for idx, gen_idx in enumerate(target_gens):
        if gen_idx < 0 or gen_idx >= total_gens:
            continue
            
        ax = axes[idx // 2, idx % 2]
        points = history[gen_idx]
        
        # Extraer puntos individuales
        accuracies = [p[0] for p in points]
        depths = [p[1] for p in points]
        
        # Calcular y extraer Frente de Pareto
        pareto = get_pareto_front(points)
        pareto_acc = [p[0] for p in pareto]
        pareto_depth = [p[1] for p in pareto]
        
        # Graficar toda la población (gris)
        ax.scatter(accuracies, depths, color='lightgray', label='Población', alpha=0.7)
        # Graficar Frente de Pareto (azul sólido) y conectar con línea
        ax.plot(pareto_acc, pareto_depth, marker='o', color='royalblue', label='Frente de Pareto', linewidth=2)
        
        ax.set_title(f'Generación {gen_idx + 1}')
        ax.set_xlabel('Accuracy (Maximizar)')
        ax.set_ylabel('Profundidad del Circuito (Minimizar)')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'pareto_evolution.pdf'), dpi=300)
    print(f"Gráfica guardada: {os.path.join(output_dir, 'pareto_evolution.pdf')}")

def plot_tradeoff_metrics(history, output_dir):
    """
    Genera un gráfico de doble eje Y mostrando cómo el Accuracy sube
    mientras la Profundidad promedio baja a lo largo de las generaciones.
    """
    generations = list(range(1, len(history) + 1))
    
    max_acc = [np.max([p[0] for p in pop]) for pop in history]
    avg_acc = [np.mean([p[0] for p in pop]) for pop in history]
    avg_depth = [np.mean([p[1] for p in pop]) for pop in history]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Generaciones')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(generations, max_acc, color='royalblue', label='Max Accuracy', linewidth=2)
    ax1.plot(generations, avg_acc, color='lightblue', label='Avg Accuracy', linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Eje secundario para la profundidad
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Profundidad Promedio', color=color)
    ax2.plot(generations, avg_depth, color='crimson', label='Avg Depth', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    # Juntar leyendas de ambos ejes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    plt.title('Trade-off Multiobjetivo: Desempeño vs Costo NISQ')
    plt.savefig(os.path.join(output_dir, 'pareto_tradeoff.pdf'), dpi=300)
    print(f"Gráfica guardada: {os.path.join(output_dir, 'pareto_tradeoff.pdf')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analizar Historial de Pareto del GA-QSVM")
    parser.add_argument("--dir", type=str, required=True, help="Ruta al directorio del experimento que contiene pareto_history.json")
    args = parser.parse_args()

    history_path = os.path.join(args.dir, 'pareto_history.json')
    if not os.path.exists(history_path):
        print(f"Error: No se encontró {history_path}")
        exit(1)
        
    with open(history_path, 'r') as f:
        history = json.load(f)
        
    plot_pareto_evolution(history, args.dir)
    plot_tradeoff_metrics(history, args.dir)
