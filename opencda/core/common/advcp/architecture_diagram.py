"""
AdvCP Module Architecture Diagram (Current Implementation)
Shows the actual data flow based on CoperceptionModelManager.make_prediction()

Key Flow:
1. CoperceptionModelManager loads batch from DataLoader
2. If AdvCP enabled: runs inference, then AdvCPManager.process_tick()
3. AdvCPManager applies attack/defense and returns modified predictions
4. Original or modified predictions used for evaluation/visualization

No OPV2V dataset dependency, no separate dataset initialization in AdvCP.
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.generic.storage import Storage

graph_attr = {
    "splines": "spline",
    "nodesep": "1.5",           # большое расстояние между узлами
    "ranksep": "1.8",           # большое расстояние между уровнями
    "fontsize": "100",           # крупный шрифт для заголовков
    "fontname": "Arial Bold",   # жирный шрифт для лучшей читаемости
    "dpi": "50",               # высокое разрешение
    "rankdir": "LR",
    "pad": "1.0",               # отступы вокруг диаграммы
    "margin": "0.5"             # поля внутри
}

edge_attr = {
    "fontsize": "40",           # крупный текст на связях
    "penwidth": "3.5",          # толстые линии
    "fontname": "Arial",
    "minlen": "3"               # минимальная длина ребра
}

node_attr = {
    "fontsize": "65",           # крупный текст в узлах
    "width": "3.5",             # широкие узлы
    "height": "2.0",            # высокие узлы
    "fontname": "Arial",
    "margin": "0.3,0.2"         # отступы внутри узлов
}
edge_font_size = 60
edge_penwidth = 10
cluster_font_size = 70

with Diagram(
    "AdvCP Module Architecture (Current)",
    show=False,
    direction="LR",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
    outformat="png"              # или "svg"
):
    # ... остальной код диаграммы остается без изменений
    # ==================== OPENCDA CORE ====================
    with Cluster("OpenCDA Core", graph_attr={"fontsize": str(cluster_font_size), "fontname": "Arial Bold"}):
        coperception_mgr = Python("\n\n\n\n\n\n\n\n\n\n\n\n\nCoperceptionModelManager\nMain Perception Manager")
    # ==================== DATA SOURCE ====================
    with Cluster("Data Source", graph_attr={"fontsize": str(cluster_font_size), "fontname": "Arial Bold"}):
        dataloader = Python("\n\n\n\n\n\n\n\n\n\n\n\n\nDataLoader\n(OpenCOOD Dataset)")
        opencood_dataset = Storage("\n\n\n\n\n\n\n\n\n\n\n\n\nOpenCOOD Dataset\n(No OPV2V)")

    # ==================== MODEL & INFERENCE ====================
    with Cluster("Perception Model", graph_attr={"fontsize": str(cluster_font_size), "fontname": "Arial Bold"}):
        opencood_model = Python("\n\n\n\n\n\n\n\n\n\n\n\n\nOpenCOOD Model\n(Perception Network)")
        inference_utils = Python("\n\n\n\n\n\n\n\n\n\n\n\n\nInference Utils\n(early/intermediate/late)")

    # ==================== ADVCP MODULE ====================
    with Cluster("AdvCP Module", graph_attr={"fontsize": str(cluster_font_size), "fontname": "Arial Bold"}):
        advcp_mgr = Python("\n\n\n\n\n\n\n\n\n\n\n\n\nAdvCPManager\nprocess_tick()")

        with Cluster("Preprocessing", graph_attr={"fontsize": str(cluster_font_size), "fontname": "Arial Bold"}):
            perception = Python("\n\n\n\n\n\n\n\n\n\n\n\n\nOpencoodPerception\nearly/intermediate_preprocess()")

        with Cluster("Attack Components", graph_attr={"fontsize": str(cluster_font_size), "fontname": "Arial Bold"}):
            attacker = Python("\n\n\n\n\n\n\n\n\n\n\n\n\nAttacker\n(lidar_remove/spoof)")

        with Cluster("Defense Components", graph_attr={"fontsize": str(cluster_font_size), "fontname": "Arial Bold"}):
            defender = Python("\n\n\n\n\n\n\n\n\n\n\n\n\nDefender\n(CAD Defense)")

    # ==================== EVALUATION & VISUALIZATION ====================
    with Cluster("Output & Evaluation", graph_attr={"fontsize": str(cluster_font_size), "fontname": "Arial Bold"}):
        eval_utils = Python("\n\n\n\n\n\n\n\n\n\n\n\n\nEval Utils\n(TP/FP calculation)")
        simple_vis = Python("\n\n\n\n\n\n\n\n\n\n\n\n\nVisualizer\n(3D/BEV)")

    # ==================== DATA FLOW ====================

    # 1. Dataset and DataLoader setup (one-time)
    opencood_dataset >> Edge(label="build_dataset()", style="dashed", color="gray", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> dataloader

    # 2. Main prediction loop - load batch
    coperception_mgr >> Edge(label="batch_data", color="blue", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> dataloader
    dataloader >> Edge(label="batch_data", color="blue", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> coperception_mgr

    # 3. Store current batch data (for AdvCP)
    coperception_mgr >> Edge(label="\n\nstore in\n_current_batch_data", color="orange", style="dashed", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> coperception_mgr

    # 4. Original inference (before AdvCP, for late attacks)
    coperception_mgr >> Edge(label="if late attack:\ninference_*_fusion()", color="blue", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> inference_utils
    inference_utils >> Edge(label="original_pred, original_gt", color="blue", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> coperception_mgr

    # 5. Model usage
    opencood_model >> Edge(label="weights", color="blue", style="dotted", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> inference_utils

    # 6. AdvCP decision point
    coperception_mgr >> Edge(label="if advcp enabled", color="orange", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> advcp_mgr

    # 7. AdvCP gets raw data (no circular dependency)
    advcp_mgr >> Edge(label="_get_coperception_data()", color="orange", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> coperception_mgr
    coperception_mgr >> Edge(label="_get_raw_data()\nreturn stored batch_data", color="orange", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> advcp_mgr

    # 8. Attack flow (conditional based on attack type)
    advcp_mgr >> Edge(label="if late attack:\n_apply_attack(predictions)", color="green", style="bold", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> attacker
    attacker >> Edge(label="modified predictions\n(pred_bboxes, pred_scores)", color="green", style="bold", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> advcp_mgr

    advcp_mgr >> Edge(label="if early/inter attack:\n_apply_attack(raw_data)", color="green", style="bold", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> attacker
    attacker >> Edge(label="modified raw data\n(LiDAR points)", color="green", style="bold", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> advcp_mgr

    # 9. Preprocessing for early/intermediate attacks
    advcp_mgr >> Edge(label="if early/inter attack:\npreprocess to OpenCOOD", color="blue", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> perception
    perception >> Edge(label="early_preprocess() or\nintermediate_preprocess()", color="blue", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> advcp_mgr

    # 10. Defense flow (conditional, after attack/preprocessing)
    advcp_mgr >> Edge(label="if defense enabled\n_apply_defense()", color="red", style="bold", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> defender
    defender >> Edge(label="defended_data, score, metrics", color="red", style="bold", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> advcp_mgr

    # 11. Return modified or original predictions
    advcp_mgr >> Edge(label="return\n(modified_data, score, metrics)", color="purple", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> coperception_mgr

    # 12. For early/intermediate attacks: run inference on preprocessed data
    coperception_mgr >> Edge(label="if early/inter attack:\ninference on modified_data", color="blue", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> inference_utils
    inference_utils >> Edge(label="predictions from\nattacked data", color="blue", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> coperception_mgr

    # 13. Evaluation
    coperception_mgr >> Edge(label="pred_box_tensor,\n pred_score, gt_box_tensor", color="blue", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> eval_utils
    eval_utils >> Edge(label="TP/FP/Stats", color="blue", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> coperception_mgr

    # 14. Visualization (optional)
    coperception_mgr >> Edge(label="if save_vis/show_vis", color="purple", style="dashed", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> simple_vis
    simple_vis >> Edge(label="rendered images", color="purple", style="dashed", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> coperception_mgr

    # 15. Model reference
    coperception_mgr >> Edge(label="model", color="blue", style="dotted", fontsize=str(edge_font_size), fontname="Arial", penwidth=str(edge_penwidth)) >> opencood_model

print("AdvCP Architecture Diagram generated successfully!")
print("\nColor Legend:")
print("- BLUE: Normal OpenCOOD data processing and inference")
print("- GREEN: Attack flow (when attack is enabled)")
print("- RED: Defense flow (when defense is enabled)")
print("- ORANGE: AdvCP control flow and decision making")
print("- PURPLE: Evaluation and visualization results")
print("- DASHED: Optional/conditional flows")
print("- DOTTED: Object references/initialization")
