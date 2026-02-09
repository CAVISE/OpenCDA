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
from diagrams.onprem.compute import Server
from diagrams.onprem.container import Docker
from diagrams.generic.compute import Rack
from diagrams.programming.language import Python
from diagrams.onprem.client import Client
from diagrams.generic.storage import Storage

graph_attr = {
    "splines": "spline",
    "nodesep": "1.5",
    "ranksep": "1.8",
    "fontsize": "12",
    "dpi": "150",
    "rankdir": "LR"
}

edge_attr = {"fontsize": "10", "penwidth": "1.2"}
node_attr = {"fontsize": "10", "width": "1.2", "height": "0.8"}

with Diagram("AdvCP Module Architecture (Current)", show=False, direction="LR",
             graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr):

    # ==================== OPENCDA CORE ====================
    with Cluster("OpenCDA Core"):
        coperception_mgr = Python("CoperceptionModelManager\nMain Perception Manager")

    # ==================== DATA SOURCE ====================
    with Cluster("Data Source"):
        dataloader = Python("DataLoader\n(OpenCOOD Dataset)")
        opencood_dataset = Storage("OpenCOOD Dataset\n(No OPV2V)")

    # ==================== MODEL & INFERENCE ====================
    with Cluster("Perception Model"):
        opencood_model = Python("OpenCOOD Model\n(Perception Network)")
        inference_utils = Python("Inference Utils\n(early/intermediate/late)")

    # ==================== ADVCP MODULE ====================
    with Cluster("AdvCP Module"):
        advcp_mgr = Python("AdvCPManager\nprocess_tick()")

        with Cluster("Attack Components"):
            attacker = Python("Attacker\n(lidar_remove/spoof)")

        with Cluster("Defense Components"):
            defender = Python("Defender\n(CAD Defense)")

    # ==================== EVALUATION & VISUALIZATION ====================
    with Cluster("Output & Evaluation"):
        eval_utils = Python("Eval Utils\n(TP/FP calculation)")
        simple_vis = Python("Visualizer\n(3D/BEV)")

    # ==================== DATA FLOW ====================

    # 1. Dataset and DataLoader setup (one-time)
    opencood_dataset >> Edge(label="build_dataset()", style="dashed", color="gray") >> dataloader

    # 2. Main prediction loop - load batch
    coperception_mgr >> Edge(label="batch_data", color="blue") >> dataloader
    dataloader >> Edge(label="batch_data", color="blue") >> coperception_mgr

    # 3. Store current batch data (for AdvCP)
    coperception_mgr >> Edge(label="store in\n_current_batch_data", color="orange", style="dashed") >> coperception_mgr

    # 4. Original inference (before AdvCP)
    coperception_mgr >> Edge(label="inference_*_fusion()", color="blue") >> inference_utils
    inference_utils >> Edge(label="original_pred, original_gt", color="blue") >> coperception_mgr

    # 5. Model usage
    opencood_model >> Edge(label="weights", color="blue", style="dotted") >> inference_utils

    # 6. AdvCP decision point
    coperception_mgr >> Edge(label="if advcp enabled", color="orange") >> advcp_mgr

    # 7. AdvCP gets raw data (no circular dependency)
    advcp_mgr >> Edge(label="_get_coperception_data()", color="orange") >> coperception_mgr
    coperception_mgr >> Edge(label="_get_raw_data()\nreturn stored batch_data", color="orange") >> advcp_mgr

    # 6. Attack flow (conditional)
    advcp_mgr >> Edge(label="if attack enabled\n_apply_attack()", color="green", style="bold") >> attacker
    attacker >> Edge(label="modified_data\n(pred_bboxes, pred_scores)", color="green", style="bold") >> advcp_mgr

    # 7. Defense flow (conditional, after attack)
    advcp_mgr >> Edge(label="if defense enabled\n_apply_defense()", color="red", style="bold") >> defender
    defender >> Edge(label="defended_data, score, metrics", color="red", style="bold") >> advcp_mgr

    # 8. Return modified or original predictions
    advcp_mgr >> Edge(label="return\n(modified_data, score, metrics)", color="purple") >> coperception_mgr

    # 9. Evaluation
    coperception_mgr >> Edge(label="pred_box_tensor,\n pred_score, gt_box_tensor", color="blue") >> eval_utils
    eval_utils >> Edge(label="TP/FP/Stats", color="blue") >> coperception_mgr

    # 10. Visualization (optional)
    coperception_mgr >> Edge(label="if save_vis/show_vis", color="purple", style="dashed") >> simple_vis
    simple_vis >> Edge(label="rendered images", color="purple", style="dashed") >> coperception_mgr

    # 11. Model reference
    coperception_mgr >> Edge(label="model", color="blue", style="dotted") >> opencood_model

print("AdvCP Architecture Diagram generated successfully!")
print("\nColor Legend:")
print("- BLUE: Normal OpenCOOD data processing and inference")
print("- GREEN: Attack flow (when attack is enabled)")
print("- RED: Defense flow (when defense is enabled)")
print("- ORANGE: AdvCP control flow and decision making")
print("- PURPLE: Evaluation and visualization results")
print("- DASHED: Optional/conditional flows")
print("- DOTTED: Object references/initialization")