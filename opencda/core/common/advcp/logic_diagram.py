""""
AdvCP Module Architecture Diagram
Simplified view focusing on data flow
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
import os

graph_attr = {
    "splines": "spline",
    "nodesep": "1.2",
    "ranksep": "1.5",
    "concentrate": "false",
    "fontsize": "24",
    "dpi": "150",
    "rankdir": "TB"
}

edge_attr = {"fontsize": "16"}
node_attr = {"fontsize": "16", "width": "1.2", "height": "1.2"}


class Icon:
    @staticmethod
    def path(filename):
        return f"./icons/{filename}"

    DATASET = "dataset.png"
    ATTACK = "cyber-attack.png"
    DEFENSE = "shield.png"
    LIDAR = "lidar.png"
    PERCEPTION = "perception.png"
    VEHICLE = "car.png"
    MODEL = "model.png"
    FUSION = "fusion.png"
    RAY_TRACING = "ray-tracing.png"
    ANOMALY_DETECTION = "anomaly-detection.png"
    CODE = "opencda.png"
    SIMULATION = "simulation.png"
    CONFIG = "config.png"


with Diagram("AdvCP Module - Data Flow", show=False, direction="LR",
             graph_attr=graph_attr, node_attr=node_attr):

    # ==================== SIMULATION & DATA ====================
    with Cluster("Simulation Layer"):
        carla = Custom("CARLA Simulator", Icon.path(Icon.SIMULATION))
        data_loader = Custom("DataLoader\n(OpenCOOD format)", Icon.path(Icon.DATASET))

        carla >> Edge(label="LiDAR poses\nvehicle states", **edge_attr) >> data_loader

    # ==================== COPERCEPTION MANAGER ====================
    with Cluster("CoperceptionModelManager"):
        coperception = Custom("CoperceptionModelManager\n.make_prediction(tick)", Icon.path(Icon.CODE))

        with Cluster("OpenCOOD Model"):
            model = Custom("Pre-trained Model", Icon.path(Icon.MODEL))
            inference = Custom("inference_utils\n(early/inter/late)", Icon.path(Icon.CODE))

        model >> Edge(label="load weights", **edge_attr) >> inference
        inference >> Edge(label="inference function", **edge_attr) >> coperception

    # ==================== ADVCP MODULE ====================
    with Cluster("AdvCP Module"):
        advcp_mgr = Custom("AdvCPManager\n.process_tick()", Icon.path(Icon.CODE))

        with Cluster("Attack Engine"):
            attacker = Custom("Attacker\n(select vehicles)", Icon.path(Icon.ATTACK))

            with Cluster("Attack Implementations"):
                early_attack = Custom("Early Attack\n(Ray tracing)", Icon.path(Icon.RAY_TRACING))
                inter_attack = Custom("Intermediate Attack\n(Adversarial)", Icon.path(Icon.ATTACK))
                late_attack = Custom("Late Attack\n(Box modify)", Icon.path(Icon.ATTACK))

        with Cluster("Defense Engine (CAD)"):
            defender = Custom("Defender\n(Consistency check)", Icon.path(Icon.DEFENSE))

            with Cluster("Defense Tools"):
                occupancy = Custom("Occupancy Map", Icon.path(Icon.LIDAR))
                ground = Custom("Ground Detection", Icon.path(Icon.LIDAR))
                segmentation = Custom("Segmentation", Icon.path(Icon.LIDAR))
                tracking = Custom("Object Tracking", Icon.path(Icon.VEHICLE))

    # ==================== EVALUATION ====================
    with Cluster("Output"):
        evaluation = Custom("Evaluation &\nVisualization", Icon.path(Icon.PERCEPTION))

    # ==================== CONFIGURATION ====================
    config = Custom("CLI Config\n(--with-advcp, --attack-type)", Icon.path(Icon.CONFIG))

    # ==================== DATA FLOW ====================

    # 1. Data loading
    data_loader >> Edge(label="batch_data", **edge_attr) >> coperception

    # 2. Store current batch data (for AdvCP)
    coperception >> Edge(label="store in\n_current_batch_data", color="orange", style="dashed", **edge_attr) >> coperception

    # 3. AdvCP integration (no circular dependency)
    coperception >> Edge(label="AdvCP: get raw data\n& apply attack/defense", color="orange", **edge_attr) >> advcp_mgr
    advcp_mgr >> Edge(label="return modified data", color="orange", **edge_attr) >> coperception

    # 4. AdvCP gets original predictions
    advcp_mgr >> Edge(color="blue", **edge_attr) >> inference
    inference >> Edge(label="original predictions", color="blue", **edge_attr) >> advcp_mgr

    # 4. Attack application
    advcp_mgr >> Edge(label="apply attack", color="red", **edge_attr) >> attacker
    attacker >> Edge(label="select attacker vehicles", color="red", **edge_attr) >> [early_attack, inter_attack, late_attack]

    # Attack-specific modifications
    early_attack >> Edge(label="modify LiDAR points\n(spoof/remove)", color="red", **edge_attr) >> advcp_mgr
    inter_attack >> Edge(label="modify features\n(adversarial)", color="red", **edge_attr) >> advcp_mgr
    late_attack >> Edge(label="modify boxes\n(spoof/remove)", color="red", **edge_attr) >> advcp_mgr

    # 5. Defense application (optional)
    advcp_mgr >> Edge(label="if apply_cad_defense:", color="green", **edge_attr) >> defender
    defender >> Edge(label="build occupancy map", color="green", **edge_attr) >> occupancy
    occupancy >> Edge(label="3D grid", color="green", **edge_attr) >> segmentation
    occupancy >> Edge(label="ground plane", color="green", **edge_attr) >> ground
    occupancy >> Edge(label="temporal data", color="green", **edge_attr) >> tracking

    [segmentation, ground, tracking] >> Edge(label="consistency check", color="green", **edge_attr) >> defender
    defender >> Edge(label="defended data\n+ trust scores", color="green", **edge_attr) >> advcp_mgr

    # 6. Return to coperception manager
    advcp_mgr >> Edge(label="modified_data,\ndefense_metrics", **edge_attr) >> coperception

    # 7. Final output
    coperception >> Edge(label="final predictions\nfor evaluation", **edge_attr) >> evaluation

    # 8. Configuration control
    config >> Edge(label="enable/disable\nattack/defense", style="dashed", color="purple", **edge_attr) >> coperception
    config >> Edge(label="attack params", style="dashed", color="purple", **edge_attr) >> advcp_mgr
