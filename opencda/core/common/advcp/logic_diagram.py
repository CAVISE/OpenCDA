""" "
AdvCP Module Architecture Diagram
Simplified view focusing on data flow
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom

graph_attr = {"splines": "spline", "nodesep": "1.2", "ranksep": "1.5", "concentrate": "false", "fontsize": "24", "dpi": "150", "rankdir": "TB"}

edge_attr = {"fontsize": "16"}
node_attr = {"fontsize": "16", "width": "1.2", "height": "1.2"}


class Icon:
    @staticmethod
    def path(filename: str) -> str:
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


with Diagram("AdvCP Module - Data Flow", show=False, direction="LR", graph_attr=graph_attr, node_attr=node_attr):
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

        with Cluster("Preprocessing"):
            perception = Custom("OpencoodPerception\n(early/intermediate_preprocess)", Icon.path(Icon.PERCEPTION))

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
    coperception >> Edge(label="AdvCP: process_tick()", color="orange", **edge_attr) >> advcp_mgr

    # 4. AdvCP gets raw data from coperception manager
    # advcp_mgr >> Edge(label="_get_coperception_data()", color="orange", **edge_attr) >> coperception
    advcp_mgr >> Edge(color="orange", **edge_attr) >> coperception
    # coperception >> Edge(label="_get_raw_data()\nreturn stored batch_data", color="orange", **edge_attr) >> advcp_mgr

    # 5. Attack application (conditional based on attack type)
    advcp_mgr >> Edge(label="if late attack:\napply to predictions", color="red", **edge_attr) >> late_attack
    advcp_mgr >> Edge(label="if early/inter attack:\napply to raw data", color="red", **edge_attr) >> [early_attack, inter_attack]

    # Late attack: modifies predictions directly
    late_attack >> Edge(label="modify predictions\n(pred_bboxes, pred_scores)", color="red", **edge_attr) >> advcp_mgr

    # Early/Intermediate attack: modifies raw data, then preprocess
    [early_attack, inter_attack] >> Edge(label="modify raw data\n(LiDAR points)", color="red", **edge_attr) >> advcp_mgr
    advcp_mgr >> Edge(label="preprocess to\nOpenCOOD format", color="blue", **edge_attr) >> perception
    perception >> Edge(label="early_preprocess() or\nintermediate_preprocess()", color="blue", **edge_attr) >> advcp_mgr

    # 6. Defense application (optional, after attack/preprocessing)
    advcp_mgr >> Edge(label="if apply_cad_defense:", color="green", **edge_attr) >> defender
    defender >> Edge(label="build occupancy map", color="green", **edge_attr) >> occupancy
    occupancy >> Edge(label="3D grid", color="green", **edge_attr) >> segmentation
    occupancy >> Edge(label="ground plane", color="green", **edge_attr) >> ground
    occupancy >> Edge(label="temporal data", color="green", **edge_attr) >> tracking

    [segmentation, ground, tracking] >> Edge(label="consistency check", color="green", **edge_attr) >> defender
    defender >> Edge(label="defended data\n+ trust scores", color="green", **edge_attr) >> advcp_mgr

    # 7. Return to coperception manager
    advcp_mgr >> Edge(label="modified_data,\ndefense_score, metrics", **edge_attr) >> coperception

    # 8. For early/intermediate attacks: run inference on preprocessed data
    coperception >> Edge(label="if early/inter attack:\nrun inference on\nmodified_data", color="blue", **edge_attr) >> inference
    inference >> Edge(label="predictions from\nattacked data", color="blue", **edge_attr) >> coperception

    # 9. Final output
    coperception >> Edge(label="final predictions\nfor evaluation", **edge_attr) >> evaluation

    # 10. Configuration control
    config >> Edge(label="enable/disable\nattack/defense", style="dashed", color="purple", **edge_attr) >> coperception
    config >> Edge(label="attack params", style="dashed", color="purple", **edge_attr) >> advcp_mgr
