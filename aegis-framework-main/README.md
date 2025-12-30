# Aegis Framework: An Adversarial Attack Toolkit for Object Detectors

Aegis is a flexible and extensible PyTorch-based framework for generating adversarial attacks against modern object detection models. It provides a unified interface for crafting and testing a variety of attack strategies, from 2D universal perturbations to sophisticated 3D texture-based attacks.

The framework is designed with a "configuration-over-code" philosophy, allowing researchers and developers to define and run complex attack experiments through simple YAML files without modifying the source code.

## Features

- **Multi-Attack Support**: Out-of-the-box implementations for several state-of-the-art attack methods:
    - **Universal 2D Noise**: Learns a single, subtle noise pattern to fool detectors.
    - **OSFD (One-Shot Fooling)**: A powerful white-box feature-level attack.
    - **2D Adversarial Patch**: Generates a physical-world-style patch that can be placed on or near an object.
    - **3D Single-Plane Texture**: Learns an adversarial texture projected onto a 3D model from a single plane.
    - **3D Tri-Plane Texture**: Learns three directional textures for more robust 3D attacks.
- **Ensemble Attacks**: Optimize a single attack to be effective against an ensemble of different victim models (e.g., YOLOv8, YOLOv5, RT-DETR, GroundingDINO) simultaneously, improving transferability.
- **Highly Configurable**: Define every aspect of your attack—from model paths and learning rates to attack-specific parameters—in a clean YAML file.
- **Flexible Data Loading**: Use a fully structured dataset directory or test attacks on-the-fly with single images, lists of images, or specific 3D poses with all required metadata.
- **Extensible by Design**: A clear class-based structure (`BaseAttack`) makes it easy to implement and integrate new, custom attack strategies.
- **Built-in Diagnostics**: Real-time monitoring of loss and gradient norms, with periodic visualization to track attack progress.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://gitlab.smartm2m.co.kr/ai-team-idn/aegis-framework.git
    cd aegis-framework
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the framework in editable mode:**
    This command installs all required dependencies from `pyproject.toml` and makes the `aegis-attack` command available in your terminal.
    ```bash
    pip install -e .
    ```
    To include support for GroundingDINO, you might need to install optional dependencies (if configured that way):
    ```bash
    pip install -e ".[dino]"
    ```

4.  **Download Assets**:
    Download the assets using
    ```bash
    python download_assets.py
    ```

## Core Concepts

- **Configuration File (`.yaml`)**: The blueprint for your experiment. It defines the attack type, its parameters, the victim models, the dataset to use, and training settings.
- **Trainer**: The core engine that reads the config, sets up the attack, and runs the main training loop.
- **Attack Strategy**: A class (e.g., `PatchAttack`) that encapsulates the logic for a specific attack method.
- **Victim Models**: The target object detectors you are trying to fool.
- **Pseudo Ground-Truth**: For disappearance attacks, the framework first runs a "guide" model on the clean image to determine the location of the target object. The attack's goal is then to minimize the detection confidence at this location.

## How to Run an Attack

The framework is controlled via the `aegis-attack` command-line interface. The primary argument is `--config`, which points to your experiment's YAML file.

```bash
aegis-attack --config path/to/your/config.yaml [OPTIONS]
```

### Command-Line Options

You can override key training parameters directly from the command line for quick experiments:

- `--config` / `-c`: (Required) Path to the YAML configuration file.
- `--epochs`: Override the number of training epochs.
- `--lr`: Override the learning rate.
- `--bs`: Override the batch size.
- `--output-path` / `-o`: Override the save path for the final adversarial artifact.
- `--clear-output-dir`: If set, deletes the entire output directory before the run starts.

---

## Example Attack Configurations

All configuration files are located in the `examples/configs/` directory.

### 1. 2D Adversarial Patch Attack

This example learns a square patch to place on a car, trained on a full dataset.

**To Run:**
```bash
aegis-attack --config examples/configs/patch_2d_attack_on_car.yaml
```

### 2. OSFD Attack on a Single Image

This example runs a white-box OSFD attack targeting the internal features of `yolov12x` on a single image.

**To Run:**
```bash
aegis-attack --config examples/configs/osfd_single_image.yaml
```

### 3. 3D Tri-Plane Attack 

This example shows how to run a powerful 3D attack.

**To Run:**
```bash
aegis-attack --config examples/configs/triplane_attack.yaml
```

## Extending the Framework

To add a new attack method (e.g., "MyNewAttack"):
1.  Create a new file `aegis/attacks/my_new_attack.py`.
2.  Implement a class `MyNewAttack` that inherits from `BaseAttack` and implements its abstract methods.
3.  Define a Pydantic model `MyNewAttackParams` in `aegis/core/config_models.py` for its specific parameters.
4.  Add `MyNewAttackParams` to the `AttackParamsUnion` in `aegis/core/config.py`.
5.  Register your new attack in the `_get_attack_strategy` factory method in `aegis/core/trainer.py`.
6.  Create a new YAML configuration file using `type: "my_new_attack"`.

