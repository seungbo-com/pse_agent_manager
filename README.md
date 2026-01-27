Agentic PES Mapper

An Autonomous Agent for Potential Energy Surface (PES) Mapping & Geometry Optimization

This project implements an Agentic Control System that uses Large Language Models (LLMs) to drive scientific simulations. Unlike traditional optimization scripts (like BFGS or Steepest Descent), this system uses a "Reasoning Loop" where an AI agent:

    Analyzes the current molecular state (Energy & Forces).

    Decides on a strategy (step size, direction).

    Executes physics-based tools (ASE).

    Adapts its strategy based on feedback (e.g., rejecting bad steps).

Architecture

The system follows a Graph-Based Control architecture using LangGraph.
Component	Technology	Role
The Brain	Llama 3.1 (via Ollama)	Reasons about the physics state and decides the next move.
The Controller	LangGraph	Manages the state machine, loops, and termination criteria.
The Hands	ASE (Atomic Simulation Environment)	Runs the actual physics calculations (DFT/Potentials).
The Memory	JSON / CSV / XYZ	Stores the trajectory and optimization history.
Getting Started
1. Prerequisites

You need Python 3.10+ and Ollama installed on your machine.

    Download Python

    Download Ollama

2. Install Dependencies

Clone the repository and install the required Python packages.
Bash

git clone https://github.com/your-username/pes-agent-mapper.git
cd pes-agent-mapper

# Install libraries
pip install langgraph langchain-ollama ase numpy

3. Setup the AI Model (Local)

This project runs locally for privacy and cost savings. You must pull the Llama 3.1 model.
Bash

# In your terminal:
ollama pull llama3.1

🏃 Usage
Step 1: Start the AI Server

Open a new terminal window and start the Ollama server. Keep this window open in the background.
Bash

ollama serve

Step 2: Run the Agent

Open a second terminal window in your project folder and run the main script.
Bash

python main.py

Step 3: Watch the Results

As the agent runs, it generates two files in real-time:

    optimization_movie.xyz: The geometry trajectory. Open this in VMD, Avogadro, or OVITO to watch the molecule relax.

    optimization_log.csv: A spreadsheet of Energy vs. Step.

Project Structure
Plaintext

pes-agent-mapper/
├── agents/
│   ├── __init__.py
│   └── optimizer.py       # The "Brain": LLM Logic & Adaptive Step Control
├── tools/
│   ├── __init__.py
│   └── calc_tools.py      # The "Hands": ASE Physics Engine (EMT/DFT)
├── workflows/
│   └── graph.py           # The "Manager": LangGraph State Machine
├── main.py                # Entry point: Logging & Execution
├── optimization_movie.xyz # Output: Visualization file
└── optimization_log.csv   # Output: Data file

Customization
Changing the Physics Engine

By default, the agent uses a fast "Toy Potential" (EMT) so you can test it on a laptop. To use real quantum chemistry (DFT), edit tools/calc_tools.py:
Python

# tools/calc_tools.py

# CHANGE THIS:
from ase.calculators.emt import EMT
atoms.calc = EMT()

# TO THIS (Example for Gaussian):
from ase.calculators.gaussian import Gaussian
atoms.calc = Gaussian(method='b3lyp', basis='6-31g*')

Adjusting Agent Aggressiveness

To change how cautious the agent is, edit agents/optimizer.py:
Python

# Cap the maximum movement per step (Angstroms)
MAX_STEP_SIZE = 0.1 

Troubleshooting

Error: [Errno 61] Connection refused

    Cause: The Ollama server is not running.

    Fix: Open a new terminal and run ollama serve.

Error: model 'llama3.1' not found

    Cause: You haven't downloaded the brain yet.

    Fix: Run ollama pull llama3.1.

Error: Agent oscillates between two energies

    Cause: The step size is too large for the potential well.

    Fix: The code now includes "Adaptive Rejection" logic. Ensure your optimizer.py contains the check: if new_e > current_e: step_size *= 0.5.