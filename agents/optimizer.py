import json
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from tools.calc_tools import calculate_single_point, update_geometry

# 1. SETUP LLM
llm = ChatOllama(model="llama3.1", format="json", temperature=0)


def optimizer_agent_node(state):
    # --- STEP 1: READ STATE ---
    current_xyz = state['xyz_string']

    # If we don't have energy yet, we MUST calculate.
    # The agent doesn't get a choice here. We just do it.
    if state.get('current_energy') is None:
        print("⚙️ SYSTEM: Initial calculation...")
        results = calculate_single_point(current_xyz)
        return {
            "current_energy": results['energy'],
            "max_force": results['max_force'],
            "forces": results['forces']
        }

    # If we are here, we HAVE energy. The Agent MUST move or finish.
    current_e = state['current_energy']
    max_f = state['max_force']
    forces = state.get('forces', [])

    # Check convergence early
    if max_f < 0.05:
        print("✅ CONVERGED! Forces are low.")
        return {"status": "converged"}

    # --- STEP 2: PREPARE DATA FOR LLM ---
    # Find the atom with the highest force to give the LLM a hint
    forces_np = np.array(forces)
    atom_indices = np.argsort(np.linalg.norm(forces_np, axis=1))[::-1]  # Sort by force magnitude
    worst_atom = atom_indices[0]
    force_vec = forces_np[worst_atom]

    # We suggest moving WITH the force (or against, depending on your physics convention)
    # Usually we move atoms following the force to lower energy.
    # suggested_move = (force_vec * 0.1).tolist()  # Scale by step size

    # 1. Calculate raw step
    step_scale = 0.1
    raw_step = force_vec * step_scale

    # 2. CLAMP THE STEP (Safety Guardrail)
    # Don't let the agent move more than 0.2 Angstroms at once,
    # no matter how hard the forces are pushing.
    max_step = 0.2
    step_norm = np.linalg.norm(raw_step)

    if step_norm > max_step:
        # Scale it down to exactly 0.2 length
        raw_step = raw_step * (max_step / step_norm)

    suggested_move = raw_step.tolist()

    # --- STEP 3: STRICT PROMPT ---
    system_prompt = """You are an Optimization Algorithm.
    The forces are too high. You MUST move an atom to relax the structure.

    You have ONE allowed action:
    "move": Shift an atom's position.

    The system suggests moving Atom {idx} by vector {vec}.

    Reply in JSON:
    {{
      "action": "move", 
      "atom_index": {idx},
      "vector": [x, y, z]
    }}
    """.format(idx=worst_atom, vec=suggested_move)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Current Force: {max_f}. Execute the move.")
    ]

    print(f"\n🤖 AGENT: Force is {max_f:.3f}. Deciding move...")

    try:
        response = llm.invoke(messages)
        decision = json.loads(response.content)
    except:
        print("❌ ERROR: Bad JSON. Falling back to default move.")
        decision = {"action": "move", "atom_index": worst_atom, "vector": suggested_move}

    # --- STEP 4: EXECUTE WITH DAMPING ---
    if decision.get('action') == 'move':
        idx = decision.get('atom_index', worst_atom)

        # 1. READ THE RAW VECTOR FROM AGENT
        # The agent usually suggests moving WITH the force to relieve stress
        agent_vec = np.array(decision.get('vector', suggested_move))

        # 2. APPLY "TRUST REGION" (Damping)
        # If the force is huge (25.0), the agent wants to move HUGE (2.5 A).
        # We must cap this.

        # Calculate how long the requested move is
        step_length = np.linalg.norm(agent_vec)

        # Define a maximum safe step per iteration (e.g., 0.1 Angstroms)
        MAX_STEP_SIZE = 0.1

        final_vec = agent_vec

        # If the step is too big, scale it down
        if step_length > MAX_STEP_SIZE:
            scale_factor = MAX_STEP_SIZE / step_length
            final_vec = agent_vec * scale_factor
            print(f"⚠️ DAMPING: Agent wanted {step_length:.3f}A step. Capped to {MAX_STEP_SIZE}A.")

        # 3. EXECUTE THE SAFE MOVE
        final_vec_list = final_vec.tolist()
        new_xyz = update_geometry(current_xyz, idx, final_vec_list)

        # 4. Recalculate
        new_results = calculate_single_point(new_xyz)

        print(f"⚡ ACTION: Moved Atom {idx} by {np.linalg.norm(final_vec):.3f}A.")
        print(f"   Old Force: {max_f:.3f} -> New Force: {new_results['max_force']:.3f}")

        # 5. OSCILLATION CHECK (Optional but smart)
        # If the energy went UP, we made a mistake. Reject the move (in a real optimizer).
        # For this simple agent, we just report it.
        if new_results['energy'] > current_e:
            print("⚠️ WARNING: Energy increased! Step was likely too big.")

        return {
            "xyz_string": new_xyz,
            "current_energy": new_results['energy'],
            "max_force": new_results['max_force'],
            "forces": new_results['forces']
        }

    return {}