# Project Vision: Symbiont

**A Neuro-Symbolic Framework for Co-Creative Scientific Discovery**

*Version 1.1 | August 2025*

---

### **1. Executive Summary**

Scientific research is on the verge of a paradigm shift, powered by generative AI. Yet, today's most powerful generative models operate as "black boxes," producing a high volume of often invalid or unfeasible results, creating a significant bottleneck. **Project Symbiont** is an open-source initiative to solve this problem. We are building a neuro-symbolic framework that fuses the creative power of generative AI with the structured, domain-specific knowledge of human scientists. Symbiont will transform generative AI from a blunt instrument into a precise, steerable partner, enabling a new era of **co-creative scientific discovery**. It acts as a "GPS" for scientific exploration, allowing researchers to guide the AI toward promising solutions, dramatically accelerating R&D in fields like drug discovery, materials science, and synthetic biology.

---

### **2. The Gap: The Generative AI Impasse in R&D**

Generative AI models like DeepMind's AlphaFold and protein generators like RFdiffusion have demonstrated a remarkable ability to create novel scientific designs. However, their practical application is hampered by a critical flaw: a lack of **steerability, interpretability, and constraint adherence**.

* **The "Black Box" Problem:** Current models are trained on vast datasets to recognize patterns. When generating new outputs (e.g., a new protein), they do so probabilistically. The scientist has little to no control over the generation process beyond the initial prompt. This lack of control is a fundamental limitation of purely neural approaches, which struggle to incorporate abstract, symbolic knowledge.

* **High Failure Rate & Wasted Resources:** This lack of control leads to a costly "generate-and-filter" workflow. A model might produce 10,000 potential protein designs, but 99% of them could be invalid, unstable, or impossible to synthesize according to known biophysical principles. This requires massive computational resources for filtering and significant human effort to validate the few promising candidates. This is a well-documented issue in generative chemistry, where models often produce chemically invalid or unsynthesizable molecules.

* **Untapped Human Expertise:** Every scientist possesses a wealth of invaluable, hard-won knowledge—the physical laws, chemical constraints, and biological principles that govern their domain. Current generative workflows have no mechanism to incorporate this expert knowledge *during* the generation process. This represents a failure to leverage the "human-in-the-loop" paradigm, which has been shown to significantly improve AI system performance and reliability.

This impasse is a major bottleneck, preventing generative AI from reaching its full potential as a transformative tool for science.

---

### **3. Our Solution: Project Symbiont**

**Mission:** To create an accessible, open-source framework that empowers scientists to directly guide generative AI models with their domain expertise, making the discovery process more efficient, targeted, and creative.

Symbiont introduces a **neuro-symbolic architecture** that acts as an intelligent bridge between the human scientist and the generative AI. This approach is grounded in the idea that intelligence requires both the pattern-matching capabilities of neural systems (System 1) and the deliberate, rule-based reasoning of symbolic systems (System 2).

Imagine a scientist designing a new enzyme. Instead of just asking the AI to "generate enzymes," they can provide a set of explicit, human-readable constraints:

> *"Generate a protein that contains the active site motif `[C-X(2)-C-X(12)-H-X(3)-H]`, remains stable at temperatures above 50°C, and does not include any disulfide bonds."*

Symbiont translates these symbolic rules into mathematical constraints that actively steer the AI's creative process, ensuring that every output is not just novel, but also valid, feasible, and aligned with the scientist's strategic goals.

---

### **4. How It Works: The Symbiont Architecture**

Symbiont is composed of four key components that work in a continuous, interactive loop.

1.  **The Generative Engine (The Creative Core):**
    * This is a **pluggable backend** that houses state-of-the-art, open-source generative models. The architecture will be model-agnostic, with standardized interfaces for different model types.
    * **Initial Integrations:**
        * **Protein/Sequence Design:** Transformers and Diffusion Models (e.g., **RFdiffusion**, **Chroma**).
        * **Small Molecule/Graph Design:** Graph Neural Networks (GNNs) and Variational Autoencoders (VAEs), using libraries like **PyTorch Geometric**.
    * The engine exposes hooks into the generation process (e.g., the latent space or the output logits) where the guidance from the neuro-symbolic bridge can be applied.

2.  **The Symbolic Scaffolding Engine (The Human Interface):**
    * This is where the scientist provides their expertise through a simple, intuitive **Domain-Specific Language (DSL)** in Python. The design philosophy is declarative—the scientist states *what* constraints must be met, not *how* to meet them.
    * **Example Rules in the DSL:**
        ```python
        # For Protein Design
        # Enforce a specific structural motif (e.g., from a PROSITE pattern)
        rules.enforce(HasMotif("C-x(2)-C-x(12)-H-x(3)-H"))
        # Constrain a calculated property
        rules.constrain(IsoelectricPoint(min=6.5, max=7.5))
        # Forbid a specific substructure
        rules.forbid(HasSubstructure("C-S-S-C"))

        # For Small Molecule Design
        # Enforce a specific chemical scaffold
        rules.enforce(HasScaffold("c1ccccc1C(=O)N")) # Benzamide
        # Constrain properties based on Lipinski's Rule of Five
        rules.constrain(MolecularWeight(max=500))
        rules.constrain(LogP(max=5))
        ```

3.  **The Neuro-Symbolic Bridge (The "Magic Glue"):**
    * This is the technical heart of Symbiont. It uses a **differentiable logic engine** to translate the symbolic rules from the DSL into a continuous loss function.
    * **Mechanism:**
        1.  The DSL rules are parsed into a formal logical representation.
        2.  This logical structure is "lifted" into a differentiable format. Logical operators (like AND, OR, NOT) are replaced by their continuous, differentiable counterparts, often using **t-norms** from fuzzy logic.
        3.  When a candidate design is generated by the neural model, it is evaluated against this differentiable logical graph. The output is a continuous "satisfaction score."
        4.  This score is converted into a **constraint violation loss**. This loss is then backpropagated through the generative neural network, updating its weights to favor rule-satisfying outputs.
    * **Core Technology:** We will build this bridge using libraries like **PyNeuraLogic** or **Logic Tensor Networks**, which are specifically designed for this type of differentiable logical reasoning within a deep learning framework.

4.  **The Interactive Discovery Dashboard (The Co-Creative Loop):**
    * A simple web-based interface built with **Streamlit** or **Gradio** to facilitate a human-in-the-loop workflow.
    * **Features:**
        * A text editor for defining and modifying DSL rules on the fly.
        * Real-time visualization of generated candidates.
        * Analytics on constraint satisfaction, showing the scientist which rules are "hardest" for the model to learn.
    * This dashboard turns the process from a one-shot generation into an iterative dialogue, embodying the principles of co-creative AI.

---

### **5. Project Roadmap & Milestones**

* **Phase 1: Core Framework & Toy Problem**
    * **Goal:** Build the foundational architecture and prove the concept.
    * **Tasks:** Develop the initial DSL parser, integrate a differentiable logic engine, build a simple generative model for a "toy problem" (e.g., DNA sequences), and create the first version of the Streamlit dashboard.

* **Phase 2: Integration & Alpha Release**
    * **Goal:** Integrate a real-world generative model and release to early adopters.
    * **Tasks:** Build a robust integration layer for a model like RFdiffusion, expand the DSL with rules specific to protein engineering, and refine the dashboard with advanced visualizations.

* **Phase 3: Community & Expansion**
    * **Goal:** Foster a thriving community and expand the framework's capabilities.
    * **Tasks:** Develop comprehensive documentation and tutorials, add support for GNN-based molecule generation, and engage with the scientific community.

---

### **6. Why Open Source?**

Science thrives on collaboration, reproducibility, and transparency. By building Symbiont as an open-source project, we will:

* **Accelerate Scientific Progress:** Provide a powerful, free tool that can be used by researchers everywhere.
* **Foster Cross-Disciplinary Collaboration:** Create a common ground where AI engineers and domain scientists can work together.
* **Ensure Transparency and Trust:** Openly share our code and methods, allowing for peer review and community-driven improvements.

---

### **7. How to Contribute**

Symbiont is an ambitious project, and its success depends on a diverse community of contributors. We are looking for collaborators with skills in AI/ML, software architecture, frontend design, various scientific domains, and technical writing.

To learn how you can get involved, please read our [**CONTRIBUTING.md**](../CONTRIBUTING.md) guide.

---

### **8. References**

[1] Heaven, W. D. (2023). "Generative AI is coming for science." *MIT Technology Review*.
[2] Kantosalo, A., & Takala, T. (2020). "The co-creative human-computer collective." *Proceedings of the 11th Conference on Creativity and Cognition*.
[3] Jumper, J., et al. (2021). "Highly accurate protein structure prediction with AlphaFold." *Nature*.
[4] Watson, J. L., et al. (2023). "De novo design of protein structure and function with RFdiffusion." *Nature*.
[5] Kautz, H. (2022). "The third AI winter." *AAAI Conference Keynote*.
[6] Ingraham, J., et al. (2023). "Illuminating protein space with a programmable generative model." *bioRxiv*.
[7] Gao, W., & Coley, C. W. (2020). "The synthesizability of molecules proposed by generative models." *Journal of Chemical Information and Modeling*.
[8] Amershi, S., et al. (2019). "Guidelines for human-AI interaction." *Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems*.
[9] Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
[10] Fey, M., & Lenssen, J. E. (2019). "Fast graph representation learning with PyTorch Geometric." *ICLR Workshop on Representation Learning on Graphs and Manifolds*.
[11] Zadeh, L. A. (1965). "Fuzzy sets." *Information and Control*.
[12] Šourek, G., et al. (2021). "Lifted Relational Neural Networks: A Differentiable Approach." *Journal of Artificial Intelligence Research*.
[13] Serafini, L., & Garcez, A. d'Avila. (2016). "Logic Tensor Networks: Deep Learning and Logical Reasoning from Data and Knowledge." *arXiv*.
[14] Rose, A. S., et al. (2018). "NGL viewer: web-based molecular graphics for large complexes." *Bioinformatics*.
[15] The Royal Society. (2023). "Science in the age of AI." *Policy Report*.
