# Symbiont: Neuro-Symbolic Generation Framework

**Symbiont is an open-source, neuro-symbolic framework that fuses the creative power of generative AI with the structured, domain-specific knowledge of human scientists. It transforms generative AI from a blunt instrument into a precise, steerable partner, enabling a new era of co-creative scientific discovery.**

## Architecture Overview

```mermaid
graph LR
    subgraph "Define Constraints"
        A[Scientific Rules<br/>& Requirements]
    end

    subgraph "Compile to Logic"
        B[Differentiable<br/>Constraint Tree]
    end

    subgraph "Guide Generation"
        C[Gradient-Steered<br/>Output]
    end

    subgraph "Validate & Refine"
        D[Constraint<br/>Satisfaction]
    end

    A -->|DSL| B
    B -->|Loss Function| C
    C -->|Backprop| D
    D -->|Iterate| C

    style A fill:#e8f5e9
    style B fill:#fff9c4
    style C fill:#e3f2fd
    style D fill:#fce4ec
```

### **The Problem: The Generative AI Impasse in R\&D**

Generative AI models have demonstrated a remarkable ability to create novel scientific designs, but their practical application is hampered by a critical flaw: a lack of **steerability and constraint adherence**.

* **The "Black Box" Problem:** Current models generate outputs probabilistically, with little to no control for the scientist beyond the initial prompt.
* **High Failure Rate:** This leads to a costly "generate-and-filter" workflow, where a majority of generated candidates are invalid, unstable, or physically impossible, wasting massive computational and experimental resources.
* **Untapped Human Expertise:** The invaluable, hard-won knowledge of domain scientists has no direct mechanism for influencing the AI's creative process.

This impasse is a major bottleneck, preventing generative AI from reaching its full potential as a transformative tool for science.

### **Our Solution: A Co-Creative Partnership**

Symbiont introduces a **neuro-symbolic architecture** that acts as an intelligent bridge between the human scientist and the generative AI. It allows scientists to define the non-negotiable rules, physical laws, and design principles of their domain as a **"symbolic scaffold."** The generative AI then explores the vast design space *within* those boundaries.

This transforms the workflow from "generate-and-filter" to **"guided-generation,"** ensuring every output is not just novel, but also valid, feasible, and aligned with the scientist's strategic goals.

### **How It Works**

Symbiont is composed of four key components that work in a continuous, interactive loop:

```mermaid
graph TB
    subgraph "Human Scientist"
        DSL[Constraint DSL]
    end

    subgraph "Symbiont Framework"
        Compiler[Differentiable<br/>Compiler]
        Bridge[Neuro-Symbolic<br/>Bridge]
    end

    subgraph "AI Model"
        Generator[Generative<br/>Model]
    end

    DSL -->|Rules & Constraints| Compiler
    Compiler -->|Differentiable Loss| Bridge
    Bridge -->|Gradient Guidance| Generator
    Generator -->|Generated Output| Bridge
    Bridge -->|Satisfaction Score| DSL

    style DSL fill:#e1f5fe
    style Compiler fill:#fff3e0
    style Bridge fill:#fce4ec
    style Generator fill:#f3e5f5
```

1. **The Generative Engine:** A pluggable backend for state-of-the-art generative models (Transformers, GNNs, Diffusion Models).
2. **The Symbolic Scaffolding Engine:** An intuitive Python DSL for scientists to declare rules and constraints in a human-readable format.
3. **The Neuro-Symbolic Bridge:** The core of the framework. It translates the symbolic rules into a differentiable loss function that can steer the generative model via backpropagation.
4. **The Interactive Discovery Dashboard:** A web-based UI for defining rules, visualizing results in real-time, and creating a rapid, iterative discovery loop.

For a complete technical breakdown, please see our full [**Project Vision Document**](/docs/project_vision.md). For the theoretical foundations and detailed framework design, see our [**Conceptual Paper**](/docs/white-papers/symbiont_conceptual_paper.pdf).

### **Getting Started**

*(This section will be updated as the project matures.)*

To get started with Symbiont, you will need Python 3.9+ and PyTorch.

1. Clone the repository:
   ```bash
   git clone https://github.com/dilee/symbiont-core.git
   cd symbiont-core
   ```

2. Install the required dependencies:
   ```bash
   poetry install
   ```

3. Run the example script:
   ```bash
   poetry run python examples/toy_sequence.py
   ```

### **Example Usage**

```python
from symbiont import Rules, MockSequenceGenerator

# Define constraints using intuitive DSL
rules = Rules()
rules.enforce(StartCodon())              # Must start with ATG
rules.constrain(GCContent(0.4, 0.6))     # GC content between 40-60%
rules.forbid(Contains("AAAA"))           # No poly-A sequences
rules.prefer(Contains("GAATTC"))         # Prefer EcoRI site

# Generate with constraints
generator = MockSequenceGenerator()
sequence = generator.constrained_generate(
    rules.compile(),
    length=100
)
```

### **Constraint Compilation Process**

```mermaid
graph TD
    subgraph "Symbolic Layer"
        R1[Hard Rule:<br/>Must have X]
        R2[Soft Rule:<br/>Should have Y]
        R3[Forbidden:<br/>Must not have Z]
    end

    subgraph "Compilation"
        C1[Fuzzy Logic<br/>t-norms]
        C2[Weight<br/>Assignment]
        C3[Differentiable<br/>Operations]
    end

    subgraph "Neural Layer"
        L[Constraint Loss<br/>Function]
        G[Gradient<br/>Computation]
        O[Optimized<br/>Output]
    end

    R1 --> C1
    R2 --> C2
    R3 --> C1
    C1 --> C3
    C2 --> C3
    C3 --> L
    L --> G
    G --> O

    style R1 fill:#ffebee
    style R2 fill:#e8f5e9
    style R3 fill:#fce4ec
```

### **Documentation**

- [**Conceptual Paper**](/docs/white-papers/symbiont_conceptual_paper.pdf) - Theoretical foundations and framework design
- [**Project Vision**](/docs/project_vision.md) - Long-term vision and strategic direction
- [**Development Roadmap**](/ROADMAP.md) - Phased development plan from prototype to production

### **How to Contribute**

We are actively looking for collaborators\! Symbiont is an ambitious project, and its success depends on a diverse community of contributors. Whether you are an AI developer, a software engineer, a domain scientist, or a technical writer, there is a place for you here.

Please read our [**CONTRIBUTING.md**](/CONTRIBUTING.md) file to learn about our development process, roadmap, and how you can get involved.

### **License**

This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.
