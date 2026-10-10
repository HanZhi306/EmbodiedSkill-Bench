# Constructing skills from local resources

The public repository includes the construction workflow and common protocol. It does not include the paper's collected corpus or authored skill packages. Supply your own local resources and keep generated corpora outside the checkout.

An embodied skill is a package `⟨m, I, S, R, A⟩`: discoverable metadata (`m`), operational instructions (`I`), scripts (`S`), references (`R`), and assets (`A`). `SKILL.md` holds metadata and instructions; optional resources go into `scripts/`, `references/`, and `assets/`.

## 1. Collect modules

Copy [inputs/sources.json](../inputs/sources.json) to your external workspace and edit `workspace`, source patterns, extra files, and `paper_zip` to point to resources you have locally. The supplied paths illustrate the collection recipe; they are not directories distributed in this public repository. Remove inapplicable entries.

```bash
python scripts/01_collect.py \
  --config ../esb-runtime/sources.json --output ../esb-runtime/corpus
```

The collector creates three-field cards (`type`, `description`, `source`), inventory/provenance records, and deduplicated source payloads stored by SHA256. A paper or repository locator records provenance; it does not imply that the upstream full text or code was retrieved. Model descriptors retain external weight/runtime dependencies.

## 2. Structure components and contracts

```bash
python scripts/02_structure.py --data ../esb-runtime/corpus
```

The common protocol maps control and perception interfaces to scripts; method references and experience material to references; and model descriptors/configuration to assets. Native bindings retain input schemas, dependencies, coordinate conventions, model profiles, and execution budgets. Source fragments requiring integration remain marked as source-only.

## 3. Prepare the authoring handoff

Adapt [inputs/request.json](../inputs/request.json) to the operational purpose. It is an example request, not a pre-authored skill.

```bash
python scripts/03_assemble.py --data ../esb-runtime/corpus prepare \
  --request ../esb-runtime/request.json
```

Give [prompts/assemble_skill.md](../prompts/assemble_skill.md) and the generated `03_handoff.json` to the authoring agent, with access to the corpus. The agent selects compatible modules and writes applicability, invocation, feedback interpretation, verification, and bounded recovery/stopping rules. Its JSON plan must satisfy `protocol.json:assembly_plan_schema`.

## 4. Materialize the plan

```bash
python scripts/03_assemble.py --data ../esb-runtime/corpus materialize \
  --plan ../esb-runtime/plan.json --repo ../esb-runtime/authored-skills
```

Materialization writes `SKILL.md`, copies selected payloads and declared dependencies, relocates links, verifies hashes and local references, and registers the result in an assembly `index.json`. Existing packages are not overwritten. Component checks do not establish physical success or implement a missing robot adapter.

The assembly `index.json` is distinct from the runtime `skills-index.json`. Runtime discovery additionally requires each package's `metadata.json` and its execution/interaction contract; do not assume an arbitrary materialized source-only package is runnable in the simulator. Consult [catalog.py](../src/embodied_skill_bench/catalog.py) and the selected native provider when integrating a new package.

For resource access, use `scripts/access.py --data ../esb-runtime/corpus --request path/to/request.json`. `read` returns a resource and `prepare` checks arguments and returns a native call descriptor. Programmatic `invoke` requires a caller-supplied runtime dispatcher.
