# Assemble an embodied skill from collected modules

You are the skill author. Read `data/03_handoff.json` for the user's intended
purpose, environment and constraints. Construct one focused embodied skill from
the classified resource pool. Choose its scope, component combination and
procedures yourself; existing skill IDs are provenance links, not required output
names or a one-module-to-one-skill mapping.

`data/` below refers to the locally collected corpus named by the handoff. The
public code release does not supply a corpus; use the caller's `--data` location.

## Inputs and progressive reading

1. Search the descriptions and types in `data/01_cards.json`. Use
   `data/02_groups.json` to locate executable interfaces, reference knowledge and
   static assets. Check an interesting module's contract in
   `data/02_contracts.json`, then read its `source.snapshot` relative to `data/`.
   `data/01_inventory.json` connects it to original files, archived assets and
   existing package versions. Prefer the requested source scope, if one is given.
2. Select a coherent subset for the intended purpose. You may combine resources
   from several source packages or use only reference material. Keep distinct
   model, budget and interface versions distinct. For native code bindings, use
   the intended package version and retain its input schema, effects, execution
   limits, model profile and dependency information. Do not infer unspecified
   units or coordinate transforms. Source-only fragments need an implementation
   adapter before being described as runnable; descriptor-only models need the
   external weights and serving runtime. A paper/repository locator provides
   provenance, not the contents of a paper or an implementation.
3. Read the selected material and author the operational instructions. Explain
   when the skill applies, what observation and runtime support it needs, how the
   chosen components are used, what feedback supports continuation, and when to
   verify, recover or stop. Derive these decisions from the actual material and
   intended purpose. Use conditional guidance where observations determine the
   next step; do not write a fixed benchmark solution or copy an old package
   wholesale. Preserve source-specific behavior, e.g. an eight-decision Jev
   waypoint and a single native VLA chunk are different execution boundaries.
4. Use links to selected components' exact `target` paths from the normalized
   contracts. Model configurations and schemas belong in `assets/`; executable
   wrappers belong in `scripts/`; procedural and empirical material belongs in
   `references/`. A repository or model can contribute multiple separately
   collected modules. Do not create empty component directories. Detailed
   supporting information stays in references; the skill body explains the
   purpose and the decisions needed to use it.

## Output and acceptance

Return a JSON object conforming to `protocol.json:assembly_plan_schema`:

- `name`: a descriptive lowercase hyphenated name, under 64 characters.
- `description`: a concise capability and its trigger conditions.
- `instructions`: the Markdown body of `SKILL.md`, without YAML frontmatter.
- `modules`: the selected module IDs, not existing skill IDs.
- `rationale`: why these resources support this purpose; mention dependencies or
  unresolved integration requirements that matter for use.

The caller saves the JSON plan and runs:

```bash
python3 -B scripts/03_assemble.py materialize --plan path/to/plan.json --repo path/to/repository
```

The materializer writes metadata and instructions into `SKILL.md`, copies the
selected payloads into their component directories, includes native-contract
dependencies, checks their hashes and local links, and registers the package in
`index.json`. It does not invent skill text, implement missing robot adapters, or
validate a physical outcome. Check the resulting skill for applicability and
coherent feedback handling before use. A controller's completion flag alone is
not evidence of task success.

The source archive includes original extracts, interface declarations, public
experience entries and already adapted guidance. Their provenance must retain
those distinctions. No simulator rollout, private task memory or benchmark
answer is needed for this assembly task.
