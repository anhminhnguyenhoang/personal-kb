# GPU kernel knowledge base

General notes on GPU hardware, kernel-optimization method, research papers, and kernel DSLs. Most notes cover AMD CDNA3 and CDNA4 (MI300X, MI350X, MI355X).

## Areas

| Area | What it covers |
| --- | --- |
| [hardware/](hardware/README.md) | AMD CDNA and NVIDIA Blackwell facts: CU, SIMD, wave, registers, occupancy, MFMA, TMEM. |
| [kernel-optimization/](kernel-optimization/README.md) | A DSL-neutral method: the loop, measurement, bottleneck diagnosis, levers, autotuning, failure modes. |
| [papers/](papers/README.md) | Notes on HipKittens (AMD tile DSL) and FlashAttention-4 (Blackwell attention). |
| [dsl/](dsl/README.md) | DSL-specific mappings for ck_dsl and FlyDSL. |

## Conventions

- Numeric prefixes show reading order.
- Every note has YAML frontmatter (`name`, `description`, `source`) and a related-links footer.
- Facts about external work cite public primary sources.
- `_sources/` keeps the author's own research captures.

### What belongs here

1. It is true without a specific machine, ticket, branch, or chat.
2. No host, container, user name, or home-directory path.
3. The subject is a concept, not an incident or a date.
4. It cites a public primary source.
5. Task numbers (shapes, speedups) are examples, not the topic.

## License

See [LICENSE](LICENSE) (CC BY 4.0).
