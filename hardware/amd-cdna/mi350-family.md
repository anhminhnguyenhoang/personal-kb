---
name: mi350-family
description: Column-wise specifications, form factors, and deployment guidance for AMD Instinct MI350P, MI350X, and MI355X.
source: _sources/amd-instinct-mi350-comparison.md
---

# AMD Instinct MI350P, MI350X, and MI355X

> MI350P brings CDNA 4 to conventional PCIe servers, while MI350X and MI355X trade deployment simplicity for larger memory, Infinity Fabric scaling, and higher power envelopes.

Sources: AMD's [MI350P](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350p.html), [MI350X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html), and [MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html) product pages and [AMD's MI350P announcement](https://www.amd.com/en/blogs/2026/amd-instinct-mi350p-pcie-gpus-run-enterprise-ai-on-your.html), with the study-session capture preserved in [`../../_sources/amd-instinct-mi350-comparison.md`](../../_sources/amd-instinct-mi350-comparison.md).

All three accelerators use AMD CDNA 4 and target the `gfx950` platform.

## Core specification comparison

| Specification | MI350P | MI350X | MI355X |
|---|---|---|---|
| Form factor | PCIe 5.0 x16, dual-slot | OAM module | OAM module |
| Compute units | 128 | 256 | 256 |
| Matrix cores | 512 | 1,024 | 1,024 |
| Peak engine clock | 2.2 GHz | 2.2 GHz | 2.4 GHz |
| HBM3E capacity | 144 GB | 288 GB | 288 GB |
| Peak memory bandwidth | 4 TB/s | 8 TB/s | 8 TB/s |
| FP16 matrix performance | 1.15 PFLOPS | 2.3 PFLOPS | approximately 2.5 PFLOPS |
| FP8 matrix performance | 2.3 PFLOPS | 4.6 PFLOPS | approximately 5.0 PFLOPS |
| MXFP4 matrix performance | 4.6 PFLOPS | 9.2 PFLOPS | approximately 10.1 PFLOPS |
| Typical board power | 600 W maximum; 450 W configurable | 1,000 W | 1,400 W |
| Cooling | Air-cooled server airflow | Air-cooled OAM platform | Direct liquid cooling |
| GPU-to-GPU fabric | PCIe only | Infinity Fabric | Infinity Fabric |

## Form-factor comparison

MI350P is a full-height, full-length, dual-slot PCIe 5.0 x16 add-in card. It uses a 12V-2x6 connector and passive cooling supplied by server airflow, making it suitable for existing PCIe servers.

MI350X and MI355X are OAM modules installed on specialized accelerator baseboards, commonly eight-GPU UBB platforms. OAM systems use higher-power 54 V infrastructure and provide AMD Infinity Fabric links between GPUs.

MI350X is designed for an approximately 1,000 W air-cooled OAM deployment. MI355X uses a 1,400 W power envelope and direct liquid cooling for maximum throughput.

PCIe prioritizes compatibility and deployment flexibility; OAM prioritizes power delivery, cooling capacity, rack density, and GPU-to-GPU scaling.

## Deployment guidance

- Choose MI350P for existing PCIe servers, enterprise inference, and RAG deployments where power and cooling changes are constrained.
- Choose MI350X for large-model AI or HPC workloads that need 288 GB of HBM3E and Infinity Fabric without the MI355X power envelope.
- Choose MI355X for the highest single-GPU throughput when direct-liquid-cooled OAM infrastructure is available.

Peak specifications are not application benchmarks; ROCm support, model quantization, kernel quality, and interconnect behavior affect real-world performance.

## Sources

- [AMD Instinct MI350P](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350p.html)
- [AMD Instinct MI350X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html)
- [AMD Instinct MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)
- [AMD: MI350P PCIe GPUs](https://www.amd.com/en/blogs/2026/amd-instinct-mi350p-pcie-gpus-run-enterprise-ai-on-your.html)
