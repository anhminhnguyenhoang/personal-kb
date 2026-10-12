# AMD Instinct MI350P, MI350X, and MI355X comparison

Captured from the Aug. 27, 2026 study session.

## Core specification comparison

- All three accelerators use AMD CDNA 4 and target the `gfx950` platform.
- MI350P: 128 compute units, 512 matrix cores, 144 GB HBM3E, 4 TB/s bandwidth, and 600 W maximum board power; 450 W is configurable.
- MI350X: 256 compute units, 1,024 matrix cores, 288 GB HBM3E, 8 TB/s bandwidth, and 1,000 W board power.
- MI355X: 256 compute units, 1,024 matrix cores, 288 GB HBM3E, 8 TB/s bandwidth, 2.4 GHz peak engine clock, and 1,400 W board power.
- MI350P peak matrix performance: 1.15 PFLOPS FP16, 2.3 PFLOPS FP8, and 4.6 PFLOPS MXFP4.
- MI350X peak matrix performance: 2.3 PFLOPS FP16, 4.6 PFLOPS FP8, and 9.2 PFLOPS MXFP4.
- MI355X peak matrix performance: approximately 2.5 PFLOPS FP16, 5.0 PFLOPS FP8, and 10.1 PFLOPS MXFP4.

## Form-factor comparison

- MI350P is a full-height, full-length, dual-slot PCIe 5.0 x16 add-in card.
- MI350P uses a 12V-2x6 connector and passive cooling supplied by server airflow.
- MI350X and MI355X are OAM modules installed on specialized accelerator baseboards, commonly eight-GPU UBB platforms.
- OAM systems use higher-power 54 V infrastructure and provide AMD Infinity Fabric links between GPUs.
- MI350X is designed for an approximately 1,000 W air-cooled OAM deployment.
- MI355X uses a 1,400 W power envelope and direct liquid cooling for maximum throughput.
- PCIe prioritizes compatibility and deployment flexibility; OAM prioritizes power delivery, cooling capacity, rack density, and GPU-to-GPU scaling.

## Deployment guidance

- Choose MI350P for existing PCIe servers, enterprise inference, and RAG deployments where power and cooling changes are constrained.
- Choose MI350X for large-model AI or HPC workloads that need 288 GB of HBM3E and Infinity Fabric without the MI355X power envelope.
- Choose MI355X for the highest single-GPU throughput when direct-liquid-cooled OAM infrastructure is available.
- Peak specifications are not application benchmarks; ROCm support, model quantization, kernel quality, and interconnect behavior affect real-world performance.

## Sources

- https://www.amd.com/en/products/accelerators/instinct/mi350/mi350p.html
- https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html
- https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html
- https://www.amd.com/en/blogs/2026/amd-instinct-mi350p-pcie-gpus-run-enterprise-ai-on-your.html
