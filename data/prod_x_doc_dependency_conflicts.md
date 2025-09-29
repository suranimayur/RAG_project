# PRODX Framework Documentation: Python Dependency Management in Infrastructure Deployments

## Overview
PRODX framework relies on shared AWS infrastructure (EMR, EC2, Lambda) for batch processing across multiple Lines of Business (LOBs). Python dependencies are managed via requirements.txt and pip in EMR Studio (ComputeHub Jupyter notebooks and bash terminals). This doc outlines best practices, common issues, and impact analysis for updates.

### Key Components
- **Service Domains**: Organized by blocks (e.g., block-a for LOB 009 - high-volume transactions; block-b for LOB 003 - compliance reporting).
- **Groups**: API-Ingestion, Group-Ingestion, Group-Transform, Egress, Orchestration.
- **AWS Integration**: EMR for compute, Glue for ETL, Lambda for orchestration, RDS/Redshift for storage.
- **YAML Wrappers**: Tenants define ingestion logic; e.g., schema for JSON/CSV/DAT files from S3 inbound to curated.

### Python Dependencies
PRODX uses Python 3.8+ base image on EMR. Common libs: boto3, pandas, cryptography (for encryption in banking), requests, numpy.

#### Version Pinning Example (requirements.txt)
```
boto3==1.28.0
pandas==1.5.3
cryptography==3.4.8  # Latest for LOB 009 (supports AES-256-GCM)
numpy==1.24.3
```

#### Conflicts and Impacts
- **Issue Example**: LOB 009 (block-a) requires `cryptography>=3.4.8` for advanced encryption in transaction processing. LOB 003 (block-b) uses `cryptography<3.3.0` for legacy compliance (avoids breaking changes in key derivation).
  - **Symptom**: Shared EMR cluster upgrade to 3.4.8 causes import errors in block-b jobs: `ImportError: cannot import name 'Fernet' from partially initialized module`.
  - **Business Impact**: Delays in compliance reports (LOB 003), potential regulatory violations. Affects Group-Transform DAGs reading from curated S3.
  - **Resolution Time**: 1-3 weeks - Infrastructure team deploys isolated envs (conda/virtualenv per LOB); capability team validates PRODX patches.
- **Simulation Scenario**: Infrastructure deployment updates `cryptography` globally. Tenant queries: "Impact of updating cryptography to 4.0 on LOB 009 and 003?"
  - **Expected RAG Response**: Detect conflict via dependency scan. Suggest: Use Docker containers per LOB or pip install --user for notebooks. Reference JIRA PRODX-456.

### Best Practices
1. **Pre-Deployment Validation**: Use `pip check` and `poetry lock` to detect conflicts before EMR spin-up.
2. **Isolation**: 
   - Virtualenvs in EMR bash: `python -m venv /tmp/lob-009-env && source /tmp/lob-009-env/bin/activate && pip install -r requirements_lob009.txt`.
   - Lambda layers for domain-specific deps.
3. **Schema Evolution**: For Glue/Athena, enable backward compatibility in YAML: `schema_evolution: true`.
4. **Monitoring**: CloudWatch alarms for import errors in Airflow logs.
5. **Update Process**: Capability team releases patches via YAML flags (e.g., `enable_new_cryptography: false`).

### Related Resources
- AWS Docs: [EMR Python Dependencies](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-python.html)
- Airflow Integration: DAGs trigger Lambda for dep validation.
- Ticketing: Log issues in JIRA under "Dependency-Conflict" label.

This doc is ingested into RAG for query resolution, e.g., analyzing update impacts to reduce tenant wait times.