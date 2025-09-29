"""Analyzer Module for PRODX RAG App

This module provides specialized analysis for PRODX framework issues, focusing on impact prediction for updates and dependency conflicts.
Uses rules-based logic to scan YAML, logs, and requirements for conflicts (e.g., Python module versions across LOBs).
Integrates with RAG for context-aware analysis.

Key Features:
- Dependency conflict detection: Parse requirements.txt/YAML for version mismatches.
- Update impact simulation: Check PRODX patches against existing jobs (e.g., schema changes).
- Suggestion generation: Propose fixes like isolation or rollbacks.
- Business Context: Analyzes infrastructure deployments and capability updates to predict Airflow failures, reducing 1-3 week delays.

Usage:
  from src.analyzer.analyzer import ImpactAnalyzer
  analyzer = ImpactAnalyzer()
  impact_report = analyzer.analyze_dependency_update('cryptography', '3.2.0', '3.4.8', lob='009')
  # impact_report: Dict with conflicts, affected_lobs, suggestions

Requires: pyyaml, json. Optional: difflib for diffs.
"""

import yaml
import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import defaultdict


class ImpactAnalyzer:
    """Analyzes impacts of updates and dependencies in PRODX framework."""

    def __init__(self, data_dir: str = 'data/'):
        """
        Initialize analyzer with access to sample data.
        
        Args:
            data_dir: Directory containing sample YAML, logs, JIRA.
        """
        self.data_dir = Path(data_dir)
        # Known LOB requirements (simulated from docs)
        self.lob_requirements = self._load_lob_reqs()
        # Known PRODX update patterns
        self.update_patterns = self._load_update_patterns()

    def _load_lob_reqs(self) -> Dict[str, Dict[str, str]]:
        """Load simulated LOB-specific requirements from data."""
        # Sample from PRODX doc: LOB 009 needs cryptography >=3.4.8, LOB 003 <3.3.0
        return {
            '009': {'cryptography': '>=3.4.8', 'pandas': '==1.5.3'},
            '003': {'cryptography': '<3.3.0', 'pandas': '==1.4.0'},
            '017': {'cryptography': '==3.2.0', 'boto3': '==1.28.0'}
        }

    def _load_update_patterns(self) -> Dict[str, Any]:
        """Load patterns for PRODX updates (e.g., schema changes)."""
        # Simulated from JIRA and logs
        return {
            'schema_evolution': {
                'pattern': r'ValidationException.*schema',
                'impact': 'Breaks Airflow jobs if not backward compatible',
                'fix': 'Add explicit casting in YAML or enable evolution flag'
            },
            'athena_parser': {
                'pattern': r'Athena parser.*implicit casting',
                'impact': 'Type mismatches in Glue/Redshift loads',
                'fix': 'Update YAML schema to explicit types'
            }
        }

    def analyze_dependency_update(self, package: str, current_version: str, new_version: str, 
                                  lob: Optional[str] = None, requirements_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze impact of updating a Python package version.
        
        Args:
            package: e.g., 'cryptography'.
            current_version, new_version: Version strings.
            lob: Specific LOB (e.g., '009') or None for all.
            requirements_file: Path to requirements.txt (optional).
        
        Returns:
            Report: {'conflicts': List, 'affected_lobs': List, 'suggestions': List}.
        """
        report = {
            'package': package,
            'current_version': current_version,
            'new_version': new_version,
            'conflicts': [],
            'affected_lobs': [],
            'suggestions': [],
            'risk_level': 'LOW'  # LOW, MEDIUM, HIGH
        }
        
        target_lobs = [lob] if lob else self.lob_requirements.keys()
        
        for l in target_lobs:
            req = self.lob_requirements.get(l, {}).get(package, '')
            if self._has_conflict(req, new_version):
                conflict = f"LOB {l}: Requirement '{req}' conflicts with {new_version}"
                report['conflicts'].append(conflict)
                report['affected_lobs'].append(l)
                report['risk_level'] = 'HIGH' if 'cryptography' == package else 'MEDIUM'
        
        if requirements_file:
            # Parse and check file
            reqs = self._parse_requirements(requirements_file)
            if package in reqs and self._has_conflict(reqs[package], new_version):
                report['conflicts'].append(f"File {requirements_file} conflicts")
        
        # Suggestions
        if report['conflicts']:
            report['suggestions'] = [
                'Use virtualenvs or conda per LOB in EMR Studio.',
                'Pin versions in YAML wrappers or Lambda layers.',
                'Run pip check pre-deployment.',
                'For cryptography: Isolate in Docker containers per service domain.'
            ]
        else:
            report['suggestions'] = ['Update safe; validate with test DAGs.']
        
        return report

    def _has_conflict(self, req_str: str, version: str) -> bool:
        """Check if requirement conflicts with version."""
        if not req_str:
            return False
        # Simple regex for version comparison (in production, use packaging library)
        if '>=' in req_str and re.match(r'^\d+\.\d+\.\d+$', version):
            # Parse rough version
            req_v = float(re.search(r'>=(\d+\.\d+)', req_str).group(1)) if '>=' in req_str else 0
            new_v = float(version.replace('.', '')[:3])  # Rough
            return new_v < req_v
        elif '<' in req_str:
            max_v = float(re.search(r'<(\d+\.\d+)', req_str).group(1))
            new_v = float(version.replace('.', '')[:3])
            return new_v >= max_v
        return False

    def _parse_requirements(self, file_path: str) -> Dict[str, str]:
        """Parse requirements.txt."""
        reqs = {}
        path = self.data_dir / file_path
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if '==' in line or '>=' in line or '<' in line:
                    pkg, ver = line.split('==', 1) if '==' in line else line.split(' ', 1)
                    reqs[pkg] = ver
        return reqs

    def analyze_log_for_impact(self, log_file: str, update_type: str = 'general') -> Dict[str, Any]:
        """
        Analyze Airflow log for update impacts using patterns.
        
        Args:
            log_file: e.g., 'sample_airflow_log.txt'.
            update_type: e.g., 'schema_evolution'.
        
        Returns:
            Report: {'matches': List, 'impacts': List, 'fixes': List}.
        """
        report = {
            'log_file': log_file,
            'update_type': update_type,
            'matches': [],
            'impacts': [],
            'fixes': [],
            'severity': 'INFO'
        }
        
        path = self.data_dir / log_file
        with open(path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        pattern_info = self.update_patterns.get(update_type, {})
        if pattern_info and re.search(pattern_info['pattern'], log_content):
            report['matches'].append(pattern_info['pattern'])
            report['impacts'].append(pattern_info['impact'])
            report['fixes'].append(pattern_info['fix'])
            report['severity'] = 'ERROR'
        else:
            report['impacts'].append('No specific impact detected.')
        
        # Scan for dependency errors
        if 'ImportError' in log_content or 'module' in log_content.lower():
            report['impacts'].append('Potential dependency conflict in EMR/ComputeHub.')
            report['fixes'].append('Check Python versions across LOBs; use --user install.')
        
        return report

    def analyze_yaml_update(self, yaml_file: str, patch_description: str) -> Dict[str, Any]:
        """Analyze YAML wrapper for impact from PRODX patch."""
        path = self.data_dir / yaml_file
        with open(path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        report = {
            'yaml_file': yaml_file,
            'patch': patch_description,
            'potential_breaks': [],
            'recommendations': []
        }
        
        # Example: Check schema for type changes
        schema = yaml_data.get('ingest_config', {}).get('schema', {})
        if schema and 'amount' in schema.get('properties', {}):
            if schema['properties']['amount']['type'] == 'number':
                if 'casting' not in patch_description:
                    report['potential_breaks'].append('Schema type mismatch if parser changes.')
                    report['recommendations'].append('Add explicit cast in schema.')
        
        # Dependency in YAML (simulated)
        if 'python_deps' in yaml_data:
            for dep, ver in yaml_data['python_deps'].items():
                report['recommendations'].append(f'Validate {dep}=={ver} with new PRODX.')
        
        return report

    def get_risk_summary(self) -> Dict[str, int]:
        """Summary of known risks from data."""
        return {
            'dependency_conflicts': len(self.lob_requirements),
            'update_patterns': len(self.update_patterns)
        }


# Example usage (for testing)
if __name__ == "__main__":
    analyzer = ImpactAnalyzer()
    # Dependency analysis
    report = analyzer.analyze_dependency_update('cryptography', '3.2.0', '3.4.8', lob='009')
    print("Dependency Report:", json.dumps(report, indent=2))
    # Log analysis
    log_report = analyzer.analyze_log_for_impact('sample_airflow_log.txt', 'schema_evolution')
    print("Log Impact:", json.dumps(log_report, indent=2))
    # Risk summary
    summary = analyzer.get_risk_summary()
    print("Risk Summary:", summary)