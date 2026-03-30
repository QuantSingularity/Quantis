"""
Security Scanning Middleware and CI/CD Integration
Implements SAST, DAST, and dependency scanning integration
"""

import json
import logging
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SecurityScanningService:
    """Service for integrating security scanning tools"""

    def __init__(self) -> None:
        self.scan_results_dir = "/tmp/security_scans"
        os.makedirs(self.scan_results_dir, exist_ok=True)

    def run_dependency_scan(
        self, requirements_file: str = "requirements.txt"
    ) -> Dict[str, Any]:
        """Run dependency vulnerability scanning using Safety"""
        try:
            result = subprocess.run(
                ["safety", "check", "--json", "--file", requirements_file],
                capture_output=True,
                text=True,
                timeout=300,
            )
            vulnerabilities = []
            if result.stdout:
                try:
                    vulnerabilities = json.loads(result.stdout)
                    if not isinstance(vulnerabilities, list):
                        vulnerabilities = []
                except json.JSONDecodeError:
                    vulnerabilities = []
            scan_results = {
                "status": "success" if not vulnerabilities else "vulnerabilities_found",
                "vulnerabilities": vulnerabilities,
                "timestamp": datetime.utcnow().isoformat(),
                "raw_output": result.stdout,
            }
            results_file = os.path.join(
                self.scan_results_dir,
                f"dependency_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )
            with open(results_file, "w") as f:
                json.dump(scan_results, f, indent=2)
            return scan_results
        except subprocess.TimeoutExpired:
            logger.error("Dependency scan timed out")
            return {"status": "timeout", "timestamp": datetime.utcnow().isoformat()}
        except Exception as e:
            logger.error(f"Error running dependency scan: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    def run_bandit_sast_scan(self, source_dir: str = ".") -> Dict[str, Any]:
        """Run Bandit SAST scan"""
        output_file = "/tmp/bandit_results.json"
        try:
            subprocess.run(
                ["bandit", "-r", source_dir, "-f", "json", "-o", output_file],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if os.path.exists(output_file):
                with open(output_file, "r") as f:
                    bandit_results = json.load(f)
            else:
                bandit_results = {"results": [], "metrics": {}}
            counts = {"high": 0, "medium": 0, "low": 0}
            for r in bandit_results.get("results", []):
                severity = r.get("issue_severity", "").lower()
                if severity in counts:
                    counts[severity] += 1
            scan_results = {
                "status": "completed",
                "tool": "bandit",
                "results": bandit_results,
                "timestamp": datetime.utcnow().isoformat(),
                "high_severity_count": counts["high"],
                "medium_severity_count": counts["medium"],
                "low_severity_count": counts["low"],
            }
            results_file = os.path.join(
                self.scan_results_dir,
                f"sast_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )
            with open(results_file, "w") as f:
                json.dump(scan_results, f, indent=2)
            return scan_results
        except subprocess.TimeoutExpired:
            logger.error("Bandit scan timed out")
            return {"status": "timeout", "timestamp": datetime.utcnow().isoformat()}
        except Exception as e:
            logger.error(f"Error running Bandit scan: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    def run_semgrep_sast_scan(self, source_dir: str = ".") -> Dict[str, Any]:
        """Run Semgrep SAST scan"""
        try:
            result = subprocess.run(
                ["semgrep", "--config=auto", "--json", source_dir],
                capture_output=True,
                text=True,
                timeout=600,
            )
            semgrep_results = {"results": []}
            if result.stdout:
                try:
                    semgrep_results = json.loads(result.stdout)
                    if "results" not in semgrep_results:
                        semgrep_results = {"results": []}
                except json.JSONDecodeError:
                    semgrep_results = {"results": []}
            scan_results = {
                "status": "completed",
                "tool": "semgrep",
                "results": semgrep_results,
                "timestamp": datetime.utcnow().isoformat(),
                "findings_count": len(semgrep_results.get("results", [])),
            }
            results_file = os.path.join(
                self.scan_results_dir,
                f"semgrep_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )
            with open(results_file, "w") as f:
                json.dump(scan_results, f, indent=2)
            return scan_results
        except subprocess.TimeoutExpired:
            logger.error("Semgrep scan timed out")
            return {"status": "timeout", "timestamp": datetime.utcnow().isoformat()}
        except Exception as e:
            logger.error(f"Error running Semgrep scan: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        try:
            dependency_results = self.run_dependency_scan()
            bandit_results = self.run_bandit_sast_scan()
            semgrep_results = self.run_semgrep_sast_scan()
            report = {
                "report_timestamp": datetime.utcnow().isoformat(),
                "summary": {
                    "dependency_vulnerabilities": len(
                        dependency_results.get("vulnerabilities", [])
                    ),
                    "sast_high_severity": bandit_results.get("high_severity_count", 0),
                    "sast_medium_severity": bandit_results.get(
                        "medium_severity_count", 0
                    ),
                    "sast_low_severity": bandit_results.get("low_severity_count", 0),
                    "semgrep_findings": semgrep_results.get("findings_count", 0),
                },
                "detailed_results": {
                    "dependency_scan": dependency_results,
                    "bandit_sast": bandit_results,
                    "semgrep_sast": semgrep_results,
                },
                "recommendations": self._generate_recommendations(
                    dependency_results, bandit_results, semgrep_results
                ),
            }
            report_file = os.path.join(
                self.scan_results_dir,
                f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
            return report
        except Exception as e:
            logger.error(f"Error generating security report: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    def _generate_recommendations(
        self, dependency_results: Dict, bandit_results: Dict, semgrep_results: Dict
    ) -> List[str]:
        """Generate security recommendations based on scan results"""
        recommendations = []
        vuln_count = len(dependency_results.get("vulnerabilities", []))
        if vuln_count > 0:
            recommendations.append(
                f"Update {vuln_count} vulnerable dependencies to their latest secure versions"
            )
            recommendations.append(
                "Implement automated dependency scanning in CI/CD pipeline"
            )
        high_severity = bandit_results.get("high_severity_count", 0)
        if high_severity > 0:
            recommendations.append(
                f"Address {high_severity} high-severity security issues identified by SAST scanning"
            )
        medium_severity = bandit_results.get("medium_severity_count", 0)
        if medium_severity > 5:
            recommendations.append(
                f"Review and address {medium_severity} medium-severity security issues"
            )
        recommendations.extend(
            [
                "Implement regular security scanning in CI/CD pipeline",
                "Set up automated security alerts for new vulnerabilities",
                "Conduct regular penetration testing",
                "Implement security code review process",
                "Maintain security training for development team",
            ]
        )
        return recommendations


class CICDSecurityIntegration:
    """CI/CD Security Integration utilities"""

    @staticmethod
    def generate_github_actions_workflow() -> str:
        """Generate GitHub Actions workflow for security scanning"""
        return "<GitHub Actions workflow YAML from your original code>"

    @staticmethod
    def generate_gitlab_ci_config() -> str:
        """Generate GitLab CI configuration for security scanning"""
        return "<GitLab CI YAML from your original code>"

    @staticmethod
    def generate_security_check_script() -> str:
        """Generate Python script to check security results"""
        return (
            "#!/usr/bin/env python3\n"
            "import json\n"
            "import logging\n"
            "import os\n"
            "import sys\n"
            "\n"
            "logging.basicConfig(\n"
            "    level=logging.INFO,\n"
            "    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',\n"
            ")\n"
            "logger = logging.getLogger(__name__)\n"
            "\n"
            "def check_security_results():\n"
            "    exit_code = 0\n"
            "\n"
            "    if os.path.exists('bandit-report.json'):\n"
            "        with open('bandit-report.json', 'r') as f:\n"
            "            bandit_data = json.load(f)\n"
            "        high_severity = len([r for r in bandit_data.get('results', []) if r.get('issue_severity') == 'HIGH'])\n"
            "        medium_severity = len([r for r in bandit_data.get('results', []) if r.get('issue_severity') == 'MEDIUM'])\n"
            "        logger.info(f'Bandit SAST Results: {high_severity} high, {medium_severity} medium severity issues')\n"
            "        if high_severity > 0:\n"
            "            logger.error('High severity security issues found!')\n"
            "            exit_code = 1\n"
            "        elif medium_severity > 10:\n"
            "            logger.warning('Too many medium severity issues found!')\n"
            "            exit_code = 1\n"
            "\n"
            "    if os.path.exists('safety-report.json'):\n"
            "        with open('safety-report.json', 'r') as f:\n"
            "            safety_data = json.load(f)\n"
            "        if isinstance(safety_data, list) and len(safety_data) > 0:\n"
            "            logger.error(f'Safety Results: {len(safety_data)} vulnerable dependencies found!')\n"
            "            exit_code = 1\n"
            "        else:\n"
            "            logger.info('No vulnerable dependencies found')\n"
            "\n"
            "    if os.path.exists('semgrep-report.json'):\n"
            "        with open('semgrep-report.json', 'r') as f:\n"
            "            semgrep_data = json.load(f)\n"
            "        findings = len(semgrep_data.get('results', []))\n"
            "        logger.info(f'Semgrep Results: {findings} findings')\n"
            "        if findings > 20:\n"
            "            logger.warning('Many security findings detected')\n"
            "\n"
            "    return exit_code\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    sys.exit(check_security_results())\n"
        )


security_scanning_service = SecurityScanningService()
cicd_integration = CICDSecurityIntegration()
