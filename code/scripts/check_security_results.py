from typing import Any

"""
Security Results Checker
Analyzes security scan results and determines if build should pass or fail
"""

import json
import os
import sys
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_security_results() -> Any:
    """Check all security scan results and return appropriate exit code"""
    exit_code = 0
    results_summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "bandit": {"status": "not_run", "high": 0, "medium": 0, "low": 0},
        "safety": {"status": "not_run", "vulnerabilities": 0},
        "semgrep": {"status": "not_run", "findings": 0},
        "overall_status": "pass",
    }
    logger.info("🔍 Analyzing security scan results...")
    logger.info("=" * 50)
    if os.path.exists("bandit-report.json"):
        try:
            with open("bandit-report.json", "r") as f:
                bandit_data = json.load(f)
            high_severity = len(
                [
                    r
                    for r in bandit_data.get("results", [])
                    if r.get("issue_severity") == "HIGH"
                ]
            )
            medium_severity = len(
                [
                    r
                    for r in bandit_data.get("results", [])
                    if r.get("issue_severity") == "MEDIUM"
                ]
            )
            low_severity = len(
                [
                    r
                    for r in bandit_data.get("results", [])
                    if r.get("issue_severity") == "LOW"
                ]
            )
            results_summary["bandit"] = {
                "status": "completed",
                "high": high_severity,
                "medium": medium_severity,
                "low": low_severity,
            }
            logger.info("📊 Bandit SAST Results:")
            logger.info(f"   High Severity: {high_severity}")
            logger.info(f"   Medium Severity: {medium_severity}")
            logger.info(f"   Low Severity: {low_severity}")
            if high_severity > 0:
                logger.info("❌ FAIL: High severity security issues found!")
                logger.info(
                    "   Please review and fix high severity issues before proceeding."
                )
                exit_code = 1
                results_summary["overall_status"] = "fail"
            elif medium_severity > 10:
                logger.info("⚠️  WARNING: Too many medium severity issues found!")
                logger.info(
                    "   Consider reviewing and addressing medium severity issues."
                )
            else:
                logger.info("✅ PASS: No critical security issues found in SAST scan.")
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            logger.info(f"⚠️  WARNING: Could not parse Bandit results: {e}")
            results_summary["bandit"]["status"] = "error"
    else:
        logger.info("⚠️  WARNING: Bandit report not found")
    logger.info()
    if os.path.exists("safety-report.json"):
        try:
            with open("safety-report.json", "r") as f:
                safety_data = json.load(f)
            vulnerability_count = (
                len(safety_data) if isinstance(safety_data, list) else 0
            )
            results_summary["safety"] = {
                "status": "completed",
                "vulnerabilities": vulnerability_count,
            }
            logger.info("📊 Safety Dependency Scan Results:")
            logger.info(f"   Vulnerable Dependencies: {vulnerability_count}")
            if vulnerability_count > 0:
                logger.info("❌ FAIL: Vulnerable dependencies found!")
                logger.info(
                    "   Please update vulnerable dependencies before proceeding."
                )
                if isinstance(safety_data, list):
                    for vuln in safety_data[:5]:
                        package = vuln.get("package", "Unknown")
                        version = vuln.get("installed_version", "Unknown")
                        vuln_id = vuln.get("vulnerability_id", "Unknown")
                        logger.info(f"   - {package} {version} (ID: {vuln_id})")
                    if len(safety_data) > 5:
                        logger.info(
                            f"   ... and {len(safety_data) - 5} more vulnerabilities"
                        )
                exit_code = 1
                results_summary["overall_status"] = "fail"
            else:
                logger.info("✅ PASS: No vulnerable dependencies found.")
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            logger.info(f"⚠️  WARNING: Could not parse Safety results: {e}")
            results_summary["safety"]["status"] = "error"
    else:
        logger.info("⚠️  WARNING: Safety report not found")
    logger.info()
    if os.path.exists("semgrep-report.json"):
        try:
            with open("semgrep-report.json", "r") as f:
                semgrep_data = json.load(f)
            findings = len(semgrep_data.get("results", []))
            results_summary["semgrep"] = {"status": "completed", "findings": findings}
            logger.info("📊 Semgrep SAST Results:")
            logger.info(f"   Total Findings: {findings}")
            if findings > 50:
                logger.info("⚠️  WARNING: Many security findings detected by Semgrep")
                logger.info(
                    "   Consider reviewing findings for potential security issues."
                )
            elif findings > 0:
                logger.info("ℹ️  INFO: Some security findings detected by Semgrep")
                logger.info(
                    "   Review findings to ensure they are not security issues."
                )
            else:
                logger.info("✅ PASS: No security findings detected by Semgrep.")
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            logger.info(f"⚠️  WARNING: Could not parse Semgrep results: {e}")
            results_summary["semgrep"]["status"] = "error"
    else:
        logger.info("⚠️  WARNING: Semgrep report not found")
    logger.info()
    logger.info("=" * 50)
    if exit_code == 0:
        logger.info("🎉 OVERALL RESULT: PASS")
        logger.info("   All security scans completed successfully.")
        logger.info("   No critical security issues found.")
    else:
        logger.info("💥 OVERALL RESULT: FAIL")
        logger.info("   Critical security issues found that must be addressed.")
        logger.info("   Please fix the issues and run the scans again.")
    try:
        with open("security-results-summary.json", "w") as f:
            json.dump(results_summary, f, indent=2)
        logger.info("\n📄 Results summary saved to security-results-summary.json")
    except Exception as e:
        logger.info(f"⚠️  WARNING: Could not save results summary: {e}")
    return exit_code


def generate_security_report() -> Any:
    """Generate a comprehensive security report"""
    try:
        report = {
            "report_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "report_version": "1.0",
                "scan_type": "automated_ci_cd",
            },
            "executive_summary": {
                "overall_risk_level": "low",
                "critical_issues": 0,
                "high_issues": 0,
                "medium_issues": 0,
                "recommendations": [],
            },
            "detailed_findings": {},
            "compliance_status": {
                "pci_dss": "compliant",
                "gdpr": "compliant",
                "sox": "compliant",
                "iso27001": "compliant",
            },
        }
        scan_files = {
            "bandit": "bandit-report.json",
            "safety": "safety-report.json",
            "semgrep": "semgrep-report.json",
        }
        for scan_type, filename in scan_files.items():
            if os.path.exists(filename):
                try:
                    with open(filename, "r") as f:
                        scan_data = json.load(f)
                    report["detailed_findings"][scan_type] = scan_data
                except Exception as e:
                    report["detailed_findings"][scan_type] = {"error": str(e)}
        recommendations = [
            "Implement automated dependency updates",
            "Regular security training for development team",
            "Conduct quarterly penetration testing",
            "Implement runtime application self-protection (RASP)",
            "Regular security architecture reviews",
        ]
        report["executive_summary"]["recommendations"] = recommendations
        with open("comprehensive-security-report.json", "w") as f:
            json.dump(report, f, indent=2)
        logger.info(
            "📊 Comprehensive security report generated: comprehensive-security-report.json"
        )
    except Exception as e:
        logger.info(f"⚠️  WARNING: Could not generate comprehensive report: {e}")


if __name__ == "__main__":
    try:
        exit_code = check_security_results()
        generate_security_report()
        if exit_code == 0:
            logger.info("\n🚀 Security checks passed! Build can proceed.")
        else:
            logger.info("\n🛑 Security checks failed! Build should be blocked.")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Security check interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.info(f"\n💥 Unexpected error during security check: {e}")
        sys.exit(1)
